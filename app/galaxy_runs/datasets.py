"""/v1/datasets: file bytes between the platform server and the thread's Galaxy history.

Neither direction writes a temporary file or holds a whole file in memory. An upload is parsed as it
arrives and forwarded to Galaxy's /api/tools/fetch as it is parsed; a download is passed through from
Galaxy chunk by chunk, with the Range header, so previews can read one region of a large file.
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
from urllib.parse import quote

import httpx
from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from app.galaxy_runs.service import GalaxyRuns, GalaxyRunsError, _NotInGalaxy

# The spec leaves the cap open ("2 GB?"); configurable until it's decided.
UPLOAD_MAX_BYTES = int(os.getenv("GALAXY_UPLOAD_MAX_BYTES", str(2 * 1024 ** 3)))
INDEX_SUFFIX = {"bam_index": "bai", "tabix_index": "tbi"}
PASS_THROUGH = ("content-type", "content-length", "content-range", "accept-ranges", "content-encoding",
                "last-modified", "etag")
METADATA_FILE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

HttpFactory = Callable[[], httpx.AsyncClient]


def _default_http() -> httpx.AsyncClient:
    # No read timeout: a large upload or download can take a long time between Galaxy's replies.
    return httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None, write=None))


class _TooLarge(Exception):
    pass


class _Form:
    """Streaming multipart parser state: text fields, and the one file part as chunks."""

    def __init__(self, boundary: bytes):
        self.fields: Dict[str, str] = {}
        self.chunks: List[bytes] = []
        self.file_name: Optional[str] = None
        self.file_started = self.file_done = self.finished = False
        self._headers: Dict[str, bytes] = {}
        self._field = b""
        self._value = b""
        self._part: Optional[str] = None
        self._is_file = False
        self._text: List[bytes] = []
        self.parser = MultipartParser(boundary, {
            "on_part_begin": self._begin, "on_header_field": self._hfield, "on_header_value": self._hvalue,
            "on_header_end": self._hend, "on_headers_finished": self._hdone, "on_part_data": self._data,
            "on_part_end": self._end, "on_end": self._finish,
        })

    def _begin(self):
        self._headers, self._field, self._value, self._text = {}, b"", b"", []

    def _hfield(self, data, start, end):
        self._field += data[start:end]

    def _hvalue(self, data, start, end):
        self._value += data[start:end]

    def _hend(self):
        self._headers[self._field.decode("latin-1").lower()] = self._value
        self._field, self._value = b"", b""

    def _hdone(self):
        _, params = parse_options_header(self._headers.get("content-disposition", b""))
        self._part = (params.get(b"name") or b"").decode("utf-8", "replace")
        self._is_file = b"filename" in params
        if self._is_file:
            if self.file_started:
                raise GalaxyRunsError(400, "Send one file per upload")
            if self._part != "file":
                raise GalaxyRunsError(400, "The file part must be named 'file'")
            self.file_name = params[b"filename"].decode("utf-8", "replace") or "upload"
            self.file_started = True

    def _data(self, data, start, end):
        if self._is_file:
            self.chunks.append(bytes(data[start:end]))
        else:
            self._text.append(bytes(data[start:end]))

    def _end(self):
        if self._is_file:
            self.file_done = True
        elif self._part:
            self.fields[self._part] = b"".join(self._text).decode("utf-8", "replace")

    def _finish(self):
        self.finished = True


async def upload(runs: GalaxyRuns, content_type: Optional[str], content_length: Optional[str],
                 stream: AsyncIterator[bytes], http: Optional[HttpFactory] = None) -> Dict[str, Any]:
    """POST /v1/datasets/upload: multipart { thread_id, file, format? } into the thread's history.

    thread_id must come before the file, so the history exists before any bytes are sent to Galaxy.
    Returns at once with the dataset; the upload itself runs as a Galaxy job (queued -> ok).
    """
    mime, params = parse_options_header((content_type or "").encode("latin-1"))
    if mime != b"multipart/form-data" or not params.get(b"boundary"):
        raise GalaxyRunsError(400, "Send multipart/form-data with thread_id, file and optional format")
    if content_length and content_length.isdigit() and int(content_length) > UPLOAD_MAX_BYTES + 1_000_000:
        raise GalaxyRunsError(413, f"Upload is over the {UPLOAD_MAX_BYTES} byte cap")

    form = _Form(params[b"boundary"])
    incoming = stream.__aiter__()

    async def read_more() -> bool:
        try:
            chunk = await incoming.__anext__()
        except StopAsyncIteration:
            return False
        form.parser.write(chunk)
        return True

    # Read up to the file part's headers: thread_id (and maybe format) arrive first.
    while not form.file_started:
        if not await read_more():
            raise GalaxyRunsError(400, "No file in the upload")
    thread_id = (form.fields.get("thread_id") or "").strip()
    if not thread_id or len(thread_id) > 256:
        raise GalaxyRunsError(400, "thread_id must come before the file")
    history_id = await runs.thread_history(thread_id)

    boundary = uuid.uuid4().hex
    sent = 0

    async def galaxy_body() -> AsyncIterator[bytes]:
        nonlocal sent
        filename = form.file_name.replace('"', "").replace("\r", "").replace("\n", "")
        yield (f'--{boundary}\r\nContent-Disposition: form-data; name="files_0|file_data"; '
               f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n').encode("utf-8")
        while True:
            while form.chunks:
                chunk = form.chunks.pop(0)
                sent += len(chunk)
                if sent > UPLOAD_MAX_BYTES:
                    raise _TooLarge()
                yield chunk
            if form.file_done:
                break
            if not await read_more():
                raise GalaxyRunsError(400, "The upload ended before the file did")
        # Fields after the file (format may come last) are read before closing the request to Galaxy.
        while not form.finished and await read_more():
            pass
        fmt = (form.fields.get("format") or "").strip() or "auto"
        targets = [{"destination": {"type": "hdas"},
                    "elements": [{"src": "files", "name": form.file_name, "ext": fmt}]}]
        yield (f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="history_id"\r\n\r\n{history_id}'
               f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="targets"\r\n\r\n{json.dumps(targets)}'
               f'\r\n--{boundary}--\r\n').encode("utf-8")

    async with (http or _default_http)() as client:
        try:
            response = await client.post(
                f"{runs.galaxy_url}/api/tools/fetch", content=galaxy_body(),
                headers={"x-api-key": runs.client.user_api_key,
                         "Content-Type": f"multipart/form-data; boundary={boundary}"})
        except _TooLarge:
            raise GalaxyRunsError(413, f"Upload is over the {UPLOAD_MAX_BYTES} byte cap")
        except httpx.HTTPError as e:
            raise GalaxyRunsError(502, f"Galaxy failed during the upload. Safe to retry. ({type(e).__name__})")
    if response.status_code >= 400:
        raise _galaxy_error(response.status_code, response.text)

    outputs = (response.json() or {}).get("outputs") or []
    if not outputs:
        raise GalaxyRunsError(502, "Galaxy accepted the upload but created no dataset")
    out = outputs[0]
    ext = out.get("file_ext") or out.get("extension")
    return {
        "id": out["id"],
        "name": out.get("name") or form.file_name,
        "format": ext if ext and ext != "auto" else "auto",
        "size": sent,
        "state": out.get("state") or "queued",
        "history_id": history_id,
        "created_at": _iso(out.get("create_time")),
    }


async def download(runs: GalaxyRuns, dataset_id: str, range_header: Optional[str],
                   metadata_file: Optional[str], http: Optional[HttpFactory] = None) -> StreamingResponse:
    """GET /v1/datasets/{id}/download: the bytes, streamed, as an attachment; Range passed through (206).

    ?metadata_file=bam_index (or tabix_index) returns that index instead, for genome browser previews.
    """
    if metadata_file is not None and not METADATA_FILE.match(metadata_file):
        raise GalaxyRunsError(400, "metadata_file must be a Galaxy metadata name such as bam_index")
    try:
        dataset = await runs._call(runs.gi.datasets.show_dataset, dataset_id)
    except _NotInGalaxy:
        raise GalaxyRunsError(404, "No such dataset for this user")

    ext = dataset.get("extension") or dataset.get("file_ext") or "data"
    name = dataset.get("name") or dataset_id
    if metadata_file:
        url = f"{runs.galaxy_url}/api/datasets/{quote(dataset_id)}/metadata_file?metadata_file={quote(metadata_file)}"
        filename = f"{name}.{INDEX_SUFFIX.get(metadata_file, metadata_file)}"
    else:
        url = f"{runs.galaxy_url}/api/datasets/{quote(dataset_id)}/display?to_ext={quote(ext)}"
        filename = name if name.endswith(f".{ext}") else f"{name}.{ext}"

    headers = {"x-api-key": runs.client.user_api_key}
    if range_header:
        headers["Range"] = range_header
    client = (http or _default_http)()
    try:
        upstream = await client.send(client.build_request("GET", url, headers=headers), stream=True)
    except httpx.HTTPError as e:
        await client.aclose()
        raise GalaxyRunsError(502, f"Galaxy failed. Safe to retry. ({type(e).__name__})")
    if upstream.status_code >= 400 and upstream.status_code != 416:
        body = (await upstream.aread()).decode("utf-8", "replace")
        await upstream.aclose()
        await client.aclose()
        raise _galaxy_error(upstream.status_code, body)

    out_headers = {k: v for k, v in upstream.headers.items() if k.lower() in PASS_THROUGH}
    out_headers.setdefault("accept-ranges", "bytes")
    out_headers["content-disposition"] = _attachment(filename)

    async def close():
        await upstream.aclose()
        await client.aclose()

    return StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code,
                             headers=out_headers, background=BackgroundTask(close))


def _galaxy_error(code: int, body: str) -> GalaxyRunsError:
    try:
        parsed = json.loads(body)
        message = parsed.get("err_msg") or parsed.get("detail") or body
    except (ValueError, AttributeError):
        message = body[:500]
    if code == 401:
        return GalaxyRunsError(401, "Galaxy did not accept the user's key")
    if code in (403, 404):
        return GalaxyRunsError(404, "No such dataset for this user")
    if code == 413:
        return GalaxyRunsError(413, "Galaxy refused the upload as too large")
    if code in (400, 422):
        return GalaxyRunsError(422, message)
    return GalaxyRunsError(502, "Galaxy failed. Safe to retry.")


def _attachment(filename: str) -> str:
    ascii_name = filename.encode("ascii", "replace").decode("ascii").replace('"', "").replace("?", "_")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


def _iso(value: Optional[str]) -> str:
    """Galaxy's naive UTC timestamps as ISO 8601 UTC, like the rest of the API."""
    if not value:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        return datetime.fromisoformat(value.replace("Z", "")).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return value
