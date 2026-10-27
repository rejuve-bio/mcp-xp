from sys import path
path.append(".")

from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import unquote

import pytest
from bioblend import ConnectionError as GalaxyConnectionError

from app.galaxy_runs import links, status as st
from app.galaxy_runs.service import GalaxyRuns, GalaxyRunsError
from app.galaxy_runs.tool_model import describe_inputs, input_types, shape_inputs, type_of


# ---------------------------------------------------------------- a fastp-like tool model

def _data(name, label, optional=False, options=None):
    return {"type": "data", "name": name, "label": label, "optional": optional,
            "extensions": ["fastqsanger.gz"], "options": options or {"hda": [], "hdca": []}}


TOOL_MODEL = {
    "id": "toolshed.g2.bx.psu.edu/repos/iuc/fastp/fastp/1.3.7+galaxy0", "version": "1.3.7+galaxy0", "name": "fastp",
    "errors": {},
    "inputs": [
        {"type": "conditional", "name": "single_paired",
         "test_param": {"type": "select", "name": "single_paired_selector", "label": "Single-end or paired reads",
                        "value": "single", "options": [["Single", "single", True], ["Paired", "paired", False]]},
         "cases": [
             {"value": "single", "inputs": [_data("in1", "Input 1", options={"hda": [{"id": "f2db", "name": "reads_R1.fastq.gz", "hid": 1}], "hdca": []})]},
             {"value": "paired", "inputs": [_data("in1", "Input 1"), _data("in2", "Input 2")]},
         ]},
        {"type": "section", "name": "filter_options", "title": "Filter options", "inputs": [
            {"type": "integer", "name": "length_required", "label": "Length required", "optional": False, "value": "15"},
            {"type": "text", "name": "adapter", "label": "Adapter", "optional": True, "value": ""},
        ]},
        {"type": "repeat", "name": "rep", "title": "Extra", "min": 0, "max": None,
         "inputs": [{"type": "text", "name": "x", "label": "X", "optional": False, "value": ""}], "cache": []},
        {"type": "hidden", "name": "secret", "value": "1"},
    ],
}


def test_describe_trimmed_keeps_required_choices_and_files():
    d = describe_inputs(TOOL_MODEL)
    paths = {e["path"]: e for e in d["inputs"]}
    assert "single_paired|single_paired_selector" in paths
    assert paths["single_paired|single_paired_selector"]["options"] == ["single", "paired"]
    # in1 is in every branch, in2 only in "paired"
    assert "when" not in paths["single_paired|in1"]
    assert paths["single_paired|in2"]["when"] == {"single_paired|single_paired_selector": "paired"}
    assert paths["single_paired|in1"]["matches"] == [{"src": "hda", "id": "f2db", "name": "reads_R1.fastq.gz", "hid": 1}]
    # settings with a default are left out of the trimmed description
    assert "filter_options|length_required" not in paths
    assert "secret" not in paths
    assert paths["rep"]["type"] == "repeat"


def test_describe_full_keeps_every_setting_with_its_section():
    d = describe_inputs(TOOL_MODEL, full=True)
    paths = {e["path"]: e for e in d["inputs"]}
    assert paths["filter_options|length_required"]["section"] == "Filter options"
    assert paths["filter_options|adapter"]["optional"] is True
    assert d["all_count"] == 5  # selector, in1, in2 and two filter options; hidden and repeat not counted


def test_input_types_cover_repeat_blocks_and_batch_shaping():
    types = input_types(TOOL_MODEL)
    assert type_of(types, "rep_3|x")["type"] == "text"
    shaped = shape_inputs({
        "single_paired|in1": {"src": "hdca", "id": "c1"},
        "single_paired|in2": {"src": "hda", "id": "d2"},
        "filter_options|length_required": 30,
    }, types)
    assert shaped["single_paired|in1"] == {"batch": True, "values": [{"src": "hdca", "id": "c1"}]}
    assert shaped["single_paired|in2"] == {"values": [{"src": "hda", "id": "d2"}]}
    assert shaped["filter_options|length_required"] == 30


# ---------------------------------------------------------------- status

@pytest.mark.parametrize("states,expected", [
    (["new"], ("queued", "new")),
    (["running"], ("running", "running")),
    (["paused"], ("running", "paused")),
    (["ok"], ("completed", "ok")),
    (["ok", "running", "queued"], ("running", "running")),
    (["ok", "queued"], ("running", "queued")),
    (["ok", "error"], ("failed", "error")),
    (["ok", "error", "running"], ("running", "running")),  # not final until every job is
    (["ok", "deleted"], ("cancelled", "deleted")),
    (["skipped"], ("completed", "skipped")),
])
def test_tool_run_status(states, expected):
    assert st.tool_run_status(states) == expected


@pytest.mark.parametrize("state,counts,expected", [
    ("new", {}, "queued"),
    ("ready", {"ok": 2}, "queued"),
    ("scheduled", {"ok": 1, "queued": 1}, "queued"),
    ("scheduled", {"running": 1}, "running"),
    ("completed", {"ok": 3}, "completed"),
    ("completed", {"ok": 2, "error": 1}, "failed"),
    ("failed", {}, "failed"),
    ("cancelling", {"running": 1}, "cancelled"),
])
def test_invocation_run_status(state, counts, expected):
    assert st.invocation_run_status(state, counts)[0] == expected


def test_paused_job_shows_as_paused():
    assert st.invocation_run_status("scheduled", {"paused": 1}) == ("running", "paused")


def test_a_run_never_goes_back():
    assert st.no_going_back("running", "queued") == "running"
    assert st.no_going_back("queued", "running") == "running"
    assert st.no_going_back(None, "queued") == "queued"


# ---------------------------------------------------------------- links

def test_tool_form_link_encodes_everything_and_lists_what_it_cannot_carry():
    url, not_carried = links.tool_form_url(
        "https://galaxy.example.org/", "toolshed.g2.bx.psu.edu/repos/iuc/fastp/fastp/1.3.7+galaxy0", "1.3.7+galaxy0", "h1",
        {"single_paired|in1": {"src": "hda", "id": "f2db"},
         "reads": {"src": "hdca", "id": "c1"},
         "many": [{"src": "hda", "id": "a"}, {"src": "hda", "id": "b"}],
         "rep_0|x": "first", "rep_1|x": "second",
         "filter_options|length_required": 30, "title": "a b+c"})
    assert url.startswith("https://galaxy.example.org/?tool_id=toolshed.g2.bx.psu.edu%2Frepos")
    assert "%2Bgalaxy0" in url and "single_paired%7Cin1=f2db" in url
    assert "rep_0%7Cx=first" in url and "title=a%20b%2Bc" in url
    assert "+" not in url.split("?", 1)[1]
    assert sorted(not_carried) == ["many", "reads", "rep_1|x"]
    assert unquote(url).count("history_id=h1") == 1


# ---------------------------------------------------------------- service with a fake Galaxy

class MemoryStore:
    def __init__(self):
        self.histories, self.runs, self.requests = {}, {}, {}

    async def get_thread_history(self, u, t):
        return self.histories.get((u, t))

    async def claim_thread_history(self, u, t, h):
        return self.histories.setdefault((u, t), h)

    async def get_run(self, u, r):
        run = self.runs.get((u, r))
        return dict(run) if run else None

    async def save_run(self, u, run):
        self.runs[(u, run["id"])] = dict(run)

    async def list_runs(self, u, t):
        runs = [dict(r) for (owner, _), r in self.runs.items() if owner == u and r["thread_id"] == t]
        return sorted(runs, key=lambda r: r["created_at"], reverse=True)

    async def claim_request(self, u, c, r):
        if (u, c) in self.requests:
            return False
        self.requests[(u, c)] = r
        return True

    async def get_request(self, u, c):
        return self.requests.get((u, c))

    async def release_request(self, u, c):
        self.requests.pop((u, c), None)


def _fake_gi():
    gi = MagicMock()
    gi.url = "https://galaxy.example.org/api"
    gi.histories.create_history.return_value = {"id": "hist1"}
    gi.tools.build.return_value = TOOL_MODEL
    gi.make_post_request.return_value = {
        "jobs": [{"id": "j1"}, {"id": "j2"}],
        "outputs": [{"id": "o1", "name": "fastp on S1", "file_ext": "fastqsanger.gz"},
                    {"id": "o2", "name": "fastp on S2", "file_ext": "fastqsanger.gz"}],
    }
    gi.datasets.show_dataset.return_value = {"id": "f2db", "name": "reads_R1", "extension": "fastqsanger.gz", "state": "ok"}
    gi.jobs.show_job.side_effect = lambda job_id, full_details=False: {
        "j1": {"id": "j1", "state": "ok", "outputs": {"out1": {"id": "o1", "src": "hda"}}},
        "j2": {"id": "j2", "state": "running", "outputs": {"out1": {"id": "o2", "src": "hda"}}},
    }[job_id]
    return gi


def _service(gi, store=None, user="alice"):
    client = SimpleNamespace(gi_client=gi, galaxy_url="https://galaxy.example.org", user_api_key="user-key")
    return GalaxyRuns(client, user, store or MemoryStore(), cache=None)


TARGET = {"type": "tool", "id": TOOL_MODEL["id"], "version": "1.3.7+galaxy0"}
INPUTS = {"single_paired|single_paired_selector": "single", "single_paired|in1": {"src": "hdca", "id": "c1"}}


async def test_start_run_returns_at_once_and_a_retry_starts_nothing():
    gi = _fake_gi()
    svc = _service(gi)
    run, repeat = await svc.start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})
    assert repeat is False
    assert run["status"] == "queued" and run["history_id"] == "hist1"
    assert run["progress"] == {"jobs_done": 0, "jobs_total": 2, "steps_done": 0, "steps_total": 1}
    payload = gi.make_post_request.call_args.kwargs["payload"]
    assert payload["inputs"]["single_paired|in1"] == {"batch": True, "values": [{"src": "hdca", "id": "c1"}]}
    assert payload["history_id"] == "hist1" and payload["tool_version"] == "1.3.7+galaxy0"

    again, repeat = await svc.start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})
    assert repeat is True and again["id"] == run["id"]
    assert gi.make_post_request.call_count == 1


async def test_failed_start_can_be_retried():
    gi = _fake_gi()
    gi.make_post_request.side_effect = GalaxyConnectionError("bad", body='{"err_msg": "Invalid input", "err_code": 400008}', status_code=400)
    svc = _service(gi)
    with pytest.raises(GalaxyRunsError) as e:
        await svc.start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})
    assert e.value.status_code == 422 and e.value.detail == "Invalid input"
    gi.make_post_request.side_effect = None
    gi.make_post_request.return_value = {"jobs": [{"id": "j1"}], "outputs": []}
    run, repeat = await svc.start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})
    assert repeat is False and run["status"] == "queued"


async def test_read_run_reports_progress_outputs_and_failure():
    gi = _fake_gi()
    svc = _service(gi)
    run, _ = await svc.start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})

    running = await svc.read_run(run["id"])
    assert running["status"] == "running"
    assert running["progress"]["jobs_done"] == 1
    assert running["outputs"] == [{"id": "o1", "name": "fastp on S1", "format": "fastqsanger.gz"}]

    jobs = {"j1": {"state": "ok", "outputs": {"out1": {"id": "o1"}}},
            "j2": {"state": "error", "outputs": {}, "tool_stderr": "line 1\nEXITING because of fatal ERROR"}}
    gi.jobs.show_job.side_effect = lambda job_id, full_details=False: {"id": job_id, **jobs[job_id]}
    failed = await svc.read_run(run["id"])
    assert failed["status"] == "failed"
    assert failed["error"] == {"step": None, "message": "line 1\nEXITING because of fatal ERROR"}

    calls = gi.jobs.show_job.call_count
    assert (await svc.read_run(run["id"]))["status"] == "failed"
    assert gi.jobs.show_job.call_count == calls  # a final run is not read from Galaxy again


async def test_a_job_galaxy_no_longer_has_ends_the_run_as_cancelled():
    gi = _fake_gi()
    svc = _service(gi)
    run, _ = await svc.start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})
    gi.jobs.show_job.side_effect = GalaxyConnectionError("gone", body="{}", status_code=404)
    removed = await svc.read_run(run["id"])
    assert removed["status"] == "cancelled"
    assert removed["error"] == {"step": None, "message": "Removed in Galaxy"}


async def test_another_users_run_is_not_found():
    store = MemoryStore()
    run, _ = await _service(_fake_gi(), store, "alice").start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})
    with pytest.raises(GalaxyRunsError) as e:
        await _service(_fake_gi(), store, "bob").read_run(run["id"])
    assert e.value.status_code == 404


async def test_cancel_asks_galaxy_and_returns_the_run():
    gi = _fake_gi()
    svc = _service(gi)
    run, _ = await svc.start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})
    cancelled = await svc.cancel_run(run["id"])
    assert gi.jobs.cancel_job.call_count == 2
    assert cancelled["id"] == run["id"]


async def test_link_opens_the_form_before_a_run_and_rerun_after():
    gi = _fake_gi()
    svc = _service(gi)
    first = await svc.link("t1", "prop_8", TARGET, INPUTS, {})
    assert first["kind"] == "tool_form" and first["not_carried"] == ["single_paired|in1"]
    await svc.start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})
    second = await svc.link("t1", "prop_8", TARGET, INPUTS, {})
    assert second == {"url": "https://galaxy.example.org/?job_id=j2", "kind": "rerun", "not_carried": []}


async def test_validate_reports_unknown_and_missing_inputs():
    gi = _fake_gi()
    svc = _service(gi)
    result = await svc.validate("t1", TARGET, {"single_paired|single_paired_selector": "paired",
                                                "single_paired|in1": {"src": "hda", "id": "a"}, "nope": 1}, {})
    assert result["valid"] is False
    assert result["errors"] == {"nope": "Unknown input", "single_paired|in2": "Required"}


async def test_thread_files_without_a_history_is_empty():
    svc = _service(_fake_gi())
    assert await svc.thread_files("t-new") == {"items": [], "next_cursor": None}


# ---------------------------------------------------------------- routes

def _client(svc):
    from fastapi import FastAPI
    from fastapi.exceptions import RequestValidationError
    from fastapi.testclient import TestClient
    from app.api.endpoints import galaxy_runs as routes
    from app.main import validation_exception_handler

    app = FastAPI()
    app.include_router(routes.router, prefix="/v1/galaxy")
    app.include_router(routes.datasets_router, prefix="/v1/datasets")
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.dependency_overrides[routes.galaxy_runs] = lambda: svc
    return TestClient(app)


def test_invalid_body_is_400_and_a_repeat_is_200():
    client = _client(_service(_fake_gi()))
    body = {"thread_id": "t1", "proposal_id": "prop_8", "client_request_id": "req-1", "target": TARGET, "inputs": INPUTS}

    assert client.post("/v1/galaxy/runs", json={**body, "target": {"type": "dataset", "id": "x"}}).status_code == 400
    first = client.post("/v1/galaxy/runs", json=body)
    assert first.status_code == 202
    again = client.post("/v1/galaxy/runs", json=body)
    assert again.status_code == 200 and again.json()["id"] == first.json()["id"]
    assert client.get(f"/v1/galaxy/runs/{first.json()['id']}").status_code == 200
    assert client.get("/v1/galaxy/runs/run_missing").status_code == 404
    listed = client.get("/v1/galaxy/runs", params={"thread_id": "t1"}).json()
    assert [r["id"] for r in listed["items"]] == [first.json()["id"]] and listed["next_cursor"] is None


# ---------------------------------------------------------------- 401/403 and file readiness

async def test_galaxy_403_is_not_found_and_401_stays_a_token_problem():
    gi = _fake_gi()
    svc = _service(gi)
    gi.tools.build.side_effect = GalaxyConnectionError("no", body='{"err_msg": "denied"}', status_code=403)
    with pytest.raises(GalaxyRunsError) as e:
        await svc.describe("t1", TARGET)
    assert e.value.status_code == 404
    gi.tools.build.side_effect = GalaxyConnectionError("no", body='{"err_msg": "bad key"}', status_code=401)
    with pytest.raises(GalaxyRunsError) as e:
        await svc.describe("t1", TARGET)
    assert e.value.status_code == 401


@pytest.mark.parametrize("state,expected", [("queued", "Still uploading"), ("running", "Still uploading"),
                                            ("error", "Upload failed"), ("ok", None), ("deferred", None)])
async def test_validate_says_still_uploading_until_the_file_is_ready(state, expected):
    gi = _fake_gi()
    gi.datasets.show_dataset.return_value = {"id": "f2db", "state": state}
    result = await _service(gi).validate("t1", TARGET, INPUTS_HDA, {})
    assert result["errors"].get("single_paired|in1") == expected


INPUTS_HDA = {"single_paired|single_paired_selector": "single", "single_paired|in1": {"src": "hda", "id": "f2db"}}


# ---------------------------------------------------------------- /v1/datasets

import httpx  # noqa: E402

from app.galaxy_runs import datasets  # noqa: E402


def _multipart(fields_before, file_bytes, fields_after=(), boundary="XyZ"):
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
             for k, v in fields_before]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="reads_R1.fastq.gz"\r\n'
                 f'Content-Type: application/gzip\r\n\r\n'.encode() + file_bytes + b"\r\n")
    parts += [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
              for k, v in fields_after]
    return b"".join(parts) + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


async def _chunks(body, size=7):
    for i in range(0, len(body), size):
        yield body[i:i + size]


def _galaxy_fetch(seen):
    async def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = await request.aread()
        return httpx.Response(200, json={"outputs": [{"id": "ds1", "name": "reads_R1.fastq.gz", "file_ext": "auto",
                                                     "state": "queued", "create_time": "2026-10-09T10:02:00.123"}],
                                         "jobs": [{"id": "j9"}]})
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_upload_streams_the_file_into_the_threads_history():
    gi = _fake_gi()
    svc = _service(gi)
    seen = {}
    payload = b"@r1\nACGT\n+\nFFFF\n" * 50
    body, ctype = _multipart([("thread_id", "t1")], payload, [("format", "fastqsanger.gz")])
    dataset = await datasets.upload(svc, ctype, str(len(body)), _chunks(body), http=_galaxy_fetch(seen))

    assert dataset == {"id": "ds1", "name": "reads_R1.fastq.gz", "format": "auto", "size": len(payload),
                       "state": "queued", "history_id": "hist1", "created_at": "2026-10-09T10:02:00Z"}
    assert seen["url"] == "https://galaxy.example.org/api/tools/fetch"
    assert seen["headers"]["x-api-key"] == "user-key"
    sent = seen["body"]
    assert b'name="files_0|file_data"; filename="reads_R1.fastq.gz"' in sent
    assert payload in sent
    assert b'name="history_id"\r\n\r\nhist1' in sent
    # format arrived after the file and still reaches Galaxy
    assert b'"ext": "fastqsanger.gz"' in sent


async def test_upload_needs_thread_id_first_a_file_and_stays_under_the_cap(monkeypatch):
    svc = _service(_fake_gi())
    body, ctype = _multipart([], b"data", [("thread_id", "t1")])
    with pytest.raises(GalaxyRunsError) as e:
        await datasets.upload(svc, ctype, None, _chunks(body), http=_galaxy_fetch({}))
    assert e.value.status_code == 400

    body = b'--XyZ\r\nContent-Disposition: form-data; name="thread_id"\r\n\r\nt1\r\n--XyZ--\r\n'
    with pytest.raises(GalaxyRunsError) as e:
        await datasets.upload(svc, "multipart/form-data; boundary=XyZ", None, _chunks(body), http=_galaxy_fetch({}))
    assert e.value.status_code == 400

    monkeypatch.setattr(datasets, "UPLOAD_MAX_BYTES", 100)
    body, ctype = _multipart([("thread_id", "t1")], b"x" * 500)
    with pytest.raises(GalaxyRunsError) as e:
        await datasets.upload(svc, ctype, None, _chunks(body), http=_galaxy_fetch({}))
    assert e.value.status_code == 413
    with pytest.raises(GalaxyRunsError) as e:  # refused before reading when Content-Length says so
        await datasets.upload(svc, ctype, str(2_000_000), _chunks(body), http=_galaxy_fetch({}))
    assert e.value.status_code == 413


def _galaxy_display(seen, status=206, body=b"ACGT"):
    async def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["range"] = request.headers.get("range")
        headers = {"content-type": "application/octet-stream", "content-length": str(len(body))}
        if status == 206:
            headers["content-range"] = f"bytes 0-{len(body) - 1}/1000"
        return httpx.Response(status, headers=headers, content=_chunks(body, 2))  # streamed, as Galaxy does
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _read(response):
    return b"".join([chunk async for chunk in response.body_iterator])


async def test_download_streams_with_range_and_names_the_file():
    gi = _fake_gi()
    svc = _service(gi)
    seen = {}
    response = await datasets.download(svc, "f2db", "bytes=0-3", None, http=_galaxy_display(seen))
    assert response.status_code == 206
    assert await _read(response) == b"ACGT"
    assert seen["range"] == "bytes=0-3"
    assert seen["url"] == "https://galaxy.example.org/api/datasets/f2db/display?to_ext=fastqsanger.gz"
    assert response.headers["content-range"] == "bytes 0-3/1000"
    assert response.headers["content-disposition"].startswith('attachment; filename="reads_R1"')


async def test_download_index_files_and_errors():
    gi = _fake_gi()
    svc = _service(gi)
    seen = {}
    response = await datasets.download(svc, "f2db", None, "bam_index", http=_galaxy_display(seen, 200))
    assert seen["url"].endswith("/api/datasets/f2db/metadata_file?metadata_file=bam_index")
    assert 'filename="reads_R1.bai"' in response.headers["content-disposition"]
    await _read(response)

    with pytest.raises(GalaxyRunsError) as e:
        await datasets.download(svc, "f2db", None, "../etc", http=_galaxy_display({}))
    assert e.value.status_code == 400

    for galaxy_code in (403, 400):  # another user's dataset; an id Galaxy can't decode
        gi.datasets.show_dataset.side_effect = GalaxyConnectionError("no", body="{}", status_code=galaxy_code)
        with pytest.raises(GalaxyRunsError) as e:
            await datasets.download(svc, "other-users", None, None, http=_galaxy_display({}))
        assert e.value.status_code == 404


def test_upload_and_download_routes(monkeypatch):
    seen = {}
    client = _client(_service(_fake_gi()))
    monkeypatch.setattr(datasets, "_default_http", _galaxy_fetch(seen))
    up = client.post("/v1/datasets/upload", data={"thread_id": "t1", "format": "fastqsanger.gz"},
                     files={"file": ("reads_R1.fastq.gz", b"@r1\nACGT\n+\nFFFF\n", "application/gzip")})
    assert up.status_code == 201 and up.json()["id"] == "ds1" and up.json()["size"] == 16

    monkeypatch.setattr(datasets, "_default_http", _galaxy_display({}))
    down = client.get("/v1/datasets/f2db/download", headers={"Range": "bytes=0-3"})
    assert down.status_code == 206 and down.content == b"ACGT"
    assert client.get("/v1/datasets/f2db/download", params={"metadata_file": "BAD!"}).status_code == 400
