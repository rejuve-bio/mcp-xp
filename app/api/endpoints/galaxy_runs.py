"""/v1/galaxy: run any Galaxy tool or workflow from a chat thread; /v1/datasets: move its files.

Every route answers at once. Runs are read again until they end; see app/galaxy_runs/service.py.
"""

import os
import time
import logging
from typing import Dict, Literal, Optional, Tuple

import redis.asyncio as aioredis
from bioblend import ConnectionError as GalaxyConnectionError
from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, Request, Response
from fastapi.concurrency import run_in_threadpool

from app.context import current_api_key
from app.galaxy import GalaxyClient
from app.persistence import MongoStore
from app.api.schemas.galaxy_runs import (
    Dataset, Description, DescribeRequest, Link, LinkRequest, Problem, Run, RunList, RunRequest, SearchList,
    ThreadFiles, Validation, ValidateRequest,
)
from app.galaxy_runs import datasets
from app.galaxy_runs.service import GalaxyRuns, GalaxyRunsError
from app.galaxy_runs.store import RunStore

logger = logging.getLogger("GalaxyRunsAPI")
router = APIRouter()

_store = RunStore(MongoStore())
_cache = aioredis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=int(os.getenv("REDIS_PORT", 6389)),
                        db=0, decode_responses=True)

# Galaxy username per API key, so each request doesn't ask Galaxy who the user is.
_USERNAMES: Dict[str, Tuple[str, float]] = {}
_USERNAME_TTL = 600


async def galaxy_runs() -> GalaxyRuns:
    api_key = current_api_key.get(None)
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing Galaxy credentials")
    client = await run_in_threadpool(GalaxyClient, api_key)
    cached = _USERNAMES.get(api_key)
    if cached and cached[1] > time.monotonic():
        username = cached[0]
    else:
        try:
            username = await run_in_threadpool(lambda: client.whoami)
        except GalaxyConnectionError as e:
            if getattr(e, "status_code", None) in (401, 403):
                raise HTTPException(status_code=401, detail="Galaxy did not accept the user's key")
            raise HTTPException(status_code=502, detail="Galaxy failed. Safe to retry.")
        if not isinstance(username, str) or not username:
            raise HTTPException(status_code=401, detail="Galaxy did not recognise the user")
        _USERNAMES[api_key] = (username, time.monotonic() + _USERNAME_TTL)
    return GalaxyRuns(client, username, _store, _cache)


def _errors(*codes: int) -> dict:
    """The spec's error statuses a route can answer, for the docs."""
    meaning = {
        400: "Invalid body", 401: "Missing or expired token", 404: "No such run, dataset, thread or proposal for this user",
        413: "Upload over the size cap", 416: "Range not satisfiable", 422: "Galaxy rejected the inputs (detail per path)",
        502: "Galaxy or mcp-xp failed. Safe to retry with the same client_request_id",
    }
    return {c: {"model": Problem, "description": meaning[c]} for c in (*codes, 401, 502)}


def _http(e: GalaxyRunsError) -> HTTPException:
    return HTTPException(status_code=e.status_code, detail=e.detail)


@router.get("/search", summary="Installed, runnable tools and workflows", response_model=SearchList,
            response_model_exclude_unset=True, responses=_errors(400))
async def search(q: str = Query(..., min_length=1), kind: Optional[Literal["tool", "workflow"]] = None,
                 svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        return await svc.search(q, kind)
    except GalaxyRunsError as e:
        raise _http(e)


@router.post("/describe", summary="What a tool or workflow needs, trimmed for the model", response_model=Description,
             response_model_exclude_unset=True, responses=_errors(400, 404))
async def describe(body: DescribeRequest, full: bool = Query(False, description="Every setting, for all_values"),
                   svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        return await svc.describe(body.thread_id, body.target.model_dump(), full=full)
    except GalaxyRunsError as e:
        raise _http(e)


@router.post("/validate", summary="Check inputs with Galaxy without running anything", response_model=Validation,
             responses=_errors(400, 404))
async def validate(body: ValidateRequest, svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        return await svc.validate(body.thread_id, body.target.model_dump(), body.inputs, body.parameters)
    except GalaxyRunsError as e:
        raise _http(e)


@router.get("/threads/{thread_id}/files", summary="The thread history's datasets and collections",
            response_model=ThreadFiles, response_model_exclude_unset=True, responses=_errors(400))
async def thread_files(thread_id: str = Path(..., min_length=1, max_length=256), svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        return await svc.thread_files(thread_id)
    except GalaxyRunsError as e:
        raise _http(e)


@router.post("/runs", status_code=202, summary="Start a run; returns at once (200 with the stored run for a repeat)",
             response_model=Run, responses={200: {"model": Run, "description": "Repeat of a client_request_id: the stored run, nothing started"},
                                           **_errors(400, 404, 422)})
async def start_run(body: RunRequest, response: Response, svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        run, repeat = await svc.start_run(body.thread_id, body.proposal_id, body.client_request_id,
                                          body.target.model_dump(), body.inputs, body.parameters)
    except GalaxyRunsError as e:
        raise _http(e)
    if repeat:
        response.status_code = 200
    return run


@router.get("/runs", summary="Every run in a thread, newest first", response_model=RunList, responses=_errors(400))
async def list_runs(thread_id: str = Query(..., min_length=1, max_length=256), svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        return await svc.list_runs(thread_id)
    except GalaxyRunsError as e:
        raise _http(e)


@router.get("/runs/{run_id}", summary="One run, as fresh as Galaxy", response_model=Run, responses=_errors(404))
async def read_run(run_id: str = Path(..., min_length=1, max_length=64), svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        return await svc.read_run(run_id)
    except GalaxyRunsError as e:
        raise _http(e)


@router.post("/runs/{run_id}/cancel", status_code=202, summary="Cancel a run; it turns cancelled once Galaxy confirms",
             response_model=Run, responses=_errors(404))
async def cancel_run(run_id: str = Path(..., min_length=1, max_length=64), svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        return await svc.cancel_run(run_id)
    except GalaxyRunsError as e:
        raise _http(e)


@router.post("/links", status_code=201, summary="A link that opens Galaxy's own form with a card's values",
             response_model=Link, responses=_errors(400, 404))
async def link(body: LinkRequest, svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        return await svc.link(body.thread_id, body.proposal_id, body.target.model_dump(), body.inputs, body.parameters)
    except GalaxyRunsError as e:
        raise _http(e)


# ---------------------------------------------------------------- /v1/datasets

datasets_router = APIRouter()


@datasets_router.post(
    "/upload", status_code=201, summary="Upload a file into the thread's Galaxy history", response_model=Dataset,
    responses=_errors(400, 404, 413, 422),
    # The body is read as a stream (no temporary file), so it is described here for the docs.
    openapi_extra={"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
        "type": "object", "required": ["thread_id", "file"],
        "properties": {
            "thread_id": {"type": "string", "description": "Send before file", "example": "8d2f0c1e-4b7a-4c39-9e21-5f6a7b8c9d0e"},
            "file": {"type": "string", "format": "binary"},
            "format": {"type": "string", "description": "Galaxy format, e.g. fastqsanger.gz; detected when left out"},
        },
        "propertyOrdering": ["thread_id", "file", "format"]}}}}})
async def upload_dataset(request: Request, svc: GalaxyRuns = Depends(galaxy_runs)):
    """Multipart { thread_id, file, format? }, thread_id before file. Streamed to Galaxy without a temporary file;
    returns at once with the dataset, whose state goes queued -> ok as Galaxy processes the upload."""
    try:
        return await datasets.upload(svc, request.headers.get("content-type"),
                                     request.headers.get("content-length"), request.stream())
    except GalaxyRunsError as e:
        raise _http(e)


@datasets_router.get(
    "/{dataset_id}/download", summary="A dataset's bytes, streamed, with Range passed through",
    response_class=Response, responses={
        200: {"description": "The whole file, as an attachment", "content": {"application/octet-stream": {}}},
        206: {"description": "The requested byte range", "content": {"application/octet-stream": {}}},
        **_errors(400, 404, 416)})
async def download_dataset(dataset_id: str = Path(..., min_length=1, max_length=64),
                           metadata_file: Optional[str] = Query(None, description="e.g. bam_index or tabix_index"),
                           range: Optional[str] = Header(None, description="e.g. bytes=0-1023; answered with 206"),
                           svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        return await datasets.download(svc, dataset_id, range, metadata_file)
    except GalaxyRunsError as e:
        raise _http(e)
