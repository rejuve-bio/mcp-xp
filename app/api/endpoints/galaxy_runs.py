"""/v1/galaxy: run any Galaxy tool or workflow from a chat thread.

Every route answers at once. Runs are read again until they end; see app/galaxy_runs/service.py.
"""

import os
import time
import logging
from typing import Dict, Literal, Optional, Tuple

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response
from fastapi.concurrency import run_in_threadpool

from app.context import current_api_key
from app.galaxy import GalaxyClient
from app.persistence import MongoStore
from app.api.schemas.galaxy_runs import DescribeRequest, LinkRequest, RunRequest, ValidateRequest
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
        username = await run_in_threadpool(lambda: client.whoami)
        if not isinstance(username, str) or not username:
            raise HTTPException(status_code=401, detail="Galaxy did not recognise the user")
        _USERNAMES[api_key] = (username, time.monotonic() + _USERNAME_TTL)
    return GalaxyRuns(client, username, _store, _cache)


def _http(e: GalaxyRunsError) -> HTTPException:
    return HTTPException(status_code=e.status_code, detail=e.detail)


@router.get("/search", summary="Installed, runnable tools and workflows")
async def search(q: str = Query(..., min_length=1), kind: Optional[Literal["tool", "workflow"]] = None,
                 svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        return await svc.search(q, kind)
    except GalaxyRunsError as e:
        raise _http(e)


@router.post("/describe", summary="What a tool or workflow needs, trimmed for the model")
async def describe(body: DescribeRequest, full: bool = Query(False, description="Every setting, for all_values"),
                   svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        return await svc.describe(body.thread_id, body.target.model_dump(), full=full)
    except GalaxyRunsError as e:
        raise _http(e)


@router.post("/validate", summary="Check inputs with Galaxy without running anything")
async def validate(body: ValidateRequest, svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        return await svc.validate(body.thread_id, body.target.model_dump(), body.inputs, body.parameters)
    except GalaxyRunsError as e:
        raise _http(e)


@router.get("/threads/{thread_id}/files", summary="The thread history's datasets and collections")
async def thread_files(thread_id: str = Path(..., min_length=1, max_length=256), svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        return await svc.thread_files(thread_id)
    except GalaxyRunsError as e:
        raise _http(e)


@router.post("/runs", status_code=202, summary="Start a run; returns at once")
async def start_run(body: RunRequest, response: Response, svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        run, repeat = await svc.start_run(body.thread_id, body.proposal_id, body.client_request_id,
                                          body.target.model_dump(), body.inputs, body.parameters)
    except GalaxyRunsError as e:
        raise _http(e)
    if repeat:
        response.status_code = 200
    return run


@router.get("/runs", summary="Every run in a thread, newest first")
async def list_runs(thread_id: str = Query(..., min_length=1, max_length=256), svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        return await svc.list_runs(thread_id)
    except GalaxyRunsError as e:
        raise _http(e)


@router.get("/runs/{run_id}", summary="One run, as fresh as Galaxy")
async def read_run(run_id: str = Path(..., min_length=1, max_length=64), svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        return await svc.read_run(run_id)
    except GalaxyRunsError as e:
        raise _http(e)


@router.post("/runs/{run_id}/cancel", status_code=202, summary="Cancel a run; it turns cancelled once Galaxy confirms")
async def cancel_run(run_id: str = Path(..., min_length=1, max_length=64), svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        return await svc.cancel_run(run_id)
    except GalaxyRunsError as e:
        raise _http(e)


@router.post("/links", status_code=201, summary="A link that opens Galaxy's own form with a card's values")
async def link(body: LinkRequest, svc: GalaxyRuns = Depends(galaxy_runs)):
    try:
        return await svc.link(body.thread_id, body.proposal_id, body.target.model_dump(), body.inputs, body.parameters)
    except GalaxyRunsError as e:
        raise _http(e)
