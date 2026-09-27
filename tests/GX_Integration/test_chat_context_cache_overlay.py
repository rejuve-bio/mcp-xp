"""Regression: chat_id must be resolved and overlaid even when the invocation
result comes from cache or Mongo (not the fresh-fetch path)."""

import pytest
from sys import path
path.append(".")

from unittest.mock import AsyncMock, MagicMock

from app.context import current_chat_id
from app.enumerations import CollectionNames
from app.GX_integration.invocations.invocation_service import InvocationService


def _make_result_dict():
    return {
        "invocation_id": "inv-1",
        "state": "Complete",
        "history_id": "hist-1",
        "create_time": "2026-09-27T00:00:00",
        "update_time": "2026-09-27T00:00:00",
        "inputs": {},
        "result": [],
        "workflow": {
            "id": "wf-1",
            "name": "wf",
            "description": None,
            "annotation": None,
            "create_time": None,
            "update_time": None,
        },
        "report": None,
    }


def _make_service():
    cache = MagicMock()
    background_tasks = MagicMock()
    mongo_client = MagicMock()
    svc = InvocationService(cache=cache, background_tasks=background_tasks, mongo_client=mongo_client)
    # Non-empty deleted list keeps the code path from falling through to a
    # Mongo lookup for deleted invocations (which would consume the first
    # side_effect entry meant for stored_result / meta).
    svc.cache.get_deleted_invocations = AsyncMock(return_value=["not-the-target"])
    svc.cache.set_invocation_result = AsyncMock(return_value=None)
    return svc


@pytest.mark.asyncio
async def test_cache_hit_still_resolves_and_overlays_chat_id():
    svc = _make_service()
    cached = _make_result_dict()
    svc.cache.get_invocation_result = AsyncMock(return_value=cached)
    svc.mongo_client.get = AsyncMock(return_value=None)  # no prior meta record
    svc.mongo_client.set = AsyncMock(return_value=None)

    token = current_chat_id.set("chat-cache-hit")
    try:
        result = await svc.get_invocation_result(
            invocation_id="inv-1",
            username="alice",
            api_key="k",
            galaxy_client=MagicMock(),
            workflow_manager=MagicMock(),
        )
    finally:
        current_chat_id.reset(token)

    assert result.chat_id == "chat-cache-hit"
    svc.mongo_client.set.assert_awaited_once()
    assert svc.mongo_client.set.await_args.kwargs["collection_name"] == CollectionNames.INVOCATION_META.value


@pytest.mark.asyncio
async def test_mongo_stored_hit_still_resolves_and_overlays_chat_id():
    svc = _make_service()
    stored = _make_result_dict()
    svc.cache.get_invocation_result = AsyncMock(return_value=None)
    svc.cache.set_invocation_result = AsyncMock(return_value=None)
    # First mongo_client.get call is stored_result, second is inside _resolve_chat_id.
    svc.mongo_client.get = AsyncMock(side_effect=[stored, None])
    svc.mongo_client.set = AsyncMock(return_value=None)

    token = current_chat_id.set("chat-mongo-hit")
    try:
        result = await svc.get_invocation_result(
            invocation_id="inv-1",
            username="alice",
            api_key="k",
            galaxy_client=MagicMock(),
            workflow_manager=MagicMock(),
        )
    finally:
        current_chat_id.reset(token)

    assert result.chat_id == "chat-mongo-hit"


@pytest.mark.asyncio
async def test_cache_hit_with_existing_meta_uses_first_seen_chat_not_current():
    svc = _make_service()
    cached = _make_result_dict()
    svc.cache.get_invocation_result = AsyncMock(return_value=cached)
    svc.mongo_client.get = AsyncMock(return_value={"chat_id": "chat-original", "first_seen_at": "t0"})
    svc.mongo_client.set = AsyncMock(return_value=None)

    token = current_chat_id.set("chat-later")
    try:
        result = await svc.get_invocation_result(
            invocation_id="inv-1",
            username="alice",
            api_key="k",
            galaxy_client=MagicMock(),
            workflow_manager=MagicMock(),
        )
    finally:
        current_chat_id.reset(token)

    assert result.chat_id == "chat-original"
    svc.mongo_client.set.assert_not_awaited()
