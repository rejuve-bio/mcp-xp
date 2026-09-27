import os
import json
import pytest
from sys import path
path.append(".")

from unittest.mock import AsyncMock, MagicMock, patch
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient as FastAPITestClient

from app.api.middleware import JWTGalaxyKeyMiddleware
from app.context import (
    CHAT_ID_MAX_LENGTH,
    current_chat_id,
    validate_chat_id,
)
from app.GX_integration.invocations.invocation_service import InvocationService
from app.enumerations import CollectionNames


# ---------- validate_chat_id ---------------------------------------------------

def test_validate_chat_id_missing_returns_none():
    assert validate_chat_id(None) is None


def test_validate_chat_id_empty_returns_none():
    assert validate_chat_id("") is None
    assert validate_chat_id("   ") is None


def test_validate_chat_id_wrong_type_returns_none():
    assert validate_chat_id(42) is None
    assert validate_chat_id(["a"]) is None


def test_validate_chat_id_oversized_returns_none():
    assert validate_chat_id("x" * (CHAT_ID_MAX_LENGTH + 1)) is None


def test_validate_chat_id_trims_and_returns_value():
    assert validate_chat_id("  abc-123  ") == "abc-123"


def test_validate_chat_id_max_length_boundary_ok():
    exactly_max = "x" * CHAT_ID_MAX_LENGTH
    assert validate_chat_id(exactly_max) == exactly_max


# ---------- JWTGalaxyKeyMiddleware sets current_chat_id ------------------------

@pytest.fixture
def middleware_client():
    fernet_secret = Fernet.generate_key()
    os.environ["SECRET_KEY"] = fernet_secret.decode()

    app = FastAPI()

    @app.get("/probe")
    async def probe():
        return {"chat_id": current_chat_id.get()}

    app.add_middleware(JWTGalaxyKeyMiddleware)
    client = FastAPITestClient(app)

    import jwt as _jwt
    token = _jwt.encode({"galaxy_api_token": "raw-key"}, key=None, algorithm="none")

    yield client, token
    del os.environ["SECRET_KEY"]


def test_middleware_reads_chat_id_from_header(middleware_client):
    client, token = middleware_client
    response = client.get(
        "/probe",
        headers={"Authorization": f"Bearer {token}", "X-Chat-Id": "chat-42"},
    )
    assert response.status_code == 200
    assert response.json()["chat_id"] == "chat-42"


def test_middleware_lowercase_header_also_read(middleware_client):
    client, token = middleware_client
    response = client.get(
        "/probe",
        headers={"Authorization": f"Bearer {token}", "x-chat-id": "chat-lower"},
    )
    assert response.status_code == 200
    assert response.json()["chat_id"] == "chat-lower"


def test_middleware_missing_header_yields_none(middleware_client):
    client, token = middleware_client
    response = client.get("/probe", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["chat_id"] is None


def test_middleware_malformed_header_yields_none_not_400(middleware_client):
    client, token = middleware_client
    response = client.get(
        "/probe",
        headers={"Authorization": f"Bearer {token}", "X-Chat-Id": "x" * 500},
    )
    assert response.status_code == 200
    assert response.json()["chat_id"] is None


# ---------- InvocationService._resolve_chat_id write-once ----------------------

@pytest.fixture
def service():
    cache = MagicMock()
    background_tasks = MagicMock()
    mongo_client = MagicMock()
    return InvocationService(
        cache=cache,
        background_tasks=background_tasks,
        mongo_client=mongo_client,
    )


@pytest.mark.asyncio
async def test_resolve_chat_id_first_call_writes_and_returns_current(service):
    service.mongo_client.get = AsyncMock(return_value=None)
    service.mongo_client.set = AsyncMock(return_value=None)

    token = current_chat_id.set("chat-first")
    try:
        result = await service._resolve_chat_id("alice", "inv-1")
    finally:
        current_chat_id.reset(token)

    assert result == "chat-first"
    service.mongo_client.set.assert_awaited_once()
    write_kwargs = service.mongo_client.set.await_args.kwargs
    assert write_kwargs["collection_name"] == CollectionNames.INVOCATION_META.value
    assert write_kwargs["key"] == "alice:inv-1"
    assert write_kwargs["value"]["chat_id"] == "chat-first"
    assert "first_seen_at" in write_kwargs["value"]


@pytest.mark.asyncio
async def test_resolve_chat_id_second_call_from_different_chat_does_not_overwrite(service):
    service.mongo_client.get = AsyncMock(return_value={"chat_id": "chat-first", "first_seen_at": "t0"})
    service.mongo_client.set = AsyncMock(return_value=None)

    token = current_chat_id.set("chat-second")
    try:
        result = await service._resolve_chat_id("alice", "inv-1")
    finally:
        current_chat_id.reset(token)

    assert result == "chat-first"
    service.mongo_client.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_chat_id_no_meta_no_context_returns_none(service):
    service.mongo_client.get = AsyncMock(return_value=None)
    service.mongo_client.set = AsyncMock(return_value=None)

    result = await service._resolve_chat_id("alice", "inv-1")

    assert result is None
    service.mongo_client.set.assert_not_awaited()


# ---------- list_invocations_by_chat cross-user scoping ------------------------

@pytest.mark.asyncio
async def test_list_by_chat_scopes_by_username_via_meta_prefix(service):
    """Two users sharing the same chat_id string must see disjoint lists."""
    alice_hit = {"key": "alice:inv-a", "value": {"chat_id": "shared"}}
    service.mongo_client.find_by_value_field = AsyncMock(return_value=[alice_hit])
    service.mongo_client.get = AsyncMock(return_value=[
        {
            "id": "inv-a",
            "workflow_name": "wf",
            "workflow_id": "wfid",
            "history_id": "hid",
            "state": "Complete",
            "outputs": [],
            "create_time": "t",
            "update_time": "t",
        }
    ])

    result = await service.list_invocations_by_chat(username="alice", chat_id="shared")

    call_kwargs = service.mongo_client.find_by_value_field.await_args.kwargs
    assert call_kwargs["key_prefix"] == "alice"
    assert call_kwargs["field_path"] == "chat_id"
    assert call_kwargs["value"] == "shared"

    assert len(result.invocations) == 1
    assert result.invocations[0].id == "inv-a"
    assert result.invocations[0].chat_id == "shared"


@pytest.mark.asyncio
async def test_list_by_chat_no_hits_returns_empty(service):
    service.mongo_client.find_by_value_field = AsyncMock(return_value=[])

    result = await service.list_invocations_by_chat(username="alice", chat_id="nope")

    assert result.invocations == []
