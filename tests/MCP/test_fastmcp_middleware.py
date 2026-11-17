import os
import time
import jwt
import json
import pytest
import logging
from sys import path
path.append(".")

from cryptography.fernet import Fernet
from fastmcp.server.middleware import MiddlewareContext
from mcp.shared.exceptions import McpError
from unittest.mock import Mock, patch, MagicMock

from app.bioblend_server.mcp_context import current_api_key_server
from app.context import current_user_email
from app.bioblend_server.mcp_middleware import JWTGalaxyKeyMiddleware as FastMCPJWTGalaxyKeyMiddleware

@pytest.fixture
def mcp_jwt_logger():
    return logging.getLogger("TestFastMCPJWTGalaxyKeyMiddleware")
        

@pytest.fixture
def fernet_secret():
    """Fixture for Fernet secret key."""
    return Fernet.generate_key()


@pytest.fixture
def fernet(fernet_secret):
    """Fixture for Fernet instance."""
    return Fernet(fernet_secret)

JWT_TEST_SECRET = "test-jwt-secret-0123456789abcdef0123456789"


def signed_token(claims: dict, secret: str = JWT_TEST_SECRET, exp_in: int = 300) -> str:
    return jwt.encode({"exp": int(time.time()) + exp_in, **claims}, secret, algorithm="HS256")


class TestFastMCPJWTGalaxyKeyMiddleware:
    """MCP middleware: same token rules as REST."""

    @pytest.fixture
    def middleware(self, fernet_secret, monkeypatch):
        monkeypatch.setenv("SECRET_KEY", fernet_secret.decode())
        monkeypatch.setenv("JWT_SECRET", JWT_TEST_SECRET)
        monkeypatch.delenv("JWT_VERIFY", raising=False)
        return FastMCPJWTGalaxyKeyMiddleware()

    @pytest.fixture
    def mock_context(self):
        context = Mock(spec=MiddlewareContext)
        context.method = "tools/call"
        return context

    @pytest.fixture
    def mock_call_next(self):
        async def mock_next(context):
            return {
                "message": "Protected endpoint",
                "api_key": current_api_key_server.get(None),
                "email": current_user_email.get(None),
            }
        return MagicMock(wraps=mock_next)

    async def _call(self, middleware, context, call_next, headers):
        with patch("app.bioblend_server.mcp_middleware.get_http_headers") as mock_headers:
            mock_headers.return_value = headers
            return await middleware.on_request(context, call_next)

    async def test_handshake_needs_no_token(self, middleware, mock_context, mock_call_next):
        mock_context.method = "initialize"
        await self._call(middleware, mock_context, mock_call_next, {})
        mock_call_next.assert_called_once()

    async def test_missing_authorization_header(self, middleware, mock_context, mock_call_next):
        with pytest.raises(McpError) as exc_info:
            await self._call(middleware, mock_context, mock_call_next, {})
        assert exc_info.value.error.code == -32001
        assert "Missing or malformed Authorization header" in exc_info.value.error.message
        mock_call_next.assert_not_called()

    async def test_malformed_token(self, middleware, mock_context, mock_call_next):
        with pytest.raises(McpError) as exc_info:
            await self._call(middleware, mock_context, mock_call_next,
                             {"Authorization": "Bearer invalid.token.here"})
        assert exc_info.value.error.code == -32001
        assert "Invalid token" in exc_info.value.error.message

    async def test_unsigned_token_rejected(self, middleware, mock_context, mock_call_next):
        token = jwt.encode({"galaxy_api_token": "raw-key", "email": "a@example.org"}, key=None, algorithm="none")
        with pytest.raises(McpError):
            await self._call(middleware, mock_context, mock_call_next, {"Authorization": f"Bearer {token}"})
        mock_call_next.assert_not_called()

    async def test_wrong_secret_rejected(self, fernet, middleware, mock_context, mock_call_next):
        claim = fernet.encrypt(json.dumps({"apikey": "k"}).encode()).decode()
        token = signed_token({"galaxy_api_token": claim}, secret="attacker-secret-0123456789abcdef01")
        with pytest.raises(McpError):
            await self._call(middleware, mock_context, mock_call_next, {"Authorization": f"Bearer {token}"})

    async def test_signed_token_sets_key_and_email(self, fernet, middleware, mock_context, mock_call_next):
        raw_apikey = "test-encrypted-api-key-456"
        claim = fernet.encrypt(json.dumps({"apikey": raw_apikey}).encode()).decode()
        token = signed_token({"galaxy_api_token": claim, "email": "Alice@Example.org"})
        result = await self._call(middleware, mock_context, mock_call_next, {"Authorization": f"Bearer {token}"})
        assert result["api_key"] == raw_apikey
        # GalaxyClient needs the email to recover from an expired key.
        assert result["email"] == "alice@example.org"
