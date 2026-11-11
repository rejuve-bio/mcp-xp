import os
import json
import logging
import asyncio
import jwt

from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from fastmcp.server.middleware import Middleware, MiddlewareContext, CallNext
from fastmcp.server.dependencies import get_http_headers
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from app.log_setup import configure_logging
from app.bioblend_server.mcp_context import current_api_key_server, current_chat_id_server
from app.context import validate_chat_id

configure_logging()

GALAXY_API_TOKEN = "galaxy_api_token"

# Per the MCP spec, initialize + the initialized notification are the pre-auth
# handshake — clients must speak them before they know what auth the server
# demands. Guarding them here breaks the whole session with -32601 because
# fastmcp turns a plain-dict middleware short-circuit into "Method not found".
_PREAUTH_METHODS = {"initialize", "notifications/initialized", "ping"}


def _unauthorized(msg: str) -> McpError:
    # -32001 is JSON-RPC "server error" range; the MCP client sees a real
    # error object instead of a spurious "Method not found".
    return McpError(ErrorData(code=-32001, message=msg))

class JWTGalaxyKeyMiddleware(Middleware):
    """
    FastMCP middleware that expects:
      Authorization: Bearer <JWT>
    The JWT must contain a claim ('galaxy_api_token') that is either:
      - a fernet-encrypted JSON payload like {"apikey": "<actual_key>"} (what the register user produces)
    The middleware will set current_api_key to the final plain api key string.
    """
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)

    async def on_request(self, context: MiddlewareContext, call_next: CallNext):
        if context.method in _PREAUTH_METHODS:
            return await call_next(context)

        headers = get_http_headers(include_all=True)
        auth = headers.get("Authorization", None) or headers.get("authorization", None)

        if auth is None or not auth.startswith("Bearer "):
            self.log.error("unauthorized, Authorization header with Bearer token is required.")
            raise _unauthorized("Missing or malformed Authorization header")

        token = auth.split(" ")[1].strip()
        try:
            payload = self._decode_jwt(token)
        except Exception as e:
            self.log.error(f"unauthorized, Invalid JWT: {e}")
            raise _unauthorized("Invalid JWT")

        if GALAXY_API_TOKEN not in payload:
            self.log.error("JWT missing API key claim '%s'", GALAXY_API_TOKEN)
            raise _unauthorized("JWT missing api-key claim")

        galaxy_jwt_token = payload[GALAXY_API_TOKEN]
        if not galaxy_jwt_token:
            self.log.error("Empty API key claim")
            raise _unauthorized("Empty api-key claim")

        apikey = await self._decrypt_api_token(galaxy_jwt_token)

        current_api_key_server.set(apikey)
        current_chat_id_server.set(
            validate_chat_id(headers.get("X-Chat-Id") or headers.get("x-chat-id"))
        )
        self.log.info("Incoming request to MCP server validated.")

        return await call_next(context)

    def _decode_jwt(self, token: str) -> dict:
        """
        Decode/verify JWT synchronously (PyJWT). Raises Exception on invalid token.
        For RS-based tokens, JWT_SECRET should contain the public key (but here signature verification is disabled).
        """
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            return payload
        except jwt.InvalidTokenError as e:
            self.log.error("Invalid JWT: %s", e)
            raise ValueError(f"Invalid JWT: {e}")

    async def _decrypt_api_token(self, token_str: str) -> Optional[str]:
        """
        If token_str is a fernet-encrypted payload (bytes when encoded),
        decrypt and parse JSON for {"apikey": "<value>"} and return the value.
        Returns None if decryption/parsing fails so caller can fallback to raw token.
        """
        if not isinstance(token_str, str) or not token_str:
            return None
        loop = asyncio.get_running_loop()
        try:
            # Environment / secrets
            FERNET_SECRET = os.getenv("SECRET_KEY")
            if not FERNET_SECRET:
                raise RuntimeError("SECRET_KEY (Fernet secret) is required in env")
            fernet = Fernet(FERNET_SECRET)
            
            decrypted = await loop.run_in_executor(None, fernet.decrypt, token_str.encode("utf-8"))
            parsed: dict = await loop.run_in_executor(None, json.loads, decrypted.decode("utf-8"))
            apikey = parsed.get("apikey")
            if apikey and isinstance(apikey, str):
                return apikey
            self.log.error("Decrypted JWT galaxy api-key payload missing 'apikey' field")
            return None
        except (InvalidToken, Exception) as e:
            # Not a fernet payload or parse failed; return None so fallback can apply
            self.log.debug("Fernet decryption/parsing failed for JWT claim: %s", e)
            return None