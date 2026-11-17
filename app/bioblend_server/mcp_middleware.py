import logging

from fastmcp.server.middleware import Middleware, MiddlewareContext, CallNext
from fastmcp.server.dependencies import get_http_headers
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from app.log_setup import configure_logging
from app.bioblend_server.mcp_context import current_api_key_server
from app.context import current_user_email
from app.auth.galaxy_accounts import GalaxyAccountError
from app.auth.identity import identify
from app.auth.tokens import TokenError, check_config

configure_logging()

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
    FastMCP twin of app/api/middleware.py's JWTGalaxyKeyMiddleware: requires
    ``Authorization: Bearer <JWT>`` from the platform's auth service and sets
    ``current_api_key_server`` plus, for verified tokens, ``current_user_email``
    (which lets GalaxyClient recover from an expired key). Keep the two in sync.
    """
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        check_config()

    async def on_request(self, context: MiddlewareContext, call_next: CallNext):
        if context.method in _PREAUTH_METHODS:
            return await call_next(context)

        headers = get_http_headers(include_all=True)
        auth = headers.get("Authorization", None) or headers.get("authorization", None)

        if auth is None or not auth.startswith("Bearer ") or not auth[7:].strip():
            self.log.error("unauthorized, Authorization header with Bearer token is required.")
            raise _unauthorized("Missing or malformed Authorization header")

        try:
            identity = await identify(auth[7:].strip())
        except TokenError as e:
            self.log.warning("Rejected bearer token: %s", e)
            raise _unauthorized(f"Invalid token: {e}")
        except GalaxyAccountError as e:
            self.log.error("Could not resolve the caller's Galaxy account: %s", e)
            raise McpError(ErrorData(code=-32002, message="Galaxy is unavailable; try again shortly."))

        current_api_key_server.set(identity.api_key)
        current_user_email.set(identity.email)
        self.log.info("Incoming request to MCP server validated.")

        return await call_next(context)
