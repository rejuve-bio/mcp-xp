import logging
from contextvars import ContextVar
from typing import Any, Optional

current_api_key: ContextVar[str] = ContextVar("current_api_key")
current_user_email: ContextVar[Optional[str]] = ContextVar("current_user_email", default=None)
current_chat_id: ContextVar[Optional[str]] = ContextVar("current_chat_id", default=None)

CHAT_ID_MAX_LENGTH = 128
_chat_id_log = logging.getLogger("chat_id")


def validate_chat_id(raw: Any) -> Optional[str]:
    """Sanitize a frontend-supplied chat id.

    Returns the trimmed value when it is a non-empty string <= CHAT_ID_MAX_LENGTH.
    Returns None for missing/empty/wrong-type/oversized inputs — with a warning
    logged for the last two cases so frontend bugs surface without breaking
    the request.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        _chat_id_log.warning("Ignoring non-string X-Chat-Id header (type=%s)", type(raw).__name__)
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    if len(stripped) > CHAT_ID_MAX_LENGTH:
        _chat_id_log.warning(
            "Ignoring oversized X-Chat-Id header (length=%d, max=%d)",
            len(stripped),
            CHAT_ID_MAX_LENGTH,
        )
        return None
    return stripped