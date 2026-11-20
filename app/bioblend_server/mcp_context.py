from contextvars import ContextVar
from typing import Optional

current_api_key_server: ContextVar[str] = ContextVar("current_api_key_server", default=None)
current_chat_id_server: ContextVar[Optional[str]] = ContextVar("current_chat_id_server", default=None)