"""Keeps credentials that travel in URLs out of the logs.

Old callers send ``/register-user?email=…&password=…``, and Galaxy client
errors quote request URLs carrying ``key=…``. uvicorn's access log and our
own log lines would otherwise write those values out in plain text.
"""

import logging
import re

_SENSITIVE_PARAMS = (
    "password", "passwd", "pwd", "key", "api_key", "apikey", "token",
    "access_token", "refresh_token", "jwt", "secret",
)
_PARAM_VALUE = re.compile(
    r"(?i)([?&](?:%s)=)[^&#\s\"']*" % "|".join(re.escape(p) for p in _SENSITIVE_PARAMS)
)
REDACTED = "<redacted>"


def redact(text: str) -> str:
    return _PARAM_VALUE.sub(r"\1" + REDACTED, text)


class RedactSecretsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(redact(a) if isinstance(a, str) else a for a in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: redact(v) if isinstance(v, str) else v for k, v in record.args.items()}
        return True


_FILTER = RedactSecretsFilter()


def install(handlers=()) -> None:
    """Attach the filter to uvicorn's loggers and to the given handlers.

    A logger's filters only see records logged on that logger, so uvicorn's
    loggers (which don't propagate) get it directly, and the root handlers get
    it for every other logger's records.
    """
    for name in ("uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(name)
        if _FILTER not in logger.filters:
            logger.addFilter(_FILTER)
    for handler in handlers:
        if _FILTER not in handler.filters:
            handler.addFilter(_FILTER)
