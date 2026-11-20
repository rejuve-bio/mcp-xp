"""Checks the platform login tokens (JWTs issued by the authentication service).

The auth service signs its tokens with HS256 and a secret it shares with each
service that accepts them. mcp-xp reads that secret from ``JWT_SECRET`` (a
comma-separated list, so a secret can be rotated or a second issuer added).

``JWT_VERIFY`` picks how strict this is:

- ``enforce`` (default): only tokens with a valid signature, an ``exp`` in the
  future and not a refresh token are accepted.
- ``report``: tokens that fail those checks are still accepted the old way and
  logged, so a deployment can confirm every caller passes before enforcing.
- ``off``: no checks (local development only).

Only a token that passed the checks is ``verified``. Callers must not act on a
token's ``email`` unless it is verified.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import jwt

GALAXY_API_TOKEN_CLAIM = "galaxy_api_token"
ALGORITHM = "HS256"
# Tolerated clock difference between the auth service and this server.
LEEWAY_SECONDS = 30

log = logging.getLogger(__name__)


class VerifyMode(str, Enum):
    ENFORCE = "enforce"
    REPORT = "report"
    OFF = "off"


class TokenError(Exception):
    """The bearer token is malformed, wrongly signed, expired or the wrong kind."""


@dataclass(frozen=True)
class PlatformToken:
    claims: dict
    verified: bool

    @property
    def email(self) -> Optional[str]:
        # The auth service names this claim `email`, except on some refreshed
        # tokens, which carry `user_email`.
        value = self.claims.get("email") or self.claims.get("user_email")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
        return None

    @property
    def galaxy_claim(self):
        return self.claims.get(GALAXY_API_TOKEN_CLAIM)


def verify_mode() -> VerifyMode:
    raw = os.getenv("JWT_VERIFY", VerifyMode.ENFORCE.value).strip().lower()
    try:
        return VerifyMode(raw)
    except ValueError:
        raise RuntimeError(f"JWT_VERIFY must be enforce, report or off, not {raw!r}") from None


def _secrets() -> list[str]:
    return [s.strip() for s in os.getenv("JWT_SECRET", "").split(",") if s.strip()]


def check_config() -> VerifyMode:
    """Fail at startup rather than on the first request when misconfigured."""
    mode = verify_mode()
    if mode is not VerifyMode.OFF and not _secrets():
        raise RuntimeError("JWT_SECRET is required unless JWT_VERIFY=off")
    if mode is VerifyMode.OFF:
        log.warning("JWT_VERIFY=off: login tokens are NOT checked. Never use this on a shared server.")
    elif mode is VerifyMode.REPORT:
        log.warning("JWT_VERIFY=report: failing tokens are logged but still accepted.")
    return mode


def _verify(token: str) -> dict:
    secrets = _secrets()
    if not secrets:
        raise RuntimeError("JWT_SECRET is required unless JWT_VERIFY=off")
    try:
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError as e:
        raise TokenError(f"malformed token: {e}") from None
    if header.get("alg") != ALGORITHM:
        raise TokenError("unsupported token algorithm")

    for secret in secrets:
        try:
            claims = jwt.decode(
                token,
                secret,
                algorithms=[ALGORITHM],
                leeway=LEEWAY_SECONDS,
                # The deployed auth service issues numeric `sub` values, which
                # PyJWT would otherwise reject.
                options={"require": ["exp"], "verify_sub": False},
            )
        except jwt.InvalidSignatureError:
            continue
        except jwt.ExpiredSignatureError:
            raise TokenError("token has expired") from None
        except jwt.InvalidTokenError as e:
            raise TokenError(f"invalid token: {e}") from None
        if claims.get("type") == "refresh":
            raise TokenError("refresh tokens are not accepted")
        return claims
    raise TokenError("invalid token signature")


def read_token(token: str) -> PlatformToken:
    """Check a bearer token according to ``JWT_VERIFY``. Raises TokenError."""
    mode = verify_mode()
    if mode is VerifyMode.ENFORCE:
        return PlatformToken(_verify(token), verified=True)

    try:
        claims = jwt.decode(token, options={"verify_signature": False})
    except jwt.InvalidTokenError as e:
        raise TokenError(f"malformed token: {e}") from None
    if mode is VerifyMode.OFF:
        return PlatformToken(claims, verified=False)

    try:
        return PlatformToken(_verify(token), verified=True)
    except TokenError as e:
        log.warning(
            "JWT_VERIFY=report: token would be rejected (%s); accepted unverified. Claim names: %s",
            e,
            sorted(claims),
        )
        return PlatformToken(claims, verified=False)
