"""Turns a bearer token into the Galaxy API key to act with.

Shared by the REST middleware, the MCP middleware and the background tasks so
all three trust tokens the same way.
"""

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from app.auth import galaxy_accounts
from app.auth.tokens import PlatformToken, TokenError, read_token

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Identity:
    api_key: str
    # Set only when the token was verified; anything keyed on the user's email
    # (key lookup, key refresh) must use this, never the raw claim.
    email: Optional[str]


def encrypt_api_key(api_key: str) -> str:
    """The `galaxy_api_token` format the auth service embeds in its tokens."""
    return _fernet().encrypt(json.dumps({"apikey": api_key}).encode("utf-8")).decode("utf-8")


def decrypt_api_key(claim) -> Optional[str]:
    if not isinstance(claim, str) or not claim:
        return None
    try:
        parsed = json.loads(_fernet().decrypt(claim.encode("utf-8")).decode("utf-8"))
    except (InvalidToken, ValueError):
        return None
    key = parsed.get("apikey") if isinstance(parsed, dict) else None
    return key if isinstance(key, str) and key else None


def _fernet() -> Fernet:
    secret = os.getenv("SECRET_KEY")
    if not secret:
        raise RuntimeError("SECRET_KEY (Fernet secret) is required in env")
    return Fernet(secret)


def import_workflows_later(api_key: str) -> None:
    """New accounts get the published workflows, as /register-user used to do."""
    def run():
        try:
            from app.utils import import_published_workflows
            import_published_workflows(galaxy_url=os.getenv("GALAXY_URL"), api_key=api_key)
        except Exception as e:  # best effort; the account works without them
            log.warning("Importing published workflows for a new account failed: %s", e)

    threading.Thread(target=run, name="import-published-workflows", daemon=True).start()


async def identify(bearer: str, *, create_account: bool = True) -> Identity:
    """Raises TokenError for an unusable token, GalaxyAccountError if Galaxy fails."""
    token: PlatformToken = read_token(bearer)
    claim_key = decrypt_api_key(token.galaxy_claim)

    if token.verified:
        email = token.email
        if email:
            cached = galaxy_accounts.cached_key(email)
            if cached:
                return Identity(cached, email)
            if claim_key:
                return Identity(claim_key, email)
            # No key in the token (social sign-in, refreshed token, or a new
            # user): the auth service vouches for the email, so look it up.
            key, created = await asyncio.to_thread(
                galaxy_accounts.key_for_email, email, create=create_account
            )
            if key is None:
                raise TokenError("no Galaxy account for this user")
            if created:
                import_workflows_later(key)
            return Identity(key, email)
        if claim_key:
            return Identity(claim_key, None)
        raise TokenError("token carries no Galaxy key and no email")

    # Unverified (JWT_VERIFY=report or off): the old behaviour, without
    # trusting the email for anything.
    if claim_key:
        return Identity(claim_key, None)
    raw = token.galaxy_claim
    if isinstance(raw, str) and raw.strip():
        return Identity(raw.strip(), None)
    raise TokenError("token carries no usable Galaxy key")
