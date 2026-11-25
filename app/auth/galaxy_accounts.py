"""Galaxy accounts for platform users, looked up with the admin API key.

Only call these with an email that came from a verified login token (or from
Galaxy itself): they act on whichever account owns that email.

Galaxy treats only a user's newest API key as valid. ``current_key`` reads that
key without creating a new one, so looking it up never logs the user out of
their other sessions.
"""

import logging
import os
import re
import secrets
import threading
import time
from typing import Optional

import httpx

log = logging.getLogger(__name__)

KEY_CACHE_SECONDS = 300
_TIMEOUT = httpx.Timeout(10.0)
# Galaxy's rule for public names (usernames).
_USERNAME_INVALID = re.compile(r"[^a-z0-9._-]+")

# Tests swap in an httpx.MockTransport here.
TRANSPORT: Optional[httpx.BaseTransport] = None

_cache: dict[str, tuple[str, float]] = {}
_cache_lock = threading.Lock()


class GalaxyAccountError(Exception):
    """Galaxy could not be reached or refused an admin request."""


def _galaxy_url() -> str:
    url = os.getenv("GALAXY_URL", "").rstrip("/")
    if not url:
        raise GalaxyAccountError("GALAXY_URL is not set")
    return url


def _admin_headers() -> dict:
    key = os.getenv("GALAXY_API_KEY")
    if not key:
        raise GalaxyAccountError("GALAXY_API_KEY (admin) is not set")
    return {"x-api-key": key}


def http_client() -> httpx.Client:
    return httpx.Client(timeout=_TIMEOUT, transport=TRANSPORT)


def normalize_email(email: str) -> str:
    return email.strip().lower()


# --- short-lived cache of each user's current key -------------------------

def cached_key(email: str) -> Optional[str]:
    email = normalize_email(email)
    with _cache_lock:
        entry = _cache.get(email)
        if entry and entry[1] > time.monotonic():
            return entry[0]
        _cache.pop(email, None)
    return None


def remember_key(email: str, key: str) -> None:
    with _cache_lock:
        _cache[normalize_email(email)] = (key, time.monotonic() + KEY_CACHE_SECONDS)


def forget_key(email: str) -> None:
    with _cache_lock:
        _cache.pop(normalize_email(email), None)


# --- admin API -------------------------------------------------------------

def find_user(client: httpx.Client, email: str) -> Optional[dict]:
    """The account whose email is exactly ``email``, or None.

    Galaxy's ``f_email`` filter is a substring match (``LIKE '%…%'``), so the
    results are filtered here; the first result is not necessarily the user.
    """
    email = normalize_email(email)
    r = client.get(f"{_galaxy_url()}/api/users", headers=_admin_headers(), params={"f_email": email})
    r.raise_for_status()
    matches = [u for u in r.json() if normalize_email(u.get("email") or "") == email]
    return matches[0] if matches else None


def current_key(client: httpx.Client, user_id: str) -> str:
    """The user's newest API key; Galaxy creates one only if they have none."""
    r = client.get(f"{_galaxy_url()}/api/users/{user_id}/api_key", headers=_admin_headers())
    r.raise_for_status()
    key = r.json()
    if not isinstance(key, str) or not key:
        raise GalaxyAccountError("Galaxy returned no API key")
    return key


def _username_for(email: str) -> str:
    base = _USERNAME_INVALID.sub("-", email.split("@", 1)[0].lower()).strip("-.") or "user"
    return base if len(base) >= 3 else f"{base}-user"


def create_user(client: httpx.Client, email: str) -> dict:
    """Create a Galaxy account for ``email`` with a random password.

    Nothing in mcp-xp logs in with the password; users who want Galaxy's own
    website use its "forgot password" flow.
    """
    email = normalize_email(email)
    base = _username_for(email)
    for attempt in range(4):
        username = base if attempt == 0 else f"{base}-{secrets.token_hex(2)}"
        r = client.post(
            f"{_galaxy_url()}/api/users",
            headers=_admin_headers(),
            json={"username": username, "email": email, "password": secrets.token_urlsafe(32)},
        )
        if r.status_code < 400:
            log.info("Created Galaxy account %s", r.json().get("username"))
            return r.json()
        if r.status_code != 400:
            r.raise_for_status()
        # 400: the email was registered meanwhile, or the username is taken.
        existing = find_user(client, email)
        if existing:
            return existing
    raise GalaxyAccountError("could not create a Galaxy account with a free username")


def key_for_email(email: str, *, create: bool) -> tuple[Optional[str], bool]:
    """The current key for ``email``'s account, and whether the account was created.

    Returns (None, False) when there is no account and ``create`` is False.
    """
    cached = cached_key(email)
    if cached:
        return cached, False
    try:
        with http_client() as client:
            user = find_user(client, email)
            created = False
            if user is None:
                if not create:
                    return None, False
                user = create_user(client, email)
                created = True
            key = current_key(client, user["id"])
    except httpx.HTTPError as e:
        raise GalaxyAccountError(f"Galaxy request failed: {e}") from e
    remember_key(email, key)
    return key, created
