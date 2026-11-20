"""Linking platform users to their Galaxy accounts and keys.

- ``POST /register-user``: called by the auth service at login with the user's
  email and password. Returns the Galaxy key for an existing account whose
  password matches. It does not create accounts: a Galaxy account is created
  the first time a verified login token arrives for an email without one
  (app/auth/identity.py), so only the auth service can cause it.
- ``GET /v1/users/me/galaxy-api-key``: the caller's current key, encrypted in
  the format the auth service embeds in its tokens.
- ``POST /v1/users/me/galaxy-api-key``: replace the caller's key with a new one.
- ``POST /refresh-galaxy-key``: deprecated alias of the GET above.

Errors on /register-user always carry ``"api_token": null``: the auth service
reads that field without checking the status and must keep logging users in.
"""

import asyncio
import hashlib
import logging
import os
from typing import Optional

import httpx
import redis
from fastapi import APIRouter, Body, Query, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.auth import galaxy_accounts
from app.auth.galaxy_accounts import GalaxyAccountError
from app.auth.identity import encrypt_api_key, import_workflows_later
from app.context import current_api_key, current_user_email

log = logging.getLogger(__name__)

router = APIRouter()

REGISTER_MAX_ATTEMPTS = 5
REGISTER_WINDOW_SECONDS = 15 * 60

# RFC 9745 / RFC 8594 headers for the routes kept only for old callers.
DEPRECATED_SINCE = "@1794873600"  # 2026-11-17
REFRESH_SUNSET = "Mon, 15 Feb 2027 00:00:00 GMT"
ME_KEY_PATH = "/v1/users/me/galaxy-api-key"


class GalaxyUserAccount(BaseModel):
    username: str
    api_token: str = Field(description="The Galaxy API key, encrypted for the auth service's tokens.")


class RegisterUserRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=1024)


class AccountError(BaseModel):
    detail: str
    username: Optional[str] = None
    api_token: Optional[str] = None


class _InvalidCredentials(Exception):
    pass


def _galaxy_url() -> str:
    return os.getenv("GALAXY_URL", "").rstrip("/")


# --- /register-user ---------------------------------------------------------

def _account_error(status: int, detail: str, headers: Optional[dict] = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"detail": detail, "username": None, "api_token": None},
        headers=headers,
    )


_redis_client = None


def _redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=os.getenv("REDIS_PORT"),
            db=0,
            decode_responses=True,
            socket_timeout=2,
        )
    return _redis_client


def _attempts_key(email: str) -> str:
    return "register_user_attempts:" + hashlib.sha256(email.encode("utf-8")).hexdigest()


def _retry_after_if_limited(email: str) -> Optional[int]:
    """Count a password attempt; return seconds to wait if over the limit."""
    try:
        r = _redis()
        key = _attempts_key(email)
        count = r.incr(key)
        if count == 1:
            r.expire(key, REGISTER_WINDOW_SECONDS)
        if count > REGISTER_MAX_ATTEMPTS:
            return max(int(r.ttl(key)), 1)
    except redis.RedisError as e:
        log.warning("register-user attempt limit unavailable (Redis): %s", e)
    return None


def _clear_attempts(email: str) -> None:
    try:
        _redis().delete(_attempts_key(email))
    except redis.RedisError:
        pass


def _key_with_password(email: str, password: str) -> str:
    """The account's current key, if Galaxy accepts this email and password."""
    try:
        with galaxy_accounts.http_client() as client:
            r = client.get(f"{_galaxy_url()}/api/authenticate/baseauth", auth=(email, password))
    except httpx.HTTPError as e:
        raise GalaxyAccountError(f"Galaxy request failed: {e}") from e
    if r.status_code in (400, 401, 403, 404):
        raise _InvalidCredentials()
    if r.status_code >= 400:
        raise GalaxyAccountError(f"Galaxy answered {r.status_code}")
    key = r.json().get("api_key")
    if not key:
        raise GalaxyAccountError("Galaxy returned no API key")
    return key


def _current_user(api_key: str) -> dict:
    try:
        with galaxy_accounts.http_client() as client:
            r = client.get(f"{_galaxy_url()}/api/users/current", headers={"x-api-key": api_key})
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise GalaxyAccountError(f"Galaxy request failed: {e}") from e
    return r.json()


@router.post(
    "/register-user",
    tags=["Signup Auth"],
    response_model=GalaxyUserAccount,
    responses={401: {"model": AccountError}, 422: {"model": AccountError},
               429: {"model": AccountError}, 502: {"model": AccountError}},
    summary="Galaxy key for an existing account, given its email and password",
)
async def register_user(
    response: Response,
    body: Optional[RegisterUserRequest] = Body(None),
    email: Optional[str] = Query(None, deprecated=True, description="Deprecated: send it in the JSON body."),
    password: Optional[str] = Query(None, deprecated=True, description="Deprecated: send it in the JSON body."),
):
    """Called by the auth service at login. Returns the Galaxy account's current
    API key (without creating a new one) when the email and password match.

    A new platform user gets ``401``: their Galaxy account is created on their
    first request with a login token instead. Every error body carries
    ``"api_token": null``.
    """
    legacy = body is None
    headers = {"Deprecation": DEPRECATED_SINCE} if legacy and (email or password) else None
    login = (body.email if body else email) or ""
    secret = (body.password if body else password) or ""
    if not login.strip() or not secret:
        return _account_error(422, "email and password are required", headers)
    login = galaxy_accounts.normalize_email(login)

    retry_after = await asyncio.to_thread(_retry_after_if_limited, login)
    if retry_after:
        return _account_error(429, "Too many attempts; try again later.",
                              {**(headers or {}), "Retry-After": str(retry_after)})

    try:
        api_key = await asyncio.to_thread(_key_with_password, login, secret)
        user = await asyncio.to_thread(_current_user, api_key)
    except _InvalidCredentials:
        return _account_error(401, "Invalid email or password, or no Galaxy account yet.", headers)
    except GalaxyAccountError as e:
        log.error("register-user: Galaxy unavailable: %s", e)
        return _account_error(502, "Galaxy is unavailable; try again shortly.", headers)

    await asyncio.to_thread(_clear_attempts, login)
    if user.get("email"):
        galaxy_accounts.remember_key(user["email"], api_key)
    import_workflows_later(api_key)
    if headers:
        response.headers.update(headers)
    log.info("register-user: Galaxy key issued for %s", user.get("username"))
    return GalaxyUserAccount(username=user.get("username") or "", api_token=encrypt_api_key(api_key))


# --- the caller's own key ---------------------------------------------------

async def _my_account(status_code: int = 200, extra_headers: Optional[dict] = None,
                      api_key: Optional[str] = None) -> JSONResponse:
    api_key = api_key or current_api_key.get()
    try:
        user = await asyncio.to_thread(_current_user, api_key)
    except GalaxyAccountError as e:
        log.error("Galaxy unavailable reading the caller's account: %s", e)
        return JSONResponse(status_code=502, content={"detail": "Galaxy is unavailable; try again shortly."})
    headers = {"Cache-Control": "no-store", **(extra_headers or {})}
    return JSONResponse(
        status_code=status_code,
        content={"username": user.get("username") or "", "api_token": encrypt_api_key(api_key)},
        headers=headers,
    )


@router.get(
    ME_KEY_PATH,
    tags=["Accounts"],
    response_model=GalaxyUserAccount,
    summary="The caller's current Galaxy API key",
)
async def get_my_galaxy_api_key():
    """The Galaxy key the caller's requests use, encrypted in the auth service's
    token format. Reading it never replaces the key."""
    return await _my_account()


@router.post(
    ME_KEY_PATH,
    tags=["Accounts"],
    status_code=201,
    response_model=GalaxyUserAccount,
    summary="Replace the caller's Galaxy API key",
)
async def create_my_galaxy_api_key():
    """Creates a new Galaxy key for the caller. Galaxy then treats the previous
    key as expired everywhere it is held, e.g. to cut off a leaked key."""
    api_key = current_api_key.get()

    def rotate() -> str:
        user = _current_user(api_key)
        try:
            with galaxy_accounts.http_client() as client:
                r = client.post(f"{_galaxy_url()}/api/users/{user['id']}/api_key", headers={"x-api-key": api_key})
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise GalaxyAccountError(f"Galaxy request failed: {e}") from e
        new_key = r.json()
        if user.get("email"):
            galaxy_accounts.remember_key(user["email"], new_key)
        return new_key

    try:
        new_key = await asyncio.to_thread(rotate)
    except GalaxyAccountError as e:
        log.error("Could not replace the caller's Galaxy key: %s", e)
        return JSONResponse(status_code=502, content={"detail": "Galaxy is unavailable; try again shortly."})
    return await _my_account(201, {"Location": ME_KEY_PATH}, api_key=new_key)


@router.post(
    "/refresh-galaxy-key",
    tags=["Signup Auth"],
    response_model=GalaxyUserAccount,
    deprecated=True,
    responses={403: {"description": "The email is not the caller's"}},
    summary="Deprecated: use GET /v1/users/me/galaxy-api-key",
)
async def refresh_galaxy_key(
    email: Optional[str] = Query(None, description="Optional; must be the caller's own email."),
):
    """The caller's current Galaxy key. Needs the caller's bearer token and acts
    only on the caller; it no longer creates a new key."""
    if email and galaxy_accounts.normalize_email(email) != current_user_email.get(None):
        return JSONResponse(status_code=403, content={"detail": "You can only read your own Galaxy key."})
    return await _my_account(extra_headers={
        "Deprecation": DEPRECATED_SINCE,
        "Sunset": REFRESH_SUNSET,
        "Link": f'<{ME_KEY_PATH}>; rel="successor-version"',
    })
