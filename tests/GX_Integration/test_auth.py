"""Login tokens, Galaxy account lookup and the account routes.

A fake Galaxy (httpx.MockTransport) mimics the behaviour these depend on:
``f_email`` is a substring match, only a user's newest key is valid, and
``GET /api/users/{id}/api_key`` reads the key without replacing it.
"""

import base64
import json
import logging
import time
from sys import path
from urllib.parse import parse_qs, urlparse

path.append(".")

import httpx
import jwt
import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import log_redaction
from app.api.endpoints import accounts
from app.api.middleware import JWTGalaxyKeyMiddleware
from app.auth import galaxy_accounts, identity, tokens
from app.context import current_api_key, current_user_email

JWT_SECRET = "test-jwt-secret-0123456789abcdef0123456789"
ADMIN_KEY = "admin-key"
GALAXY = "http://galaxy.test"


# --- fake Galaxy ---------------------------------------------------------------

class FakeGalaxy:
    def __init__(self):
        self.users = {}
        self.keys = {}  # user id -> list of keys, newest last
        self.minted = 0
        self.created_users = []

    def add_user(self, email, password, username=None, key=None):
        uid = f"u{len(self.users) + 1}"
        self.users[uid] = {"id": uid, "email": email, "username": username or email.split("@")[0],
                           "password": password}
        self.keys[uid] = [key] if key else []
        return uid

    def newest(self, uid):
        if not self.keys[uid]:
            self._mint(uid)
        return self.keys[uid][-1]

    def _mint(self, uid):
        self.minted += 1
        key = f"key-{uid}-{self.minted}"
        self.keys[uid].append(key)
        return key

    def _user_for_key(self, key):
        for uid, keys in self.keys.items():
            if key in keys:
                return uid, keys[-1] == key
        return None, False

    def _public(self, uid):
        u = self.users[uid]
        return {"id": u["id"], "email": u["email"], "username": u["username"]}

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = urlparse(str(request.url))
        key = request.headers.get("x-api-key")
        admin = key == ADMIN_KEY
        route = url.path

        if route == "/api/authenticate/baseauth":
            decoded = base64.b64decode(request.headers["authorization"].split()[1]).decode()
            email, password = decoded.split(":", 1)
            for uid, u in self.users.items():
                if u["email"] == email and u["password"] == password:
                    return httpx.Response(200, json={"api_key": self.newest(uid)})
            return httpx.Response(401, json={"err_msg": "Invalid password.", "err_code": 401000})

        if route == "/api/users" and request.method == "GET" and admin:
            needle = parse_qs(url.query).get("f_email", [""])[0]
            found = sorted((self._public(uid) for uid, u in self.users.items() if needle in u["email"]),
                           key=lambda u: u["email"])
            return httpx.Response(200, json=found)

        if route == "/api/users" and request.method == "POST" and admin:
            body = json.loads(request.content)
            if any(u["email"] == body["email"] or u["username"] == body["username"] for u in self.users.values()):
                return httpx.Response(400, json={"err_msg": "taken"})
            uid = self.add_user(body["email"], body["password"], body["username"])
            self.created_users.append(uid)
            return httpx.Response(200, json=self._public(uid))

        if route == "/api/users/current":
            uid, newest = self._user_for_key(key)
            if uid and newest:
                return httpx.Response(200, json=self._public(uid))
            if uid:
                return httpx.Response(401, json={"err_msg": "Provided API key has expired.", "err_code": 401001})
            return httpx.Response(401, json={"err_msg": "Invalid key", "err_code": 401000})

        if route.startswith("/api/users/") and route.endswith("/api_key"):
            uid = route.split("/")[3]
            owner, newest = self._user_for_key(key)
            if not (admin or (owner == uid and newest)):
                return httpx.Response(403, json={"err_msg": "forbidden"})
            if request.method == "GET":
                return httpx.Response(200, json=self.newest(uid))
            if request.method == "POST":
                return httpx.Response(200, json=self._mint(uid))

        return httpx.Response(404, json={"err_msg": f"no fake route {request.method} {route}"})


@pytest.fixture
def fernet_key():
    return Fernet.generate_key().decode()


@pytest.fixture
def galaxy(monkeypatch, fernet_key):
    fake = FakeGalaxy()
    monkeypatch.setenv("SECRET_KEY", fernet_key)
    monkeypatch.setenv("JWT_SECRET", JWT_SECRET)
    monkeypatch.delenv("JWT_VERIFY", raising=False)
    monkeypatch.setenv("GALAXY_URL", GALAXY)
    monkeypatch.setenv("GALAXY_API_KEY", ADMIN_KEY)
    monkeypatch.setattr(galaxy_accounts, "TRANSPORT", httpx.MockTransport(fake.handler))
    monkeypatch.setattr(identity, "import_workflows_later", lambda key: None)
    monkeypatch.setattr(accounts, "import_workflows_later", lambda key: None)
    monkeypatch.setattr(accounts, "_retry_after_if_limited", lambda email: None)
    monkeypatch.setattr(accounts, "_clear_attempts", lambda email: None)
    with galaxy_accounts._cache_lock:
        galaxy_accounts._cache.clear()
    yield fake
    with galaxy_accounts._cache_lock:
        galaxy_accounts._cache.clear()


def token_for(claims, secret=JWT_SECRET, exp_in=300, **extra):
    return jwt.encode({"exp": int(time.time()) + exp_in, **claims, **extra}, secret, algorithm="HS256")


def encrypted(key):
    return identity.encrypt_api_key(key)


@pytest.fixture
def client(galaxy):
    app = FastAPI()

    @app.get("/whoami")
    async def whoami():
        return {"api_key": current_api_key.get(None), "email": current_user_email.get(None)}

    app.include_router(accounts.router)
    app.add_middleware(JWTGalaxyKeyMiddleware)
    return TestClient(app)


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


# --- token checks --------------------------------------------------------------

class TestTokens:
    @pytest.fixture(autouse=True)
    def env(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET", JWT_SECRET)
        monkeypatch.delenv("JWT_VERIFY", raising=False)

    def test_valid_token_is_verified(self):
        t = tokens.read_token(token_for({"email": "A@x.org", "sub": 7}))
        assert t.verified and t.email == "a@x.org"

    def test_user_email_claim_from_refreshed_tokens(self):
        assert tokens.read_token(token_for({"user_email": "b@x.org"})).email == "b@x.org"

    @pytest.mark.parametrize("make", [
        lambda: jwt.encode({"email": "a@x.org", "exp": int(time.time()) + 60}, None, algorithm="none"),
        lambda: token_for({"email": "a@x.org"}, secret="another-secret-0123456789abcdef0123"),
        lambda: token_for({"email": "a@x.org"}, exp_in=-120),
        lambda: jwt.encode({"email": "a@x.org"}, JWT_SECRET, algorithm="HS256"),
        lambda: token_for({"email": "a@x.org", "type": "refresh"}),
        lambda: jwt.encode({"email": "a@x.org", "exp": int(time.time()) + 60}, JWT_SECRET, algorithm="HS512"),
        lambda: "not-a-jwt",
    ], ids=["alg-none", "wrong-secret", "expired", "no-exp", "refresh-token", "other-alg", "garbage"])
    def test_rejected_when_enforced(self, make):
        with pytest.raises(tokens.TokenError):
            tokens.read_token(make())

    def test_any_listed_secret_is_accepted(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET", f"old-secret-0123456789abcdef0123456, {JWT_SECRET}")
        assert tokens.read_token(token_for({"email": "a@x.org"})).verified

    def test_report_mode_accepts_failing_tokens_unverified(self, monkeypatch, caplog):
        monkeypatch.setenv("JWT_VERIFY", "report")
        bad = token_for({"email": "a@x.org"}, secret="another-secret-0123456789abcdef0123")
        with caplog.at_level(logging.WARNING):
            t = tokens.read_token(bad)
        assert not t.verified
        assert "would be rejected" in caplog.text

    def test_report_mode_still_verifies_good_tokens(self, monkeypatch):
        monkeypatch.setenv("JWT_VERIFY", "report")
        assert tokens.read_token(token_for({"email": "a@x.org"})).verified

    def test_config_requires_a_secret(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET")
        with pytest.raises(RuntimeError):
            tokens.check_config()
        monkeypatch.setenv("JWT_VERIFY", "off")
        assert tokens.check_config() is tokens.VerifyMode.OFF

    def test_unknown_mode_is_an_error(self, monkeypatch):
        monkeypatch.setenv("JWT_VERIFY", "maybe")
        with pytest.raises(RuntimeError):
            tokens.check_config()


# --- account lookup ------------------------------------------------------------

class TestGalaxyAccounts:
    def test_find_user_needs_an_exact_email(self, galaxy):
        galaxy.add_user("bob@x.org", "pw")
        galaxy.add_user("ob@x.org", "pw")
        with galaxy_accounts.http_client() as c:
            assert galaxy_accounts.find_user(c, "ob@x.org")["email"] == "ob@x.org"
            # "%" or a fragment matches others in Galaxy's filter, but no one exactly.
            assert galaxy_accounts.find_user(c, "%") is None
            assert galaxy_accounts.find_user(c, "x.org") is None

    def test_key_lookup_does_not_rotate(self, galaxy):
        uid = galaxy.add_user("a@x.org", "pw", key="k1")
        key, created = galaxy_accounts.key_for_email("a@x.org", create=False)
        assert (key, created) == ("k1", False)
        assert galaxy.keys[uid] == ["k1"]

    def test_creates_account_only_when_asked(self, galaxy):
        assert galaxy_accounts.key_for_email("new@x.org", create=False) == (None, False)
        key, created = galaxy_accounts.key_for_email("new@x.org", create=True)
        assert created and key
        assert galaxy.users[galaxy.created_users[0]]["email"] == "new@x.org"

    def test_username_collision_gets_a_suffix(self, galaxy):
        galaxy.add_user("alice@other.org", "pw", username="alice")
        galaxy_accounts.key_for_email("alice@x.org", create=True)
        assert galaxy.users[galaxy.created_users[0]]["username"].startswith("alice-")


# --- middleware + identity -------------------------------------------------------

class TestIdentity:
    def test_key_from_token(self, client, galaxy):
        r = client.get("/whoami", headers=bearer(token_for({"email": "a@x.org", "galaxy_api_token": encrypted("k1")})))
        assert r.json() == {"api_key": "k1", "email": "a@x.org"}

    def test_token_without_key_uses_the_account_by_email(self, client, galaxy):
        galaxy.add_user("a@x.org", "pw", key="k1")
        for claim in (None, ""):
            r = client.get("/whoami", headers=bearer(token_for({"email": "a@x.org", "galaxy_api_token": claim})))
            assert r.json()["api_key"] == "k1"
        assert galaxy.minted == 0

    def test_first_request_creates_the_galaxy_account(self, client, galaxy):
        r = client.get("/whoami", headers=bearer(token_for({"email": "new@x.org", "galaxy_api_token": None})))
        assert r.status_code == 200
        assert len(galaxy.created_users) == 1

    def test_forged_token_cannot_create_or_reach_accounts(self, client, galaxy):
        galaxy.add_user("victim@x.org", "pw", key="k1")
        forged = token_for({"email": "victim@x.org"}, secret="attacker-secret-0123456789abcdef012")
        assert client.get("/whoami", headers=bearer(forged)).status_code == 401
        forged_new = token_for({"email": "someone@x.org"}, secret="attacker-secret-0123456789abcdef012")
        assert client.get("/whoami", headers=bearer(forged_new)).status_code == 401
        assert galaxy.created_users == [] and galaxy.minted == 0

    def test_report_mode_keeps_old_tokens_working_without_trusting_email(self, client, galaxy, monkeypatch):
        monkeypatch.setenv("JWT_VERIFY", "report")
        galaxy.add_user("victim@x.org", "pw", key="k1")
        unsigned = jwt.encode({"email": "victim@x.org", "galaxy_api_token": "tester-raw-key"}, None, algorithm="none")
        r = client.get("/whoami", headers=bearer(unsigned))
        assert r.json() == {"api_key": "tester-raw-key", "email": None}
        no_key = jwt.encode({"email": "victim@x.org"}, None, algorithm="none")
        assert client.get("/whoami", headers=bearer(no_key)).status_code == 401
        assert galaxy.minted == 0

    def test_galaxy_down_is_503(self, client, galaxy, monkeypatch):
        def down(request):
            raise httpx.ConnectError("down")
        monkeypatch.setattr(galaxy_accounts, "TRANSPORT", httpx.MockTransport(down))
        r = client.get("/whoami", headers=bearer(token_for({"email": "a@x.org"})))
        assert r.status_code == 503


class TestExpiredKeyRecovery:
    def test_refresh_reads_the_current_key_instead_of_minting(self, galaxy):
        from app.galaxy import GalaxyClient
        uid = galaxy.add_user("a@x.org", "pw", key="old")
        galaxy.keys[uid].append("current")  # e.g. the user made a new key in Galaxy
        gc = GalaxyClient("old", galaxy_url=GALAXY, user_email="a@x.org")
        assert gc._refresh_user_key() == "current"
        assert galaxy.minted == 0
        assert galaxy_accounts.cached_key("a@x.org") == "current"

    def test_no_refresh_without_a_verified_email(self, galaxy):
        from app.galaxy import GalaxyClient
        gc = GalaxyClient("old", galaxy_url=GALAXY, user_email=None)
        with pytest.raises(RuntimeError):
            gc._refresh_user_key()


# --- /register-user ------------------------------------------------------------

class TestRegisterUser:
    def test_json_body_with_right_password(self, client, galaxy):
        uid = galaxy.add_user("a@x.org", "pw", key="k1")
        r = client.post("/register-user", json={"email": "a@x.org", "password": "pw"})
        assert r.status_code == 200
        assert identity.decrypt_api_key(r.json()["api_token"]) == "k1"
        assert galaxy.keys[uid] == ["k1"]  # no new key minted
        assert "Deprecation" not in r.headers

    def test_query_params_still_work_but_are_deprecated(self, client, galaxy):
        galaxy.add_user("a@x.org", "pw", key="k1")
        r = client.post("/register-user?email=a%40x.org&password=pw")
        assert r.status_code == 200
        assert r.headers["Deprecation"] == accounts.DEPRECATED_SINCE

    def test_wrong_password_gets_no_key(self, client, galaxy):
        galaxy.add_user("a@x.org", "pw", key="k1")
        r = client.post("/register-user", json={"email": "a@x.org", "password": "guess"})
        assert r.status_code == 401
        assert r.json()["api_token"] is None
        assert galaxy.minted == 0

    def test_does_not_create_accounts(self, client, galaxy):
        r = client.post("/register-user", json={"email": "new@x.org", "password": "chosen-by-anyone"})
        assert r.status_code == 401 and r.json()["api_token"] is None
        assert galaxy.created_users == []

    def test_wildcard_email_matches_nobody(self, client, galaxy):
        galaxy.add_user("admin@x.org", "pw", key="k1")
        r = client.post("/register-user?email=%25&password=x")
        assert r.status_code == 401 and r.json()["api_token"] is None

    def test_missing_fields(self, client, galaxy):
        r = client.post("/register-user")
        assert r.status_code == 422 and r.json()["api_token"] is None

    def test_rate_limited(self, client, galaxy, monkeypatch):
        monkeypatch.setattr(accounts, "_retry_after_if_limited", lambda email: 42)
        r = client.post("/register-user", json={"email": "a@x.org", "password": "pw"})
        assert r.status_code == 429
        assert r.headers["Retry-After"] == "42" and r.json()["api_token"] is None

    def test_galaxy_down(self, client, galaxy, monkeypatch):
        def down(request):
            raise httpx.ConnectError("down")
        monkeypatch.setattr(galaxy_accounts, "TRANSPORT", httpx.MockTransport(down))
        r = client.post("/register-user", json={"email": "a@x.org", "password": "pw"})
        assert r.status_code == 502 and r.json()["api_token"] is None


# --- the caller's own key -----------------------------------------------------------

class TestMyGalaxyKey:
    def test_get_returns_current_key_without_rotating(self, client, galaxy):
        galaxy.add_user("a@x.org", "pw", key="k1")
        r = client.get(accounts.ME_KEY_PATH, headers=bearer(token_for({"email": "a@x.org"})))
        assert r.status_code == 200
        assert identity.decrypt_api_key(r.json()["api_token"]) == "k1"
        assert r.headers["Cache-Control"] == "no-store"
        assert galaxy.minted == 0

    def test_post_replaces_the_key(self, client, galaxy):
        uid = galaxy.add_user("a@x.org", "pw", key="k1")
        r = client.post(accounts.ME_KEY_PATH, headers=bearer(token_for({"email": "a@x.org"})))
        assert r.status_code == 201
        assert r.headers["Location"] == accounts.ME_KEY_PATH
        new_key = identity.decrypt_api_key(r.json()["api_token"])
        assert new_key != "k1" and galaxy.keys[uid][-1] == new_key

    def test_requires_a_token(self, client, galaxy):
        assert client.get(accounts.ME_KEY_PATH).status_code == 401


class TestRefreshGalaxyKey:
    def test_needs_a_token(self, client, galaxy):
        galaxy.add_user("victim@x.org", "pw", key="k1")
        r = client.post("/refresh-galaxy-key?email=victim%40x.org")
        assert r.status_code == 401
        assert galaxy.minted == 0

    def test_cannot_read_someone_else(self, client, galaxy):
        galaxy.add_user("victim@x.org", "pw", key="k1")
        galaxy.add_user("me@x.org", "pw", key="k2")
        r = client.post("/refresh-galaxy-key?email=victim%40x.org", headers=bearer(token_for({"email": "me@x.org"})))
        assert r.status_code == 403

    def test_own_key_with_deprecation_headers(self, client, galaxy):
        galaxy.add_user("me@x.org", "pw", key="k2")
        r = client.post("/refresh-galaxy-key?email=Me%40x.org", headers=bearer(token_for({"email": "me@x.org"})))
        assert r.status_code == 200
        assert identity.decrypt_api_key(r.json()["api_token"]) == "k2"
        assert r.headers["Deprecation"] == accounts.DEPRECATED_SINCE
        assert r.headers["Sunset"] == accounts.REFRESH_SUNSET
        assert 'rel="successor-version"' in r.headers["Link"]
        assert galaxy.minted == 0


# --- what the deployed auth service versions do with the answers ---------------------

class TestAuthServiceContract:
    """Replays the auth service's own handling of our responses; neither version
    may crash its login or refresh because of a change here."""

    def test_login_at_f017088_reads_api_token_without_checking_status(self, client, galaxy):
        # galaxy_user = response.json()["api_token"]  -- a KeyError would 500 the login.
        for body in ({"email": "new@x.org", "password": "p"}, {"email": "a@x.org", "password": "wrong"}):
            r = client.post(f"/register-user?email={body['email']}&password={body['password']}")
            assert r.json()["api_token"] is None

    def test_refresh_at_main_treats_401_as_no_key(self, client, galaxy):
        # requests: response.raise_for_status() -> HTTPError (a RequestException), caught.
        r = client.post("/refresh-galaxy-key?email=a%40x.org")
        with pytest.raises(httpx.HTTPStatusError):
            r.raise_for_status()

    def test_token_issued_after_that_still_works(self, client, galaxy):
        galaxy.add_user("a@x.org", "pw", key="k1")
        refreshed = token_for({"user_id": 1, "user_email": "a@x.org", "galaxy_api_token": None}, sub=1)
        assert client.get("/whoami", headers=bearer(refreshed)).json()["api_key"] == "k1"


# --- logs ---------------------------------------------------------------------------

class TestLogRedaction:
    def test_redacts_credentials_in_urls(self):
        line = '1.2.3.4 - "POST /register-user?email=a%40x.org&password=hunter2 HTTP/1.1" 200'
        assert "hunter2" not in log_redaction.redact(line)
        assert "email=a%40x.org" in log_redaction.redact(line)
        assert "abc123" not in log_redaction.redact("GET https://g.org/api/tools?key=abc123&io=1")

    def test_access_log_record_args_are_redacted(self, caplog):
        log_redaction.install()
        access = logging.getLogger("uvicorn.access")
        with caplog.at_level(logging.INFO, logger="uvicorn.access"):
            access.info('%s - "%s %s HTTP/%s" %d', "1.2.3.4", "POST",
                        "/register-user?email=a%40x.org&password=hunter2", "1.1", 200)
        assert "hunter2" not in caplog.text and "password=<redacted>" in caplog.text
