from sys import path
path.append(".")

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from app.GX_integration import toolshed
from app.galaxy_runs import installs
from app.galaxy_runs.service import GalaxyRuns
from test_galaxy_runs import MemoryStore, _client, _fake_gi  # the fake Galaxy and store the run routes use

SHED = "toolshed.g2.bx.psu.edu"
FASTP_TOOL = {"id": f"{SHED}/repos/iuc/fastp/fastp/1.3.7+galaxy0", "version": "1.3.7+galaxy0",
              "tool_shed_repository": {"tool_shed": SHED, "owner": "iuc", "name": "fastp", "changeset_revision": "e6bf0b6e0ad4"}}
OTHER_TOOL = {"id": "cat1", "version": "1.0.0"}
BODY = {"owner": "iuc", "name": "fastp", "changeset_revision": "e6bf0b6e0ad4", "client_request_id": "req-1"}


class InstallMemoryStore(MemoryStore):
    """The run store plus install records, in memory."""

    def __init__(self):
        super().__init__()
        self.installs = {}

    async def get_install(self, install_id):
        record = self.installs.get(("install", install_id))
        return dict(record) if record else None

    async def save_install(self, install):
        self.installs[("install", install["id"])] = dict(install)

    async def delete_install(self, install_id):
        self.installs.pop(("install", install_id), None)

    async def claim_install_revision(self, key, install_id):
        return self.installs.setdefault(("revision", key), install_id) == install_id

    async def get_install_revision(self, key):
        return self.installs.get(("revision", key))

    async def release_install_revision(self, key, install_id):
        if self.installs.get(("revision", key)) == install_id:
            del self.installs[("revision", key)]

    async def claim_install_request(self, u, c, install_id):
        return self.installs.setdefault(("request", u, c), install_id) == install_id

    async def get_install_request(self, u, c):
        return self.installs.get(("request", u, c))


def _admin_gi():
    """Galaxy as the admin key sees it: the fastp tool shows up once its repository is installed."""
    gi = MagicMock()
    gi.toolshed.get_repositories.return_value = []
    installed = []
    gi.toolshed.install_repository_revision.side_effect = lambda **kw: installed.append(kw) or [{"status": "Installed"}]
    gi.tools.get_tools.side_effect = lambda **kw: [OTHER_TOOL] + ([FASTP_TOOL] if installed else [])
    return gi


def _service(admin_gi, store, user="alice"):
    client = SimpleNamespace(gi_client=_fake_gi(), galaxy_url="https://galaxy.example.org", user_api_key="user-key",
                             gi_admin=SimpleNamespace(gi=admin_gi), admin_api_key="admin-key")
    return GalaxyRuns(client, user, store, cache=None)


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    monkeypatch.setattr(toolshed, "RELOAD_PAUSE", 0)
    monkeypatch.setattr(toolshed, "PRESENT_PAUSE", 0)


def _settled(client, install_id, timeout=5):
    """Read the install until it stops installing."""
    deadline = time.monotonic() + timeout
    while True:
        install = client.get(f"/v1/galaxy/installs/{install_id}").json()
        if install["status"] != "installing" or time.monotonic() > deadline:
            return install
        time.sleep(0.02)


def test_an_install_starts_at_once_and_lists_the_installed_tools():
    gi, store = _admin_gi(), InstallMemoryStore()
    with _client(_service(gi, store)) as client:
        started = client.post("/v1/galaxy/installs", json=BODY)
        assert started.status_code == 202
        assert started.json()["status"] == "installing"
        assert started.json()["owner"] == "iuc" and started.json()["changeset_revision"] == "e6bf0b6e0ad4"

        install = _settled(client, started.json()["id"])

    assert install["status"] == "installed" and install["error"] is None
    assert install["tools"] == [{"id": FASTP_TOOL["id"], "version": "1.3.7+galaxy0"}]
    sent = gi.toolshed.install_repository_revision.call_args.kwargs
    assert sent["tool_shed_url"] == f"https://{SHED}" and sent["name"] == "fastp" and sent["owner"] == "iuc"
    assert sent["changeset_revision"] == "e6bf0b6e0ad4" and sent["install_resolver_dependencies"] is True


def _blocking(gi):
    """Hold Galaxy's install call until the test lets it go."""
    release = threading.Event()
    install = gi.toolshed.install_repository_revision.side_effect
    gi.toolshed.install_repository_revision.side_effect = lambda **kw: release.wait(5) and install(**kw)
    return release


def test_a_second_request_for_the_same_revision_joins_the_running_install():
    gi, store = _admin_gi(), InstallMemoryStore()
    release = _blocking(gi)
    with _client(_service(gi, store)) as alice, _client(_service(gi, store, user="bob")) as bob:
        first = alice.post("/v1/galaxy/installs", json=BODY)
        joined = bob.post("/v1/galaxy/installs", json={**BODY, "client_request_id": "bob-1"})
        retried = alice.post("/v1/galaxy/installs", json=BODY)
        assert first.status_code == 202
        assert joined.status_code == 200 and joined.json()["id"] == first.json()["id"]
        assert joined.json()["status"] == "installing"
        assert retried.status_code == 200 and retried.json()["id"] == first.json()["id"]

        release.set()
        assert _settled(alice, first.json()["id"])["status"] == "installed"
        later = bob.post("/v1/galaxy/installs", json={**BODY, "client_request_id": "bob-2"})

    assert later.status_code == 200 and later.json()["status"] == "installed"
    assert gi.toolshed.install_repository_revision.call_count == 1


def test_a_failed_install_says_why_and_a_new_request_starts_again():
    gi, store = _admin_gi(), InstallMemoryStore()
    working = gi.toolshed.install_repository_revision.side_effect
    gi.toolshed.install_repository_revision.side_effect = RuntimeError("conda could not resolve fastp")
    with _client(_service(gi, store)) as client:
        first = client.post("/v1/galaxy/installs", json=BODY)
        failed = _settled(client, first.json()["id"])
        assert failed["status"] == "failed" and failed["tools"] == []
        assert "conda could not resolve fastp" in failed["error"]

        gi.toolshed.install_repository_revision.side_effect = working
        again = client.post("/v1/galaxy/installs", json={**BODY, "client_request_id": "req-2"})
        assert again.status_code == 202 and again.json()["id"] != first.json()["id"]
        assert _settled(client, again.json()["id"])["status"] == "installed"


def test_an_install_whose_tools_never_show_up_fails():
    gi, store = _admin_gi(), InstallMemoryStore()
    gi.tools.get_tools.side_effect = lambda **kw: [OTHER_TOOL]
    with _client(_service(gi, store)) as client:
        install = _settled(client, client.post("/v1/galaxy/installs", json=BODY).json()["id"])
    assert install["status"] == "failed" and "none of its tools" in install["error"]


def test_an_install_past_the_timeout_fails_and_frees_the_revision(monkeypatch):
    gi, store = _admin_gi(), InstallMemoryStore()
    release = _blocking(gi)
    with _client(_service(gi, store)) as client:
        first = client.post("/v1/galaxy/installs", json=BODY).json()
        monkeypatch.setattr(installs, "INSTALL_TIMEOUT_SECONDS", 0)
        timed_out = client.get(f"/v1/galaxy/installs/{first['id']}").json()
        assert timed_out["status"] == "failed" and "30 minutes" in timed_out["error"]

        monkeypatch.setattr(installs, "INSTALL_TIMEOUT_SECONDS", 60)
        again = client.post("/v1/galaxy/installs", json={**BODY, "client_request_id": "req-2"})
        assert again.status_code == 202 and again.json()["id"] != first["id"]
        release.set()
        assert _settled(client, again.json()["id"])["status"] == "installed"
        # the first install finishing late doesn't undo its timeout
        assert _settled(client, first["id"])["status"] == "failed"


@pytest.mark.parametrize("tool_shed", ["testtoolshed.g2.bx.psu.edu", "https://toolshed.example.org/"])
def test_another_tool_shed_is_refused(tool_shed):
    gi = _admin_gi()
    with _client(_service(gi, InstallMemoryStore())) as client:
        refused = client.post("/v1/galaxy/installs", json={**BODY, "tool_shed": tool_shed})
    assert refused.status_code == 400 and SHED in refused.json()["detail"]
    gi.toolshed.install_repository_revision.assert_not_called()


def test_the_main_tool_shed_may_be_given_as_a_url():
    with _client(_service(_admin_gi(), InstallMemoryStore())) as client:
        started = client.post("/v1/galaxy/installs", json={**BODY, "tool_shed": f"https://{SHED}/"})
    assert started.status_code == 202 and started.json()["tool_shed"] == SHED


def test_without_a_revision_the_newest_installable_one_is_installed(monkeypatch):
    seen = []

    def toolshed_api(request: httpx.Request):
        seen.append(str(request.url))
        known = request.url.params["owner"] == "iuc" and request.url.params["name"] == "fastp"
        return httpx.Response(200, json=["0a1b2c3d4e5f", "e6bf0b6e0ad4"] if known else [])

    monkeypatch.setattr(installs, "_default_http",
                        lambda: httpx.AsyncClient(transport=httpx.MockTransport(toolshed_api)))
    body = {k: v for k, v in BODY.items() if k != "changeset_revision"}
    with _client(_service(_admin_gi(), InstallMemoryStore())) as client:
        started = client.post("/v1/galaxy/installs", json=body)
        unknown = client.post("/v1/galaxy/installs", json={**body, "name": "nope", "client_request_id": "req-2"})
    assert started.status_code == 202 and started.json()["changeset_revision"] == "e6bf0b6e0ad4"
    assert seen[0].startswith(f"https://{SHED}/api/repositories/get_ordered_installable_revisions")
    assert unknown.status_code == 404


def test_an_unknown_install_is_not_found():
    with _client(_service(_admin_gi(), InstallMemoryStore())) as client:
        assert client.get("/v1/galaxy/installs/install_missing").status_code == 404
