"""GET /v1/galaxy/toolshed/search: ToolShed repositories, with whether this Galaxy has them installed.

The ToolShed is faked at the HTTP level with answers shaped like the real ones; Galaxy is the fake from
test_galaxy_runs, with an admin client that lists installed repositories.
"""
from sys import path
path.append(".")

from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlparse

import httpx
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from app.api.endpoints import galaxy_runs as routes
from app.exceptions import validation_exception_handler
from app.galaxy_runs.service import GalaxyRuns
from tests.GX_Integration.test_galaxy_runs import MemoryStore, _fake_gi

SHED = "https://toolshed.g2.bx.psu.edu/api/repositories"


def _hit(repo_id, name, owner, description):
    return {"score": 10.0, "repository": {
        "id": repo_id, "name": name, "repo_owner_username": owner, "description": description,
        "long_description": description, "approved": False, "times_downloaded": 1, "categories": "sequence analysis"}}


SEARCH = {"total_results": "2", "page": "1", "page_size": "10", "hits": [
    _hit("3a57c9beb4edc913", "fastp", "iuc", "Fast all-in-one preprocessing for FASTQ files"),
    _hit("057f7ad94e138e19", "fastplong", "iuc", "Fastplong is used for preprocessing long reads.\n"),
]}
REVISIONS = {("iuc", "fastp"): ["988729b728f0", "4990e6f1348f", "bca43867b6f4"],
             ("iuc", "fastplong"): ["c3e3eecc8fe9", "085d03d88643"]}
TOOLS = {"bca43867b6f4": [("fastp", "toolshed.g2.bx.psu.edu/repos/iuc/fastp/fastp/1.4.0+galaxy1", "1.4.0+galaxy1")],
         "085d03d88643": [("fastplong", "toolshed.g2.bx.psu.edu/repos/iuc/fastplong/fastplong/0.3.0+galaxy0",
                           "0.3.0+galaxy0")]}


def _install_info(name, owner, revision):
    """get_repository_revision_install_info: [repository, revision metadata, {name: [..., revision, ...]}]."""
    tools = [{"id": short, "guid": guid, "name": short, "version": version, "description": "", "requirements": []}
             for short, guid, version in TOOLS[revision]]
    return [{"model_class": "Repository", "name": name, "owner": owner},
            {"model_class": "RepositoryMetadata", "changeset_revision": revision, "downloadable": True,
             "malicious": False, "includes_tools": True, "valid_tools": tools},
            {name: ["", f"https://toolshed.g2.bx.psu.edu/repos/{owner}/{name}", revision, "1", owner, {}, {}]}]


def _toolshed(seen, down=False):
    async def handler(request: httpx.Request):
        seen.append(str(request.url))
        if down:
            raise httpx.ConnectError("unreachable", request=request)
        url = urlparse(str(request.url))
        args = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/api/repositories":
            return httpx.Response(200, json=SEARCH)
        if url.path == "/api/repositories/get_ordered_installable_revisions":
            return httpx.Response(200, json=REVISIONS[(args["owner"], args["name"])])
        if url.path == "/api/repositories/get_repository_revision_install_info":
            return httpx.Response(200, json=_install_info(args["name"], args["owner"], args["changeset_revision"]))
        return httpx.Response(404, json={"err_msg": "not found"})
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


INSTALLED = [
    {"id": "r1", "name": "fastp", "owner": "iuc", "tool_shed": "toolshed.g2.bx.psu.edu",
     "changeset_revision": "4990e6f1348f", "status": "Installed", "deleted": False, "uninstalled": False},
    # the same name from another owner and an uninstalled fastplong don't count
    {"id": "r2", "name": "fastplong", "owner": "someone", "tool_shed": "toolshed.g2.bx.psu.edu",
     "changeset_revision": "aaaa", "status": "Installed", "deleted": False, "uninstalled": False},
    {"id": "r3", "name": "fastplong", "owner": "iuc", "tool_shed": "toolshed.g2.bx.psu.edu",
     "changeset_revision": "085d03d88643", "status": "Uninstalled", "deleted": True, "uninstalled": True},
]


def _client(seen, down=False, installed=INSTALLED):
    admin = MagicMock()
    admin.toolshed.get_repositories.return_value = installed
    client = SimpleNamespace(gi_client=_fake_gi(), gi_admin=SimpleNamespace(gi=admin),
                             galaxy_url="https://galaxy.example.org", user_api_key="user-key")
    svc = GalaxyRuns(client, "alice", MemoryStore(), cache=None)

    app = FastAPI()
    app.include_router(routes.router, prefix="/v1/galaxy")
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.dependency_overrides[routes.galaxy_runs] = lambda: svc
    app.dependency_overrides[routes.toolshed_http] = lambda: _toolshed(seen, down)
    return TestClient(app)


def test_search_returns_latest_installable_revision_tools_and_whether_it_is_installed():
    seen = []
    response = _client(seen).get("/v1/galaxy/toolshed/search", params={"q": "fastp"})

    assert response.status_code == 200
    assert response.json() == {"items": [
        {"owner": "iuc", "name": "fastp", "description": "Fast all-in-one preprocessing for FASTQ files",
         "changeset_revision": "bca43867b6f4",
         "tool_ids": ["toolshed.g2.bx.psu.edu/repos/iuc/fastp/fastp/1.4.0+galaxy1"],
         "installed": False, "installed_revision": "4990e6f1348f"},  # an older revision is installed
        {"owner": "iuc", "name": "fastplong", "description": "Fastplong is used for preprocessing long reads.",
         "changeset_revision": "085d03d88643",
         "tool_ids": ["toolshed.g2.bx.psu.edu/repos/iuc/fastplong/fastplong/0.3.0+galaxy0"],
         "installed": False, "installed_revision": None},
    ]}
    assert parse_qs(urlparse(seen[0]).query)["q"] == ["fastp"]
    assert len(seen) == 1 + 2 * 2  # one search, then two small calls per repository


def test_an_empty_query_is_400_and_asks_nothing():
    seen = []
    client = _client(seen)
    assert client.get("/v1/galaxy/toolshed/search", params={"q": ""}).status_code == 400
    assert client.get("/v1/galaxy/toolshed/search", params={"q": "   "}).status_code == 400
    assert client.get("/v1/galaxy/toolshed/search").status_code == 400
    assert seen == []


def test_toolshed_down_is_502_with_a_short_message():
    seen = []
    response = _client(seen, down=True).get("/v1/galaxy/toolshed/search", params={"q": "fastp"})
    assert response.status_code == 502
    assert response.json() == {"detail": "The ToolShed could not be reached. Safe to retry."}


def test_a_repository_whose_latest_revision_is_installed_is_installed():
    latest = {**INSTALLED[0], "id": "r4", "changeset_revision": "bca43867b6f4"}
    response = _client([], installed=[INSTALLED[0], latest]).get("/v1/galaxy/toolshed/search", params={"q": "fastp"})

    fastp = response.json()["items"][0]
    assert fastp["installed"] is True and fastp["installed_revision"] == "bca43867b6f4"
