"""Installing a ToolShed repository on request, so a user can run a tool that isn't on the server yet.

Installs are server-wide: one install per repository revision, joined by anyone who asks for the same one.
Starting an install returns at once; the install itself runs in the background with the admin key.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.GX_integration import toolshed
from app.galaxy_runs.service import GalaxyRuns, GalaxyRunsError, _now

log = logging.getLogger("GalaxyToolInstalls")

TOOL_SHED = "toolshed.g2.bx.psu.edu"
INSTALL_TIMEOUT_SECONDS = 30 * 60
TIMED_OUT = "The install did not finish within 30 minutes"
PUBLIC_INSTALL_FIELDS = ("id", "tool_shed", "owner", "name", "changeset_revision", "status", "tools", "error",
                         "created_at", "updated_at")

_running: set = set()  # background installs, kept so they aren't garbage collected
_in_progress: Dict[str, str] = {}  # revision key -> install id whose background task is still running here


def _default_http() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=30.0)


def public_install(install: Dict[str, Any]) -> Dict[str, Any]:
    return {k: install.get(k) for k in PUBLIC_INSTALL_FIELDS}


def _shed_host(tool_shed: str) -> str:
    return tool_shed.strip().removeprefix("https://").removeprefix("http://").rstrip("/")


def _revision_key(install: Dict[str, Any]) -> str:
    return f"{install['tool_shed']}/{install['owner']}/{install['name']}/{install['changeset_revision']}"


def _stopped(key: str, install_id: str) -> None:
    if _in_progress.get(key) == install_id:
        del _in_progress[key]


class ToolInstalls:

    def __init__(self, runs: GalaxyRuns, http=None):
        self.store = runs.store
        self.username = runs.username
        admin = getattr(runs.client, "gi_admin", None)
        if admin is None:
            log.error("ToolShed installs need GALAXY_API_KEY (admin)")
            raise GalaxyRunsError(502, "Galaxy failed. Safe to retry.")
        self.gi = admin.gi
        self.galaxy_url = runs.galaxy_url
        self.admin_key = runs.client.admin_api_key
        self.http = http or _default_http

    async def start(self, owner: str, name: str, changeset_revision: Optional[str], client_request_id: str,
                    tool_shed: str = TOOL_SHED) -> Tuple[Dict[str, Any], bool]:
        """Start an install, or return the one already running or done. The bool is True when nothing started."""
        shed = _shed_host(tool_shed)
        if shed != TOOL_SHED:
            raise GalaxyRunsError(400, f"Only tools from {TOOL_SHED} can be installed")
        asked = await self.store.get_install_request(self.username, client_request_id)
        if asked:
            return await self.read(asked), True

        revision = changeset_revision or await self._latest_revision(owner, name)
        now = _now()
        install = {"id": f"install_{uuid.uuid4().hex[:12]}", "tool_shed": shed, "owner": owner, "name": name,
                   "changeset_revision": revision, "status": "installing", "tools": [], "error": None,
                   "requested_by": self.username, "started": time.time(), "created_at": now, "updated_at": now}
        key = _revision_key(install)
        await self.store.save_install(install)
        while not await self.store.claim_install_revision(key, install["id"]):
            current = await self._current(key)
            if current:
                await self.store.delete_install(install["id"])
                if current["status"] != "failed":  # a timed-out one still finishing: a retry may start anew later
                    await self.store.claim_install_request(self.username, client_request_id, current["id"])
                return public_install(current), True
        await self.store.claim_install_request(self.username, client_request_id, install["id"])

        _in_progress[key] = install["id"]
        task = asyncio.create_task(self._install(install))
        _running.add(task)
        task.add_done_callback(_running.discard)
        task.add_done_callback(lambda _: _stopped(key, install["id"]))  # also when cancelled
        return public_install(install), False

    async def _current(self, key: str) -> Optional[Dict[str, Any]]:
        """The install holding the revision, unless it failed and is done (then a new one may start)."""
        install_id = await self.store.get_install_revision(key)
        install = await self._settle(await self.store.get_install(install_id)) if install_id else None
        if install and (install["status"] != "failed" or _in_progress.get(key) == install_id):
            return install
        if install_id:
            await self.store.release_install_revision(key, install_id)
        return None

    async def read(self, install_id: str) -> Dict[str, Any]:
        install = await self._settle(await self.store.get_install(install_id))
        if not install:
            raise GalaxyRunsError(404, "No such install")
        return public_install(install)

    async def _settle(self, install: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """An install still running past the timeout failed, even if mcp-xp restarted in the meantime.

        If its background task is still running here, the revision stays held until that task ends, so a new
        install can't clear the repository row Galaxy is still filling in."""
        if install and install["status"] == "installing" and time.time() - install["started"] > INSTALL_TIMEOUT_SECONDS:
            install = await self._finish(install, "failed", error=TIMED_OUT)
        return install

    async def _finish(self, install: Dict[str, Any], status: str, tools: Optional[List[Dict[str, str]]] = None,
                      error: Optional[str] = None) -> Dict[str, Any]:
        install = {**install, "status": status, "tools": tools or [], "error": error, "updated_at": _now()}
        await self.store.save_install(install)
        key = _revision_key(install)
        if status == "failed" and _in_progress.get(key) != install["id"]:
            await self.store.release_install_revision(key, install["id"])
        return install

    # ------------------------------------------------------------------ the install itself

    async def _install(self, install: Dict[str, Any]) -> None:
        try:
            tools, galaxy_error = await self._run_install(install)
            if tools:
                status, error = "installed", None
            elif galaxy_error:
                status, error = "failed", f"Galaxy could not install {install['name']}: {galaxy_error}"
            else:
                status, error = "failed", f"Galaxy installed {install['name']} but none of its tools showed up"
        except Exception as e:
            log.error("Install of %s failed: %s", _revision_key(install), e)
            tools, status, error = [], "failed", f"Galaxy could not install {install['name']}: {e}"
        key = _revision_key(install)
        _stopped(key, install["id"])
        try:
            current = await self.store.get_install(install["id"])
            if current and current["status"] == "installing":  # not already failed by the timeout
                await self._finish(current, status, tools, error)
            elif not current or current["status"] == "failed":  # timed out while running: free the revision now
                await self.store.release_install_revision(key, install["id"])
        except Exception as e:
            log.error("Could not save the end of install %s (%s, %s): %s", install["id"], key, status, e)

    async def _run_install(self, install: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Optional[Exception]]:
        """The revision's tools once Galaxy shows them, and Galaxy's error if its install call failed."""
        if await toolshed.clear_ghost_install(self.gi, self.galaxy_url, self.admin_key, install,
                                              lambda: self._tools(install)):
            await toolshed.reload_toolbox(self.gi)
        galaxy_error = None
        try:
            await toolshed.install_revision(self.gi.toolshed, install)
        except Exception as e:  # e.g. a proxy timing out on the long call while Galaxy carries on
            log.warning("Galaxy's install call for %s failed, checking its tools anyway: %s", _revision_key(install), e)
            galaxy_error = e
        return await toolshed.reload_until_present(self.gi, lambda: self._tools(install)) or [], galaxy_error

    async def _tools(self, install: Dict[str, Any]) -> List[Dict[str, str]]:
        """The installed tools that come from this repository revision."""
        tools = await asyncio.to_thread(self.gi.tools.get_tools)
        found = []
        for tool in tools or []:
            repo = tool.get("tool_shed_repository") or {}
            if (repo.get("owner"), repo.get("name"), repo.get("changeset_revision")) == (
                    install["owner"], install["name"], install["changeset_revision"]):
                found.append({"id": tool["id"], "version": tool.get("version")})
        return found

    async def _latest_revision(self, owner: str, name: str) -> str:
        """The repository's newest installable revision on the ToolShed."""
        try:
            async with self.http() as client:
                resp = await client.get(f"https://{TOOL_SHED}/api/repositories/get_ordered_installable_revisions",
                                        params={"owner": owner, "name": name})
                resp.raise_for_status()
                revisions = resp.json()
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON
            log.error("ToolShed lookup of %s/%s failed: %s", owner, name, e)
            raise GalaxyRunsError(502, "The ToolShed failed. Safe to retry.")
        if not isinstance(revisions, list) or not revisions:
            raise GalaxyRunsError(404, f"No installable revision of {owner}/{name} on {TOOL_SHED}")
        return revisions[-1]
