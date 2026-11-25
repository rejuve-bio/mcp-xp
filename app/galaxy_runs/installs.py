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


def _default_http() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=30.0)


def public_install(install: Dict[str, Any]) -> Dict[str, Any]:
    return {k: install.get(k) for k in PUBLIC_INSTALL_FIELDS}


def _shed_host(tool_shed: str) -> str:
    return tool_shed.strip().removeprefix("https://").removeprefix("http://").rstrip("/")


def _revision_key(install: Dict[str, Any]) -> str:
    return f"{install['tool_shed']}/{install['owner']}/{install['name']}/{install['changeset_revision']}"


class ToolInstalls:

    def __init__(self, runs: GalaxyRuns, http=None):
        self.store = runs.store
        self.username = runs.username
        self.gi = runs.client.gi_admin.gi
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
                await self.store.claim_install_request(self.username, client_request_id, current["id"])
                return public_install(current), True
        await self.store.claim_install_request(self.username, client_request_id, install["id"])

        task = asyncio.create_task(self._install(install))
        _running.add(task)
        task.add_done_callback(_running.discard)
        return public_install(install), False

    async def _current(self, key: str) -> Optional[Dict[str, Any]]:
        """The install holding the revision, unless it failed (then a new one may start)."""
        install_id = await self.store.get_install_revision(key)
        install = await self._settle(await self.store.get_install(install_id)) if install_id else None
        if install and install["status"] != "failed":
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
        """An install still running past the timeout failed, even if mcp-xp restarted in the meantime."""
        if install and install["status"] == "installing" and time.time() - install["started"] > INSTALL_TIMEOUT_SECONDS:
            install = await self._finish(install, "failed", error=TIMED_OUT)
        return install

    async def _finish(self, install: Dict[str, Any], status: str, tools: Optional[List[Dict[str, str]]] = None,
                      error: Optional[str] = None) -> Dict[str, Any]:
        install = {**install, "status": status, "tools": tools or [], "error": error, "updated_at": _now()}
        await self.store.save_install(install)
        if status == "failed":
            await self.store.release_install_revision(_revision_key(install), install["id"])
        return install

    # ------------------------------------------------------------------ the install itself

    async def _install(self, install: Dict[str, Any]) -> None:
        try:
            tools = await self._run_install(install)
            status, error = ("installed", None) if tools else (
                "failed", f"Galaxy installed {install['name']} but none of its tools showed up")
        except Exception as e:
            log.error("Install of %s failed: %s", _revision_key(install), e)
            tools, status, error = [], "failed", f"Galaxy could not install {install['name']}: {e}"
        current = await self.store.get_install(install["id"])
        if current and current["status"] == "installing":  # not already failed by the timeout
            await self._finish(current, status, tools, error)

    async def _run_install(self, install: Dict[str, Any]) -> List[Dict[str, str]]:
        if await toolshed.clear_ghost_install(self.gi, self.galaxy_url, self.admin_key, install,
                                              lambda: self._tools(install)):
            await toolshed.reload_toolbox(self.gi)
        await toolshed.install_revision(self.gi.toolshed, install)
        return await toolshed.reload_until_present(self.gi, lambda: self._tools(install)) or []

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
        except httpx.HTTPError as e:
            log.error("ToolShed lookup of %s/%s failed: %s", owner, name, e)
            raise GalaxyRunsError(502, "The ToolShed failed. Safe to retry.")
        if not isinstance(revisions, list) or not revisions:
            raise GalaxyRunsError(404, f"No installable revision of {owner}/{name} on {TOOL_SHED}")
        return revisions[-1]
