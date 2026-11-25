"""Find tools on the main ToolShed that may not be installed on this Galaxy yet.

One search call, then two small calls per repository (its installable revisions, then the latest one's tools),
at most MAX_RESULTS repositories, PARALLEL at a time. Which revisions are installed comes from Galaxy's
admin list of installed ToolShed repositories.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from app.galaxy_runs.service import GalaxyRuns, GalaxyRunsError

log = logging.getLogger("GalaxyRuns")

TOOL_SHED = "toolshed.g2.bx.psu.edu"  # the only tool shed users may search and install from
API = f"https://{TOOL_SHED}/api/repositories"
MAX_RESULTS = 10
PARALLEL = 4  # repositories looked up at the same time

HttpFactory = Callable[[], httpx.AsyncClient]


def default_http() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(15.0))


class _ToolShedDown(Exception):
    pass


async def _get(http: httpx.AsyncClient, path: str, **params) -> Any:
    try:
        response = await http.get(f"{API}{path}", params=params)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as e:
        log.warning("ToolShed call %s failed: %s", path or "search", e)
        raise _ToolShedDown() from e


async def _latest(http: httpx.AsyncClient, slots: asyncio.Semaphore, owner: str,
                  name: str) -> Tuple[List[str], List[str]]:
    """The installable revisions, oldest first, and the latest one's tool ids; ([], []) when none is installable."""
    async with slots:
        revisions = await _get(http, "/get_ordered_installable_revisions", owner=owner, name=name)
        if not isinstance(revisions, list) or not revisions:
            return [], []
        revision = revisions[-1]
        info = await _get(http, "/get_repository_revision_install_info", owner=owner, name=name,
                          changeset_revision=revision)
    metadata = info[1] if isinstance(info, list) and len(info) > 1 and isinstance(info[1], dict) else {}
    return revisions, [t["guid"] for t in metadata.get("valid_tools") or [] if t.get("guid")]


async def _installed(runs: GalaxyRuns) -> Dict[Tuple[str, str], List[str]]:
    """The installed revisions of every main-ToolShed repository on this Galaxy, by (owner, name)."""
    admin = getattr(runs.client, "gi_admin", None)
    if admin is None:
        log.error("ToolShed search needs GALAXY_API_KEY (admin) to list installed repositories")
        raise GalaxyRunsError(502, "Galaxy failed. Safe to retry.")
    repos = await runs._call(admin.gi.toolshed.get_repositories)
    installed: Dict[Tuple[str, str], List[str]] = {}
    for r in repos or []:
        if (r.get("tool_shed") == TOOL_SHED and r.get("status") == "Installed"
                and not r.get("deleted") and not r.get("uninstalled")):
            installed.setdefault((r.get("owner"), r.get("name")), []).append(r.get("changeset_revision"))
    return installed


def _installed_revision(revisions: List[str], installed: List[str]) -> Optional[str]:
    """The newest installed revision: the latest installable one if installed, else the newest older one."""
    for revision in reversed(revisions):
        if revision in installed:
            return revision
    return installed[0] if installed else None


async def search(runs: GalaxyRuns, q: str, http: HttpFactory = default_http) -> Dict[str, Any]:
    q = q.strip()
    if not q:
        raise GalaxyRunsError(400, "q must not be empty")
    try:
        async with http() as client:
            found = await _get(client, "", q=q, page_size=MAX_RESULTS)
            repos = [h["repository"] for h in (found.get("hits") or [])[:MAX_RESULTS] if h.get("repository")]
            slots = asyncio.Semaphore(PARALLEL)
            latest, installed = await asyncio.gather(
                asyncio.gather(*(_latest(client, slots, r["repo_owner_username"], r["name"]) for r in repos)),
                _installed(runs))
    except (_ToolShedDown, KeyError, AttributeError, TypeError):
        raise GalaxyRunsError(502, "The ToolShed could not be reached. Safe to retry.")
    items = []
    for r, (revisions, tool_ids) in zip(repos, latest):
        revision = revisions[-1] if revisions else None
        here = _installed_revision(revisions, installed.get((r["repo_owner_username"], r["name"]), []))
        items.append({"owner": r["repo_owner_username"], "name": r["name"],
                      "description": (r.get("description") or "").strip(), "changeset_revision": revision,
                      "tool_ids": tool_ids, "installed": revision is not None and here == revision,
                      "installed_revision": here})
    return {"items": items}
