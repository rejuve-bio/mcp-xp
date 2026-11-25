"""Installing a ToolShed repository revision into Galaxy with the admin key.

Shared by workflow import (WorkflowInstaller) and single-tool installs (app/galaxy_runs/installs.py).
`gi` is a plain bioblend GalaxyInstance on the admin key.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict

import httpx

log = logging.getLogger("ToolShedInstall")

UNHEALTHY_REPOSITORY_STATUSES = {"Error", "Uninstalled", "New"}
RELOAD_PAUSE = 2  # seconds before asking Galaxy to reload its toolbox
PRESENT_ATTEMPTS = 6
PRESENT_PAUSE = 10  # seconds between checks that an installed tool shows up


async def install_revision(client, repo: Dict[str, Any], retry: bool = True):
    """Install one repository revision with its tool, repository and resolver dependencies. Blocks for minutes.

    `client` is Galaxy's ToolShed client on the admin key (gi.toolshed)."""
    try:
        return await asyncio.to_thread(
            client.install_repository_revision,
            tool_shed_url=f'https://{repo["tool_shed"]}',
            name=repo["name"],
            owner=repo["owner"],
            changeset_revision=repo["changeset_revision"],
            install_tool_dependencies=True,
            install_repository_dependencies=True,
            install_resolver_dependencies=True,
            tool_panel_section_id=None,
            new_tool_panel_section_label=None,
        )
    except Exception as e:
        if not retry:
            log.error(f"tool installation failing: {e}")
            raise
        log.warning(f"tool installation failed, trying again. {e}")
        return await install_revision(client, repo, retry=False)


async def reload_toolbox(gi) -> None:
    await asyncio.sleep(RELOAD_PAUSE)
    await asyncio.to_thread(gi.config.reload_toolbox)


async def reload_until_present(gi, present: Callable[[], Awaitable[Any]]) -> Any:
    """Reload the toolbox until `present()` is truthy, up to about a minute. Returns its last answer.

    Galaxy can answer "already installed" while another revision is the one installed, so its answer alone
    is not trusted."""
    found = None
    for attempt in range(PRESENT_ATTEMPTS):
        await reload_toolbox(gi)
        found = await present()
        if found:
            return found
        if attempt < PRESENT_ATTEMPTS - 1:
            await asyncio.sleep(PRESENT_PAUSE)
    return found


async def clear_ghost_install(gi, galaxy_url: str, admin_key: str, repo: Dict[str, Any],
                              tool_present: Callable[[], Awaitable[bool]]) -> bool:
    """Detect a 'ghost install' and remove its stale row so a fresh install runs. Returns True if it removed one;
    the caller then reloads the toolbox.

    Galaxy's `tool_shed_repositories` table can mark a repository revision as `Installed` even when files were
    never downloaded, conda env never resolved, or the install was interrupted. In those cases,
    `install_repository_revision` silently no-ops because the DB row already exists.

    Galaxy 25.0 has no `repair_repository_revision` endpoint, so the only remedy for a stuck row is to DELETE it
    (`uninstall_repository`) and install again.

    Uninstall side-effects to be aware of:
    - `remove_from_disk=true` removes files and conda envs for this repo.
    - Shared dependencies may be cascaded off; if another healthy repo relied on them, they will be reinstalled
      on next need. That's an accepted cost of clearing the ghost.
    """
    label = f"{repo['name']}@{repo['changeset_revision']}"
    try:
        # gi.toolShed is the client for repositories installed in Galaxy.
        repos = await asyncio.to_thread(gi.toolShed.get_repositories)
        for r in repos:
            if not (r.get('name') == repo['name']
                    and r.get('owner') == repo['owner']
                    and r.get('changeset_revision') == repo['changeset_revision']):
                continue

            status = r.get('status', '')
            tool_actually_present = await tool_present()
            repo_deleted = bool(r.get('deleted'))
            repo_uninstalled = bool(r.get('uninstalled'))
            needs_cleanup = (
                repo_deleted
                or repo_uninstalled
                or status in UNHEALTHY_REPOSITORY_STATUSES
                or (status == 'Installed' and not tool_actually_present)
            )
            if not needs_cleanup:
                return False

            repo_id = r.get('id')
            if not repo_id:
                log.warning(f"Cannot uninstall {label}: matching repo row has no 'id' field. "
                            "Falling through to install attempt.")
                return False

            log.warning(f"Repo {label} status='{status}' deleted={repo_deleted} uninstalled={repo_uninstalled} "
                        f"tool_present={tool_actually_present} — uninstalling stale row")
            try:
                # DELETE /api/tool_shed_repositories/{id}; remove_from_disk cleans the row, files and conda envs.
                async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
                    resp = await client.delete(
                        f"{galaxy_url}/api/tool_shed_repositories/{repo_id}",
                        headers={"x-api-key": admin_key},
                        params={"remove_from_disk": "true"},
                    )
                    resp.raise_for_status()
                    log.info(f"Uninstall API call succeeded for {label}")
                return True
            except Exception as uninstall_err:
                log.warning(f"Uninstall HTTP call failed for {label}: {uninstall_err}. Falling through to install attempt.")
                return False
    except Exception as e:
        # Best-effort: if we can't enumerate repos, don't block the install path.
        log.debug(f"Repo state check skipped: {e}")
    return False
