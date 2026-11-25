import pytest

from unittest.mock import AsyncMock, MagicMock, patch

from app.GX_integration.workflows.worklfow_installer import WorkflowInstaller


@pytest.fixture
def mock_galaxy_client():
    mock_client = MagicMock()
    mock_client.gi_object = MagicMock()
    mock_client.gi_admin = MagicMock()
    mock_client.admin_api_key = "admin-key"
    mock_client.galaxy_url = "https://galaxy.example"
    return mock_client


@pytest.fixture
def workflow_installer(mock_galaxy_client):
    return WorkflowInstaller(mock_galaxy_client)


def _step():
    return {
        "id": "1",
        "tool_id": "toolshed.g2.bx.psu.edu/repos/dev/tool/tool/1.0",
        "tool_shed_repository": {
            "tool_shed": "toolshed.g2.bx.psu.edu",
            "name": "tool",
            "owner": "dev",
            "changeset_revision": "rev123",
        },
    }


def _mock_httpx_client(monkey_patches):
    """Common httpx.AsyncClient mock plumbing.

    Returns (mock_client_cls_patcher, mock_client). Enter mock_client_cls_patcher
    in a with-block; then set up mock_client.delete/post as needed on the returned
    mock_client.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_client.delete = AsyncMock(return_value=mock_response)
    mock_client.post = AsyncMock(return_value=mock_response)
    monkey_patches["mock_client_cls"].return_value.__aenter__.return_value = mock_client
    return mock_client


@pytest.mark.asyncio
async def test_ensure_repository_installable_uninstalls_deleted_repo(workflow_installer):
    step = _step()
    repo = {
        "id": "repo-1",
        "name": "tool",
        "owner": "dev",
        "changeset_revision": "rev123",
        "status": "Installed",
        "deleted": True,
        "uninstalled": False,
    }
    workflow_installer.gi_admin.gi.toolShed.get_repositories.return_value = [repo]

    with patch("app.GX_integration.toolshed.httpx.AsyncClient") as mock_client_cls, \
         patch.object(workflow_installer, "_tool_exists", new=AsyncMock(return_value=False)), \
         patch.object(workflow_installer, "_reload_toolbox", new=AsyncMock()) as mock_reload:
        mock_client = _mock_httpx_client({"mock_client_cls": mock_client_cls})

        await workflow_installer._ensure_repository_installable(step, step["tool_shed_repository"])

    mock_client.delete.assert_awaited_once()
    delete_call = mock_client.delete.call_args
    assert "repo-1" in delete_call.args[0], "repo id must be in DELETE URL path"
    assert "/api/tool_shed_repositories/" in delete_call.args[0]
    assert delete_call.kwargs["params"]["remove_from_disk"] == "true"
    mock_client.post.assert_not_awaited()
    mock_reload.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_repository_installable_uninstalls_uninstalled_flag_repo(workflow_installer):
    """Repo row with `uninstalled=True` is treated as a ghost and cleared."""
    step = _step()
    repo = {
        "id": "repo-2",
        "name": "tool",
        "owner": "dev",
        "changeset_revision": "rev123",
        "status": "Installed",
        "deleted": False,
        "uninstalled": True,
    }
    workflow_installer.gi_admin.gi.toolShed.get_repositories.return_value = [repo]

    with patch("app.GX_integration.toolshed.httpx.AsyncClient") as mock_client_cls, \
         patch.object(workflow_installer, "_tool_exists", new=AsyncMock(return_value=False)), \
         patch.object(workflow_installer, "_reload_toolbox", new=AsyncMock()):
        mock_client = _mock_httpx_client({"mock_client_cls": mock_client_cls})

        await workflow_installer._ensure_repository_installable(step, step["tool_shed_repository"])

    mock_client.delete.assert_awaited_once()
    assert "repo-2" in mock_client.delete.call_args.args[0]


@pytest.mark.asyncio
async def test_ensure_repository_installable_uninstalls_installed_but_missing(workflow_installer):
    """Repo status=Installed but tool not in toolbox → still a ghost → DELETE it."""
    step = _step()
    repo = {
        "id": "repo-3",
        "name": "tool",
        "owner": "dev",
        "changeset_revision": "rev123",
        "status": "Installed",
        "deleted": False,
        "uninstalled": False,
    }
    workflow_installer.gi_admin.gi.toolShed.get_repositories.return_value = [repo]

    with patch("app.GX_integration.toolshed.httpx.AsyncClient") as mock_client_cls, \
         patch.object(workflow_installer, "_tool_exists", new=AsyncMock(return_value=False)), \
         patch.object(workflow_installer, "_reload_toolbox", new=AsyncMock()):
        mock_client = _mock_httpx_client({"mock_client_cls": mock_client_cls})

        await workflow_installer._ensure_repository_installable(step, step["tool_shed_repository"])

    mock_client.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_repository_installable_uninstalls_error_status(workflow_installer):
    """Repo with status='Error' is treated as unhealthy and DELETE'd."""
    step = _step()
    repo = {
        "id": "repo-4",
        "name": "tool",
        "owner": "dev",
        "changeset_revision": "rev123",
        "status": "Error",
        "deleted": False,
        "uninstalled": False,
    }
    workflow_installer.gi_admin.gi.toolShed.get_repositories.return_value = [repo]

    with patch("app.GX_integration.toolshed.httpx.AsyncClient") as mock_client_cls, \
         patch.object(workflow_installer, "_tool_exists", new=AsyncMock(return_value=False)), \
         patch.object(workflow_installer, "_reload_toolbox", new=AsyncMock()):
        mock_client = _mock_httpx_client({"mock_client_cls": mock_client_cls})

        await workflow_installer._ensure_repository_installable(step, step["tool_shed_repository"])

    mock_client.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_repository_installable_skips_when_repo_healthy(workflow_installer):
    """Healthy repo (Installed + tool present, no deleted/uninstalled) → no HTTP call."""
    step = _step()
    repo = {
        "id": "repo-5",
        "name": "tool",
        "owner": "dev",
        "changeset_revision": "rev123",
        "status": "Installed",
        "deleted": False,
        "uninstalled": False,
    }
    workflow_installer.gi_admin.gi.toolShed.get_repositories.return_value = [repo]

    with patch("app.GX_integration.toolshed.httpx.AsyncClient") as mock_client_cls, \
         patch.object(workflow_installer, "_tool_exists", new=AsyncMock(return_value=True)):
        await workflow_installer._ensure_repository_installable(step, step["tool_shed_repository"])

    mock_client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_repository_installable_bails_when_repo_id_missing(workflow_installer):
    """If the matching repo row has no `id`, we log and fall through — no DELETE fired."""
    step = _step()
    repo = {
        # note: no "id"
        "name": "tool",
        "owner": "dev",
        "changeset_revision": "rev123",
        "status": "Error",
        "deleted": False,
        "uninstalled": False,
    }
    workflow_installer.gi_admin.gi.toolShed.get_repositories.return_value = [repo]

    with patch("app.GX_integration.toolshed.httpx.AsyncClient") as mock_client_cls, \
         patch.object(workflow_installer, "_tool_exists", new=AsyncMock(return_value=False)):
        await workflow_installer._ensure_repository_installable(step, step["tool_shed_repository"])

    mock_client_cls.assert_not_called()
