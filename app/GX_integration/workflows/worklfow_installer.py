import logging
import asyncio
import traceback

from dotenv import load_dotenv

load_dotenv()

from sys import path
path.append('.')

from app.galaxy import GalaxyClient
from app.GX_integration import toolshed
from app.api.socket_manager import SocketManager
from app.enumerations import (
    NumericLimits,
    SocketMessageEvent,
    SocketMessageType
    )

class WorkflowInstaller:
    """Handles tool installation and workflow upload operations"""

    UNHEALTHY_REPOSITORY_STATUSES = toolshed.UNHEALTHY_REPOSITORY_STATUSES
    
    def __init__(self, galaxy_client: GalaxyClient):

        self.galaxy_client = galaxy_client

        self.gi_object=self.galaxy_client.gi_object 
        self.gi_admin = self.galaxy_client.gi_admin # For administrative functionalitites like toolshed instantiation and tool installing
        self.toolshed = self.gi_admin.gi.toolshed  # Galaxy's ToolShed client, on the admin key

        self.log = logging.getLogger(__class__.__name__)
    
    
    async def _install_galaxy_tool(self, toolshed_info):
        """install a tool to galaxy using a toolshed information of a galaxy worklfow ga file."""
        return await toolshed.install_revision(self.toolshed, toolshed_info)

    async def _reload_toolbox(self):
        """ Reload the galaxy toolbox. """
        await toolshed.reload_toolbox(self.gi_admin.gi)

    async def _tool_exists(self, step: dict) -> bool:
        """Checks if a specific version of a tool is installed within the galaxy instance"""

        tool_id = step.get('tool_id')
        if not tool_id:
            return True

        try:
            tool = await asyncio.to_thread(self.gi_admin.gi.tools.show_tool, tool_id)
        except Exception as e:
            self.log.debug(f"Could not find tool in search: {e}")
            return False
        if not tool:
            return False

        # Grab repository info (None if local tool)
        step_repo = step.get('tool_shed_repository')
        tool_repo = tool.get('tool_shed_repository')
        
        # If the step was defined to come from a Tool Shed, enforce that
        if step_repo:
            # tool must also be from a Tool Shed
            if not tool_repo:
                return False
            # revisions must match exactly
            if tool_repo.get('changeset_revision') != step_repo.get('changeset_revision'):
                return False

        return True

    async def _ensure_repository_installable(self, step: dict, toolshed_info: dict) -> None:
        """Clear a 'ghost install' of the step's repository so a fresh install runs (see toolshed.clear_ghost_install)."""
        if await toolshed.clear_ghost_install(self.gi_admin.gi, self.galaxy_client.galaxy_url,
                                              self.galaxy_client.admin_api_key, toolshed_info,
                                              lambda: self._tool_exists(step)):
            await self._reload_toolbox()

    # Function that installs tools missing in the galaxy instance for the workflow invocation
    # Need  administrator api
    async def _tool_check_install(self, step: dict, ws_manager: SocketManager, tracker_id: str):
        """Check and install if a tool in a workflow is missing.

        After install, re-verifies the exact revision is present by reloading the toolbox
        and re-checking up to 6 times (~60s). The Tool Shed API can return "already
        installed" while a different revision is the one actually installed, so the
        response message alone is not trustworthy."""

        # Recurse into subworkflow steps
        if step.get('type') == 'subworkflow':
            for sub_step in step['subworkflow']['steps'].values():
                await self._tool_check_install(sub_step, ws_manager, tracker_id)
            return  # Skip install for subworkflow container itself

        # Skip steps without a tool_id
        if not step.get('tool_id'):
            return

        # Already present at the right revision — nothing to do
        if await self._tool_exists(step):
            return

        toolshed_info = step['tool_shed_repository']
        self.log.info(f"Installing tool for step {step['id']}: {step['tool_id']}")

        # Repair stale/ghost DB rows so the install call below actually runs
        # instead of being silently shortcut by the Tool Shed API.
        await self._ensure_repository_installable(step, toolshed_info)

        try:
            await self._install_galaxy_tool(toolshed_info)
        except Exception as e:
            self.log.error(f"Failed to install tool '{toolshed_info['name']}': {str(e)}  traceback:{traceback.format_exc()}")
            if ws_manager:
                await ws_manager.broadcast(
                    event=SocketMessageEvent.workflow_upload.value,
                    data={
                        "type": SocketMessageType.TOOL_INSTALL.value,
                        "payload": {"message": f"Failed to install {toolshed_info['name']}: {e}"}
                    },
                    tracker_id=tracker_id,
                )
            raise

        # Post-install verification: reload toolbox and confirm the exact
        # revision is present. Galaxy may need a moment for conda resolution.
        if await toolshed.reload_until_present(self.gi_admin.gi, lambda: self._tool_exists(step)):
            self.log.info(f"Verified install of {step['tool_id']}")
            if ws_manager:
                await ws_manager.broadcast(
                    event=SocketMessageEvent.workflow_upload.value,
                    data={
                        "type": SocketMessageType.TOOL_INSTALL.value,
                        "payload": {"message": f"Tool Name: {toolshed_info.get('name', 'N/A')}, installed successfully."}
                    },
                    tracker_id=tracker_id,
                )
            return

        msg = (
            f"Tool {step['tool_id']} not present after install + 60s wait. "
            f"A different revision of '{toolshed_info['name']}' may be installed; "
            f"uninstall the existing revision and retry, or install manually."
        )
        self.log.error(msg)
        if ws_manager:
            await ws_manager.broadcast(
                event=SocketMessageEvent.workflow_upload.value,
                data={
                    "type": SocketMessageType.TOOL_INSTALL.value,
                    "payload": {"message": msg}
                },
                tracker_id=tracker_id,
            )
        raise RuntimeError(msg)
    
    async def upload_workflow(self, workflow_json: dict, ws_manager: SocketManager = None, tracker_id: str = None, retry_count: int = 1, installer_count = 1):
        """Upload workflow from a ga file json."""
        
        semaphore = asyncio.Semaphore(installer_count) # Limit semaphores for tool installation.
        async def limited_install(step):
            async with semaphore:
                return await self._tool_check_install(step, ws_manager, tracker_id)

        if ws_manager:
            await ws_manager.broadcast(
                event = SocketMessageEvent.workflow_upload.value,
                data = {
                    "type": SocketMessageType.UPLOAD_WORKFLOW.value,
                    "payload": {"message": "Workflow upload started, checking and installing missing tools."}
                    },
                tracker_id=tracker_id
            )
        self.log.info("Workflow upload started, checking and installing missing tools.")
        # Check if the tools are installed and install all missing tools
        try:
            workflow_steps=workflow_json.get('steps', None)

            if workflow_steps:
                    await asyncio.gather(*[limited_install(step) for step in workflow_steps.values()])
                    
        except Exception as e:
            if ws_manager:
                await ws_manager.broadcast(
                    event = SocketMessageEvent.workflow_upload.value,
                    data = {
                        "type": SocketMessageType.UPLOAD_FAILURE.value,
                        "payload": {"message": f"Error installing missing tools in the uploaded workflow: {e}"}
                        },
                    tracker_id=tracker_id
                )
            self.log.error(f"Error installing missing tools in the uploaded workflow: {e} traceback:{traceback.format_exc()}")
            
            # return {"error": f"Error installing missing tools in the uploaded workflow: {e}"}
            
        await self._reload_toolbox()

        workflow = await asyncio.to_thread(
            self.gi_object.workflows.import_new, 
            src=workflow_json, 
            publish=False
            )
        
        # Extract workflow id for uploaded worklflow checking.
        workflow_id = workflow.id
        
        retry_count += 1
        # Check if the workflow is considered runnable by the instance
        if workflow.is_runnable:
            if ws_manager:
                self.log.info("workflow Uploaded successfully")
                await ws_manager.broadcast(
                    event = SocketMessageEvent.workflow_upload.value,
                    data = {
                        "type": SocketMessageType.UPLOAD_COMPLETE.value,
                        "payload": {"message": "Workflow successfully uploaded."}
                        },
                    tracker_id=tracker_id
                    )
        else:
            self.log.debug("Workflow is not runnable, deleting failed workflow.")
            await asyncio.to_thread(self.gi_object.gi.workflows.delete_workflow, workflow_id=workflow_id)
            if retry_count > 3:
                self.log.error("Workflow is not runnable, failed to upload correctly.")
                if ws_manager:
                    await ws_manager.broadcast(
                        event = SocketMessageEvent.workflow_upload.value,
                        data = {
                            "type": SocketMessageType.UPLOAD_FAILURE.value,
                            "payload": {"message": "Workflow upload failed."}
                            },
                        tracker_id=tracker_id
                        )
            else:
                self.log.error(f"Workflow is not runnable, failed to upload correctly. Retrying... (attempt {retry_count})")
                await asyncio.sleep(NumericLimits.SHORT_SLEEP.value)
                await self.upload_workflow(workflow_json=workflow_json, ws_manager=ws_manager, tracker_id=tracker_id, retry_count=retry_count)
