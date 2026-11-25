import re
import os
import tempfile
import redis
import pathlib

from anyio.to_thread import run_sync
import logging

from sys import path
path.append(".")

from fastapi import APIRouter, Path, Query, BackgroundTasks, Response
from fastapi.responses import  FileResponse
from fastapi.concurrency import run_in_threadpool
from starlette.status import HTTP_204_NO_CONTENT

from app.context import current_api_key
from app.galaxy import GalaxyClient
from app.persistence import MongoStore
from app.api.schemas import invocation
from app.api.socket_manager import ws_manager
from app.orchestration.invocation_cache import InvocationCache
from app.orchestration.invocation_tasks import InvocationBackgroundTasks

from app.exceptions import InternalServerErrorException
from app.GX_integration.workflows.workflow_manager import WorkflowManager
from app.GX_integration.invocations.invocation_service import InvocationService
from app.GX_integration.invocations.utils import _rmtree_sync

# Helper functions and redis instantiation
logger = logging.getLogger("Invocation")
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=os.environ.get("REDIS_PORT"), db=0, decode_responses=True)
invocation_cache = InvocationCache(redis_client)
invocation_background = InvocationBackgroundTasks(cache = invocation_cache, redis_client=redis_client)
mongo_client = MongoStore()

invocation_service = InvocationService(
                cache = invocation_cache,
                background_tasks = invocation_background,
                mongo_client = mongo_client
                )
router = APIRouter()

@router.get(
    "",
    response_model=invocation.InvocationList,
    summary="List all workflow invocations",
    tags=["Invocation"]
)
async def list_invocations(
    workflow_id: str | None = Query(None, description="Filter by workflow ID"),
    history_id: str | None = Query(None, description="Filter by History ID"),
):
    """
    Retrieves a list of workflow invocations from the Galaxy instance.
    
    Features:
    - Multi-level caching for optimal performance
    - Request deduplication to prevent duplicate processing
    - Parallel processing of data fetching
    - Graceful error handling and partial results
    """
    api_key = current_api_key.get()
    
    # Initialize cache and clients
    try:
        galaxy_client = GalaxyClient(api_key)
        username = galaxy_client.whoami
        workflow_manager = WorkflowManager(galaxy_client)
        
        return await invocation_service.list_invocations(
            username = username,
            api_key = api_key,
            galaxy_client = galaxy_client,
            workflow_manager = workflow_manager,
            workflow_id = workflow_id,
            history_id = history_id,
            ws_manager = ws_manager
        )
        
    except Exception as e:
        logger.error(f"Failed to list invocations: {e}")
        raise  InternalServerErrorException("Failed to list invocations")

@router.get(
    "/by-chat/{chat_id}",
    response_model=invocation.InvocationList,
    summary="List invocations tagged with a given chat_id",
    tags=["Invocation"],
)
async def list_invocations_by_chat(
    chat_id: str = Path(..., description="Chat ID whose invocations to fetch"),
):
    """Reverse lookup: return every invocation tagged with `chat_id` for the
    JWT-authenticated user. Cross-user scoping is enforced server-side."""
    try:
        api_key = current_api_key.get()
        galaxy_client = GalaxyClient(api_key)
        username = galaxy_client.whoami
        return await invocation_service.list_invocations_by_chat(
            username=username,
            chat_id=chat_id,
        )
    except Exception as e:
        logger.error(f"Failed to list invocations by chat_id: {e}")
        raise InternalServerErrorException("Failed to list invocations by chat_id")

@router.get(
    "/{invocation_id}/invocation_pdf",
    response_class = FileResponse,
    summary= "Get invocation pdf report",
    tags=["Invocation", "Workflows"]
)
async def invocation_report_pdf(
    invocation_id: str = Path(..., description="The ID of the invocation from a certain workflow")
):

    galaxy_client = GalaxyClient(current_api_key.get())
    workflow_manager = WorkflowManager(galaxy_client)

    tmpdir = tempfile.mkdtemp(prefix="galaxy_pdf_")
    tmpdir_path = pathlib.Path(tmpdir)

    try:

        inv = await run_in_threadpool(
            workflow_manager.gi_object.gi.invocations.show_invocation,
            invocation_id=invocation_id
        )
        
        try:
            # get workflow object 
            workflow_obj = await run_in_threadpool(
                workflow_manager.gi_object.workflows.get,
                id_ = inv.get("workflow_id")
            )
            workflow_name = workflow_obj.name
        except Exception as e:
            logger.error(f"error finding workflow name: {e} defaulting to id.")
            workflow_name = inv.get("workflow_id")

        pdf_report_name = re.sub(r'[\\/*?:"<>|]', '', f"{workflow_name}_invocation_report.pdf")
        pdf_path = tmpdir_path / pdf_report_name

        await run_in_threadpool(
            workflow_manager.gi_object.gi.invocations.get_invocation_report_pdf,
            invocation_id=invocation_id,
            file_path=str(pdf_path)
        )
        background = BackgroundTasks()
        background.add_task(run_sync, _rmtree_sync, tmpdir)

        return FileResponse(
            path=pdf_path,
            filename=pdf_report_name,
            media_type="application/octet-stream",
            background=background
        )
    except Exception as exc:
        # Clean up immediately on error
        await run_sync(_rmtree_sync, tmpdir)
        raise InternalServerErrorException("Failed to get PDF invocation report") 

@router.get(
    "/{invocation_id}/result",
    response_model=invocation.InvocationResult,
    summary="Result of a certain workflow invocations",
    tags=["Invocation"]
)
async def show_invocation_result(
    invocation_id: str = Path(..., description=""),
    internal_api: str | None = None
):
    try:
        if internal_api:
            logger.info("Calling invocation result internally.")
            api_key = internal_api
        else:
            api_key = current_api_key.get()
            
        galaxy_client = GalaxyClient(api_key)
        username = galaxy_client.whoami
        workflow_manager = WorkflowManager(galaxy_client)
        
        return await invocation_service.get_invocation_result(
            invocation_id = invocation_id,
            username = username,
            api_key = api_key,
            galaxy_client = galaxy_client,
            workflow_manager = workflow_manager,
            ws_manager = ws_manager
        )
    except Exception as e:
        logger.error(f"Failed to show invocation result: {e}")
        raise  InternalServerErrorException("Failed to show invocation result")
        
@router.post(
    "/{invocation_id}/cancel",
    summary="Cancel a running workflow invocation",
    tags=["Invocation"],
    status_code=HTTP_204_NO_CONTENT
)
async def cancel_workflow_invocation(
    invocation_id: str = Path(..., description="The ID of the invocation to cancel")
) -> Response:
    try:
        api_key = current_api_key.get()
        galaxy_client = GalaxyClient(api_key)
        username = galaxy_client.whoami
        workflow_manager = WorkflowManager(galaxy_client)

        return await invocation_service.cancel_invocation(
            invocation_id=invocation_id,
            username=username,
            workflow_manager=workflow_manager
        )
    except Exception as e:
        logger.error(f"Failed to cancel invocation: {e}")
        raise InternalServerErrorException("Failed to cancel invocation")


@router.delete(
    "/DELETE",
    summary="Delete workflow invocations",
    tags=["Invocation"],
    status_code=HTTP_204_NO_CONTENT
)
async def delete_invocations(
    invocation_ids: str = Query(..., description="Comma-separated IDs of the workflow invocations to delete")
) -> Response:
    """
    Simulates deletion of workflow invocations in the middleware layer since Galaxy does not support
    permanent deletion of invocation records via API. Cancels the invocation(s) if running, purges
    associated datasets to free space, and marks them as deleted in persistent storage to filter
    from listings.
    """

    try:
        
        api_key = current_api_key.get()
        galaxy_client = GalaxyClient(api_key)
        username = galaxy_client.whoami
        
        workflow_manager = WorkflowManager(galaxy_client)
        return await invocation_service.delete_invocations(
            invocation_ids = invocation_ids,
            username = username,
            workflow_manager= workflow_manager
        )
        
    except Exception as e:
        logger.error(f"Failed to delete invocations: {e}")
        raise InternalServerErrorException("Failed to delete invocations")