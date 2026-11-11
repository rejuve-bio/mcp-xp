from sys import path
path.append('.')
import tempfile
import os
from dotenv import load_dotenv
import logging
import uuid
import redis
import hashlib
import json
import asyncio
import aiofiles

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Path,
    Query,
    Form,
    Request,
    Response
    )
from fastapi.responses import HTMLResponse
from fastapi.concurrency import run_in_threadpool
from bioblend.galaxy.objects.wrappers import HistoryDatasetAssociation, HistoryDatasetCollectionAssociation
from starlette.status import HTTP_204_NO_CONTENT

from app.context import current_api_key
from app.galaxy import GalaxyClient
from app.GX_integration.workflows.workflow_manager import WorkflowManager
from app.api.schemas import workflow
from app.api.socket_manager import ws_manager
from app.enumerations import SocketMessageEvent, SocketMessageType
from app.orchestration.invocation_cache import InvocationCache
from app.orchestration.invocation_tasks import InvocationBackgroundTasks
from app.enumerations import NumericLimits

from app.exceptions import InternalServerErrorException

load_dotenv()

logger = logging.getLogger('workflow_endpoint')
router = APIRouter()
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=os.environ.get("REDIS_PORT"), db=0, decode_responses=True)
invocation_cache = InvocationCache(redis_client)
invocation_background = InvocationBackgroundTasks(cache = invocation_cache, redis_client=redis_client)

@router.get(
    "",
    response_model=workflow.WorkflowList,
    summary="List all workflows",
    tags=["Workflows"]
)
async def list_workflows():
    """
    Retrieves a list of all workflows available in the Galaxy instance.
    Returns basic information including workflow ID, name, and description.
    """
    
    api_key = current_api_key.get()

    try:
        galaxy_client = GalaxyClient(api_key)
        username = galaxy_client.whoami
        
        # Step 1: Request deduplication
        request_hash = hashlib.md5(api_key.encode()).hexdigest()
        if await invocation_cache.is_duplicate_workflow_request(username, request_hash):
            logger.info("Duplicate workflow list request detected")

        # Step 2: Check for cached responses
        cached_data = await invocation_cache.get_workflows_cache(username)
        if cached_data:
            deleted_workflow_ids = await invocation_cache.get_deleted_workflows(username)
            # Filter out deleted workflows
            filtered_workflows = [
                w for w in cached_data if w.get("id") not in deleted_workflow_ids
            ]
             
            logger.info("Serving workflows from cache")
            return workflow.WorkflowList(workflows=filtered_workflows)

        # Step 3: If no cache is found then retreive workflows list from galaxy, and set cache.
        workflow_manager = WorkflowManager(galaxy_client)
        workflow_list = await invocation_background.fetch_workflows_safely(workflow_manager, fetch_details=True)
        await invocation_cache.set_workflows_cache(username,workflow_list)
        
        # Remove and clear deleted workflow list cache
        await invocation_cache.clear_deleted_workflows(username)
        return workflow.WorkflowList(workflows=workflow_list)
    except Exception as e:
        raise InternalServerErrorException("Failed to list workflows")
    
@router.post(
        "/upload-workflow",
        response_model = workflow.WorkflowUploadResponse,
        summary = "Upload an external workflow ga file",
        tags=["Workflows"]
)
async def upload_workflow(
        file: UploadFile = File(..., description="The Workflow ga file to upload."),
        tracker_id: str | None = Query(None, description="Client-supplied tracker ID for WebSocket updates"),
):
    """Uploads an external workflow from a ga file into the galaxy instance."""

    galaxy_client = GalaxyClient(current_api_key.get())
    workflow_manager = WorkflowManager(galaxy_client)
    tracker_id = tracker_id or str(uuid.uuid4())

    try:
        # Use a temporary file to handle the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            tmp.write(await file.read())    
            tmp_path = tmp.name
            
        async with aiofiles.open(tmp_path, 'r') as f:
            workflow_json: dict = json.loads(await f.read())

        workflow_name = workflow_json.get("name")
        # Upload the galaxt workflow into the instance
        asyncio.create_task(
            workflow_manager.upload_workflow(
                        workflow_json=workflow_json,
                        ws_manager=ws_manager,
                        tracker_id=tracker_id
                    )
                )

        # Clean up the temporary file and return the workflow name as response.
        os.remove(tmp_path)
        return workflow.WorkflowUploadResponse(workflow_name=workflow_name)
    
    except Exception as e:
        # Clean up in case of error
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise InternalServerErrorException("An error occurred")  
 
# @router.get(
#     "/{workflow_id}/form",
#     response_class=HTMLResponse,
#     summary="Get Dynamic Workflow Form",
#     tags=["Workflows"]
# )
async def get_workflow_form(
    workflow_id: str = Path(..., description="The ID of the Galaxy workflow."),
    history_id: str = Query(..., description="The ID of the history to select inputs from.")
):
    """
    Generates and returns a dynamic HTML form for a specific workflow.
    The form is populated with available datasets from the provided history.
    """

    galaxy_client = GalaxyClient(current_api_key.get())
    workflow_manager = WorkflowManager(galaxy_client)

    try:
        # Get the wrapper objects needed by your service method
        workflow_obj = await run_in_threadpool(workflow_manager.gi_object.workflows.get, workflow_id)
        history_obj = await run_in_threadpool(workflow_manager.gi_object.histories.get, history_id)

        # Run the synchronous HTML build in a thread pool
        html_form = await run_in_threadpool(
            workflow_manager.build_input,
            workflow=workflow_obj,
            history=history_obj
        )
        return HTMLResponse(content=html_form)
    except Exception as e:
        raise InternalServerErrorException("Failed to build workflow form")

@router.post(
    "/{workflow_id}/histories/{history_id}/execute",
    response_model=workflow.WorkflowExecutionResponse,
    summary="Execute a Workflow",
    tags=["Workflows"]
)
async def execute_workflow(
    request: Request,
    dummy_input: str = Form(None, description="Dummy input to force form rendering the form input for the galaxy execution"), # Adding dummy form value to input request form.
    workflow_id: str = Path(..., description="The ID of the Galaxy workflow to execute."),
    history_id: str = Path(..., description="The ID of the history for execution."),
    tracker_id: str | None = Query(None, description="Client-supplied tracker ID for WebSocket updates"),
):
    """
    Executes a workflow using input data submitted via a form.

    This endpoint expects a `multipart/form-data` submission, as generated by the `/form` endpoint.
    It parses the form, invokes the workflow, tracks it to completion, and returns the results.
    """

    galaxy_client = GalaxyClient(current_api_key.get())
    workflow_manager = WorkflowManager(galaxy_client)
    tracker_id = tracker_id or str(uuid.uuid4())

    try:
        await ws_manager.broadcast(event = SocketMessageEvent.workflow_execute.value,
                             data = {
                                 "type": SocketMessageType.WORKFLOW_EXECUTE.value,
                                 "payload": {"message" : "Execution started."}   
                                },
                             tracker_id=tracker_id
        )
        form_data = await request.form()
        workflow_obj = await run_in_threadpool(workflow_manager.gi_object.workflows.get, workflow_id)
        history_obj = await run_in_threadpool(workflow_manager.gi_object.histories.get, history_id)
        workflow_details = await run_in_threadpool(workflow_manager.gi_object.gi.workflows.show_workflow, workflow_id)

        # Reconstruct the 'inputs' dictionary for BioBlend
        inputs = {}
        steps: dict = workflow_details['steps']
        for step_id, step_details in steps.items():

            # Check if this step is an input step and is in the form data
            form_value = form_data.get(step_id)
            if form_value is None:
                continue

            # Differentiate between data inputs and parameter inputs
            if step_details['type'] in ['data_input', 'data_collection_input']:
                src = 'hdca' if step_details['type'] == 'data_collection_input' else 'hda'
                inputs[step_id] = {'src': src, 'id': form_value}

            elif step_details['type'] == 'parameter_input':
                inputs[step_id] = form_value

        logger.info(f"inputs applied: {inputs}")

        # Run the entire synchronous workflow execution and tracking in a thread pool
        invocation_id, report, intermediate_outputs, final_outputs = await workflow_manager.run_track_workflow(
            inputs=inputs,
            workflow=workflow_obj,
            history=history_obj,
            ws_manager = ws_manager,
            tracker_id = tracker_id
            )

        # Format the outputs using Pydantic schemas
        intermediate_outputs_formatted = []
        final_outputs_formatted = []

        for ds in final_outputs:
            if isinstance(ds, HistoryDatasetAssociation):
                final_outputs_formatted.append(
                        {
                            "type" : "dataset",
                            "id": ds.id,
                            "name": ds.name,
                            "visible": ds.visible,
                            "peek": workflow_manager.gi_object.gi.datasets.show_dataset(ds.id)['peek'],
                            "data_type": workflow_manager.gi_object.gi.datasets.show_dataset(ds.id)['data_type']
                        }
                    )
            elif isinstance(ds, HistoryDatasetCollectionAssociation):
                final_outputs_formatted.append(
                        {
                            "type" : "collection",
                            "id": ds.id,
                            "name": ds.name,
                            "visible": ds.visible,
                            "collection_type": ds.collection_type,
                            "elements": [
                                {
                                    "identifier": e["element_identifier"],
                                    "name": e["object"]["name"],
                                    "id": e["object"]["id"],
                                    "peek": e["object"]["peek"],
                                    "data_type": workflow_manager.gi_object.gi.datasets.show_dataset(e["object"]["id"])['data_type']
                                }

                                for e in ds.elements
                            ]
                        }
                    )
                
        for ds in intermediate_outputs:
            if isinstance(ds, HistoryDatasetAssociation):
                intermediate_outputs_formatted.append(
                        {
                            "type" : "dataset",
                            "id": ds.id,
                            "name": ds.name,
                            "visible": ds.visible,
                            "peek": workflow_manager.gi_object.gi.datasets.show_dataset(ds.id)['peek'],
                            "data_type": workflow_manager.gi_object.gi.datasets.show_dataset(ds.id)['data_type']
                        }
                    )
            elif isinstance(ds, HistoryDatasetCollectionAssociation):
                intermediate_outputs_formatted.append(
                        {
                            "type" : "collection",
                            "id": ds.id,
                            "name": ds.name,
                            "visible": ds.visible,
                            "collection_type": ds.collection_type,
                            "elements": [
                                {
                                    "identifier": e["element_identifier"],
                                    "name": e["object"]["name"],
                                    "id": e["object"]["id"],
                                    "peek": e["object"]["peek"],
                                    "data_type": workflow_manager.gi_object.gi.datasets.show_dataset(e["object"]["id"])['data_type']
                                }

                                for e in ds.elements
                            ]
                        }
                    )
                                        
        return {
                "invocation_id": invocation_id,
                "history_id": history_id,
                "report": report,
                "final_outputs": final_outputs_formatted,
                "intermediate_outputs": intermediate_outputs_formatted
            }
    except Exception as e:
        await ws_manager.broadcast(
            event= SocketMessageEvent.workflow_execute.value,
            data = {
                "type": SocketMessageType.WORKFLOW_FAILURE.value,
                "payload" : f"Workflow execution failed: {e}"
            },
            tracker_id = tracker_id
        )
        # Provide a detailed error message for debugging
        raise InternalServerErrorException("Workflow execution failed")
    
@router.get(
    "/{workflow_id}/details",
    response_model=workflow.WorkflowDetails,
    summary="Get detailed information about the Galaxy workflow",
    tags=["Workflows"]
)
async def get_workflow_details(
    workflow_id: str = Path(..., description="The ID of the Galaxy workflow.")
):
    """
    Shows the details of a workflow including id, created time,
    annotations, published or not, creator, url, license, steps and inputs
    """

    galaxy_client = GalaxyClient(current_api_key.get())
    workflow_manager = WorkflowManager(galaxy_client)

    try:
        workflow_details  = await run_in_threadpool(workflow_manager.gi_object.gi.workflows.show_workflow, workflow_id)
       
        return {
            "id": workflow_details.get("id"),
            "tags": workflow_details.get("tags", None),
            "create_time": workflow_details.get("create_time"),
            "annotations": workflow_details.get("annotations", None),
            "published": workflow_details.get("published"),
            "license": workflow_details.get("license", None),
            "galaxy_url": workflow_details.get("url"),
            "creator": workflow_details.get("creator", None),
            "steps": workflow_details.get("steps"),
            "inputs": workflow_details.get("inputs"),
        }
                
    except Exception as e:
        # detailed error responses
        raise InternalServerErrorException('Show workflow failed')
    
@router.delete(
   "/DELETE",
    summary="Delete workflows",
    tags=["Workflows"],
    status_code=HTTP_204_NO_CONTENT
)
async def delete_workflows(
    workflow_ids: str = Path(..., description="Comma-separated IDs of the Galaxy workflows to delete")
) -> Response:
    """ Delete a galaxy workflow from a users galaxy instance. """
    
    api_key = current_api_key.get()
    galaxy_client = GalaxyClient(api_key)
    username = galaxy_client.whoami
    workflow_manager = WorkflowManager(galaxy_client)

    try:
        # Parse comma-separated string into list
        ids_list = [i.strip() for i in workflow_ids.split(",") if i.strip()]

        # Define semaphore with a limit
        semaphore = asyncio.Semaphore(NumericLimits.SEMAPHORE_LIMIT.value)

        # Define background task for deleting workflows with semaphore
        async def delete_workflow_with_semaphore(workflow_id: str):
            async with semaphore:
                await asyncio.to_thread(workflow_manager.gi_object.gi.workflows.delete_workflow, workflow_id)

        async def delete_workflows_task():
            try:
                await asyncio.gather(
                    *[delete_workflow_with_semaphore(workflow_id) for workflow_id in ids_list]
                )
                logger.info(f"Workflows successfully deleted: {ids_list}")
            except Exception as e:
                workflow_manager.log.error(f"Background task failed to delete workflows {ids_list} for user {username}: {e}")

        # Schedule deletion task in the background
        asyncio.create_task(delete_workflows_task())

        # Add workflow IDs44 to deleted set in Redis (foreground)
        await invocation_cache.add_deleted_workflows(username, ids_list)

        return Response(status_code=HTTP_204_NO_CONTENT)

    except Exception as e:
        raise InternalServerErrorException("Failed to process workflow deletion request")
        # TODO: Need to find way to make the deletion not affect the published workflow usecase(workflow publication).