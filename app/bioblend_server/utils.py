import os
import json
import logging
import asyncio
import httpx
import redis

from typing import List, Optional, Literal, Any
from pydantic import BaseModel 

from app.log_setup import configure_logging
from app.llm_config import LLMModelConfig, LLMConfiguration
from app.llm_provider import GeminiProvider, OpenAIProvider

from app.enumerations import InvocationStates

from app.galaxy import GalaxyClient
from app.bioblend_server.mcp_context import current_api_key_server
from app.orchestration.invocation_cache import InvocationCache
from app.orchestration.invocation_tasks import InvocationBackgroundTasks

from app.persistence import MongoStore
from app.GX_integration.invocations.invocation_service import InvocationService
from app.GX_integration.workflows.workflow_manager import WorkflowManager
from app.GX_integration.invocations.data_manager import InvocationDataManager

configure_logging()
logger = logging.getLogger("fastmcp_bioblend_server")

mongo_client = MongoStore()
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=os.getenv("REDIS_PORT"), db=0, decode_responses=True)
invocation_cache = InvocationCache(redis_client)
invocation_background = InvocationBackgroundTasks(cache = invocation_cache, redis_client = redis_client)
invocation_service = InvocationService(cache = invocation_cache, background_tasks= invocation_background, mongo_client= mongo_client)
inv_data_manager = InvocationDataManager(cache = invocation_cache, background_tasks = invocation_background)



# Pydantic model for the MCP server responses
class MCPActions(BaseModel):
    """Represents an actionable operation returned by the Galaxy Informer tool."""

    action: Literal["Execute", "Import"]
    link: Optional[str] = None

class InformerResponse(BaseModel):
    """Response schema returned by the Galaxy Informer tool."""

    response: str
    actions: Optional[List[MCPActions]] = None
    
class DefaultTextResponses(BaseModel):
    """Default Text repsponse of the MCP server."""
    
    response: str
    
    
async def get_llm_response(message, llm_provider = os.environ.get("CURRENT_LLM", "gemini")):
    
    model_config_data = LLMConfiguration().data
    
    if llm_provider == "gemini":
        gemini_cfg = LLMModelConfig(model_config_data['providers']['gemini'])
        llm = GeminiProvider(model_config=gemini_cfg)
    elif llm_provider == "openai":
        openai_cfg = LLMModelConfig(model_config_data['providers']['openai'])
        llm = OpenAIProvider(model_config=openai_cfg)

    # Accept either a raw string or already-formatted list[dict]
    if isinstance(message, str):
        message = [{"role": "user", "content": message}]
    return await llm.get_response(message)


async def fetch_workflow_json_async(url: str) -> dict[str, Any]:
            """Fetch and load JSON from a given URL using httpx (async)."""
            
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                response = await http_client.get(url)
                response.raise_for_status()
                return json.loads(response.text)
            
   
async def analyze_invocation(invocation_id: str, user_api_key: str, failure: bool) -> str:
    # instantiate galaxy client and invocation cacher classes
    galaxy_client = GalaxyClient(user_api_key)
    username = galaxy_client.whoami
    
    logger.info(f"Loading workflow Invocation with ID: {invocation_id} for explanationn for current Galaxy MCP server user: {username}")
    
    # Retrieve invocation details and workflow title concurrently
    try:
        def get_invocation_details():
            return galaxy_client.gi_client.invocations.show_invocation(invocation_id)
        
        def get_invocation_report():
            return galaxy_client.gi_client.invocations.get_invocation_report(invocation_id)
        
        invocation_details, invocation_report = await asyncio.gather(
            asyncio.to_thread(get_invocation_details),
            asyncio.to_thread(get_invocation_report)
        )
    except Exception as e:
        logger.error(f"Error retrieving invocation details or report: {str(e)}")
        return f"Error retrieving invocation details or report for ID {invocation_id}: {str(e)}"
    
    # Retrieve workflow details using the invocation report
    workflow_name = invocation_report.get("title", "Unknown")
    
    # Initialize explanation string
    explanation = "\n\n**Invocation details**"
    explanation += f"\n\nWorkflow Invocation ID: {invocation_id}\nWorkflow Name: {workflow_name}\n"
    
    # Check invocation state
    invocation_state = await invocation_cache.get_invocation_state(username, invocation_id)
    if invocation_state is None:
        invocation_raw_state = invocation_details.get('state', "Unknown")
    
        if invocation_raw_state == "scheduled":
            invocation_state = InvocationStates.PENDING.value
            
            try:
                workflow_manager = WorkflowManager(galaxy_client=galaxy_client)

                asyncio.create_task(
                    invocation_service.get_invocation_result(
                        invocation_id = invocation_id,
                        username=username,
                        api_key = current_api_key_server,
                        galaxy_client=galaxy_client,
                        workflow_manager= workflow_manager,
                        ws_manager=None
                        )
                    )
                explanation += "Workflow invocation still in Pending state. tracking workflow invocation."
            except Exception as e:
                logger.error(f"Error retrieving or tracking scheduled invocation: {str(e)}")
                explanation += f"\nError tracking scheduled invocation: {str(e)}\n"
        else:
            invocation_state = InvocationStates.FAILED.value
    
    invocation_inputs: dict = invocation_details.get("inputs", {})
    invocation_input_parameters: dict = invocation_details.get("input_step_parameters", {})
    if invocation_inputs:
        explanation += "\n\nInvocation input datasets:\n"
        explanation += "\n".join(f"  - {label.get('label')}" for label in invocation_inputs.values()) + "\n"

    if invocation_input_parameters:
        explanation += "\n\nInvocation input parameters:\n"
        explanation += "\n".join(f"  - {label.get('label')}: {label.get('parameter_value')}" for label in invocation_input_parameters.values()) + "\n"
    
    explanation += f"\n\nInvocation State: {invocation_state}\n\n"
    
    # If failure is indicated, focus on errors; otherwise, report on outputs
    if failure or invocation_state not in [InvocationStates.PENDING.value, InvocationStates.COMPLETE.value]:
        
        failure = True
        explanation += "Analysis for failures in the workflow invocation:\n"
        
        # Get failure reports or the invocation.
        explanation += await inv_data_manager.report_invocation_failure(galaxy_client, invocation_id)
        
        return explanation
    if invocation_state == "pending":
        explanation += "\n\nInvocation still Pending but no indicated failure. Analyzing output datasets:\n"
    else:
        explanation += "\n\nInvocation completed without indicated failure. Analyzing output datasets:\n"
        
    invocation_outputs = invocation_details.get('outputs', {})
    invocation_output_collections = invocation_details.get("output_collections", {})
    
    if invocation_outputs:
        explanation += "\n\nInvocation output datasets:\n"
        explanation += "\n".join(f"  - {label}" for label in invocation_outputs.keys()) + "\n"
    
    if invocation_output_collections:
        explanation += "\n\nInvocation output dataset collections:\n"
        explanation += "\n".join(f"  - {label}" for label in invocation_output_collections.keys()) + "\n" 
        
    if not invocation_outputs and not invocation_output_collections:
        explanation += "\nNo output datasets or collections found for this invocation.\n"
    
    return explanation