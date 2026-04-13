import os
import re
import json
from fastmcp import FastMCP
import logging
from typing import Optional
import asyncio
import httpx
from contextlib import asynccontextmanager

from bioblend.galaxy.client import ConnectionError as GalaxyConnectionError
from qdrant_client.models import PointStruct
from qdrant_client.http.exceptions import ApiException

from app.log_setup import configure_logging
from app.galaxy import GalaxyClient

from app.bioblend_server.mcp_middleware import JWTGalaxyKeyMiddleware
from app.bioblend_server.mcp_context import current_api_key_server

from app.bioblend_server.utils import (
    InformerResponse,
    DefaultTextResponses,
    get_llm_response,
    analyze_invocation,
    fetch_workflow_json_async
    )

from app.bioblend_server.background_runner import BackgroundIndexer
from app.bioblend_server.informer.informer import GalaxyInformer
from app.GX_integration.workflows.workflow_manager import WorkflowManager

from app.bioblend_server.GraphRAG.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
)
from app.bioblend_server.GraphRAG.neo4j_connector import Neo4jGraphConnector
from app.bioblend_server.GraphRAG.semantic_adapter import InformerSemanticAdapter
from app.bioblend_server.GraphRAG.pipeline import GraphRAGPipeline

# Module-level lazy singletons for connection pooling
_neo4j_connector: Neo4jGraphConnector | None = None
_neo4j_lock = asyncio.Lock()
_informer_manager = None
_informer_lock = asyncio.Lock()


async def _get_neo4j_connector() -> Neo4jGraphConnector:
    global _neo4j_connector
    if _neo4j_connector is None:
        async with _neo4j_lock:
            if _neo4j_connector is None:
                _neo4j_connector = Neo4jGraphConnector(
                    uri=NEO4J_URI,
                    user=NEO4J_USER,
                    password=NEO4J_PASSWORD,
                    database=NEO4J_DATABASE,
                )
    return _neo4j_connector


async def _get_informer_manager():
    global _informer_manager
    if _informer_manager is None:
        async with _informer_lock:
            if _informer_manager is None:
                from app.bioblend_server.informer.manager import InformerManager
                _informer_manager = await InformerManager.create()
    return _informer_manager

configure_logging()
logger = logging.getLogger("fastmcp_bioblend_server")

if not os.environ.get("GALAXY_API_KEY") or not os.environ.get("QDRANT_HTTP_PORT") or not os.environ.get("CURRENT_LLM"):
    logger.warning("MCP server environment variables are not set.")

# Context Manager for the MCP server.

@asynccontextmanager
async def mcp_galaxy_lifespan(server: FastMCP):
    """ 
    Manages the lifecycle of the background indexer.
    Ensures it starts with the server and shuts down gracefully.
    """
    # 1. Initialize the worker and start loop
    indexer = BackgroundIndexer()
    loop_task = asyncio.create_task(indexer.run_loop())
    
    yield 
    
    # 3. Graceful Shutdown
    logger.info("Server shutting down, stopping background tasks...")
    loop_task.cancel()
    try:
        # Wait for the task to acknowledge cancellation
        await loop_task
    except asyncio.CancelledError:
        pass
    logger.info("Background tasks stopped cleanly.")
    
    
# ==================================== #
     ## The Galaxy MCP Server ##
# ==================================== #

bioblend_app = FastMCP(
                    name="galaxyTools",
                    instructions="""
                            Galaxy MCP assistant.
                            Provide information on Galaxy tools, datasets, workflows, and invocations.
                            Explain failures and recommend fixes.
                            Import recommended workflows when requested.

                            """,
                    middleware=[JWTGalaxyKeyMiddleware()],
                    lifespan=mcp_galaxy_lifespan,
                    )


# =============================================================================================================================================================== #
    ## Tool 1: Galaxy assitant recommendation tool, gives details on galaxy datasets, tools, and workflows both in and outside of the connected galaxy instance ##
# =============================================================================================================================================================== #

@bioblend_app.tool()
async def get_galaxy_information_tool(
    query: str,
    query_type: str,
    entity_id: str = None
) -> InformerResponse:
    """
    Fetch detailed information about Galaxy tools, workflows, datasets, and workflow invocations.

    This tool performs semantic and fuzzy search over Galaxy entities and can access
    information from the connected Galaxy instance (user-specific, real-time state) 
    as well as global Galaxy data. It is intended for retrieving factual metadata 
    about specific entities and interacting with the user's Galaxy environment.

    Use this tool when:
    - The user asks for a recommendation of a tool/workflow.
    - The user asks for details about a specific tool, workflow, or dataset.
    - The query involves workflow invocations or dataset metadata.
    - The user may want to execute or import a workflow/tool.
    - Real-time or instance-specific information is required.

    Do NOT use this tool when:
    - The query focuses on relationships between multiple tools or workflows.
    - The question requires structural reasoning or ecosystem-level analysis.

    Capabilities:
    - Semantic and fuzzy search over Galaxy tools, workflows, and datasets.
    - Retrieves metadata: descriptions, parameters, dataset info, workflow structure.
    - Returns actionable links when available (e.g., execute workflow, import workflow, open tool).

    Args:
        query (str): The user's natural language query with full context.
        query_type (str): Galaxy entity type ("tool", "dataset", or "workflow").
        entity_id (Optional[str]): Explicit Galaxy entity ID if provided.

    Returns:
        InformerResponse containing:
            - response: Detailed textual answer.
            - actions: Optional actionable operations (Execute or Import) with links.
    """
    
    logger.info(f"Calling get_galaxy_information with query='{query}', query_type='{query_type}', entity_id='{entity_id}'")
    try:
        # Get current user
        user_api_key = current_api_key_server.get()
        if user_api_key is None:
            raise ValueError("current user api-key is missing")        

        # Create galaxy instances
        galaxy_client = GalaxyClient(user_api_key)
        logger.info( f"current Galaxy MCP server user: {galaxy_client.whoami}")
        # Create GalaxyInformer object and execute informer
        informer = await GalaxyInformer.create(galaxy_client=galaxy_client, entity_type=query_type)
        response, actions  = await informer.get_entity_info(search_query = query, entity_id = entity_id)
        
        return InformerResponse(response = response, actions = actions)
    
    except GalaxyConnectionError as e:
        logger.error(f"Failed to connect to Galaxy: {e}")
        return InformerResponse(response="Failed to connect to Galaxy.", actions=None)
    except Exception as e:
        logger.error(f"Error in get_galaxy_information_tool: {e}", exc_info=True)
        return InformerResponse(response="An error occurred while fetching Galaxy information.", actions=None)


# ==================================================================================================================================================================== #
    ## Tool 2: GraphRAG knowledge retrieval — planner-driven graph reasoning ##
# ==================================================================================================================================================================== #

@bioblend_app.tool()
async def graph_rag_query(
    query: str,
    debug: bool = False,
) -> DefaultTextResponses:
    """
    Retrieve contextual knowledge from the Galaxy knowledge graph using GraphRAG.

    This tool uses an LLM-based planner to automatically determine the best
    retrieval strategy for any query over the Galaxy knowledge graph.  It handles
    entity lookups, multi-hop reasoning, workflow/tool comparisons, path finding,
    category exploration, and ecosystem analytics — all from a single natural
    language query.

    The planner generates structured query schemas that are converted to safe
    parameterized Cypher, executed against Neo4j, and rendered as evidence
    context.

    Args:
        query: The user's natural language question with full context.
        debug: If True, include planner reasoning and per-query timing in
            the response.

    Returns:
        DefaultTextResponses with structured knowledge graph evidence.
    """
    logger.info(f"GraphRAG query: '{query}', debug={debug}")
    try:
        from app.bioblend_server.informer.search.semantic_searcher import SemanticSearcher

        # 1. Reusable singletons (module-level, thread-safe)
        connector = await _get_neo4j_connector()
        manager = await _get_informer_manager()

        user_api_key = current_api_key_server.get()
        galaxy_client = GalaxyClient(user_api_key) if user_api_key else None
        username = galaxy_client.whoami if galaxy_client else "default_user"

        semantic_searcher = SemanticSearcher(
            vector_manager=manager,
            entity_type="tool",
            username=username,
            score_threshold=0.3,
            limit=10,
        )

        adapter = InformerSemanticAdapter(
            semantic_searcher=semantic_searcher,
            entity_types=["tool", "workflow"],
        )

        # 3. Pipeline
        pipeline = GraphRAGPipeline(
            connector=connector,
            semantic_adapter=adapter,
        )

        # 4. Execute
        result = await pipeline.run(query=query, debug=debug)

        # 5. Build response — include debug details when requested
        if debug:
            import json
            debug_sections = [result.answer]
            if result.raw_evidence:
                debug_sections.append(f"\n--- Raw Evidence ---\n{result.raw_evidence}")
            if result.plan_summary:
                debug_sections.append(f"\n--- Plan Summary ---\n{result.plan_summary}")
            if result.limitations:
                debug_sections.append(f"\n--- Limitations ---\n" + "\n".join(f"- {l}" for l in result.limitations))
            if result.debug_trace:
                debug_sections.append(f"\n--- Debug Trace ---\n{json.dumps(result.debug_trace, indent=2, default=str)}")
            return DefaultTextResponses(response="\n".join(debug_sections))

        return DefaultTextResponses(response=result.answer)

    except Exception as e:
        logger.error(f"GraphRAG query failed: {e}", exc_info=True)
        return DefaultTextResponses(response=f"GraphRAG query failed: {str(e)}")
    

# ========================================================================================================== #
    ## Tool 3: Invocation Analyzing tool, analyzes, summarizes and recommends fixes for failed invocaiton. ##
# ========================================================================================================== #

@bioblend_app.tool()
async def explain_galaxy_workflow_invocation(
    invocation_id: str,
    failure: bool
) -> DefaultTextResponses:
    """
    Analyze a Galaxy workflow invocation and generate a detailed report.

    This tool retrieves metadata for a specific Galaxy workflow invocation. 
    It summarizes output datasets for successful jobs or provides diagnostic details 
    and actionable suggestions for failed jobs.

    Use this tool when:
    - The user wants a summary of workflow outputs.
    - The user needs diagnostics and suggested fixes for failed workflow runs.

    Args:
        invocation_id (str): Unique identifier of the Galaxy workflow invocation to analyze.
        failure (bool, optional): Focus on failed job diagnostics (`True`) or output summaries (`False`). Defaults to `False`.

    Returns:
        DefaultTextResponses: A detailed report of workflow outputs or failure diagnostics with actionable suggestions.
    """
    
    # Get current user
    user_api_key = current_api_key_server.get()
    if user_api_key is None:
        raise ValueError("current user api-key is missing")
    
    try:
        
        invocation_analysis: str = await analyze_invocation(invocation_id = invocation_id, user_api_key = user_api_key, failure=failure)
        
        if failure:
            logger.info("Loading failure explanation and suggestions for invocation.")
            invocation_prompt = f"""
                You are a Galaxy workflow expert.

                Analyze the following workflow invocation report.
                Identify why the workflow failed and suggest clear, actionable fixes.

                Report:
                {invocation_analysis}

                Respond with:
                - Root cause of failure(s)
                - Recommended fix or next step
                """
        else:
            logger.info("Loading summarized report for successful invocation.")
            invocation_prompt = f"""
                You are a Galaxy workflow expert.

                Summarize this successful workflow invocation report.

                Report:
                {invocation_analysis}

                Respond with:
                - What the workflow accomplished
                - Key output datasets or collections
                - Next logical steps for the user
                """
        try:
            response = await get_llm_response(message = invocation_prompt)
        except Exception as e:
            logger.error(f"Error preparing structured suggestions, returning full report. {e}")
            response = invocation_analysis
            
        return DefaultTextResponses(response = response)
    
    except GalaxyConnectionError as e:
        logger.error(f"Failed to connect to Galaxy: {e}")
        return DefaultTextResponses(response = "Failed to connect to Galaxy please try again.")
    except Exception as e:
        logger.error(f"Error caused whn trying to fetch invocation details: {e}")
        return DefaultTextResponses(response = "Error caused whn trying to fetch invocation details.")


# ============================================================ #
    ## Tool 4: Workflow Importing tool after recommendation. ##
# ============================================================ #

@bioblend_app.tool()
async def import_workflow_to_galaxy_instance(
    workflow_name: str
) -> DefaultTextResponses:
    # TODO: No Galaxy duplicate check add that.
    
    """
    Import a Galaxy workflow from the IWC or WorkflowHUB repository into the user's Galaxy instance.

    This tool fetches the workflow JSON, uploads it to the connected Galaxy instance,
    and ensures any required tools are installed. The workflow is then added to the
    user's workflow list.

    Use this tool when:
    - The user wants to import an workflow.

    Args:
        workflow_name (str): Full and exact name of the workflow to import.

    Returns:
        DefaultTextResponses: Message indicating the import status or an error description.
    """
    try:
        
        from app.bioblend_server.informer.manager import InformerManager
        
        # Validate API key and initialize clients
        user_api_key: str = current_api_key_server.get()
        if not user_api_key:
            raise ValueError("User API key is not provided.")
        
        galaxy_client: GalaxyClient = GalaxyClient(user_api_key)
        username = galaxy_client.whoami
        workflow_manager: WorkflowManager = WorkflowManager(galaxy_client)
        qdrant_client: InformerManager = await InformerManager().create()

        # TODO: Fill the workflow collection name (has to be user-specific, so ...)
        workflow_collection_name: str = "generic_galaxy_workflow"
        workflow_name_alternative = re.sub(r'workflow', '', workflow_name, flags= re.IGNORECASE) # if there are any uneccessary strings included in the parameter.
        
        # Step 1: Search for the workflow by name in metadata (synchronous call in thread pool)
        logger.info(f"Searching for workflow '{workflow_name}' in collection '{workflow_collection_name}'.")
        logger.info(f"current Galaxy MCP server user: {username}")
        hits = await qdrant_client.match_name_from_collection(
            workflow_collection_name=workflow_collection_name,
            workflow_name = workflow_name
            )

        if not hits or not hits[0]:
            
            hits = await qdrant_client.match_name_from_collection(
            workflow_collection_name=workflow_collection_name,
            workflow_name = workflow_name_alternative
            )
            
            if not hits or not hits[0]:
                logger.warning(f"Workflow '{workflow_name}' not found in collection '{workflow_collection_name}'")
                response =  f"Workflow '{workflow_name}' not found in available workflow collection for import."
                return DefaultTextResponses(response = response)
            
            # Extract workflow download URL from point payload
        point: PointStruct = hits[0][0]
        workflow_url: Optional[str] = point.payload.get("raw_download_url")
        workflow_source = point.payload.get("source")
        
        if not workflow_url:
            logger.error(f"No download link found for workflow '{workflow_name}'")
            response = f"Couldn't import workflow '{workflow_name}'."
            return DefaultTextResponses(response = response)
        
        # Fetch the workflow JSON
        logger.info(f"Fetching workflow JSON from IWC repository using URL: {workflow_url}")
        workflow_json: dict = await fetch_workflow_json_async(workflow_url)

        if workflow_source == "workflow_hub":
            workflow_json: dict = workflow_json.get("content")
            if isinstance(workflow_json, str):
                workflow_json = json.loads(workflow_json)

        ga_workflow_name: str = workflow_json.get("name", workflow_json.get("workflow_name", ""))
        if not ga_workflow_name:
            logger.error(f"Workflow JSON does not contain a 'name' field for '{workflow_name}'")
            raise ValueError(f"Workflow JSON does not contain a 'name' field for '{workflow_name}'.")

        # Background upload task
        logger.info(f"Initiating upload of workflow '{ga_workflow_name}'")
        
        asyncio.create_task(
            workflow_manager.upload_workflow(
                workflow_json=workflow_json
                )
            )

        response = (
            f"{ga_workflow_name} workflow is being imported,"
            "mssing tools are being checked and installed,"
            "and the workflow will be added to your workflow list shortly."
        )
        
        return DefaultTextResponses(response = response)
    
    except GalaxyConnectionError as e:
        logger.error(f"Failed to connect to Galaxy: {e}")
        return DefaultTextResponses(response = "Failed to connect to Galaxy.")
    
    except httpx.HTTPStatusError as http_err:
        logger.error(f"HTTP error fetching workflow from IWC repository: {str(http_err)}")
        return DefaultTextResponses(response = "HTTP error occurred while fetching workflow from IWC repository.")
    
    except httpx.RequestError as req_err:
        logger.error(f"Network error fetching workflow from IWC repository: {str(req_err)}")
        return DefaultTextResponses(response = "Request HTTP error occurred while fetching workflow from IWC repository.")
    
    except ApiException as qdrant_err:
        logger.error(f"Qdrant error occurred during workflow search: {str(qdrant_err)}")
        return DefaultTextResponses(response ="Error occured during workflow search.")
    
    except ValueError as val_err:
        logger.error(f"Validation error: {str(val_err)}")
        return DefaultTextResponses(response = "An unexpected error occurred during workflow import.")
    
    except Exception as exc:
        logger.exception(f"Unexpected error during workflow import: {str(exc)}")
        return DefaultTextResponses(response = "An unexpected error occurred during workflow import.")