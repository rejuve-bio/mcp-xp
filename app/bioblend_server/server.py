import os
import re
import json
from fastmcp import FastMCP
import logging
from typing import Literal, Optional
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

from app.bioblend_server.GraphRAG.config import(
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    NEO4J_DATABASE
)

from app.bioblend_server.GraphRAG.neo4j_connector import Neo4jGraphConnector
from app.bioblend_server.GraphRAG.semantic_adapter import InformerSemanticAdapter
from app.bioblend_server.GraphRAG.pipeline import GraphRAGPipeline

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
) -> DefaultTextResponses:
    """
    Fetch detailed information on Galaxy tools, workflows, datasets, and invocations.

    This tool handles all information requests about Galaxy entities, based on
    the `query_type` (tool, workflow, dataset) and the user's `query`.
    Use `entity_id` only when the user's query explicitly includes an ID.

    Args:
        query: The user's query message that needs a response, accompanied by full and detailed contextual information.
        query_type: The type of Galaxy entity the query needs a response for, with one of three values: "tool", "dataset", or "workflow".
                    Select "workflow" for general workflow details and specific workflow invocation details.
        entity_id: Optional parameter. Provide this only when the user's query explicitly includes an ID,
                   allowing retrieval of information by that specific entity ID.

    Returns:
       DefaultTextResponses: A string containing the detailed Galaxy information and the response to the user's query. and a dict with action links of the information fetched.
        
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
        return DefaultTextResponses(response=f"Failed to connect to Galaxy: {e}")
    except Exception as e:
        logger.error(f"Error in get_galaxy_information_tool: {e}", exc_info=True)
        return DefaultTextResponses(response=f"An error occurred while fetching Galaxy information: {e}")


# ========================================================================================= #
    ## Tool 2: GraphRAG knowledge retrieval, returns context from the Galaxy knowledge graph ##
# ========================================================================================= #

@bioblend_app.tool()
async def graph_rag_query(
    query: str,
    query_type: Literal["local", "global", "complex"] = "local",
    compare_workflows_a: str = None,
    compare_workflows_b: str = None,
    connect_tools_a: str = None,
    connect_tools_b: str = None,
    category: str = None,
) -> DefaultTextResponses:
    """
    Retrieves context from the Galaxy knowledge graph using GraphRAG.

    Performs semantic search over tools and workflows, expands results through
    targeted Cypher queries on the Neo4j knowledge graph, and returns a
    structured context string that can be used to answer the user's query.

    Supports three query modes:
    - "local":   Standard semantic search + graph expansion for specific entities.
    - "global":  Ecosystem-wide analytics (most used tools, community clusters).
    - "complex": Multi-hop relationship queries (workflow comparisons, tool connections, category drill-downs).

    Args:
        query: The user's natural language question about Galaxy tools, workflows, or pipelines.
        query_type: One of "local", "global", or "complex". Use "complex" when the query
                    involves comparing workflows, finding tool relationships, or drilling into categories.
        compare_workflows_a: For workflow comparison queries, the name/keyword for the first workflow.
        compare_workflows_b: For workflow comparison queries, the name/keyword for the second workflow.
        connect_tools_a: For tool connection queries, the name of the first tool.
        connect_tools_b: For tool connection queries, the name of the second tool.
        category: For category drill-down queries, the category name to explore.

    Returns:
        DefaultTextResponses: The retrieved knowledge graph context as structured text.
    """
    logger.info(f"GraphRAG query: '{query}', type='{query_type}'")
    try:
        
        from app.bioblend_server.informer.manager import InformerManager
        from app.bioblend_server.informer.search.semantic_searcher import SemanticSearcher

        # 1. Neo4j connector
        connector = Neo4jGraphConnector(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
        )

        # 2. Semantic adapter
        manager = await InformerManager.create()

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
            graph_connector=connector,
            semantic_adapter=adapter,
            config={"log_level": "INFO"},
        )

        # 4. Build optional complex query params
        compare_workflows = None
        if compare_workflows_a and compare_workflows_b:
            compare_workflows = (compare_workflows_a, compare_workflows_b)

        connect_tools = None
        if connect_tools_a and connect_tools_b:
            connect_tools = (connect_tools_a, connect_tools_b)

        # 5. Execute
        result = await pipeline.retrieve_context(
            query=query,
            query_type=query_type,
            top_k=15,
            compare_workflows=compare_workflows,
            connect_tools=connect_tools,
            category=category,
        )

        context = result.get("context", "No context found.")
        connector.close()

        return DefaultTextResponses(response=context)

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
    Generates a detailed explanation of a Galaxy workflow invocation.

    This function retrieves and analyzes metadata for a given Galaxy workflow invocation.
    It either summarizes successful outputs or provides diagnostic details for failed jobs,
    and suggest fixes for workflow invocation.

    Args:
        invocation_id (str): 
            The unique identifier of the Galaxy workflow invocation to analyze.
        failure (bool): 
            Indicates whether to focus on failed job diagnostics (`True`) or 
            output dataset summaries (`False`), if empty it defaults to false.

    Returns:
        DefaultTextResponses: A clear report of the workflow invocation results or a report explaining failure causes with actionable suggestions.
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
    Imports a Galaxy workflow from the IWC workflow repository or the WorkflowHUB repository, fetching the workflow JSON,
    and uploading it to the Galaxy instance. Handles tool installation and ensures the workflow is added to the user's list.

    Args:
        workflow_name (str): The Full and exact name of the workflow to import.

    Returns:
        DefaultTextResponses: A message indicating the import status or an error description.
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