import pytest
import httpx
import logging

from unittest.mock import Mock, AsyncMock, patch
from qdrant_client.models import ScoredPoint
from sys import path

path.append(".")

from app.log_setup import configure_logging
from app.bioblend_server import server


configure_logging()

# Extract the actual functions from the decorated tools
get_galaxy_information_tool = server.get_galaxy_information_tool.fn
explain_galaxy_workflow_invocation = server.explain_galaxy_workflow_invocation.fn
import_workflow_to_galaxy_instance = server.import_workflow_to_galaxy_instance.fn
fetch_workflow_json_async = server.fetch_workflow_json_async


@pytest.fixture
def mock_current_api_key_server():
    with patch("app.bioblend_server.server.current_api_key_server") as mock:
        yield mock


@pytest.fixture
def mock_galaxy_client():
    with patch("app.bioblend_server.server.GalaxyClient") as mock:
        yield mock


@pytest.fixture
def mock_galaxy_informer():
    with patch("app.bioblend_server.server.GalaxyInformer") as mock:
        yield mock


@pytest.fixture
def mock_invocation_cache():
    with patch("app.bioblend_server.utils.invocation_cache") as mock:
        yield mock


@pytest.fixture
def mock_to_thread():
    with patch("app.bioblend_server.server.asyncio.to_thread") as mock:
        yield mock


@pytest.fixture
def mock_create_task():
    with patch("app.bioblend_server.server.asyncio.create_task") as mock:
        def _close_scheduled_coroutine(coro):
            if hasattr(coro, "close"):
                coro.close()
            return Mock(name="scheduled_task")

        mock.side_effect = _close_scheduled_coroutine
        yield mock


@pytest.fixture
def mock_httpx_async_client():
    with patch("httpx.AsyncClient") as mock:
        yield mock


@pytest.fixture
def mock_workflow_manager():
    with patch("app.bioblend_server.server.WorkflowManager") as mock:
        yield mock


@pytest.fixture
def mock_informer_manager():
    with patch("app.bioblend_server.informer.manager.InformerManager") as mock:
        yield mock


@pytest.fixture
def mock_fetch_workflow_json_async():
    with patch("app.bioblend_server.server.fetch_workflow_json_async") as mock:
        yield mock


@pytest.fixture
def informer_test_log():
    return logging.getLogger("TestGetGalaxyInformationTool")


@pytest.fixture
def inv_explainer_test_log():
    return logging.getLogger("TestExplainGalaxyWorkflowInvocationTool")


@pytest.fixture
def importer_test_log():
    return logging.getLogger("TestImportWorkflowToGalaxyInstanceTool")


class TestGetGalaxyInformationTool:
    """Tests for get_galaxy_information_tool function"""

    @pytest.mark.asyncio
    async def test_successful_tool_query(
        self,
        mock_current_api_key_server,
        mock_galaxy_client,
        mock_galaxy_informer,
        informer_test_log,
    ):
        """Test successful tool information retrieval"""

        informer_test_log.info("TEST: test_successful_tool_query starting")
        # Setup mocks
        mock_current_api_key_server.get.return_value = "test_api_key"
        mock_client_instance = Mock()
        mock_client_instance.whoami = "test_user"
        mock_galaxy_client.return_value = mock_client_instance

        # Create informer instance mock
        mock_informer_instance = AsyncMock()
        mock_informer_instance.get_entity_info = AsyncMock(
            return_value=("Tool information result", [])
        )

        # Make create an AsyncMock (not just set return_value)
        mock_galaxy_informer.create = AsyncMock(return_value=mock_informer_instance)

        # Execute
        result = await get_galaxy_information_tool(
            query="Show me alignment tools", query_type="tool", entity_id=None
        )

        # Assert - result is now an InformerResponse Pydantic model
        assert result.response == "Tool information result"
        assert result.actions == []
        mock_galaxy_informer.create.assert_called_once()
        mock_informer_instance.get_entity_info.assert_called_once_with(
            search_query="Show me alignment tools", entity_id=None
        )
        informer_test_log.info("TEST: test_successful_tool_query PASSED")

    @pytest.mark.asyncio
    async def test_workflow_query_with_entity_id(
        self,
        mock_current_api_key_server,
        mock_galaxy_client,
        mock_galaxy_informer,
        informer_test_log,
    ):
        """Test workflow query with specific entity ID"""

        informer_test_log.info("TEST: test_workflow_query_with_entity_id starting")

        mock_current_api_key_server.get.return_value = "test_api_key"
        mock_client_instance = Mock()
        mock_client_instance.whoami = "test_user"
        mock_galaxy_client.return_value = mock_client_instance

        mock_informer_instance = AsyncMock()
        mock_informer_instance.get_entity_info = AsyncMock(
            return_value=("Workflow details", [])
        )
        mock_galaxy_informer.create = AsyncMock(return_value=mock_informer_instance)

        result = await get_galaxy_information_tool(
            query="Get workflow details",
            query_type="workflow",
            entity_id="workflow_123",
        )

        # Assert - result is now an InformerResponse Pydantic model
        assert result.response == "Workflow details"
        assert result.actions == []
        mock_informer_instance.get_entity_info.assert_called_once_with(
            search_query="Get workflow details", entity_id="workflow_123"
        )
        informer_test_log.info("TEST: test_workflow_query_with_entity_id PASSED")

    @pytest.mark.asyncio
    async def test_missing_api_key(
        self, mock_current_api_key_server, informer_test_log, caplog
    ):
        """Test error handling when API key is missing"""

        informer_test_log.info("TEST: test_missing_api_key starting")
        mock_current_api_key_server.get.return_value = None
        caplog.set_level(logging.ERROR)

        result = await get_galaxy_information_tool(
            query="test query", query_type="tool"
        )

        assert result.response == "An error occurred while fetching Galaxy information."
        assert result.actions is None
        assert "current user api-key is missing" in caplog.text
        informer_test_log.info("TEST: test_missing_api_key PASSED")

    @pytest.mark.asyncio
    async def test_galaxy_client_exception(
        self,
        mock_current_api_key_server,
        mock_galaxy_client,
        informer_test_log,
        caplog,
    ):
        """Test exception handling during Galaxy client operations"""

        informer_test_log.info("TEST: test_galaxy_client_exception starting.")
        mock_current_api_key_server.get.return_value = "test_api_key"
        mock_galaxy_client.side_effect = Exception("Connection failed")
        caplog.set_level(logging.ERROR)

        result = await get_galaxy_information_tool(
            query="test query", query_type="tool"
        )

        assert result.response == "An error occurred while fetching Galaxy information."
        assert result.actions is None
        assert "Connection failed" in caplog.text
        informer_test_log.info("TEST: test_galaxy_client_exception PASSED.")


class TestExplainGalaxyWorkflowInvocationTool:
    """Tests for explain_galaxy_workflow_invocation function"""

    @pytest.mark.asyncio
    async def test_successful_invocation_explanation(
        self,
        mock_current_api_key_server,
        mock_galaxy_client,
        mock_invocation_cache,
        mock_to_thread,
        inv_explainer_test_log,
        caplog,
    ):
        """Test successful workflow invocation explanation"""

        inv_explainer_test_log.info(
            "TEST: test_successful_invocation_explanation starting."
        )
        mock_current_api_key_server.get.return_value = "test_api_key"
        mock_client_instance = Mock()
        mock_client_instance.whoami = "test_user"
        mock_client_instance.gi_client.url = (
            "http://test-galaxy"  # Fix for BioBlend URL join
        )
        caplog.set_level(logging.INFO)

        invocation_data = {
            "state": "ok",
            "inputs": {"0": {"label": "Input Dataset"}},
            "input_step_parameters": {
                "1": {"label": "param1", "parameter_value": "value1"}
            },
            "outputs": {"output1": "dataset1"},
            "output_collections": {},
        }
        report_data = {"title": "Test Workflow", "markdown": "reports"}

        mock_client_instance.gi_client.invocations.show_invocation.return_value = (
            invocation_data
        )
        mock_client_instance.gi_client.invocations.get_invocation_report.return_value = (
            report_data
        )
        mock_galaxy_client.return_value = mock_client_instance

        mock_invocation_cache.get_invocation_state = AsyncMock(return_value="Complete")

        # Mock asyncio.to_thread - return result directly, NOT as coroutine
        async def side_effect(func, *args, **kwargs):
            return func(*args, **kwargs)

        mock_to_thread.side_effect = side_effect

        # Patch GalaxyClient in utils module as well
        with patch("app.bioblend_server.utils.GalaxyClient", return_value=mock_client_instance):
            with patch(
                "app.bioblend_server.server.get_llm_response", new_callable=AsyncMock
            ) as mock_llm_response:

                mock_llm_response.return_value = "successful invocation report"
                result = await explain_galaxy_workflow_invocation(
                    invocation_id="inv_123", failure=False
                )

        # The log message is in utils.py analyze_invocation function, which is called
        # Check for the actual log messages from the server function
        assert f"Loading summarized report for successful invocation." in caplog.text
        assert result.response == "successful invocation report"
        inv_explainer_test_log.info(
            "TEST: test_successful_invocation_explanation PASSED."
        )

    @pytest.mark.asyncio
    async def test_failed_invocation_with_error_jobs(
        self,
        mock_current_api_key_server,
        mock_galaxy_client,
        mock_invocation_cache,
        mock_to_thread,
        inv_explainer_test_log,
        caplog,
    ):
        """Test explanation of failed invocation with error jobs"""

        inv_explainer_test_log.info(
            "TEST: test_failed_invocation_with_error_jobs starting."
        )
        mock_current_api_key_server.get.return_value = "test_api_key"
        mock_client_instance = Mock()
        mock_client_instance.whoami = "test_user"

        # Mock invocation details
        mock_client_instance.gi_client.invocations.show_invocation.return_value = {
            "state": "error",
            "inputs": {},
            "input_step_parameters": {},
            "outputs": {},
            "output_collections": {},
        }
        mock_client_instance.gi_client.invocations.get_invocation_report.return_value = {
            "title": "Failed Workflow",
            "markdown": "reports",
        }

        # Mock failed jobs
        mock_client_instance.gi_client.jobs.get_jobs.return_value = [
            {"id": "job_1", "state": "error"},
            {"id": "job_2", "state": "ok"},
        ]
        mock_client_instance.gi_client.jobs.show_job.return_value = {
            "tool_id": "failed_tool",
            "exit_code": 1,
            "stderr": "Tool execution failed",
            "stdout": "Processing data...",
        }

        mock_galaxy_client.return_value = mock_client_instance

        mock_invocation_cache.get_invocation_state = AsyncMock(return_value="Failed")

        # Mock asyncio.to_thread
        def mock_thread_sync(func):
            # Call the function and return its result directly (not wrapped in coroutine)
            return func()

        mock_to_thread.side_effect = lambda func: mock_thread_sync(func)
        
        # Patch GalaxyClient in utils module as well
        with patch("app.bioblend_server.utils.GalaxyClient", return_value=mock_client_instance):
            with patch(
                "app.bioblend_server.server.get_llm_response", new_callable=AsyncMock
            ) as mock_llm_response:

                mock_llm_response.return_value = "Failed invocation report"
                result = await explain_galaxy_workflow_invocation(
                    invocation_id="inv_456", failure=True
                )

        # The log message is in utils.py analyze_invocation function
        # Check for the actual log messages from the server function
        assert (
            f"Loading failure explanation and suggestions for invocation."
            in caplog.text
        )
        assert result.response == "Failed invocation report"

        inv_explainer_test_log.info(
            "TEST: test_failed_invocation_with_error_jobs PASSED."
        )

    @pytest.mark.asyncio
    async def test_scheduled_invocation(
        self,
        mock_current_api_key_server,
        mock_galaxy_client,
        mock_invocation_cache,
        mock_to_thread,
        mock_create_task,
        inv_explainer_test_log,
        caplog,
    ):
        """Test handling of scheduled (pending) invocation"""
        inv_explainer_test_log.info("TEST: test_scheduled_invocation starting.")

        mock_current_api_key_server.get.return_value = "test_api_key"
        mock_client_instance = Mock()
        mock_client_instance.whoami = "test_user"

        mock_client_instance.gi_client.invocations.show_invocation.return_value = {
            "state": "scheduled",
            "inputs": {},
            "input_step_parameters": {},
            "outputs": {},
            "output_collections": {},
        }
        mock_client_instance.gi_client.invocations.get_invocation_report.return_value = {
            "title": "Scheduled Workflow",
            "markdown": "reports",
        }

        mock_galaxy_client.return_value = mock_client_instance

        mock_invocation_cache.get_invocation_state = AsyncMock(return_value=None)

        def mock_thread_sync(func):
            # Call the function and return its result directly (not wrapped in coroutine)
            return func()

        mock_to_thread.side_effect = lambda func: mock_thread_sync(func)

        # Patch GalaxyClient in utils module as well
        with patch("app.bioblend_server.utils.GalaxyClient", return_value=mock_client_instance):
            with patch(
                "app.bioblend_server.server.get_llm_response", new_callable=AsyncMock
            ) as mock_llm_response:

                mock_llm_response.return_value = "Pending invocation report"
                result = await explain_galaxy_workflow_invocation(
                    invocation_id="inv_789", failure=False
                )

        # The log message is in utils.py analyze_invocation function
        # Check for the actual log messages from the server function
        assert f"Loading summarized report for successful invocation." in caplog.text
        assert result.response == "Pending invocation report"
        mock_create_task.assert_called_once()
        inv_explainer_test_log.info("TEST: test_scheduled_invocation PASSED.")

    @pytest.mark.asyncio
    async def test_missing_api_key_invocation(
        self, mock_current_api_key_server, inv_explainer_test_log
    ):
        """Test error when API key is missing for invocation"""

        inv_explainer_test_log.info("TEST: test_missing_api_key_invocation starting.")

        mock_current_api_key_server.get.return_value = None

        with pytest.raises(ValueError, match="current user api-key is missing"):
            await explain_galaxy_workflow_invocation(
                invocation_id="inv_test", failure=False
            )
        inv_explainer_test_log.info("TEST: test_missing_api_key_invocation PASSED.")


class TestImportWorkflowToGalaxyInstanceTool:
    """Tests for import_workflow_to_galaxy_instance function"""

    @pytest.mark.asyncio
    async def test_successful_workflow_import(
        self,
        mock_current_api_key_server,
        mock_galaxy_client,
        mock_workflow_manager,
        mock_informer_manager,
        mock_fetch_workflow_json_async,
        mock_create_task,
        importer_test_log,
    ):
        """Test successful workflow import"""

        importer_test_log.info("TEST: test_successful_workflow_import starting.")
        mock_current_api_key_server.get.return_value = "test_api_key"
        mock_galaxy_client.return_value = Mock()
        mock_workflow_manager.return_value = Mock()

        mock_scored_point = ScoredPoint(
            id=1,
            score=0.95,
            version=1,
            payload={"raw_download_url": "https://example.com/workflow.ga"},
        )

        # Properly mock the InformerManager chain
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.match_name_from_collection = AsyncMock(
            return_value=[[mock_scored_point]]
        )

        mock_manager = Mock()
        mock_manager.create = AsyncMock(return_value=mock_qdrant_instance)
        mock_informer_manager.return_value = mock_manager

        mock_fetch_workflow_json_async.return_value = {
            "workflow_name": "Imported Workflow",
            "steps": {},
        }
        result = await import_workflow_to_galaxy_instance(workflow_name="Test Workflow")

        assert "workflow is being imported" in result.response.lower()
        importer_test_log.info("TEST: test_successful_workflow_import PASSED.")

    @pytest.mark.asyncio
    async def test_workflow_not_found_in_collection(
        self,
        mock_current_api_key_server,
        mock_galaxy_client,
        mock_workflow_manager,
        mock_informer_manager,
        importer_test_log,
    ):
        """Test workflow not found in Qdrant collection"""
        importer_test_log.info("TEST: test_workflow_not_found_in_collection starting.")

        mock_current_api_key_server.get.return_value = "test_api_key"
        mock_galaxy_client.return_value = Mock()
        mock_workflow_manager.return_value = Mock()

        # Mock empty search results
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.match_name_from_collection = AsyncMock(return_value=[])

        mock_informer_instance = Mock()
        mock_informer_instance.create = AsyncMock(return_value=mock_qdrant_instance)
        mock_informer_manager.return_value = mock_informer_instance

        result = await import_workflow_to_galaxy_instance(
            workflow_name="Nonexistent Workflow"
        )

        assert "not found in available workflow collection" in result.response.lower()
        importer_test_log.info("TEST: test_workflow_not_found_in_collection PASSED.")

    @pytest.mark.asyncio
    async def test_http_error_during_fetch(
        self,
        mock_current_api_key_server,
        mock_galaxy_client,
        mock_workflow_manager,
        mock_informer_manager,
        mock_fetch_workflow_json_async,
        importer_test_log,
    ):
        """Test HTTP error during workflow JSON fetch"""

        importer_test_log.info("TEST: test_http_error_during_fetch starting.")
        mock_current_api_key_server.get.return_value = "test_api_key"
        mock_galaxy_client.return_value = Mock()
        mock_workflow_manager.return_value = Mock()

        mock_scored_point = ScoredPoint(
            id=1,
            score=0.95,
            version=1,
            payload={"raw_download_url": "https://example.com/workflow.ga"},
        )

        mock_qdrant_instance = Mock()
        mock_qdrant_instance.match_name_from_collection = AsyncMock(
            return_value=[[mock_scored_point]]
        )

        mock_manager = Mock()
        mock_manager.create = AsyncMock(return_value=mock_qdrant_instance)
        mock_informer_manager.return_value = mock_manager

        # Simulate HTTP error
        mock_fetch_workflow_json_async.side_effect = httpx.HTTPStatusError(
            "500 Server Error", request=Mock(), response=Mock()
        )

        result = await import_workflow_to_galaxy_instance(workflow_name="Test Workflow")

        assert "http error" in result.response.lower()
        importer_test_log.info("TEST: test_http_error_during_fetch PASSED.")
