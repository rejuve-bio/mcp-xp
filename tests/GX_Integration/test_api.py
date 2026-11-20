import pytest
import asyncio
import logging
import jwt


from fastapi.testclient import TestClient as FastAPITestClient
from unittest.mock import patch, AsyncMock, MagicMock
from contextlib import asynccontextmanager
import httpx
from cryptography.fernet import Fernet
import json

from sys import path

path.append(".")

import os
import time

os.environ["SECRET_KEY"] = (
    "test_secret_key_must_be_long_enough_for_fernet_key_generation_or_mocked"
)
# Actually Fernet key needs to be valid base64 32 bytes.
# But the code might just check existence.
# Let's check app/orchestration/invocation_tasks.py to see what it expects.
# If it uses Fernet(key), it needs a valid key.
# I'll use a valid key just in case.
from cryptography.fernet import Fernet

os.environ["SECRET_KEY"] = Fernet.generate_key().decode()

import app.main
from app.api.endpoints.invocation import invocation_service

from app.log_setup import configure_logging
from app.context import current_api_key
from app.exceptions import InternalServerErrorException

# Configure test logging
configure_logging()
logger = logging.getLogger("Test APIs")


@pytest.fixture
def secret_key():
    """Generate a Fernet key for testing."""
    logger.debug("Creating secret key")
    return Fernet.generate_key().decode()


@pytest.fixture
def mock_fernet(secret_key):
    """Create a Fernet instance with the test secret key."""
    logger.debug("Creating mock_fernet")
    return Fernet(secret_key.encode())


@pytest.fixture
def mock_env(monkeypatch, secret_key):
    """Mock environment variables."""
    logger.info("Setting up mock_env")
    monkeypatch.setenv("SECRET_KEY", secret_key)
    monkeypatch.setenv("GALAXY_API_KEY", "test-admin-key")
    monkeypatch.setenv("GALAXY_URL", "http://test-galaxy")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-0123456789abcdef0123456789")
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("REDIS_PORT", "6379")
    logger.info("mock_env setup complete")


@pytest.fixture
def mock_import_workflows():
    """Mock import_published_workflows before app import"""
    logger.debug("Setting up mock_import_workflows")
    with patch("app.main.import_published_workflows") as mock:
        logger.debug("mock_import_workflows patched")
        yield mock
    logger.debug("mock_import_workflows teardown")


@pytest.fixture
def mock_galaxy_client():
    """Mock GalaxyClient and its dependencies."""
    with patch("app.api.endpoints.invocation.GalaxyClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.whoami = "test_user"
        mock_gi = MagicMock()
        mock_invocations = MagicMock()
        mock_gi.invocations = mock_invocations
        mock_client.gi_object.gi = mock_gi
        mock_client_class.return_value = mock_client
        yield mock_client, mock_invocations


@pytest.fixture
def mock_workflow_manager():
    """Mock WorkflowManager."""
    with patch("app.api.endpoints.invocation.WorkflowManager") as mock_wm_class:
        mock_wm = MagicMock()
        mock_wm_class.return_value = mock_wm
        yield mock_wm


@pytest.fixture
def mock_current_api_key():
    """Set current_api_key context value."""
    token = current_api_key.set("test_user_key")
    try:
        yield
    finally:
        current_api_key.reset(token)


@pytest.fixture
def auth_headers(mock_fernet):
    """Generate mock auth headers with JWT containing encrypted token claim."""

    inner_payload = json.dumps({"apikey": "test_user_key"})
    encrypted = mock_fernet.encrypt(inner_payload.encode()).decode()
    jwt_payload = {"galaxy_api_token": encrypted, "exp": int(time.time()) + 300}
    token = jwt.encode(jwt_payload, os.environ["JWT_SECRET"], algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client(mock_env, mock_import_workflows, mock_galaxy_client, mock_workflow_manager):
    logger.info("Starting client fixture")

    with patch("redis.Redis") as mock_redis_class:
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping = lambda: True
        mock_redis_instance.close.return_value = None
        mock_redis_instance.get = MagicMock(return_value=0)
        mock_redis_instance.incr = MagicMock(side_effect=lambda key: 1)
        mock_redis_instance.zadd = MagicMock(return_value=1)
        mock_redis_instance.ttl = MagicMock(return_value=60)
        mock_redis_instance.exists = MagicMock(return_value=True)
        mock_redis_instance.zremrangebyscore = MagicMock(return_value=0)
        mock_redis_instance.zcount = MagicMock(return_value=1)
        mock_redis_instance.expire = MagicMock(return_value=True)
        mock_pipeline = MagicMock()
        mock_pipeline.incr = MagicMock(return_value=1)
        mock_pipeline.expire = MagicMock(return_value=True)
        mock_pipeline.execute = MagicMock(return_value=[1, True, 1, True, 1, True])
        mock_redis_instance.pipeline = MagicMock(return_value=mock_pipeline)
        mock_redis_class.return_value = mock_redis_instance
        logger.info("Redis mock configured")

        with patch("app.api.middleware.RateLimiterMiddleware", lambda app: app):
            logger.info("RateLimiterMiddleware disabled for tests")

            # Mock lifespan to do nothing
            @asynccontextmanager
            async def mock_lifespan(app):
                logger.info("Mock lifespan called")
                yield
                logger.info("Mock lifespan shutdown")

            # Mock wait_shutdown to handle CancelledError safely (no-op)
            async def mock_wait_shutdown(self):
                logger.debug("Mock wait_shutdown called (noop)")
                try:
                    await asyncio.sleep(0)
                except asyncio.CancelledError:
                    logger.debug("CancelledError caught in mock_wait_shutdown")
                logger.debug("Mock wait_shutdown completed")

            try:
                from app.main import app

                logger.info("app.main imported successfully!")
            except Exception as e:
                logger.error(f"ERROR importing app.main: {e}")
                import traceback

                traceback.print_exc()
                raise

            # Patch the app.lifespan attribute directly
            logger.debug("Patching app.lifespan")
            app.lifespan = mock_lifespan
            logger.info("app.lifespan patched")

            # Patch wait_shutdown
            logger.debug("Patching starlette.testclient.TestClient.wait_shutdown")
            with patch(
                "starlette.testclient.TestClient.wait_shutdown", mock_wait_shutdown
            ):
                logger.debug("wait_shutdown patched")

                with patch(
                    "app.orchestration.invocation_cache.InvocationCache"
                ) as mock_cache_class:
                    logger.debug("InvocationCache patched")
                    with patch(
                        "app.orchestration.invocation_tasks.InvocationBackgroundTasks"
                    ) as mock_bg_tasks_class:
                        logger.debug("InvocationBackgroundTasks patched")
                        mock_cache = MagicMock()
                        mock_bg_tasks = MagicMock()
                        mock_cache_class.return_value = mock_cache
                        mock_bg_tasks_class.return_value = mock_bg_tasks
                        logger.info("Cache and background tasks mocked")

                        logger.debug("Creating FastAPITestClient")
                        try:
                            with FastAPITestClient(app) as test_client:
                                logger.info("TestClient created successfully!")
                                test_client.mock_import_workflows = (
                                    mock_import_workflows
                                )
                                test_client.mock_redis = mock_redis_instance
                                logger.debug("Yielding test_client")
                                yield test_client
                                logger.debug("Test completed, tearing down")
                        except Exception as e:
                            logger.error(f"ERROR creating TestClient: {e}")
                            import traceback

                            traceback.print_exc()
                            raise
    logger.info("client fixture teardown complete")


# /register-user, /refresh-galaxy-key and /v1/users/me/galaxy-api-key are
# covered against a fake Galaxy in test_auth.py.


class TestInvocationList:
    """Unit tests for the /api/invocation/ endpoint."""

    def test_list_invocations_cache_hit(
        self, client, mock_current_api_key, mock_fernet, auth_headers, caplog
    ):
        """Test response served from cache hit."""
        logger.info("TEST: test_list_invocations_cache_hit starting")
        caplog.set_level(logging.INFO)

        # Mock cache to return a cached response
        mock_cache = AsyncMock()
        mock_cache.get_deleted_invocations = AsyncMock(return_value=[])
        mock_cache.is_duplicate_request = AsyncMock(return_value=False)
        mock_cache.get_response_cache = AsyncMock(
            return_value={
                "invocations": [
                    {
                        "id": "inv1",
                        "workflow_name": "wf1",
                        "workflow_id": "wf_id1",
                        "history_id": "hist1",
                        "state": "Complete",
                        "outputs": [],
                        "create_time": "2023-01-01T00:00:00",
                        "update_time": "2023-01-01T01:00:00",
                    }
                ]
            }
        )
        mock_mongo = AsyncMock()
        mock_mongo.get = AsyncMock(return_value=None)
        with patch.object(invocation_service, "cache", mock_cache), \
             patch.object(invocation_service, "mongo_client", mock_mongo):
            response = client.get("/api/invocation/", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data["invocations"]) == 1
        assert data["invocations"][0]["id"] == "inv1"
        assert (
            "serving response from cache" in caplog.text
        )  # Note: adjusted log message to match code
        logger.info("TEST: test_list_invocations_cache_hit PASSED")

    def test_list_invocations_cache_miss_success(
        self,
        client,
        mock_galaxy_client,
        mock_workflow_manager,
        mock_current_api_key,
        mock_fernet,
        auth_headers,
        caplog,
    ):
        """Test cache miss, successful fetch and formatting."""
        logger.info("TEST: test_list_invocations_cache_miss_success starting")
        caplog.set_level(logging.INFO)

        mock_client, mock_invocations = mock_galaxy_client
        mock_invocations.get_invocations.return_value = [
            {
                "id": "inv1",
                "workflow_id": "wf_id1",
                "history_id": "hist1",
                "state": "new",
                "create_time": "2023-01-01T00:00:00",
                "update_time": "2023-01-01T01:00:00",
            }
        ]
        
        # Mock workflow_manager.gi_object.gi.invocations.get_invocations
        mock_workflow_manager.gi_object.gi.invocations.get_invocations.return_value = mock_invocations.get_invocations.return_value

        # Mock step jobs summary to ensure "Pending" state (running jobs)
        mock_workflow_manager.gi_object.gi.invocations.get_invocation_step_jobs_summary.return_value = [
            {"states": {"running": 1}}
        ]

        # Mock background tasks
        mock_bg_tasks = AsyncMock()
        mock_bg_tasks.fetch_workflows_safely = AsyncMock(
            return_value=[{"id": "wf_id1", "name": "wf1"}]
        )
        mock_bg_tasks.build_invocation_workflow_mapping = AsyncMock(
            return_value=(
                {"inv1": {"workflow_name": "wf1", "workflow_id": "wf_id1"}},
                mock_invocations.get_invocations.return_value,
            )
        )
        with (
            patch.object(invocation_service, "background_tasks", mock_bg_tasks),
            patch.object(
                invocation_service.inv_data_manager, "background_tasks", mock_bg_tasks
            ),
        ):

            # Mock cache misses
            mock_cache = AsyncMock()
            mock_cache.get_deleted_invocations = AsyncMock(return_value=[])
            mock_cache.is_duplicate_request = AsyncMock(return_value=False)
            mock_cache.get_response_cache = AsyncMock(return_value=None)
            mock_cache.get_invocations_cache = AsyncMock(return_value=None)
            mock_cache.get_workflows_cache = AsyncMock(return_value=None)
            mock_cache.get_invocation_workflow_mapping = AsyncMock(return_value=None)
            mock_cache.get_invocation_state = AsyncMock(
                return_value=None
            )  # Force raw state mapping
            mock_cache.set_invocations_cache = AsyncMock()
            mock_cache.set_workflows_cache = AsyncMock()
            mock_cache.set_invocation_workflow_mapping = AsyncMock()
            mock_cache.set_response_cache = AsyncMock()
            mock_cache.set_invocation_state = AsyncMock()
            # Mock acquire_lock as an async context manager
            from contextlib import asynccontextmanager
            @asynccontextmanager
            async def mock_acquire_lock(*args, **kwargs):
                yield
            mock_cache.acquire_lock = mock_acquire_lock
            mock_mongo = AsyncMock()
            mock_mongo.get = AsyncMock(return_value=None)
            mock_mongo.get_element = AsyncMock(return_value=None)
            mock_mongo.exists = AsyncMock(return_value=False)
            mock_mongo.set = AsyncMock()
            mock_mongo.set_element = AsyncMock()
            mock_mongo.add_to_set = AsyncMock()
            with (
                patch.object(invocation_service, "cache", mock_cache),
                patch.object(invocation_service.inv_data_manager, "cache", mock_cache),
                patch.object(invocation_service.inv_tracker, "cache", mock_cache),
                patch.object(invocation_service.background_tasks, "cache", mock_cache),
                patch.object(invocation_service, "mongo_client", mock_mongo),
            ):
                with patch(
                    "fastapi.concurrency.run_in_threadpool", new_callable=AsyncMock
                ) as mock_run_thread:
                    mock_run_thread.side_effect = lambda func, *args, **kwargs: func(
                        *args, **kwargs
                    )
                    response = client.get("/api/invocation/", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data["invocations"]) == 1
        assert data["invocations"][0]["id"] == "inv1"
        assert data["invocations"][0]["state"] == "Pending"
        mock_cache.set_response_cache.assert_called()
        assert "Successfully retrieved invocations (total: 1)" in caplog.text
        logger.info("TEST: test_list_invocations_cache_miss_success PASSED")

    def test_list_invocations_with_filters(
        self,
        client,
        mock_galaxy_client,
        mock_workflow_manager,
        mock_current_api_key,
        mock_fernet,
        auth_headers,
    ):
        """Test with workflow_id and history_id filters."""
        logger.info("TEST: test_list_invocations_with_filters starting")

        mock_client, mock_invocations = mock_galaxy_client
        mock_invocations.get_invocations.return_value = [{"id": "inv1", "state": "new"}]

        mock_bg_tasks = AsyncMock()
        mock_bg_tasks.fetch_workflows_safely = AsyncMock(return_value=[])
        mock_bg_tasks.build_invocation_workflow_mapping = AsyncMock(
            return_value=({}, mock_invocations.get_invocations.return_value)
        )
        with patch.object(invocation_service, "background_tasks", mock_bg_tasks):

            mock_cache = AsyncMock()
            mock_cache.get_deleted_invocations = AsyncMock(return_value=[])
            mock_cache.is_duplicate_request = AsyncMock(return_value=False)
            mock_cache.get_response_cache = AsyncMock(return_value=None)
            mock_cache.get_invocations_cache = AsyncMock(return_value=None)
            mock_cache.get_workflows_cache = AsyncMock(return_value=None)
            mock_cache.get_invocation_state = AsyncMock(return_value="Pending")
            # Mock acquire_lock as an async context manager
            from contextlib import asynccontextmanager
            @asynccontextmanager
            async def mock_acquire_lock(*args, **kwargs):
                yield
            mock_cache.acquire_lock = mock_acquire_lock
            mock_mongo = AsyncMock()
            mock_mongo.get = AsyncMock(return_value=None)
            mock_mongo.exists = AsyncMock(return_value=False)
            with patch.object(invocation_service, "cache", mock_cache), \
                 patch.object(invocation_service, "mongo_client", mock_mongo):
                with patch(
                    "fastapi.concurrency.run_in_threadpool", new_callable=AsyncMock
                ) as mock_run_thread:
                    mock_run_thread.side_effect = lambda func, *args, **kwargs: func(
                        *args, **kwargs
                    )
                    mock_workflow_manager.gi_object = mock_client.gi_object
                    response = client.get(
                        "/api/invocation/",
                        params={"workflow_id": "wf_id1", "history_id": "hist1"},
                        headers=auth_headers,
                    )

        assert response.status_code == 200
        data = response.json()
        assert len(data["invocations"]) == 0
        mock_invocations.get_invocations.assert_called_with(
            workflow_id="wf_id1", history_id="hist1", limit=100
        )
        logger.info("TEST: test_list_invocations_with_filters PASSED")

    def test_list_invocations_no_data(
        self,
        client,
        mock_galaxy_client,
        mock_workflow_manager,
        mock_current_api_key,
        mock_fernet,
        auth_headers,
        caplog,
    ):
        """Test when no invocations are retrieved."""
        logger.info("TEST: test_list_invocations_no_data starting")

        mock_client, mock_invocations = mock_galaxy_client
        mock_invocations.get_invocations.return_value = []
        mock_workflow_manager.gi_object.gi.invocations.get_invocations.return_value = []

        mock_bg_tasks = AsyncMock()
        mock_bg_tasks.fetch_workflows_safely = AsyncMock(return_value=[])
        mock_bg_tasks.build_invocation_workflow_mapping = AsyncMock(
            return_value=({}, [])
        )
        with patch.object(invocation_service, "background_tasks", mock_bg_tasks):

            mock_cache = AsyncMock()
            mock_cache.get_deleted_invocations = AsyncMock(return_value=[])
            mock_cache.is_duplicate_request = AsyncMock(return_value=False)
            mock_cache.get_response_cache = AsyncMock(return_value=None)
            mock_cache.get_invocations_cache = AsyncMock(return_value=None)
            mock_cache.get_workflows_cache = AsyncMock(return_value=None)
            mock_cache.get_invocation_workflow_mapping = AsyncMock(return_value=None)
            mock_cache.set_invocations_cache = AsyncMock()
            mock_cache.set_workflows_cache = AsyncMock()
            mock_cache.set_invocation_workflow_mapping = AsyncMock()
            mock_cache.set_response_cache = AsyncMock()
            # Mock acquire_lock as an async context manager
            from contextlib import asynccontextmanager
            @asynccontextmanager
            async def mock_acquire_lock(*args, **kwargs):
                yield
            mock_cache.acquire_lock = mock_acquire_lock
            mock_mongo = AsyncMock()
            mock_mongo.get = AsyncMock(return_value=None)
            mock_mongo.get_element = AsyncMock(return_value=None)
            mock_mongo.exists = AsyncMock(return_value=False)
            mock_mongo.set_element = AsyncMock()
            with patch.object(invocation_service, "cache", mock_cache), \
                 patch.object(invocation_service, "mongo_client", mock_mongo):
                with patch(
                    "fastapi.concurrency.run_in_threadpool", new_callable=AsyncMock
                ) as mock_run_thread:
                    mock_run_thread.side_effect = lambda func, *args, **kwargs: func(
                        *args, **kwargs
                    )
                    logger.info("fecthing response")
                    response = client.get("/api/invocation/", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        # When no invocations are retrieved, the service returns empty list
        assert len(data["invocations"]) == 0
        assert "No invocations data retrieved" in caplog.text
        logger.info("TEST: test_list_invocations_no_data PASSED")

    def test_list_invocations_deleted_filtered(
        self,
        client,
        mock_galaxy_client,
        mock_workflow_manager,
        mock_current_api_key,
        mock_fernet,
        auth_headers,
    ):
        """Test filtering out deleted invocations."""
        logger.info("TEST: test_list_invocations_deleted_filtered starting")

        mock_client, mock_invocations = mock_galaxy_client
        mock_invocations.get_invocations.return_value = [
            {
                "id": "inv1",
                "workflow_id": "wf_id_1",
                "workflow_name": "wf1",
                "history_id": "hist1",
                "state": "new",
                "create_time": "2025-11-05T00:00:00Z",
                "update_time": "2025-11-05T00:00:00Z",
            },
            {
                "id": "inv2",
                "workflow_id": "wf_id_2",
                "workflow_name": "wf2",
                "history_id": "hist2",
                "state": "scheduled",
                "create_time": "2025-11-05T00:00:00Z",
                "update_time": "2025-11-05T00:00:00Z",
            },
        ]
        
        # Mock workflow_manager.gi_object.gi.invocations.get_invocations
        mock_workflow_manager.gi_object.gi.invocations.get_invocations.return_value = mock_invocations.get_invocations.return_value

        mock_bg_tasks = AsyncMock()
        mock_bg_tasks.fetch_workflows_safely = AsyncMock(
            return_value=[
                {"id": "wf_id1", "name": "wf1"},
                {"id": "wf_id2", "name": "wf2"},
            ]
        )
        mock_bg_tasks.build_invocation_workflow_mapping = AsyncMock(
            return_value=(
                {
                    "inv1": {"workflow_name": "wf1", "workflow_id": "wf_id1"},
                    "inv2": {"workflow_name": "wf2", "workflow_id": "wf_id2"},
                },
                mock_invocations.get_invocations.return_value,
            )
        )
        with patch.object(invocation_service, "background_tasks", mock_bg_tasks):

            mock_cache = AsyncMock()
            mock_cache.get_deleted_invocations = AsyncMock(return_value=["inv2"])
            mock_cache.is_duplicate_request = AsyncMock(return_value=False)
            mock_cache.get_response_cache = AsyncMock(return_value=None)
            mock_cache.get_invocations_cache = AsyncMock(return_value=None)
            mock_cache.get_workflows_cache = AsyncMock(return_value=None)
            mock_cache.get_invocation_workflow_mapping = AsyncMock(return_value=None)
            mock_cache.get_invocation_state = AsyncMock(
                side_effect=["Failed", "Complete"]
            )
            mock_cache.set_invocations_cache = AsyncMock()
            mock_cache.set_workflows_cache = AsyncMock()
            mock_cache.set_invocation_workflow_mapping = AsyncMock()
            mock_cache.set_response_cache = AsyncMock()
            # Mock acquire_lock as an async context manager
            from contextlib import asynccontextmanager
            @asynccontextmanager
            async def mock_acquire_lock(*args, **kwargs):
                yield
            mock_cache.acquire_lock = mock_acquire_lock
            mock_mongo = AsyncMock()
            mock_mongo.get = AsyncMock(return_value=None)
            mock_mongo.get_element = AsyncMock(return_value=None)
            mock_mongo.exists = AsyncMock(return_value=False)
            mock_mongo.set = AsyncMock()
            mock_mongo.set_element = AsyncMock()
            mock_mongo.add_to_set = AsyncMock()
            with (
                patch.object(invocation_service, "cache", mock_cache),
                patch.object(invocation_service.inv_data_manager, "cache", mock_cache),
                patch.object(invocation_service.inv_tracker, "cache", mock_cache),
                patch.object(invocation_service.background_tasks, "cache", mock_cache),
                patch.object(invocation_service, "mongo_client", mock_mongo),
            ):
                with patch(
                    "fastapi.concurrency.run_in_threadpool", new_callable=AsyncMock
                ) as mock_run_thread:
                    mock_run_thread.side_effect = lambda func, *args, **kwargs: func(
                        *args, **kwargs
                    )
                    response = client.get("/api/invocation/", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data["invocations"]) == 1
        assert data["invocations"][0]["id"] == "inv1"
        logger.info("TEST: test_list_invocations_deleted_filtered PASSED")

    def test_list_invocations_partial_failure(
        self,
        client,
        mock_galaxy_client,
        mock_workflow_manager,
        mock_current_api_key,
        mock_fernet,
        auth_headers,
        caplog,
    ):
        """Test partial failure handling (e.g., workflows fetch fails)."""
        logger.info("TEST: test_list_invocations_partial_failure starting")

        mock_client, mock_invocations = mock_galaxy_client
        mock_invocations.get_invocations.return_value = [
            {
                "id": "inv1",
                "workflow_id": "wf_id_1",
                "workflow_name": "wf1",
                "history_id": "hist1",
                "state": "new",
                "create_time": "2025-11-05T00:00:00Z",
                "update_time": "2025-11-05T00:00:00Z",
            }
        ]
        
        # Mock workflow_manager.gi_object.gi.invocations.get_invocations for partial failure path
        mock_workflow_manager.gi_object.gi.invocations.get_invocations.return_value = mock_invocations.get_invocations.return_value
        
        # Mock step jobs summary to ensure "Pending" state (running jobs)
        mock_workflow_manager.gi_object.gi.invocations.get_invocation_step_jobs_summary.return_value = [
            {"states": {"running": 1}}
        ]

        # Mock background to raise on workflow mapping build
        mock_bg_tasks = AsyncMock()
        mock_bg_tasks.fetch_workflows_safely = AsyncMock(
            return_value=[]
        )
        mock_bg_tasks.build_invocation_workflow_mapping = AsyncMock(
            side_effect=Exception("Workflow mapping failed")
        )
        with (
            patch.object(invocation_service, "background_tasks", mock_bg_tasks),
            patch.object(
                invocation_service.inv_data_manager, "background_tasks", mock_bg_tasks
            ),
        ):

            mock_cache = AsyncMock()
            mock_cache.get_deleted_invocations = AsyncMock(return_value=[])
            mock_cache.is_duplicate_request = AsyncMock(return_value=False)
            mock_cache.get_response_cache = AsyncMock(return_value=None)
            mock_cache.get_invocations_cache = AsyncMock(return_value=None)
            mock_cache.get_workflows_cache = AsyncMock(return_value=None)
            mock_cache.get_invocation_workflow_mapping = AsyncMock(return_value=None)
            mock_cache.get_invocation_state = AsyncMock(return_value=None)
            mock_cache.set_invocations_cache = AsyncMock()
            mock_cache.set_workflows_cache = AsyncMock()
            mock_cache.set_invocation_workflow_mapping = AsyncMock()
            mock_cache.set_response_cache = AsyncMock()
            mock_cache.set_invocation_state = AsyncMock()
            # Mock acquire_lock as an async context manager
            from contextlib import asynccontextmanager
            @asynccontextmanager
            async def mock_acquire_lock(*args, **kwargs):
                yield
            mock_cache.acquire_lock = mock_acquire_lock
            mock_mongo = AsyncMock()
            mock_mongo.get = AsyncMock(return_value=None)
            mock_mongo.get_element = AsyncMock(return_value=None)
            mock_mongo.exists = AsyncMock(return_value=False)
            mock_mongo.set_element = AsyncMock()
            with (
                patch.object(invocation_service, "cache", mock_cache),
                patch.object(invocation_service.inv_data_manager, "cache", mock_cache),
                patch.object(invocation_service.inv_tracker, "cache", mock_cache),
                patch.object(invocation_service.background_tasks, "cache", mock_cache),
                patch.object(invocation_service, "mongo_client", mock_mongo),
            ):
                with patch(
                    "fastapi.concurrency.run_in_threadpool", new_callable=AsyncMock
                ) as mock_run_thread:
                    mock_run_thread.side_effect = lambda func, *args, **kwargs: func(
                        *args, **kwargs
                    )
                    response = client.get("/api/invocation/", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        # When workflow fetch fails, the service uses partial failure handling
        # which returns invocations with "Unknown (partial failure)" workflow names
        assert len(data["invocations"]) == 1
        assert data["invocations"][0]["workflow_name"] == "Unknown (partial failure)"
        assert "Workflow mapping failed" in caplog.text
        logger.info("TEST: test_list_invocations_partial_failure PASSED")

    def test_list_invocations_full_failure(
        self,
        client,
        mock_galaxy_client,
        mock_current_api_key,
        mock_fernet,
        auth_headers,
        caplog,
    ):
        """Test full failure with 500 response."""
        logger.info("TEST: test_list_invocations_full_failure starting")

        mock_cache = AsyncMock()
        mock_cache.get_deleted_invocations = AsyncMock(return_value=[])
        mock_cache.is_duplicate_request = AsyncMock(return_value=False)
        mock_cache.get_response_cache = AsyncMock(return_value=None)
        mock_mongo = AsyncMock()
        mock_mongo.get = AsyncMock(return_value=None)
        with (
            patch.object(invocation_service, "cache", mock_cache),
            patch.object(invocation_service.inv_data_manager, "cache", mock_cache),
            patch.object(invocation_service.inv_tracker, "cache", mock_cache),
            patch.object(invocation_service.background_tasks, "cache", mock_cache),
            patch.object(invocation_service, "mongo_client", mock_mongo),
        ):
            with patch.object(
                invocation_service,
                "list_invocations",
                side_effect=Exception("Galaxy API failed"),
            ):
                with patch(
                    "fastapi.concurrency.run_in_threadpool", new_callable=AsyncMock
                ) as mock_run_thread:
                    mock_run_thread.side_effect = lambda func, *args, **kwargs: func(
                        *args, **kwargs
                    )
                    response = client.get("/api/invocation/", headers=auth_headers)

        assert response.status_code == 500
        assert "Failed to list invocations" in response.json()["detail"]
        assert "Failed to list invocations" in caplog.text
        logger.info("TEST: test_list_invocations_full_failure PASSED")

    def test_list_invocations_duplicate_request(
        self, client, mock_current_api_key, mock_fernet, auth_headers, caplog
    ):
        """Test duplicate request detection."""
        logger.info("TEST: test_list_invocations_duplicate_request starting")
        caplog.set_level(logging.INFO)

        mock_cache = AsyncMock()
        mock_cache.get_deleted_invocations = AsyncMock(return_value=[])
        mock_cache.is_duplicate_request = AsyncMock(return_value=True)
        mock_cache.get_response_cache = AsyncMock(
            return_value={
                "invocations": [
                    {
                        "id": "inv1",
                        "workflow_id": "wf_id_1",
                        "workflow_name": "wf1",
                        "history_id": "hist1",
                        "state": "Pending",
                        "outputs": [],
                        "create_time": "2025-11-05T00:00:00Z",
                        "update_time": "2025-11-05T00:00:00Z",
                    }
                ]
            }
        )
        mock_mongo = AsyncMock()
        mock_mongo.get = AsyncMock(return_value=None)
        with patch.object(invocation_service, "cache", mock_cache), \
             patch.object(invocation_service, "mongo_client", mock_mongo):
            response = client.get("/api/invocation/", headers=auth_headers)

        assert response.status_code == 200
        assert len(response.json()["invocations"]) == 1
        assert "Duplicate request detected" in caplog.text
        logger.info("TEST: test_list_invocations_duplicate_request PASSED")

    def test_list_invocations_state_mapping_cached(
        self,
        client,
        mock_galaxy_client,
        mock_workflow_manager,
        mock_current_api_key,
        mock_fernet,
        auth_headers,
    ):
        """Test state mapping using cached state."""
        logger.info("TEST: test_list_invocations_state_mapping_cached starting")

        mock_client, mock_invocations = mock_galaxy_client
        mock_invocations.get_invocations.return_value = [
            {
                "id": "inv1",
                "workflow_id": "wf_id_1",
                "workflow_name": "wf1",
                "history_id": "hist1",
                "state": "new",
                "create_time": "2025-11-05T00:00:00Z",
                "update_time": "2025-11-05T00:00:00Z",
            }
        ]
        
        # Mock workflow_manager.gi_object.gi.invocations.get_invocations
        mock_workflow_manager.gi_object.gi.invocations.get_invocations.return_value = mock_invocations.get_invocations.return_value

        mock_bg_tasks = AsyncMock()
        mock_bg_tasks.fetch_workflows_safely = AsyncMock(
            return_value=[{"id": "wf_id1", "name": "wf1"}]
        )
        mock_bg_tasks.build_invocation_workflow_mapping = AsyncMock(
            return_value=(
                {"inv1": {"workflow_name": "wf1", "workflow_id": "wf_id1"}},
                mock_invocations.get_invocations.return_value,
            )
        )
        with (
            patch.object(invocation_service, "background_tasks", mock_bg_tasks),
            patch.object(
                invocation_service.inv_data_manager, "background_tasks", mock_bg_tasks
            ),
        ):

            mock_cache = AsyncMock()
            mock_cache.get_deleted_invocations = AsyncMock(return_value=[])
            mock_cache.is_duplicate_request = AsyncMock(return_value=False)
            mock_cache.get_response_cache = AsyncMock(return_value=None)
            mock_cache.get_invocations_cache = AsyncMock(return_value=None)
            mock_cache.get_workflows_cache = AsyncMock(return_value=None)
            mock_cache.get_invocation_state = AsyncMock(return_value="Pending")
            # Mock acquire_lock as an async context manager
            from contextlib import asynccontextmanager
            @asynccontextmanager
            async def mock_acquire_lock(*args, **kwargs):
                yield
            mock_cache.acquire_lock = mock_acquire_lock
            mock_mongo = AsyncMock()
            mock_mongo.get = AsyncMock(return_value=None)
            mock_mongo.exists = AsyncMock(return_value=False)
            mock_mongo.set = AsyncMock()
            mock_mongo.add_to_set = AsyncMock()
            with (
                patch.object(invocation_service, "cache", mock_cache),
                patch.object(invocation_service.inv_data_manager, "cache", mock_cache),
                patch.object(invocation_service.inv_tracker, "cache", mock_cache),
                patch.object(invocation_service.background_tasks, "cache", mock_cache),
                patch.object(invocation_service, "mongo_client", mock_mongo),
            ):
                with patch(
                    "fastapi.concurrency.run_in_threadpool", new_callable=AsyncMock
                ) as mock_run_thread:
                    mock_run_thread.side_effect = lambda func, *args, **kwargs: func(
                        *args, **kwargs
                    )
                    response = client.get("/api/invocation/", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["invocations"][0]["state"] == "Pending"
        logger.info("TEST: test_list_invocations_state_mapping_cached PASSED")

    def test_list_invocations_unauthorized(self, client):
        """Test unauthorized access (no auth header)."""
        logger.info("TEST: test_list_invocations_unauthorized starting")

        response = client.get("/api/invocation/")

        assert response.status_code == 401
        logger.info("TEST: test_list_invocations_unauthorized PASSED")
