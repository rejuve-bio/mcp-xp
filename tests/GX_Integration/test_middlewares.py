import os
import jwt
import json
import pytest
import redis
import logging
from datetime import datetime
from sys import path
path.append(".")

from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient as FastAPITestClient
from unittest.mock import Mock, patch, MagicMock
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_429_TOO_MANY_REQUESTS

from app.log_setup import configure_logging
from app.api.middleware import JWTGalaxyKeyMiddleware, RateLimiterMiddleware
from app.context import current_api_key


# Configure test logging
configure_logging()


@pytest.fixture
def jwt_logger():
    return logging.getLogger("TestJWTGalaxyKeyMiddleware")

@pytest.fixture
def rate_logger():
    return logging.getLogger("TestRateLimiterMiddleware")


@pytest.fixture
def fernet_secret():
    """Fixture for Fernet secret key."""
    return Fernet.generate_key()


@pytest.fixture
def fernet(fernet_secret):
    """Fixture for Fernet instance."""
    return Fernet(fernet_secret)


@pytest.fixture
def test_app():
    """Basic FastAPI app for testing middlewares."""
    app = FastAPI()
    
    @app.get("/")
    async def root():
        return {"message": "Hello World", "api_key": current_api_key.get(None)}
       
    @app.get("/protected")
    async def protected():
        return {"message": "Protected endpoint", "api_key": current_api_key.get(None)}
    
    @app.get("/health")
    async def health():
        return {"status": "ok"}
    
    return app
JWT_TEST_SECRET = "test-jwt-secret-0123456789abcdef0123456789"


def signed_token(claims: dict, secret: str = JWT_TEST_SECRET, exp_in: int = 300) -> str:
    claims = {"exp": int(datetime.now().timestamp()) + exp_in, **claims}
    return jwt.encode(claims, secret, algorithm="HS256")


class TestJWTGalaxyKeyMiddleware:
    """REST middleware: only tokens signed by the auth service get through."""

    @pytest.fixture
    def app_with_jwt_middleware(self, test_app, fernet_secret, monkeypatch):
        monkeypatch.setenv("SECRET_KEY", fernet_secret.decode())
        monkeypatch.setenv("JWT_SECRET", JWT_TEST_SECRET)
        monkeypatch.delenv("JWT_VERIFY", raising=False)
        test_app.add_middleware(JWTGalaxyKeyMiddleware)
        yield FastAPITestClient(test_app)

    def _get(self, client, token):
        return client.get("/protected", headers={"Authorization": f"Bearer {token}"})

    def test_public_paths_allowed_without_auth(self, app_with_jwt_middleware):
        response = app_with_jwt_middleware.get("/")
        assert response.status_code == 200

    def test_missing_authorization_header(self, app_with_jwt_middleware):
        response = app_with_jwt_middleware.get("/protected")
        assert response.status_code == HTTP_401_UNAUTHORIZED
        assert response.json()["detail"] == "Authorization header with Bearer token is required."
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_malformed_token(self, app_with_jwt_middleware):
        response = self._get(app_with_jwt_middleware, "invalid.token.here")
        assert response.status_code == HTTP_401_UNAUTHORIZED
        assert response.headers["WWW-Authenticate"] == 'Bearer error="invalid_token"'

    def test_unsigned_token_rejected(self, app_with_jwt_middleware):
        token = jwt.encode({"galaxy_api_token": "raw-key"}, key=None, algorithm="none")
        assert self._get(app_with_jwt_middleware, token).status_code == HTTP_401_UNAUTHORIZED

    def test_token_signed_with_other_secret_rejected(self, app_with_jwt_middleware, fernet):
        claim = fernet.encrypt(json.dumps({"apikey": "k"}).encode()).decode()
        token = signed_token({"galaxy_api_token": claim}, secret="attacker-secret-0123456789abcdef01")
        response = self._get(app_with_jwt_middleware, token)
        assert response.status_code == HTTP_401_UNAUTHORIZED
        assert "signature" in response.json()["detail"]

    def test_expired_token_rejected(self, app_with_jwt_middleware, fernet):
        claim = fernet.encrypt(json.dumps({"apikey": "k"}).encode()).decode()
        token = signed_token({"galaxy_api_token": claim}, exp_in=-3600)
        response = self._get(app_with_jwt_middleware, token)
        assert response.status_code == HTTP_401_UNAUTHORIZED
        assert "expired" in response.json()["detail"]

    def test_raw_key_claim_no_longer_accepted(self, app_with_jwt_middleware):
        token = signed_token({"galaxy_api_token": "raw-galaxy-key"})
        assert self._get(app_with_jwt_middleware, token).status_code == HTTP_401_UNAUTHORIZED

    def test_signed_token_with_encrypted_key(self, app_with_jwt_middleware, fernet):
        raw_apikey = "test-encrypted-api-key-456"
        claim = fernet.encrypt(json.dumps({"apikey": raw_apikey}).encode()).decode()
        response = self._get(app_with_jwt_middleware, signed_token({"galaxy_api_token": claim}))
        assert response.status_code == 200
        assert response.json()["api_key"] == raw_apikey

    def test_refresh_galaxy_key_is_not_public(self, test_app, fernet_secret, monkeypatch):
        monkeypatch.setenv("SECRET_KEY", fernet_secret.decode())
        monkeypatch.setenv("JWT_SECRET", JWT_TEST_SECRET)

        @test_app.post("/refresh-galaxy-key")
        async def refresh():
            return {"api_key": current_api_key.get(None)}

        test_app.add_middleware(JWTGalaxyKeyMiddleware)
        response = FastAPITestClient(test_app).post("/refresh-galaxy-key?email=a@example.org")
        assert response.status_code == HTTP_401_UNAUTHORIZED

    def test_startup_fails_without_secret(self, test_app, fernet_secret, monkeypatch):
        monkeypatch.setenv("SECRET_KEY", fernet_secret.decode())
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.delenv("JWT_VERIFY", raising=False)
        test_app.add_middleware(JWTGalaxyKeyMiddleware)
        with pytest.raises(RuntimeError, match="JWT_SECRET"):
            FastAPITestClient(test_app).get("/")


class TestRateLimiterMiddleware:
    """Unit tests for RateLimiterMiddleware."""
    
    
    @pytest.fixture
    def mock_redis_client(self,rate_logger):
        """Mock Redis client."""
        rate_logger.debug("Creating mock_redis_client")
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.pipeline.return_value = MagicMock()
        rate_logger.debug("mock_redis_client setup complete")
        return mock_redis
    
    @pytest.fixture
    def app_with_rate_limiter(self, test_app, mock_redis_client, rate_logger):
        """App with RateLimiter middleware added."""
        rate_logger.debug("Setting up app_with_rate_limiter")
        test_app.add_middleware(RateLimiterMiddleware, redis_client=mock_redis_client)
        client = FastAPITestClient(test_app)
        rate_logger.debug("app_with_rate_limiter setup complete")
        return client
    
    def test_skip_rate_limiting_for_health_paths(self, app_with_rate_limiter, caplog,rate_logger):
        """Test that rate limiting is skipped for health check paths."""
        rate_logger.info("TEST: test_skip_rate_limiting_for_health_paths starting")
        caplog.set_level(logging.INFO)
        rate_logger.debug("Making request to /health")
        response = app_with_rate_limiter.get("/health")
        assert response.status_code == 200
        rate_logger.info("TEST: test_skip_rate_limiting_for_health_paths PASSED")

    def test_no_api_key_skips_rate_limiting(self, app_with_rate_limiter, mock_redis_client, caplog, rate_logger):
        """Test that requests without API key skip rate limiting."""
        rate_logger.info("TEST: test_no_api_key_skips_rate_limiting starting")
        caplog.set_level(logging.INFO)
        with patch.object(mock_redis_client, 'pipeline') as mock_pipeline:
            rate_logger.debug("Making request without API key to /")
            response = app_with_rate_limiter.get("/")
            assert response.status_code == 200
            mock_pipeline.assert_not_called()  # No Redis interaction
        rate_logger.info("TEST: test_no_api_key_skips_rate_limiting PASSED")

    @patch('time.time')
    def test_rate_limit_exceeded_for_endpoint(self, mock_time, app_with_rate_limiter, mock_redis_client, caplog, rate_logger):
        """Test 429 response when endpoint rate limit is exceeded."""
        rate_logger.info("TEST: test_rate_limit_exceeded_for_endpoint starting")
        caplog.set_level(logging.ERROR)
        mock_time.return_value = 1000

        pipeline_mock = MagicMock()
        pipeline_mock.execute.return_value = [31, 1, 1, 1, 1, 1]
        mock_redis_client.pipeline.return_value = pipeline_mock
        
        rate_logger.debug("Making request to /invocations/ with exceeded endpoint limit")
        response = app_with_rate_limiter.get(
            "/invocations/",
            headers={"Authorization": "Bearer test-key"}
        )
        assert response.status_code == HTTP_429_TOO_MANY_REQUESTS
        assert response.json()["limit_type"] == "endpoint"
        assert int(response.headers["X-RateLimit-Limit"]) == 30
        assert "endpoint" in caplog.text
        rate_logger.info("TEST: test_rate_limit_exceeded_for_endpoint PASSED")

    @patch('time.time')
    def test_rate_limit_exceeded_for_user(self, mock_time, app_with_rate_limiter, mock_redis_client, caplog, rate_logger):
        """Test 429 response when user global rate limit is exceeded."""
        rate_logger.info("TEST: test_rate_limit_exceeded_for_user starting")
        caplog.set_level(logging.ERROR)
        mock_time.return_value = 1000
        # endpoint ok (1), user=101 >100
        pipeline_mock = MagicMock()
        pipeline_mock.execute.return_value = [1, 1, 101, 1, 1, 1]
        mock_redis_client.pipeline.return_value = pipeline_mock
        
        rate_logger.debug("Making request to /protected with exceeded user limit")
        response = app_with_rate_limiter.get(
            "/protected",
            headers={"Authorization": "Bearer test-user-key"}
        )
        assert response.status_code == HTTP_429_TOO_MANY_REQUESTS
        assert response.json()["limit_type"] == "user"
        assert int(response.headers["X-RateLimit-Limit"]) == 100
        assert "user" in caplog.text
        rate_logger.info("TEST: test_rate_limit_exceeded_for_user PASSED")

    @patch('time.time')
    def test_rate_limit_exceeded_for_global(self, mock_time, app_with_rate_limiter, mock_redis_client, caplog, rate_logger):
        """Test 429 response when global rate limit is exceeded."""
        rate_logger.info("TEST: test_rate_limit_exceeded_for_global starting")
        caplog.set_level(logging.ERROR)
        mock_time.return_value = 1000
        pipeline_mock = MagicMock()
        pipeline_mock.execute.return_value = [1, 1, 1, 1, 2001, 1]
        mock_redis_client.pipeline.return_value = pipeline_mock
        
        rate_logger.debug("Making request to /protected with exceeded global limit")
        response = app_with_rate_limiter.get(
            "/protected",
            headers={"Authorization": "Bearer test-global-key"}
        )
        assert response.status_code == HTTP_429_TOO_MANY_REQUESTS
        assert response.json()["limit_type"] == "global"
        assert int(response.headers["X-RateLimit-Limit"]) == 2000
        assert "global" in caplog.text
        rate_logger.info("TEST: test_rate_limit_exceeded_for_global PASSED")

    @patch('time.time')
    def test_within_rate_limits_adds_headers(self, mock_time, app_with_rate_limiter, mock_redis_client, caplog, rate_logger):
        """Test that successful requests add rate limit headers."""
        rate_logger.info("TEST: test_within_rate_limits_adds_headers starting")
        caplog.set_level(logging.INFO)
        mock_time.return_value = 1000
        # All counters within limits
        pipeline_mock = MagicMock()
        pipeline_mock.execute.return_value = [5, 1, 10, 1, 500, 1]
        mock_redis_client.pipeline.return_value = pipeline_mock
        
        rate_logger.debug("Making request to /protected within limits")
        response = app_with_rate_limiter.get(
            "/protected",
            headers={"Authorization": "Bearer test-within-limit-key"}
        )
        assert response.status_code == 200
        assert int(response.headers["X-RateLimit-Limit"]) == 100  # Default limit
        assert int(response.headers["X-RateLimit-Remaining"]) == 95
        assert "X-RateLimit-Reset" in response.headers
        rate_logger.info("TEST: test_within_rate_limits_adds_headers PASSED")

    @patch('time.time')
    def test_redis_error_allows_request(self, mock_time, app_with_rate_limiter, mock_redis_client, caplog, rate_logger):
        """Test that Redis errors allow the request to proceed."""
        rate_logger.info("TEST: test_redis_error_allows_request starting")
        caplog.set_level(logging.ERROR)
        mock_time.return_value = 1000
        mock_redis_client.pipeline.side_effect = redis.RedisError("Connection error")
        
        rate_logger.debug("Making request to /protected with Redis error")
        response = app_with_rate_limiter.get(
            "/protected",
            headers={"Authorization": "Bearer test-redis-error-key"}
        )
        assert response.status_code == 200  # Proceeds despite error
        assert "Connection error" in caplog.text
        rate_logger.info("TEST: test_redis_error_allows_request PASSED")

    @pytest.mark.asyncio
    async def test_mark_user_active(self, mock_redis_client, caplog, rate_logger):
        """Test marking user as active in Redis sorted set."""
        rate_logger.info("TEST: test_mark_user_active starting")
        caplog.set_level(logging.INFO)
        middleware = RateLimiterMiddleware(FastAPI(), redis_client = mock_redis_client)
        rate_logger.debug("Marking user active: test-active-key")
        await middleware._mark_user_active("test-active-key")
        mock_redis_client.zadd.assert_called_once_with(
            "rate_limit:active_users_last_10_minutes",
            {"test-active-key": pytest.approx(datetime.now().timestamp())}
        )
        rate_logger.info("TEST: test_mark_user_active PASSED")