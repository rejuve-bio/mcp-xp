import os
import json
import logging
import asyncio
import time
import redis
import jwt
from datetime import datetime

from typing import Optional, Dict

from fastapi import Request, HTTPException, status, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from cryptography.fernet import Fernet, InvalidToken
from urllib.parse import urlparse
from starlette.types import ASGIApp
from starlette.status import HTTP_204_NO_CONTENT

import jwt
from app.context import (
    current_api_key,
    current_user_email,
    current_chat_id,
    validate_chat_id,
)

from app.exceptions import UnauthorizedException

GALAXY_API_TOKEN = "galaxy_api_token"


class JWTGalaxyKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware that expects:
      Authorization: Bearer <JWT>
    The JWT must contain a claim ('galaxy_api_token') that is either:
      - a fernet-encrypted JSON payload like {"apikey": "<actual_key>"} (what the register user produces)
    The middleware will set current_api_key to the final plain api key string.
    """

    def __init__(self, app):
        super().__init__(app)
        self.log = logging.getLogger(self.__class__.__name__)

    async def dispatch(self, request: Request, call_next):
        # Allow public paths if you need them (adjust to your app). Remove if unwanted.
        try:
            if request.method == "OPTIONS":
                self.log.info("Skipping options method")
                return await call_next(request)
            
            public_paths = {"/", "/docs", "/redoc", "/openapi.json", "/register-user", "/refresh-galaxy-key"}
            if request.url.path in public_paths or request.url.path.startswith("/static/"):
                return await call_next(request)

            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                raise UnauthorizedException("Authorization header with Bearer token is required.")

            token = auth.split(" ")[1].strip()
            try:
                payload = self._decode_jwt(token)
            except HTTPException as e:
                return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

            # Extract the API token claim
            if GALAXY_API_TOKEN not in payload:
                self.log.error("JWT missing API key claim '%s'", GALAXY_API_TOKEN)
                raise UnauthorizedException("JWT missing required claim")

            galaxy_jwt_token = payload[GALAXY_API_TOKEN]
            if not galaxy_jwt_token:
                self.log.error("Empty API key claim")

                raise UnauthorizedException("API key claim is empty.")

            # Try to decrypt claim_value (it might be the fernet token string produced by register-user)
            apikey = await self._decrypt_api_token(galaxy_jwt_token)
            if not apikey:
                # fallback: treat as raw api key string
                if isinstance(galaxy_jwt_token, str) and galaxy_jwt_token.strip():
                    apikey = galaxy_jwt_token.strip()
                else:
                    self.log.error("Unable to obtain Galaxy API key from JWT claim")
                    raise UnauthorizedException("Invalid API key in JWT claim.")

            # set the context for downstream handlers
            current_api_key.set(apikey)
            current_user_email.set(payload.get("email"))
            current_chat_id.set(
                validate_chat_id(request.headers.get("X-Chat-Id") or request.headers.get("x-chat-id"))
            )
            return await call_next(request)
        
        except UnauthorizedException as e:
            return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

    def _decode_jwt(self, token: str) -> dict:
        """
        Decode/verify JWT synchronously (PyJWT). Raises HTTPException(401) on invalid token.
        For RS-based tokens, JWT_SECRET should contain the public key.
        """
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            return payload
        except jwt.InvalidTokenError as e:
            self.log.error("Invalid JWT: %s", e)
            raise UnauthorizedException("Invalid JWT token")

    async def _decrypt_api_token(self, token_str: str) -> Optional[str]:
        """
        If token_str is a fernet-encrypted payload (bytes when encoded),
        decrypt and parse JSON for {"apikey": "<value>"} and return the value.
        Returns None if decryption/parsing fails so caller can fallback to raw token.
        """
        # Environment / secrets
        FERNET_SECRET = os.getenv("SECRET_KEY")
        if not FERNET_SECRET:
            raise RuntimeError("SECRET_KEY (Fernet secret) is required in env")
        
        fernet = Fernet(FERNET_SECRET)
        
        if not isinstance(token_str, str) or not token_str:
            return None
        loop = asyncio.get_running_loop()
        try:
            decrypted = await loop.run_in_executor(None, fernet.decrypt, token_str.encode("utf-8"))
            parsed = await loop.run_in_executor(None, json.loads, decrypted.decode("utf-8"))
            apikey = parsed.get("apikey")
            if apikey and isinstance(apikey, str):
                return apikey
            self.log.error("Decrypted JWT galaxy api-key payload missing 'apikey' field")
            return None
        except (InvalidToken, Exception) as e:
            # Not a fernet payload or parse failed; return None so fallback can apply
            self.log.debug("Fernet decryption/parsing failed for JWT claim: %s", e)
            return None
        
        
        
class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Redis-based rate limiter middleware with per-endpoint and per-user limits
    """
    
    def __init__(self, app, default_rate_limit: int = 100, redis_client = None):
        super().__init__(app)
        self.redis = redis_client or redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=os.getenv("REDIS_PORT"), db=0, decode_responses=True)
        self.default_rate_limit = default_rate_limit
        self.log = logging.getLogger(__class__.__name__)
        
        # Define rate limits per endpoint (requests per minute)
        self.endpoint_limits = {
            "GET /invocations/": 30,
            "GET /invocations/{invocation_id}/result": 10,
            "POST /invocations/{invocation_id}/report-pdf": 5,
            "DELETE /invocations/": 3,
            # Add more endpoints as needed
        }
        
        # Global rate limits (requests per minute)
        self.global_limits = {
            "per_user": 100,  # 200 requests per minute per user
            "total": 2000     # 2000 requests per minute globally
        }
    
    async def dispatch(self, request: Request, call_next):
        """Main middleware dispatcher"""
        
        # Skip rate limiting for certain paths (health checks, etc.)
        if self._should_skip_rate_limiting(request):
            return await call_next(request)
        
        try:
            # Get user identifier (API key)
            api_key = self._get_api_key(request)
            if not api_key:
                return await call_next(request)  # Skip if no API key
            
            # Track active user
            await self._mark_user_active(api_key)
            
            # Get endpoint key
            endpoint_key = self._get_endpoint_key(request)
            
            # Check rate limits
            rate_limit_result = await self._check_rate_limits(api_key, endpoint_key)
            
            if not rate_limit_result["allowed"]:
                return self._create_rate_limit_response(rate_limit_result)
            
            # Add rate limit headers to response
            response = await call_next(request)
            self._add_rate_limit_headers(response, rate_limit_result)
            
            return response
            
        except Exception as e:
            self.log.error(f"Rate limiter middleware error: {e}")
            # Continue without rate limiting if there's an error
            return await call_next(request)
    
    def _should_skip_rate_limiting(self, request: Request) -> bool:
        """Check if rate limiting should be skipped for this request"""
        skip_paths = ["/health", "/docs", "/redoc", "/openapi.json"]
        return any(request.url.path.startswith(path) for path in skip_paths)
    
    def _get_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request"""
        # Try to get from header
        api_key = request.headers.get("Authorization")
        if api_key and api_key.startswith("Bearer "):
            return api_key[7:]  # Remove "Bearer " prefix
        
        # Try to get from query parameter
        return request.query_params.get("api_key")
    
    def _get_endpoint_key(self, request: Request) -> str:
        """Generate endpoint key for rate limiting"""
        method = request.method
        path = request.url.path
        
        # Normalize path parameters (replace IDs with placeholders)
        normalized_path = self._normalize_path(path)
        
        return f"{method} {normalized_path}"
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path by replacing IDs with placeholders"""
        import re
        
        # Replace UUIDs and other IDs with placeholders
        patterns = [
            (r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{id}'),
            (r'/[0-9a-f]{16,}', '/{id}'),  # Long hex IDs
            (r'/\d+', '/{id}'),            # Numeric IDs
        ]
        
        normalized = path
        for pattern, replacement in patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized
    
    async def _check_rate_limits(self, api_key: str, endpoint_key: str) -> Dict:
        """Check all applicable rate limits"""
        current_time = int(time.time())
        minute_window = current_time // 60  # Current minute
        
        try:
            # Get limits
            endpoint_limit = self.endpoint_limits.get(endpoint_key, self.default_rate_limit)
            user_global_limit = self.global_limits["per_user"]
            
            # Redis keys
            endpoint_key_redis = f"rate_limit:endpoint:{api_key}:{endpoint_key}:{minute_window}"
            user_key_redis = f"rate_limit:user:{api_key}:{minute_window}"
            global_key_redis = f"rate_limit:global:{minute_window}"
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis.pipeline()
            
            # Increment counters
            pipe.incr(endpoint_key_redis)
            pipe.expire(endpoint_key_redis, 60)
            pipe.incr(user_key_redis)
            pipe.expire(user_key_redis, 60)
            pipe.incr(global_key_redis)
            pipe.expire(global_key_redis, 60)
            
            # Execute pipeline
            results = pipe.execute()
            
            endpoint_count = results[0]
            user_count = results[2]
            global_count = results[4]
            
            # Check limits
            if endpoint_count > endpoint_limit:
                return {
                    "allowed": False,
                    "limit_type": "endpoint",
                    "limit": endpoint_limit,
                    "current": endpoint_count,
                    "reset_time": (minute_window + 1) * 60
                }
            
            if user_count > user_global_limit:
                return {
                    "allowed": False,
                    "limit_type": "user",
                    "limit": user_global_limit,
                    "current": user_count,
                    "reset_time": (minute_window + 1) * 60
                }
            
            # Check global limit (optional - can be disabled)
            global_limit = self.global_limits["total"]
            if global_count > global_limit:
                return {
                    "allowed": False,
                    "limit_type": "global",
                    "limit": global_limit,
                    "current": global_count,
                    "reset_time": (minute_window + 1) * 60
                }
            
            return {
                "allowed": True,
                "endpoint_limit": endpoint_limit,
                "endpoint_current": endpoint_count,
                "user_limit": user_global_limit,
                "user_current": user_count,
                "reset_time": (minute_window + 1) * 60
            }
            
        except redis.RedisError as e:
            self.log.error(f"Redis error in rate limiter: {e}")
            # Allow request if Redis fails
            return {"allowed": True}
    
    def _create_rate_limit_response(self, rate_limit_result: Dict) -> Response:
        """Create 429 Too Many Requests response"""
        reset_time = rate_limit_result.get("reset_time", int(time.time()) + 60)
        retry_after = reset_time - int(time.time())
        
        error_detail = {
            "error": "Rate limit exceeded",
            "limit_type": rate_limit_result.get("limit_type"),
            "limit": rate_limit_result.get("limit"),
            "current": rate_limit_result.get("current"),
            "retry_after": retry_after
        }
        
        headers = {
            "X-RateLimit-Limit": str(rate_limit_result.get("limit", 0)),
            "X-RateLimit-Remaining": str(max(0, rate_limit_result.get("limit", 0) - rate_limit_result.get("current", 0))),
            "X-RateLimit-Reset": str(reset_time),
            "Retry-After": str(retry_after)
        }
        
        return Response(
            content=json.dumps(error_detail),
            status_code=429,
            headers=headers,
            media_type="application/json"
        )
    
    def _add_rate_limit_headers(self, response: Response, rate_limit_result: Dict):
        """Add rate limit headers to successful responses"""
        if rate_limit_result.get("allowed"):
            response.headers["X-RateLimit-Limit"] = str(rate_limit_result.get("endpoint_limit", 0))
            response.headers["X-RateLimit-Remaining"] = str(
                max(0, rate_limit_result.get("endpoint_limit", 0) - rate_limit_result.get("endpoint_current", 0))
            )
            response.headers["X-RateLimit-Reset"] = str(rate_limit_result.get("reset_time", 0))
            
    async def _mark_user_active(self, api_key: str):
        """Mark user as active in the last hour using a Redis sorted set"""
        
        active_users_key = "rate_limit:active_users_last_10_minutes"
        current_time = int(datetime.now().timestamp())
        
        # Store or update the user's timestamp
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.redis.zadd,
            active_users_key,
            {api_key: current_time}
        )
class DomainCORSMiddleware(BaseHTTPMiddleware):
    """ CORS middleware that only allows requests from the same domain/hostname. """

    def __init__(
        self,
        app: ASGIApp,
        allow_methods: list[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        # Range: partial reads of /v1/datasets downloads (previews, resumed downloads).
        allow_headers: list[str] = ["Authorization", "Content-Type", "X-Requested-With", "Range"],
        # Headers browser code may read: a download's name and the byte range it got.
        expose_headers: list[str] = ["Content-Disposition", "Content-Range", "Content-Length", "Accept-Ranges"],
        max_age: int = 600,
        allow_schemes: list[str] = ["http", "https"],
        trust_proxy_headers: bool = False,  # Set True if behind reverse proxy
    ):
        super().__init__(app)
        self.allow_methods = allow_methods
        self.allow_headers = allow_headers
        self.expose_headers = expose_headers
        self.max_age = max_age
        self.allow_schemes = set(scheme.lower() for scheme in allow_schemes)
        self.trust_proxy_headers = trust_proxy_headers
        self.log = logging.getLogger(__class__.__name__)

    def _get_server_hostname(self, request: Request) -> str:
        """Extract server hostname, handling proxy headers if configured."""
        if self.trust_proxy_headers:
            host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
        else:
            host = request.headers.get("host", "")
        
        # Handle IPv6 and port parsing safely
        try:
            parsed = urlparse(f"http://{host}")
            return (parsed.hostname or host).lower()
        except Exception as e:
            self.log.error(f"Failed to parse host header: {host}, error: {e}")
            return host.split(":")[0].lower()  # Fallback

    def _validate_origin(self, origin_str: str | None, server_hostname: str) -> str | None:
        """
        Validate origin against server hostname.
        Returns the origin if valid, None otherwise.
        """
        if not origin_str:
            return None

        # Handle null origin (file://, sandboxed contexts)
        if origin_str == "null":
            self.log.warning("Blocked null origin request")
            return None

        try:
            origin_parsed = urlparse(origin_str)
            
            # Validate origin has all required components
            if not origin_parsed.hostname or not origin_parsed.scheme:
                self.log.warning(f"Invalid origin format: {origin_str}")
                return None

            # Check scheme is allowed
            if origin_parsed.scheme.lower() not in self.allow_schemes:
                self.log.warning(
                    f"Origin scheme not allowed: {origin_parsed.scheme} (origin: {origin_str})"
                )
                return None

            # Check hostname matches (case-insensitive)
            origin_hostname = origin_parsed.hostname.lower()
    
            # Exact match or subdomain match (e.g., api.example.com matches example.com)
            if origin_hostname == server_hostname or origin_hostname.endswith(f".{server_hostname}"):
                return origin_str
            
            self.log.warning(f"Origin hostname mismatch: {origin_parsed.hostname} != {server_hostname}")
            return None

        except Exception as e:
            self.log.error(f"Failed to parse origin: {origin_str}, error: {e}")
            return None

    async def dispatch(self, request: Request, call_next):
        origin_str = request.headers.get("origin")
        server_hostname = self._get_server_hostname(request)
        
        # Validate origin
        allow_origin = self._validate_origin(origin_str, server_hostname)
        self.log.debug(f"Origin: {origin_str}")
        self.log.debug(f"Server hostname: {server_hostname}")
        self.log.debug(f"Allow origin: {allow_origin}")

        # Handle preflight requests
        if request.method == "OPTIONS":
            if allow_origin:
                return self._preflight_response(allow_origin)
            else:
                # Return 204 with no CORS headers - browser will block
                return Response(status_code=HTTP_204_NO_CONTENT)

        # Process actual request
        response = await call_next(request)

        # Add CORS headers if origin is valid
        if allow_origin:
            self._add_cors_headers(response, allow_origin)

        return response

    def _add_cors_headers(self, response: Response, origin: str):
        """Add CORS headers to response."""
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)
        
        # Append to existing Vary header if present
        existing_vary = response.headers.get("Vary", "")
        if existing_vary:
            if "Origin" not in existing_vary:
                response.headers["Vary"] = f"{existing_vary}, Origin"
        else:
            response.headers["Vary"] = "Origin"

    def _preflight_response(self, origin: str) -> Response:
        """Create preflight response with CORS headers."""
        headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": ", ".join(self.allow_methods),
            "Access-Control-Allow-Headers": ", ".join(self.allow_headers),
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": str(self.max_age),
            "Vary": "Origin",
        }
        return Response(status_code=HTTP_204_NO_CONTENT, headers=headers)