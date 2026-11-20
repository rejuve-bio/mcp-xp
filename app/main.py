import os
import sys
import json
import httpx
import asyncio
import logging
import redis

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from pydantic import BaseModel
from cryptography.fernet import Fernet
from contextlib import asynccontextmanager
from starlette.status import (
    HTTP_401_UNAUTHORIZED, 
    HTTP_500_INTERNAL_SERVER_ERROR, 
    HTTP_502_BAD_GATEWAY
    )

from fastapi import (
    FastAPI,
    Request,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    Response
    )
from fastapi.openapi.utils import get_openapi
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.mcp_client.chatbot import ChatSession, initialize_session

from app.utils import import_published_workflows
from app.log_setup import configure_logging
from app.api.middleware import (
    JWTGalaxyKeyMiddleware,
    RateLimiterMiddleware,
    DomainCORSMiddleware
    )
from app.api.api import api_router 
from app.api.endpoints import galaxy_runs
from app.api.endpoints.accounts import router as accounts_router
from app.api.socket_manager import ws_manager
from app.enumerations import SocketMessageEvent
from app.orchestration.invocation_cache import InvocationCache
from app.orchestration.invocation_tasks import InvocationBackgroundTasks

from app.exceptions import (
    http_exception_handler,
    generic_exception_handler,
    validation_exception_handler,
    UnauthorizedException,
    InternalServerErrorException
    )

load_dotenv()

# Global variables for cache and background tasks
# Initialize Redis client
redis_client = None
invocation_cache = None
background_tasks = None
sessions = {}

logger= logging.getLogger('main')

GALAXY_URL = os.getenv("GALAXY_URL")
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY is missing from environment")
fernet = Fernet(SECRET_KEY.encode() if isinstance(SECRET_KEY, str) else SECRET_KEY)


configure_logging()

# Base Schemas

class MessageRequest(BaseModel):
    message: str

APP_DESCRIPTION = """
- This **FastAPI application** provides a RESTful API for dynamically interacting with the **Galaxy platform** and the **Galaxy Agent**.\n
- It supports operations such as **executing Galaxy tools and workflows**, **managing history and datasets**, and **coordinating tasks** through the agent.\n\n  
- All API requests require **authentication** via an encrypted **USER-API-KEY token** in the request header.\n
- This token is obtained by **registering and validating your Galaxy user API key**.\n
- Requests **without a valid registered key will be rejected**.

"""

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    # Generate default schema
    openapi_schema = get_openapi(
        title="Galaxy Interaction API",
        version="1.0.0",
        description= APP_DESCRIPTION,
        routes=app.routes,
    )

    # Bearer JWT security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }

    # Apply globally
    openapi_schema["security"] = [{"BearerAuth": []}]

    # /v1 answers an invalid body with 400 (app.exceptions.validation_exception_handler), so FastAPI's automatic
    # "422 Validation Error" would misdocument it; 422 there means only "Galaxy rejected the inputs".
    for path, operations in openapi_schema["paths"].items():
        if path.startswith("/v1/"):
            for operation in operations.values():
                ref = operation.get("responses", {}).get("422", {}).get("content", {}) \
                    .get("application/json", {}).get("schema", {}).get("$ref", "")
                if ref.endswith("/HTTPValidationError"):
                    del operation["responses"]["422"]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks"""
    global redis_client, invocation_cache, background_tasks
    
    # Startup
    try:        
        # Test Redis connection
        redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=os.getenv("REDIS_PORT"), db=0, decode_responses=True)
        await asyncio.get_event_loop().run_in_executor(None, redis_client.ping)
        logger.info("Redis connection established")
        
        # Initialize cache
        invocation_cache = InvocationCache(redis_client)
        
        # Initialize background tasks
        background_tasks = InvocationBackgroundTasks(invocation_cache, redis_client)
        
        # Start background tasks
        asyncio.create_task(background_tasks.start_cache_warming_task())
        asyncio.create_task(background_tasks.start_cleanup_task())
        
        logger.info("Background tasks started")
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown
    try:
        logger.info("Shutting down application...")
        
        # Cancel background tasks
        for task in asyncio.all_tasks():
            if not task.done():
                task.cancel()
        
        # Close Redis connection
        if redis_client:
            redis_client.close()
        
        logger.info("Shutdown complete")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")



# Initialize the FastAPI application with a lifespan context
app = FastAPI(lifespan=lifespan)

# Add Middlewares, in the order CORS - Rate limiter - Auth
app.add_middleware(JWTGalaxyKeyMiddleware)
app.add_middleware(RateLimiterMiddleware)
app.add_middleware(DomainCORSMiddleware)

app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Include the API router
app.include_router(api_router, prefix="/api")
app.include_router(galaxy_runs.router, prefix="/v1/galaxy", tags=["Galaxy runs"])
app.include_router(galaxy_runs.datasets_router, prefix="/v1/datasets", tags=["Galaxy runs"])

app.add_exception_handler(RequestValidationError, validation_exception_handler)

# Override FastAPI’s OpenAPI generator
app.openapi = custom_openapi


@app.get("/", tags=["Root"])
async def read_root():
    """Add a simple root endpoint for a health check"""

    return {"message": "Welcome to the Galaxy API!"}


@app.get("/chat_history", tags=["Agent"])
async def get_chat_history(request: Request):
    """Conversation history in a session"""

    user_ip = request.client.host
    if user_ip not in sessions:
        chat_session = await initialize_session(user_ip)
        sessions[user_ip] = chat_session
        # return {"error": "Chat session not initiated for this IP"}

    return {"memory": sessions[user_ip].memory}


@app.post("/send_message", tags=["Agent"])
async def send_message(request: Request, message: MessageRequest):
    """Conversate with the Galaxy Agent"""

    user_ip = request.client.host
    if user_ip not in sessions:
            chat_session: ChatSession = await initialize_session(user_ip)
            sessions[user_ip] = chat_session
            # return {"error": "Chat session not initiated for this IP"}

    chat_session = sessions[user_ip]
    response = await chat_session.respond(model_id=os.getenv("CURRENT_LLM", "gemini"), user_input=message.message)
    return {"response": response}


@app.get("/list_tools", tags=["Agent"])
async def list_tools(request: Request):
    """List MCP server tools available for the LLM"""

    logger.info("listing tools")
    
    user_ip = request.client.host
    if user_ip not in sessions:
            chat_session = await initialize_session(user_ip)
            sessions[user_ip] = chat_session

    chat_session = sessions[user_ip]
    all_tools = []
    for server in chat_session.servers:
        logger.info(f"server {server.name}")
        tools = await server.list_tools()   
        logger.info(f'found tools: {[tool.name for tool in tools]}')
        all_tools.extend([tool.__dict__ for tool in tools])
    return {"tools": all_tools}


app.include_router(accounts_router)


@app.post("/galaxy_auth",
          tags =["Signup Auth"]
          )
async def galaxy_proxy_login(request: Request):
    
    """ CORS proxy for Galaxy user authentication endpoint. """
    
    try:
        # Read the body
        body = await request.body()
        
        # Forward only necessary headers
        headers = {
            "content-type": "application/json",
            "authorization": request.headers.get("authorization", ""),
        }
        
        logger.info(f"Forwarding login request to {GALAXY_URL}/user/login")
        
        # Forward to Galaxy
        async with httpx.AsyncClient() as client:
            target_response = await client.post(
                f"{GALAXY_URL}/user/login",
                content=body,
                headers=headers,
                follow_redirects=False,
            )
            
        # Create response
        proxy_response = Response(
            content=target_response.content,
            status_code=target_response.status_code,
            media_type=target_response.headers.get("content-type")
        )
        
        # Forward Set-Cookie headers from Galaxy to client for session management
        for key, value in target_response.headers.multi_items():
            if key.lower() == "set-cookie":
                proxy_response.headers.append(key, value)
        
        # Forward other important headers
        important_headers = ["content-type", "www-authenticate"]
        for header in important_headers:
            if header in target_response.headers:
                proxy_response.headers[header] = target_response.headers[header]
        
        logger.info(f"Login response: {target_response.status_code}")
        return proxy_response
    
    except httpx.ConnectError as e:
        logger.error(f"Connection error: {e}")
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Proxy connetcion error: {e}")
    except httpx.RequestError as e:
        logger.error(f"Error connecting to Galaxy: {e}")
        raise HTTPException(status_code=HTTP_502_BAD_GATEWAY, detail=f"Galaxy server error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Proxy error: {e}")

@app.websocket("/ws/{tracker_id}")
async def websocket_endpoint(websocket: WebSocket, tracker_id: str):
    # Accept and register socket in the tracker room
    await ws_manager.connect(websocket, tracker_id)
    
    # Background keepalive pinger
    async def ping_loop():
        try:
            while True:
                await asyncio.sleep(30)  
                # Keeping shape consisitent
                await ws_manager.broadcast( 
                    event=SocketMessageEvent.ping.value,
                    data={},
                    tracker_id=tracker_id
                )
        except WebSocketDisconnect:
            # Exit silently
            pass
        except Exception as e:
            logger.error(f"error in websocket pinging: {e}")
            pass

    ping_task = asyncio.create_task(ping_loop())

    try:
        # Block until the client disconnects (close frame)
        while True:
            try:
                # keeps the connection open
                await websocket.receive_json()  
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                logger.info(f"RuntimeError error caused: {e} ")
                break
    finally:
        ping_task.cancel()
        try:
            await ping_task
        except asyncio.CancelledError as e:
            logger.error(f"asyncio CancelledError error: {e}")
            pass
        await ws_manager.disconnect(websocket, tracker_id)