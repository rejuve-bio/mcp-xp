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
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
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
from app.api.socket_manager import ws_manager
from app.enumerations import SocketMessageEvent
from app.orchestration.invocation_cache import InvocationCache
from app.orchestration.invocation_tasks import InvocationBackgroundTasks

from app.exceptions import (
    http_exception_handler,
    generic_exception_handler,  
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

class GalaxyUserAccount(BaseModel):
    username: str
    api_token: str


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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """/v1/galaxy answers an invalid body with 400, keeping 422 for inputs Galaxy rejects."""
    if request.url.path.startswith("/v1/galaxy"):
        return JSONResponse(status_code=400, content={"detail": jsonable_encoder(exc.errors())})
    return await request_validation_exception_handler(request, exc)

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


@app.post("/register-user",
          response_model=GalaxyUserAccount,
          tags =["Signup Auth"]
           )
async def get_create_galaxy_user_and_key(
    email: str = Query(description="email of the user to be registered to our galaxy instance."),
    password: str = Query(description="passcode to use for our galaxy isntance.")
    ) -> tuple[str, str]:

    """ Register Galaxy user from platform using the same credentials   """

    # 1. Create user via admin API
    username=None
    api_token=None

    try:
        logger.info("creating galaxy user account from galaxy service.")
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                url = f"{GALAXY_URL}/api/users",
                headers={"x-api-key": os.getenv("GALAXY_API_KEY")},
                json={
                    "username": email.split("@")[0],  # or whatever scheme you like
                    "email":    email,
                    "password": password
                }
            )
                
        resp.raise_for_status()
        galaxy_user_id = resp.json()["id"]
        username= resp.json()["username"]
        logger.info(f"Galaxy account created with username {username}")
    
    except httpx.ConnectError as e:
        logger.error(f"Error connecting to galaxy instance when generating api: {e}")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:

            # Else Fetch user by email if already exists
            try:
                logger.info("account already exists, getting api.")
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(
                        f"{GALAXY_URL}/api/users",
                        headers={"x-api-key": os.getenv("GALAXY_API_KEY")},
                        params={"f_email": email}
                    )
                resp.raise_for_status()
                users = resp.json()
                galaxy_user_id = users[0]["id"]
                username= users[0]["username"]
                logger.info(f"Galaxy User fetched with username {username}")
            
            except httpx.ConnectError as e:
                logger.error(f"Error connecting to galaxy instance when generating api: {e}")   

            except HTTPException as e: 
                raise InternalServerErrorException("Error getting/creating galaxy user")
            except Exception as e:
                logger.error(f"Error: {e}")
                raise
        elif e.response.status_code == HTTP_401_UNAUTHORIZED:
            logger.error(f"Unauthorized admin id: {e}")
            raise UnauthorizedException("Unauthorized admin id")
        else:
            raise InternalServerErrorException("Error caused during getting api_key for the user")

    except HTTPException as e:
        logger.error(f"errror creating user acount: {e}")
        raise InternalServerErrorException("error creating user account")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
 
    # 2. Return generated galaxy api-key
    try:
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            key_resp = await client.post(
                url = f"{GALAXY_URL}/api/users/{galaxy_user_id}/api_key",
                headers={"x-api-key": os.getenv("GALAXY_API_KEY")},
                json={
                    "name": "auto-generated from platform"
                    }
            )

        key_resp.raise_for_status()
        api_key = key_resp.json()

        payload = json.dumps({"apikey": api_key}).encode("utf-8")
        api_token = fernet.encrypt(payload).decode("utf-8")
        logger.info(f"Galaxy api-key extracted and encrypted for user with galaxy id {galaxy_user_id}")

    except HTTPException as e:
        logger.error(f"error creating galaxy user api key: {e}")
        raise InternalServerErrorException("error getting galaxy user api-key")
    except Exception as e:
        logger.error(f"Error: {e}")

    # import missing public workflows into the users account.    
    import_published_workflows(galaxy_url=GALAXY_URL, api_key=api_key)

    return  GalaxyUserAccount(
        username = username,
        api_token = api_token
    )


@app.post("/refresh-galaxy-key",
          response_model=GalaxyUserAccount,
          tags=["Signup Auth"])
async def refresh_galaxy_key(
    email: str = Query(description="Email of the existing Galaxy user whose API key to refresh.")
) -> GalaxyUserAccount:
    """Mint a fresh Galaxy API key for an existing user. No password required —
    uses the admin key to look up the user by email and generate a new key."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Look up user by email
            user_resp = await client.get(
                f"{GALAXY_URL}/api/users",
                headers={"x-api-key": os.getenv("GALAXY_API_KEY")},
                params={"f_email": email}
            )
            user_resp.raise_for_status()
            users = user_resp.json()
            if not users:
                logger.error(f"refresh-galaxy-key: no Galaxy user for {email}")
                raise UnauthorizedException(f"Galaxy user {email} not found")
            galaxy_user_id = users[0]["id"]
            username = users[0]["username"]

            # Mint new API key
            key_resp = await client.post(
                f"{GALAXY_URL}/api/users/{galaxy_user_id}/api_key",
                headers={"x-api-key": os.getenv("GALAXY_API_KEY")},
                json={"name": "refresh"}
            )
            key_resp.raise_for_status()
            api_key = key_resp.json()

        # Encrypt with Fernet (same format as register-user)
        payload = json.dumps({"apikey": api_key}).encode("utf-8")
        api_token = fernet.encrypt(payload).decode("utf-8")
        logger.info(f"refresh-galaxy-key: fresh API key issued for {username}")
        return GalaxyUserAccount(username=username, api_token=api_token)

    except UnauthorizedException:
        raise
    except Exception as e:
        logger.error(f"refresh-galaxy-key failed for {email}: {e}")
        raise InternalServerErrorException("Failed to refresh Galaxy API key")


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