import redis
import asyncio
import logging
import jwt
import os
import json
from dotenv import load_dotenv

from cryptography.fernet import Fernet, InvalidToken
from typing import Dict, List, Optional
from datetime import datetime
from starlette.concurrency import run_in_threadpool
from app.orchestration.invocation_cache import InvocationCache
from app.enumerations import NumericLimits
from app.galaxy import GalaxyClient
from app.auth.galaxy_accounts import GalaxyAccountError
from app.auth.identity import identify
from app.auth.tokens import TokenError
from app.GX_integration.workflows.workflow_manager import WorkflowManager

load_dotenv()

GALAXY_API_TOKEN = "galaxy_api_token"
FERNET_SECRET = os.getenv("SECRET_KEY")
if not FERNET_SECRET:
    raise RuntimeError("SECRET_KEY (Fernet secret) is required in env")

class InvocationBackgroundTasks:
    """Background tasks for maintaining invocation cache"""
    
    def __init__(self, cache: InvocationCache, redis_client: redis.Redis):
        self.cache = cache
        self.redis = redis_client
        self.log = logging.getLogger(__class__.__name__)
        
    async def start_cache_warming_task(self):
        """Start the background cache warming task"""
        while True:
            try:
                await self.warm_all_user_caches()
                await asyncio.sleep(NumericLimits.BACKGROUND_INTERVAL.value)
            except Exception as e:
                self.log.error(f"Error in cache warming task: {e}")
                # Wait 1 minute before retrying on error
                await asyncio.sleep(NumericLimits.LONG_SLEEP.value)
    
    async def warm_all_user_caches(self):
        """Warm cache for all active users"""
        try:
            # Get all active API keys from Redis
            active_users = await self.get_active_users()
            
            self.log.debug(f"Warming cache for {len(active_users)} active users")
            
            # Process users in batches to avoid overwhelming the Galaxy API
            batch_size = NumericLimits.BATCH_SIZE.value
            for i in range(0, len(active_users), batch_size):
                batch = active_users[i:i + batch_size]
                tasks = [self.warm_user_cache(token) for token in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log any exceptions from the batch
                for result in results:
                    if isinstance(result, Exception):
                        self.log.error(f"Error in batch cache warming: {result}")
                
                # Small delay between batches
                if i + batch_size < len(active_users):
                    await asyncio.sleep(NumericLimits.SHORT_SLEEP.value)
            
        except Exception as e:
            self.log.error(f"Error warming all user caches: {e}")
    
    async def get_active_users(self) -> List[str]:
        """Get list of active users from Redis (users who made requests in the last hour)"""
        try:
            current_time = int(datetime.now().timestamp())
            ten_minutes_ago = current_time - NumericLimits.BACKGROUND_INTERVAL.value
            active_users_key = "rate_limit:active_users_last_10_minutes"
            
            # Clean old entries (remove users with timestamps older than 10 minutes)
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.redis.zremrangebyscore, 
                active_users_key, 
                '-inf', 
                ten_minutes_ago
            )
            
            # Get remaining members (active API keys)
            active_users = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.redis.zrange, 
                active_users_key, 
                0, 
                -1
            )
            
            return active_users  # Already strings due to decode_responses=True
        
        except Exception as e:
            self.log.error(f"Error getting active users: {e}")
            return []
        
    async def _decrypt_api_token(self, token_str: str) -> Optional[str]:
        """
        If token_str is a fernet-encrypted payload (bytes when encoded),
        decrypt and parse JSON for {"apikey": "<value>"} and return the value.
        Returns None if decryption/parsing fails so caller can fallback to raw token.
        """
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
            return None
        except (InvalidToken, Exception) as e:
            # Not a fernet payload or parse failed; return None so fallback can apply
            return None
        
    async def warm_user_cache(self, token: str, api_key= None):
        """Warm cache for a specific user"""
        
        try:
            # The active-user set records every bearer token the rate limiter
            # saw, including ones the JWT middleware then rejected; only warm
            # caches for tokens that still pass the same checks.
            user_email = None
            if not api_key:
                try:
                    identity = await identify(token, create_account=False)
                except (TokenError, GalaxyAccountError) as e:
                    self.log.debug(f"Skipping cache warm for an unusable token: {e}")
                    return
                api_key, user_email = identity.api_key, identity.email

            galaxy_client = GalaxyClient(api_key, user_email=user_email)
            username = galaxy_client.whoami
            
            # Avoid warming cache for the same user too frequently
            last_warm_key = f"last_cache_warm:{api_key}"
            last_warm = await asyncio.get_event_loop().run_in_executor(None, self.redis.get, last_warm_key)
            
            if last_warm:
                last_warm_time = datetime.fromisoformat(last_warm)
                if (datetime.now() - last_warm_time).total_seconds() < NumericLimits.WARM_CHECK.value:
                    self.log.debug(f"Skipping cache warm for user {username}: recently warmed")
                    return  # Skip if warmed recently
            
            self.log.debug(f"Warming cache for user: {username}")
            
            # Initialize workflow manager
            workflow_manager = WorkflowManager(galaxy_client)
            
            # Warm workflows cache with full details
            workflows = await self.fetch_workflows_safely(workflow_manager, fetch_details=True)
            if workflows:
                await self.cache.set_workflows_cache(username, workflows)
            
            # Build and cache invocation-workflow mapping and invocations list
            mapping, all_invocations = await self.build_invocation_workflow_mapping(workflow_manager, workflows)
            if mapping:
                await self.cache.set_invocation_workflow_mapping(username, mapping)
            if all_invocations:
                await self.cache.set_invocations_cache(username, all_invocations, filters={"workflow_id": None, "history_id": None})
            
            # Update last warm timestamp
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.redis.setex, 
                last_warm_key, 
                NumericLimits.WARM_TIMESTAMP.value, 
                datetime.now().isoformat()
            )
            
            self.log.info(f"Cache warmed for user {username}: {len(workflows or [])} workflows, {len(mapping)} mappings, {len(all_invocations)} invocations")
            
        except Exception as e:
            self.log.error(f"Error warming cache for user: {e}")
    
    async def fetch_workflows_safely(self, workflow_manager: WorkflowManager, fetch_details: bool = False) -> List[Dict]:
        """Safely fetch workflows with timeout and error handling, optionally with full details"""
        try:
            # Fetch raw workflows
            raw_workflows = await asyncio.wait_for(
                run_in_threadpool(workflow_manager.gi_object.gi.workflows.get_workflows),
                timeout=NumericLimits.TIMEOUT.value
            )
            if not raw_workflows:
                self.log.debug("No workflows found for user")
                return []
            if not fetch_details:
                return raw_workflows

            # Parallel fetch full details
            semaphore = asyncio.Semaphore(NumericLimits.SEMAPHORE_LIMIT.value)
            async def fetch_full(wf_id):
                async with semaphore:
                    try:
                        return await asyncio.wait_for(
                            run_in_threadpool(
                                workflow_manager.gi_object.gi.workflows.show_workflow,
                                workflow_id=wf_id
                            ),
                            timeout=NumericLimits.TIMEOUT.value
                        )
                    except Exception as e:
                        self.log.warning(f"Failed to fetch full workflow {wf_id}: {e}")
                        return None

            tasks = [fetch_full(wf['id']) for wf in raw_workflows]
            full_workflows = await asyncio.gather(*tasks)

            # Process with partial handling
            processed = []
            for idx, full_wf in enumerate(full_workflows):
                if full_wf is None:
                    raw_wf = raw_workflows[idx]
                    processed.append({
                        'id': raw_wf['id'],
                        'name': raw_wf['name'],
                        'description': "Unknown",
                        'tags': []
                    })
                else:
                    processed.append({
                        'id': full_wf['id'],
                        'name': full_wf['name'],
                        'description': full_wf.get('annotation') or full_wf.get('description', None),
                        'tags': full_wf.get('tags', [])
                    })
            return processed

        except asyncio.TimeoutError:
            self.log.warning("Timeout fetching workflows")
            return []
        except Exception as e:
            self.log.error(f"Error fetching workflows: {e}")
            return []
    
    async def build_invocation_workflow_mapping(self, workflow_manager: WorkflowManager, workflows: List[Dict]):
        """Build invocation-workflow mapping and collect all invocations efficiently"""
        mapping: Dict = {}
        all_invocations: List = []
        
        if not workflows:
            self.log.debug("No workflows provided for invocation mapping")
            return mapping, all_invocations
        
        try:
            # Process workflows in parallel, but limit concurrency
            semaphore = asyncio.Semaphore(NumericLimits.SEMAPHORE_LIMIT.value)
            
            async def process_workflow(workflow):
                async with semaphore:
                    try:
                        # Add timeout for individual workflow invocation fetching
                        wf_invocations = await asyncio.wait_for(
                            run_in_threadpool(
                                workflow_manager.gi_object.gi.invocations.get_invocations,
                                workflow_id=workflow['id'],
                                limit = NumericLimits.INVOCATION_LIMIT.value
                            ),
                            timeout=NumericLimits.TIMEOUT.value
                        )
                        
                        wf_mapping = {}
                        for inv in wf_invocations or []:
                            wf_mapping[inv['id']] = {
                                'workflow_name': workflow['name'],
                                'workflow_id': workflow['id']
                            }
                        return wf_mapping, wf_invocations
                        
                    except asyncio.TimeoutError:
                        self.log.warning(f"Timeout fetching invocations for workflow {workflow['id']}")
                        return {}, []
                    except Exception as e:
                        self.log.warning(f"Error processing workflow {workflow['id']}: {e}")
                        return {}, []
            
            # Process workflows in parallel
            tasks = [process_workflow(wf) for wf in workflows]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            for result in results:
                if isinstance(result, tuple):
                    mapping.update(result[0])
                    all_invocations.extend(result[1])
                elif isinstance(result, Exception):
                    self.log.warning(f"Workflow processing failed: {result}")
            
            return mapping, all_invocations
            
        except Exception as e:
            self.log.error(f"Error building invocation-workflow mapping: {e}")
            return mapping, all_invocations
    
    async def cleanup_expired_cache_entries(self):
        """Clean up expired cache entries (runs less frequently)"""
        try:
            # Prune active users sorted set as a safety measure
            current_time = int(datetime.now().timestamp())
            hour_ago = current_time - NumericLimits.WARM_TIMESTAMP.value
            active_users_key = "rate_limit:active_users_last_hour"
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.redis.zremrangebyscore, 
                active_users_key, 
                '-inf', 
                hour_ago
            )
            
            # This could be expanded to clean up orphaned cache entries
            # For now, Redis TTL handles most cleanup automatically
            self.log.debug("Cache cleanup task completed")
        except Exception as e:
            self.log.error(f"Error in cache cleanup: {e}")
    
    async def start_cleanup_task(self):
        """Start background cleanup task (runs every hour)"""
        while True:
            try:
                await self.cleanup_expired_cache_entries()
                await asyncio.sleep(NumericLimits.WARM_TIMESTAMP.value)  # Run every hour
            except Exception as e:
                self.log.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(NumericLimits.LONG_SLEEP.value)