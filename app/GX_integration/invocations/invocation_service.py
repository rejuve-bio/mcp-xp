import asyncio
from datetime import datetime, timezone
from typing import List, Optional
import logging

from sys import path
path.append(".")

from fastapi import Response
from starlette.status import HTTP_204_NO_CONTENT

from app.context import current_chat_id
from app.galaxy import GalaxyClient
from app.persistence import MongoStore
from app.api.schemas import invocation, workflow
from app.orchestration.invocation_cache import InvocationCache
from app.orchestration.invocation_tasks import InvocationBackgroundTasks

from app.enumerations import (
    NumericLimits,
    InvocationStates,
    CollectionNames
    )
from app.api.socket_manager import SocketManager

from app.exceptions import (
    InternalServerErrorException,
    NotFoundException
    )

from app.GX_integration.workflows.workflow_manager import WorkflowManager
from app.GX_integration.invocations.data_manager import InvocationDataManager
from app.GX_integration.invocations.tracker import InvocationTracker
from app.GX_integration.invocations.utils import generate_request_hash, log_task_error

class InvocationService:
    """High-level invocation operations - orchestrates the main workflows"""
    
    def __init__(
        self, 
        cache: InvocationCache, 
        background_tasks: InvocationBackgroundTasks,
        mongo_client: MongoStore
        ):
        
        self.cache = cache
        self.background_tasks = background_tasks
        self.inv_data_manager = InvocationDataManager(cache = self.cache, background_tasks = self.background_tasks)
        self.inv_tracker = InvocationTracker(cache = self.cache, redis_client = self.cache.redis, inv_data_manager = self.inv_data_manager)
        self.mongo_client = mongo_client
        self.log = logging.getLogger(__class__.__name__)

    async def _resolve_chat_id(self, username: str, invocation_id: str) -> Optional[str]:
        """Return the chat_id associated with this invocation using write-once semantics.

        If a meta record already exists for `(username, invocation_id)`, its chat_id
        wins — later views from a different chat context do not overwrite it. If no
        meta record exists and the current request has a chat_id, we write one now
        and return it. Otherwise return None.
        """
        meta_key = f"{username}:{invocation_id}"
        existing = await self.mongo_client.get(
            collection_name=CollectionNames.INVOCATION_META.value,
            key=meta_key,
        )
        if isinstance(existing, dict) and existing.get("chat_id"):
            return existing["chat_id"]

        request_chat_id = current_chat_id.get()
        if not request_chat_id:
            return None

        await self.mongo_client.set(
            collection_name=CollectionNames.INVOCATION_META.value,
            key=meta_key,
            value={
                "chat_id": request_chat_id,
                "first_seen_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return request_chat_id

    async def list_invocations(
        self,
        username: str,
        api_key:str,
        galaxy_client: GalaxyClient,
        workflow_manager: WorkflowManager,
        workflow_id: Optional[str] = None,
        history_id: Optional[str]= None,
        ws_manager : Optional[SocketManager] = None
    ) -> invocation.InvocationList:
        """Main method for listing invocations with caching"""   
         
        try:
            # Get deleted invocations list.
            deleted_invocation_ids= await self.cache.get_deleted_invocations(username)
            
            if not deleted_invocation_ids:
                deleted_invocation_ids = await self.mongo_client.get(collection_name= CollectionNames.DELETED_INVOCATIONS.value, key= username)
                if deleted_invocation_ids:
                    await self.cache.add_to_deleted_invocations(username = username, invocation_ids = deleted_invocation_ids)
                else:
                    deleted_invocation_ids = []
            
            # Step 1: Request deduplication check
            request_hash = generate_request_hash(username, workflow_id, history_id)
            if await self.cache.is_duplicate_request(username, request_hash):
                self.log.info("Duplicate request detected.")
            
            # Step 2: Try to get cached response first
            cached_response = await self.cache.get_response_cache(
                username, workflow_id, history_id
            )
            if cached_response:
                self.log.info("Filtering and serving response from cache")
                invocations = cached_response.get("invocations", [])
                filtered_cache_response = [
                    inv for inv in invocations if inv["id"] not in deleted_invocation_ids
                ]
                response_data = {
                    "invocations": filtered_cache_response,
                }

                return invocation.InvocationList(**response_data)
            
            # Serialize concurrent calls to the locked section for the same request.
            async with self.cache.acquire_lock(f"list_invocations:{username}:{request_hash}"):
                
                # Step 3: Initialize Galaxy client and workflow manager give it time for new invocations to be registerd under list
                # TODO: sleeping is a temporary solution, find a permanent solution for this.
                await asyncio.sleep(NumericLimits.SHORT_SLEEP.value)
                
                # Step 4: Fetch data with parallel processing and caching
                invocations_data, workflows_data = await self.inv_data_manager.fetch_core_data(
                    username = username,
                    workflow_manager = workflow_manager,
                    workflow_id = workflow_id,
                    history_id = history_id
                )
                
                if not invocations_data:
                    self.log.warning("No invocations data retrieved found.")
                    return invocation.InvocationList(invocations=[])
                
                structured_invocation_ids = await self.mongo_client.get(
                    collection_name=CollectionNames.INVOCATION_IDS.value,
                    key = username
                    ) or []
                               
                # filter out already structured invocations...
                excluded_ids = set(deleted_invocation_ids) | set(structured_invocation_ids)
                
                invocations_data = [
                inv for inv in invocations_data 
                if inv["id"] not in excluded_ids
                ]
                
                # Get structured invocations.
                structured_invocations = await self.mongo_client.get(
                    collection_name=CollectionNames.INVOCATION_LISTS.value,
                    key=username
                ) or []
                
                # If no new invocaitons exist then return as is.
                if not invocations_data:
                    self.log.debug("Returning invocation list from database.")
                    return invocation.InvocationList(invocations = structured_invocations)
                    
                # Step 5: Get invocation-workflow mapping (cached or build)
                # TODO: Need to find a better way to find a invocations workflow details.
                workflow_mapping, all_invocations = await self.background_tasks.build_invocation_workflow_mapping(
                    workflow_manager = workflow_manager,
                    workflows = workflows_data
                    )
                
                if workflow_mapping:
                    await self.cache.set_invocation_workflow_mapping(username, workflow_mapping)
                if not workflow_id and not history_id:
                    if all_invocations:
                        await self.cache.set_invocations_cache(username, all_invocations, filters={"workflow_id": None, "history_id": None})
                
                    # Step 6: Filter out deleted and already structured invocations
                    invocations_data = [inv for inv in all_invocations if inv.get("id") not in excluded_ids]
                else:
                    invocations_data = [
                        inv for inv in invocations_data 
                        if inv.get('id') not in excluded_ids
                    ]
                    
                # Step 7: Format invocations and filter deleted invocatoins optimized processing
                invocation_list = await self.inv_tracker.format_invocations(
                username = username, 
                invocations_data = invocations_data,
                workflow_mapping = workflow_mapping,
                workflow_manager = workflow_manager,
                mongo_client = self.mongo_client
                )
                                
                # Step 8: Build response
                response_data = {
                    "invocations": [inv.model_dump() for inv in invocation_list],
                }
                
                # Extendthe stored nd structured invocaitons back to the result..
                response_data["invocations"].extend(structured_invocations)

                # Step 9: Cache the response
                await self.cache.set_response_cache(
                    username, response_data, workflow_id, history_id
                )
                
                # TODO: Persistence should also extend to handle filters (worklfow_id and history_id)
                # now we are just storing invocaiton info witout filters
                async def store_inv():
                    """ Store structured invocation info """
                    
                    # Persist only complete or failed invocations.
                    store_invocations = [inv.model_dump() for inv in invocation_list if inv.state in [InvocationStates.FAILED.value, InvocationStates.COMPLETE.value]]
                    if store_invocations:
                        self.log.debug(f"found {len(store_invocations)} invocations to persist. storing structured invocations")
                        await self.mongo_client.add_to_set(
                            collection_name=CollectionNames.INVOCATION_IDS.value, 
                            key = username, 
                            elements = [inv["id"] for inv in store_invocations]
                            )
                        if await self.mongo_client.exists(collection_name= CollectionNames.INVOCATION_LISTS.value, key = username):
                            await self.mongo_client.extend(
                                collection_name = CollectionNames.INVOCATION_LISTS.value,
                                key = username,
                                new_values = store_invocations
                                )
                        else:
                            await self.mongo_client.set(
                                collection_name = CollectionNames.INVOCATION_LISTS.value,
                                key = username,
                                value = store_invocations
                                )
                    
                # Start tracking pending invocaitons early.
                async def track_pending():
                    """ Start tracking pending invocations."""
                    
                    pending_invocations = [inv.id for inv in invocation_list if inv.state == InvocationStates.PENDING.value]
                    if pending_invocations:
                        self.log.debug(f"Pending invocations found: {len(pending_invocations)}, tracking them now.")

                    for i in range(0, len(pending_invocations), NumericLimits.BATCH_LIMIT.value):
                        batch = pending_invocations[i : i + NumericLimits.BATCH_LIMIT.value]

                        results = await asyncio.gather(
                            *(
                                self.get_invocation_result(
                                    invocation_id=inv_id,
                                    username=username,
                                    api_key=api_key,
                                    galaxy_client=galaxy_client,
                                    workflow_manager=workflow_manager,
                                    ws_manager=ws_manager,
                                )
                                for inv_id in batch
                            ),
                            return_exceptions=True,
                        )

                        for inv_id, result in zip(batch, results):
                            if isinstance(result, Exception):
                                self.log.error(
                                    "Invocation failed",
                                    extra={"invocation_id": inv_id},
                                )
            
                # Persist structured invocation info in the background.
                asyncio.create_task(store_inv())
                
                # Start tracking pending workflow invocaiton (limited to 2 per tracking step so it wont explode).
                asyncio.create_task(track_pending())
                
                self.log.info(f"Successfully retrieved invocations (total: {len(response_data['invocations'])})")
                return invocation.InvocationList(**response_data)
            
        except Exception as e:
            self.log.error(f"Error in list_invocations: {e}", exc_info=True)
            # Try to return partial results if possible
            try:
                return await self.handle_partial_failure(
                    username = username,
                    workflow_manager = workflow_manager,
                    mongo_client = self.mongo_client,
                    workflow_id = workflow_id,
                    history_id = history_id
                    )
            except Exception as fallback_error:
                self.log.error(f"Fallback also failed: {fallback_error}")
                raise InternalServerErrorException("Failed to list invocations")
    
    async def get_invocation_result(
        self,
        invocation_id: str,
        username: str,
        api_key: str,
        galaxy_client: GalaxyClient,
        workflow_manager: WorkflowManager,
        ws_manager: Optional[SocketManager] = None
    ) -> invocation.InvocationResult:
        """Get detailed invocation result with tracking"""
        try:
            # Step 1: Check if deleted
            deleted = await self.cache.get_deleted_invocations(username)
            
            if not deleted:
                deleted = await self.mongo_client.get(collection_name = CollectionNames.DELETED_INVOCATIONS.value, key = username)
                
                if deleted:
                    await self.cache.add_to_deleted_invocations(username = username, invocation_ids = deleted)
                else:
                    deleted = []
                
            if invocation_id in deleted:
                raise NotFoundException("Invocation not found")

            # Step 2: Check cache for full result
            cached_result = await self.cache.get_invocation_result(username, invocation_id)
            if cached_result:
                self.log.info("fetching workflow invocation result from cache.")
                return invocation.InvocationResult(**cached_result)
            
            stored_result = await self.mongo_client.get(collection_name = CollectionNames.INVOCATION_RESULTS.value, key = f"{username}:{invocation_id}")
            
            if stored_result:
                await self.cache.set_invocation_result(username = username, invocation_id = invocation_id, result = stored_result)
                self.log.info("fetching workflow invocation result from database.")
                return invocation.InvocationResult(**stored_result)

            # Step 3: Fetch invocation for preliminary info

            _invocation, invocation_details = await asyncio.gather(
                asyncio.to_thread(
                    workflow_manager.gi_object.invocations.get,
                    id_=invocation_id
                ),
                asyncio.to_thread(
                    workflow_manager.gi_object.gi.invocations.show_invocation,
                    invocation_id =invocation_id
                )
            )
            if not _invocation:
                raise NotFoundException("Invocation not found")

            # Fetch common details for partial response
            mapping = await self.cache.get_invocation_workflow_mapping(username)
            if not mapping:
                self.log.warning("workflow to invocation map not found, warming user cache.")
                await self.background_tasks.warm_user_cache(token = "dummytoken", api_key=api_key) # using a dummy token to fill the functionality
                mapping = await self.cache.get_invocation_workflow_mapping(username)
                
            if not mapping or invocation_id not in mapping:
                raise InternalServerErrorException("could not find workflow details with the invocation id inputted.")
            
            stored_workflow_id = mapping.get(invocation_id, {}).get('workflow_id')

            # Retrieve workflow description. It's usually already in cache
            # from the /api/workflows warm-up path, but for a newly-scheduled
            # invocation whose result is fetched BEFORE the workflows cache
            # is warmed for this workflow_id, that cache is empty and we'd
            # return 500 on the very first fetch. Mirror the same warm-then-
            # retry fallback used above for the invocation→workflow mapping
            # (see lines 319-323): if the initial cache lookup misses, warm
            # once and read again. Prevents the "first call errors, retry
            # passes" pattern the platform kept surfacing.
            def _resolve_workflow_description(workflow_list):
                if not workflow_list:
                    return None
                for _workflow in workflow_list:
                    try:
                        cached_workflow_id = _workflow.get("source_workflow_id", _workflow.get("id"))
                        if stored_workflow_id == cached_workflow_id:
                            return workflow.WorkflowListItem(**_workflow)
                    except Exception as e:
                        self.log.warning(f"couldn't retrieve from cache: {e}")
                return None

            workflow_description_list = await self.cache.get_workflows_cache(username)
            workflow_description = _resolve_workflow_description(workflow_description_list)

            if workflow_description is None:
                self.log.warning(
                    "workflow description cache miss for workflow_id "
                    f"{stored_workflow_id}, warming user cache."
                )
                await self.background_tasks.warm_user_cache(
                    token="dummytoken",  # same dummy-token trick used above
                    api_key=api_key,
                )
                workflow_description_list = await self.cache.get_workflows_cache(username)
                workflow_description = _resolve_workflow_description(workflow_description_list)

            if workflow_description is None:
                raise InternalServerErrorException(
                    "Could not locate workflow description in cache after warm attempt."
                )

            inputs_formatted = await self.inv_data_manager.structure_inputs(inv=invocation_details, workflow_manager=workflow_manager)
            
            # Compute preliminary state (without full check)
            state = invocation_details.get("state")
            if state in ["cancelled", "failed"]:
                inv_state = InvocationStates.FAILED.value
            elif state in ["requires_materialization", "ready", "new", "scheduled"]:
                inv_state = InvocationStates.PENDING.value
            else:
                self.log.warning(f"Unknown invocation state: {state}")
                inv_state = InvocationStates.FAILED.value
            
            # If pending-like, trigger background tracking if not already
            tracking_key = f"tracking:{username}:{invocation_id}"
            
            resolved_chat_id = await self._resolve_chat_id(username, invocation_id)

            if inv_state == InvocationStates.PENDING.value:
                invocation_result = {
                    "invocation_id" : invocation_details.get("id"),
                    "state" : inv_state,
                    "history_id" : invocation_details.get("history_id"),
                    "create_time" : invocation_details.get("create_time"),
                    "update_time" :invocation_details.get("update_time"),
                    "inputs" : inputs_formatted,
                    "result" : [],
                    "workflow" : workflow_description.model_dump(),
                    "report" : None,
                    "chat_id" : resolved_chat_id,
                }
                
                await self.cache.set_invocation_result(
                    username = username, 
                    invocation_id = invocation_details.get("id"),
                    result = invocation_result,
                    )
                
                set_result = await asyncio.to_thread(
                    self.cache.redis.set, tracking_key, "1", ex=NumericLimits.BACKGROUND_INVOCATION_TRACK.value, nx=True
                )
                if set_result: 
                    
                    background_task = asyncio.create_task(
                        self.inv_tracker.background_track_and_cache(
                            invocation_id = invocation_id,
                            galaxy_client = galaxy_client,
                            workflow_manager = workflow_manager,
                            mongo_client = self.mongo_client,
                            history_id = invocation_details.get("history_id"),
                            create_time = invocation_details.get("create_time"),
                            last_update_time = invocation_details.get("update_time"),
                            inputs_formatted = inputs_formatted,
                            workflow_description= workflow_description,
                            ws_manager = ws_manager,
                            tracking_key = tracking_key
                            )
                        )
                    background_task.add_done_callback(lambda t: log_task_error(t, task_name="track invocation"))
            else:
                self.log.info(f"Loading invocation failure report for invocation {invocation_id}")
                invocation_report = await self.inv_data_manager.report_invocation_failure(galaxy_client = galaxy_client, invocation_id = invocation_id)
                invocation_result = {
                    "invocation_id" : invocation_details.get("id"),
                    "state" : inv_state,
                    "history_id" : invocation_details.get("history_id"),
                    "create_time" : invocation_details.get("create_time"),
                    "update_time" :invocation_details.get("update_time"),
                    "inputs" : inputs_formatted,
                    "result" : [],
                    "workflow" : workflow_description.model_dump(),
                    "report" : invocation_report,
                    "chat_id" : resolved_chat_id,
                }
                
                await self.cache.set_invocation_result(
                    username = username, 
                    invocation_id = invocation_details.get("id"),
                    result = invocation_result,
                    )
                
                asyncio.create_task(
                    self.mongo_client.set(
                        collection_name= CollectionNames.INVOCATION_RESULTS.value, 
                        key = f"{username}:{invocation_id}", 
                        value = invocation_result
                        )
                    )
            
            return invocation.InvocationResult(**invocation_result)
        except Exception as e:
            self.log.error(f"Error getting invocation result: {e}")
            raise InternalServerErrorException("Error getting invocation result")

        
    
    async def delete_invocations(
        self,
        invocation_ids: List[str],
        username: str,
        workflow_manager: WorkflowManager
    ) -> None:
        """Mark invocations as deleted and clean up resources"""
        try:
            
            # Parse comma-separated string into list
            ids_list = [i.strip() for i in invocation_ids.split(",") if i.strip()]

            # Mark as deleted using cache method and store deleted invocation.
            await asyncio.gather(
                self.cache.add_to_deleted_invocations(username, ids_list),
                self.mongo_client.add_to_set(collection_name = CollectionNames.DELETED_INVOCATIONS.value, key = f"deleted_invocations:{username}", elements = ids_list)
            )
            
            
            # Spawn async deletion tasks
            background_task = asyncio.create_task(
                self.inv_tracker.cancel_invocation_and_delete_data(
                    invocation_ids = ids_list, 
                    workflow_manager = workflow_manager,
                    username = username,
                    mongo_client = self.mongo_client
                    )
                )
            background_task.add_done_callback(lambda t: log_task_error(t, task_name="Invocation cancelling and data deletion"))

            return Response(status_code=HTTP_204_NO_CONTENT)
        
        except Exception as e:
            raise InternalServerErrorException("Failed to delete invocations")
    
    async def cancel_invocation(
        self,
        invocation_id: str,
        username: str,
        workflow_manager: WorkflowManager
    ) -> None:
        """Cancel a running invocation; preserves data, updates state to Failed."""
        try:
            deleted = await self.cache.get_deleted_invocations(username)
            if not deleted:
                deleted = await self.mongo_client.get(
                    collection_name=CollectionNames.DELETED_INVOCATIONS.value, key=username
                ) or []
            if invocation_id in deleted:
                raise NotFoundException("Invocation not found")

            await self.inv_tracker.cancel_invocation(
                invocation_id=invocation_id,
                workflow_manager=workflow_manager,
                username=username,
                mongo_client=self.mongo_client
            )

            cached_result = await self.cache.get_invocation_result(username, invocation_id)
            if cached_result:
                cached_result["state"] = InvocationStates.FAILED.value
                asyncio.create_task(
                    self.cache.set_invocation_result(username, invocation_id, cached_result)
                )

            asyncio.create_task(self.cache.delete_response_cache(username))

            return Response(status_code=HTTP_204_NO_CONTENT)

        except NotFoundException:
            raise
        except Exception as e:
            self.log.error(f"Failed to cancel invocation {invocation_id}: {e}")
            raise InternalServerErrorException("Failed to cancel invocation")

    async def handle_partial_failure(
        self,
        username: str,
        workflow_manager: WorkflowManager,
        mongo_client: MongoStore,
        workflow_id: str | None,
        history_id: str | None
    ) -> invocation.InvocationList:
        """Handle partial failures by returning minimal data"""

        try:           
            if workflow_id and history_id:
                invocations = await asyncio.to_thread(
                    workflow_manager.gi_object.gi.invocations.get_invocations,
                    workflow_id=workflow_id,
                    history_id = history_id
                )  
            elif workflow_id:
                invocations = await asyncio.to_thread(
                    workflow_manager.gi_object.gi.invocations.get_invocations,
                    workflow_id=workflow_id
                )
            elif history_id:
                invocations = await asyncio.to_thread(
                    workflow_manager.gi_object.gi.invocations.get_invocations,
                    history_id=history_id
                )
            else:
                invocations = await asyncio.to_thread(
                    workflow_manager.gi_object.gi.invocations.get_invocations
                )
            
            # Create minimal invocation list without workflow names
            state_tasks = [
                self.inv_tracker.map_invocation_state(
                    username=username,
                    raw_state=inv.get('state'),
                    invocation_id=inv.get('id'),
                    workflow_manager=workflow_manager,
                    mongo_client = mongo_client
                )
                for inv in invocations
            ]
            states = await asyncio.gather(*state_tasks)

            # Build the list in one go
            invocation_list = [
                invocation.InvocationListItem(
                    id=inv.get('id'),
                    workflow_name="Unknown (partial failure)",
                    workflow_id='Unknown (partial failure)',
                    history_id=inv.get('history_id'),
                    state=state,
                    outputs = [],
                    create_time=inv.get('create_time'),
                    update_time=inv.get('update_time')
                )
                for inv, state in zip(invocations, states)
    ]
            return invocation.InvocationList(
                invocations=invocation_list
            )
            
        except Exception as e:
            self.log.error(f"Partial failure handling also failed: {e}")
            raise

    async def list_invocations_by_chat(
        self,
        username: str,
        chat_id: str,
    ) -> invocation.InvocationList:
        """List invocations tagged with `chat_id` for the given `username`.

        Cross-user scoping is enforced by the meta collection key prefix
        (`{username}:{invocation_id}`), so two users sharing the same chat_id
        string still see disjoint results. Full invocation shape is hydrated
        from the persisted INVOCATION_LISTS entry when present; otherwise a
        minimal placeholder is returned so pending/unlisted invocations still
        surface.
        """
        try:
            meta_hits = await self.mongo_client.find_by_value_field(
                collection_name=CollectionNames.INVOCATION_META.value,
                field_path="chat_id",
                value=chat_id,
                key_prefix=username,
            )
            if not meta_hits:
                return invocation.InvocationList(invocations=[])

            prefix = f"{username}:"
            invocation_ids = {
                hit["key"][len(prefix):]
                for hit in meta_hits
                if isinstance(hit.get("key"), str) and hit["key"].startswith(prefix)
            }

            stored_list = await self.mongo_client.get(
                collection_name=CollectionNames.INVOCATION_LISTS.value,
                key=username,
            ) or []

            items: List[invocation.InvocationListItem] = []
            seen_ids = set()
            for stored in stored_list:
                stored_id = stored.get("id")
                if stored_id in invocation_ids:
                    stored = {**stored, "chat_id": chat_id}
                    try:
                        items.append(invocation.InvocationListItem(**stored))
                        seen_ids.add(stored_id)
                    except Exception as parse_exc:
                        self.log.warning(
                            f"Skipping malformed stored invocation {stored_id}: {parse_exc}"
                        )

            missing_ids = invocation_ids - seen_ids
            for missing_id in missing_ids:
                cached_result = await self.cache.get_invocation_result(username, missing_id)
                if not cached_result:
                    cached_result = await self.mongo_client.get(
                        collection_name=CollectionNames.INVOCATION_RESULTS.value,
                        key=f"{username}:{missing_id}",
                    )
                if not cached_result:
                    continue
                workflow_info = cached_result.get("workflow") or {}
                try:
                    items.append(invocation.InvocationListItem(
                        id=missing_id,
                        workflow_name=workflow_info.get("name", "Unknown"),
                        workflow_id=workflow_info.get("id", ""),
                        history_id=cached_result.get("history_id", ""),
                        state=cached_result.get("state", InvocationStates.PENDING.value),
                        outputs=[],
                        create_time=cached_result.get("create_time", ""),
                        update_time=cached_result.get("update_time", ""),
                        chat_id=chat_id,
                    ))
                except Exception as parse_exc:
                    self.log.warning(
                        f"Skipping unlisted invocation {missing_id}: {parse_exc}"
                    )

            return invocation.InvocationList(invocations=items)
        except Exception as e:
            self.log.error(f"Failed to list invocations by chat_id: {e}", exc_info=True)
            raise InternalServerErrorException("Failed to list invocations by chat_id")
