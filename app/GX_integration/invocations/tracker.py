import redis
import asyncio
from typing import List
import logging

from sys import path
path.append(".")

from fastapi.concurrency import run_in_threadpool
from bioblend.galaxy.objects.wrappers import (
    HistoryDatasetAssociation,
    HistoryDatasetCollectionAssociation
    )

from app.galaxy import GalaxyClient
from app.persistence import MongoStore
from app.GX_integration.workflows.workflow_manager import WorkflowManager
from app.GX_integration.invocations.output_indexer import OutputIndexer
from app.api.schemas import invocation
from app.api.socket_manager import SocketManager
from app.orchestration.invocation_cache import InvocationCache
from app.enumerations import (
    NumericLimits,
    JobState,
    InvocationStates,
    CollectionNames,
    SocketMessageEvent,
    SocketMessageType
    )

from app.GX_integration.invocations.data_manager import InvocationDataManager
from app.GX_integration.invocations.utils import GALAXY_PENDING_INVOCATION_STATES, resolve_galaxy_invocation_state

# invocation_tracker.py
class InvocationTracker:
    """Handles state tracking, background monitoring, and cleanup operations"""
    
    def __init__(self, cache: InvocationCache, redis_client: redis.Redis, inv_data_manager:InvocationDataManager):
        self.cache = cache
        self.redis_client = redis_client
        self.inv_data_manager = inv_data_manager
        self.log = logging.getLogger(__class__.__name__)
        

    async def format_invocations(
        self,
        username: str,
        invocations_data: list[dict],
        workflow_mapping: dict[str, dict],
        workflow_manager: WorkflowManager,
        mongo_client: MongoStore
    ) -> list[invocation.InvocationListItem]:
        """Format invocations with optimized state mapping"""
    
        self.log.info(f"length og mappings: {len(workflow_mapping)}  and invocations: {len(invocations_data)}")
        length = 0
        filtered_invocations = []
        for inv in invocations_data:
            inv_id = inv.get('id')
            
            # Get workflow info from mapping
            workflow_info = workflow_mapping.get(inv_id, {})
            workflow_name = workflow_info.get('workflow_name', None)
            workflow_id = workflow_info.get("workflow_id", None)
            
            # Skip if no workflow name : Likely due to the invocation being a subworkflow invocation or workflow being deleted.
            # NOTE: Maybe in the future if invocation are still needed even after the workflow has been deleted, we can refactor implemenation.
            if not workflow_name:
                length += 1
                continue
            
            filtered_invocations.append((inv, workflow_name, workflow_id))

        # Gather all state mappings concurrently
        state_tasks = [
            self.map_invocation_state(
                username=username,
                raw_state=inv.get('state'),
                invocation_id=inv.get("id"),
                workflow_manager=workflow_manager,
                mongo_client = mongo_client
            )
            for inv, _, _ in filtered_invocations
        ]
        states = await asyncio.gather(*state_tasks)

        # Build the final list
        invocation_list = [
            invocation.InvocationListItem(
                id=inv.get('id'),
                workflow_name=workflow_name,
                workflow_id=workflow_id,
                history_id=inv.get('history_id'),
                state=state,
                outputs=[],
                create_time=inv.get('create_time'),
                update_time=inv.get('update_time')
            )
            for (inv, workflow_name, workflow_id), state in zip(filtered_invocations, states)
        ]
                
        self.log.debug(f"skipped {length} invocation becasue workflowname was missing.")
        
        return invocation_list

    
    async def map_invocation_state(
        self,
        username: str,
        raw_state: str,
        invocation_id: str,
        workflow_manager: WorkflowManager,
        mongo_client: MongoStore
    ) -> str:
        """Map Galaxy invocation states to user-friendly states"""
        
        # Get invocation state if saved for accurate state.
        cached_state = await self.cache.get_invocation_state(username, invocation_id)
        if cached_state:
            self.log.debug(f"Getting cached invocation state: {cached_state}")
            return cached_state
        
        stored_state = await mongo_client.get_element(collection_name = CollectionNames.INVOCATION_STATES.value, key = username, field = invocation_id)
        
        if stored_state:
            self.log.debug("Getting stored invocation state.")
            return stored_state

        # Pending Galaxy states get a deeper look at their step jobs.
        if raw_state in GALAXY_PENDING_INVOCATION_STATES:
            return await self.compute_deep_invocation_state(
                workflow_manager = workflow_manager,
                mongo_client = mongo_client,
                invocation_id = invocation_id,
                username = username
            )

        try:
            inv_state = await resolve_galaxy_invocation_state(
                raw_state = raw_state,
                invocation_id = invocation_id,
                gi = workflow_manager.gi_object.gi
            )
        except Exception as e:
            # One failed lookup must not fail the whole invocation list.
            self.log.error(f"Error resolving state for {invocation_id}: {e}")
            return InvocationStates.PENDING.value

        # Final states are stored so the jobs summary is fetched only once.
        if inv_state != InvocationStates.PENDING.value:
            asyncio.create_task(mongo_client.set_element(collection_name = CollectionNames.INVOCATION_STATES.value, key = username, field = invocation_id, value = inv_state))
            asyncio.create_task(self.cache.set_invocation_state(username, invocation_id, inv_state))

        return inv_state
        
    async def compute_deep_invocation_state(
        self,
        workflow_manager: WorkflowManager,
        mongo_client: MongoStore,
        invocation_id: str,
        username: str
    ) -> str:
        """Compute detailed invocation state from step jobs"""
        try:
            # Fetch step jobs summary
            step_jobs = await asyncio.to_thread(
                workflow_manager.gi_object.gi.invocations.get_invocation_step_jobs_summary,
                invocation_id=invocation_id
            )

            all_ok = True
            has_failed = False

            for step in step_jobs:
                states = step.get('states', {})

                ok_count = states.get(JobState.OK.value, 0)
                skipped_count = states.get(JobState.SKIPPED.value, 0)
                failed_count = states.get(JobState.FAILED.value, 0)
                error_count = states.get(JobState.ERROR.value, 0)
                running_count = states.get(JobState.RUNNING.value, 0)

                total_jobs = sum(states.values())
                completed_jobs = ok_count + skipped_count
                failed_jobs = failed_count + error_count
                all_jobs_failed = (failed_jobs == total_jobs and total_jobs > 0)

                # Set flags
                if all_jobs_failed:
                    has_failed = True

                # Determine step state (similar to _analyze_step_jobs)
                if all_jobs_failed:
                    step_state = JobState.ERROR.value
                elif completed_jobs == total_jobs:
                    step_state = JobState.OK.value
                elif running_count > 0:
                    step_state = JobState.RUNNING.value
                elif failed_jobs > 0 and failed_jobs < total_jobs:
                    step_state = JobState.RUNNING.value
                else:
                    step_state = InvocationStates.PENDING.value

                if step_state != JobState.OK.value:
                    all_ok = False

            if has_failed:
                inv_state = InvocationStates.FAILED.value
            elif all_ok:
                inv_state = InvocationStates.COMPLETE.value
            else:
                inv_state = InvocationStates.PENDING.value
            
            asyncio.create_task(mongo_client.set_element(collection_name = CollectionNames.INVOCATION_STATES.value, key = username, field = invocation_id, value = inv_state))
            asyncio.create_task(self.cache.set_invocation_state(username, invocation_id, inv_state))
            
            return inv_state
            
        except Exception as e:
            self.log.error(f"Error computing deep state for {invocation_id}: {e}")

    
    async def background_track_and_cache(
        self,
        invocation_id: str,
        galaxy_client: GalaxyClient,
        workflow_manager: WorkflowManager,
        mongo_client: MongoStore,
        history_id: str,
        create_time: str,
        last_update_time: str,
        inputs_formatted: dict,
        workflow_description,
        ws_manager: SocketManager,
        tracking_key: str
    ):
        """Background task to track invocation and cache results"""
        username = galaxy_client.whoami
        output_indexer = OutputIndexer(username = username, galaxy_client = galaxy_client, cache = self.cache, mongo_client = mongo_client, ws_manager = ws_manager)
        
        try:
    
            _invocation = await run_in_threadpool(
                galaxy_client.gi_object.invocations.get,
                id_=invocation_id
            )
            if not _invocation:
                self.log.error(f"Invocation {invocation_id} not found in background")
                return

            outputs, inv_state, update_time = await workflow_manager.track_invocation(
                invocation=_invocation,
                tracker_id=invocation_id,
                ws_manager=ws_manager,
                invocation_check=True
            )
            
            # Set invocation to cache and persist as well.
        
            await asyncio.gather(
                self.cache.set_invocation_state(username, invocation_id, inv_state),
                mongo_client.set_element(collection_name = CollectionNames.INVOCATION_STATES.value, key = username, field = invocation_id, value = inv_state)
                )

            if inv_state == InvocationStates.FAILED.value:
                
                invocation_report = await self.inv_data_manager.report_invocation_failure(galaxy_client, invocation_id)
                workflow_result, _ = await self.inv_data_manager.structure_outputs(
                _invocation=_invocation, outputs=outputs, workflow_manager=workflow_manager, failure = True
                )
            else:
                workflow_result, invocation_report = await self.inv_data_manager.structure_outputs(
                _invocation=_invocation, outputs=outputs, workflow_manager=workflow_manager
                )
                
            result_dict = {
                "invocation_id": invocation_id,
                "state": inv_state,
                "history_id": history_id,
                "create_time": create_time,
                "update_time": update_time if update_time else last_update_time,
                "inputs": inputs_formatted,
                "result": workflow_result,
                "workflow": workflow_description.model_dump(),
                "report": invocation_report
            }
            
            asyncio.create_task(self.cache.set_invocation_result(username, invocation_id, result_dict))
            asyncio.create_task(mongo_client.set(collection_name = CollectionNames.INVOCATION_RESULTS.value, key = f"{username}:{invocation_id}", value = result_dict))
            
            self.log.info("Invocation results are complete and ready.")
            if ws_manager:
                ws_data = {
                    "type": SocketMessageType.INVOCATION_COMPLETE.value,
                    "payload": {"message": "Invocation results are complete and ready."}
                }
                await ws_manager.broadcast(
                    event=SocketMessageEvent.workflow_execute.value,
                    data=ws_data,
                    tracker_id=invocation_id
                )
            
            # Index invocation outputs
            await output_indexer.index_datasets_and_register(invocation_result= result_dict)


        except Exception as e:
            self.log.error(f"Error in background tracking for {invocation_id}: {e}")
            result_dict = {
                "invocation_id": invocation_id,
                "state": InvocationStates.FAILED.value,
                "history_id": history_id,
                "create_time": create_time,
                "update_time": last_update_time,
                "inputs": inputs_formatted,
                "result": [],
                "workflow": workflow_description.model_dump(),
                "report": None
            }
            
            await asyncio.gather(
                self.cache.set_invocation_result(username, invocation_id, result_dict),
                mongo_client.set(collection_name = CollectionNames.INVOCATION_RESULTS.value, key = f"{username}:{invocation_id}", value = result_dict)
                )

        finally:
            await asyncio.to_thread(self.redis_client.delete, tracking_key)

    
    async def cancel_invocation_and_delete_data(
        self,
        invocation_ids: List[str],
        workflow_manager: WorkflowManager,
        username: str,
        mongo_client: MongoStore
    ):
        """Cancel invocations and delete associated data"""
        initial_deletions = [
            mongo_client.delete_specific(
                collection_name=CollectionNames.INVOCATION_LISTS.value,
                key=username,
                elements=invocation_ids
            ),
            mongo_client.remove_from_set(
                collection_name=CollectionNames.INVOCATION_IDS.value,
                key=username,
                elements=invocation_ids
            ),
            mongo_client.delete_elements(
                collection_name= CollectionNames.INVOCATION_STATES.value,
                key = username,
                fields=invocation_ids
            )
        ]

        semaphore = asyncio.Semaphore(NumericLimits.SEMAPHORE_LIMIT.value)

        async def handle_invocation(invocation_id: str):
            async with semaphore:
                # Get the invocation object
                _invocation = await asyncio.to_thread(
                    workflow_manager.gi_object.invocations.get,
                    id_=invocation_id
                )

                # Cancel if not in a terminal state
                state = _invocation.state
                terminal_states = {'cancelled', 'failed', 'scheduled', 'error'}  # 'scheduled' often means completed
                if state not in terminal_states:
                    await asyncio.to_thread(
                        workflow_manager.gi_object.gi.invocations.cancel_invocation,
                        invocation_id=invocation_id
                    )
                    # Short wait for cancellation to propagate (or implement polling for state change)
                    await asyncio.sleep(NumericLimits.SHORT_SLEEP.value)

                # TODO: Check cache first before doing a full track for output id extraction 
                # Get outputs and purge datasets/collections to free space
                outputs, _, _ = await workflow_manager.track_invocation(
                    invocation=_invocation,
                    tracker_id=invocation_id,
                    ws_manager=None,
                    invocation_check=True
                )
                
                # Prepare concurrent deletions for invocation outputs
                delete_datasets = [asyncio.to_thread(ds.delete, purge=True) for ds in outputs if isinstance(ds, HistoryDatasetAssociation)]
                delete_collections = [asyncio.to_thread(ds.delete) for ds in outputs if isinstance(ds, HistoryDatasetCollectionAssociation)]
                
                # Mongo and cache deletions
                mongo_del = mongo_client.delete(
                    collection_name=CollectionNames.INVOCATION_RESULTS.value,
                    key=f"{username}:{invocation_id}",
                )
                cache_del = self.cache.delete_invocation_state(username, invocation_id=invocation_id)
                
                # Gather all per-invocation deletions concurrently
                await asyncio.gather(
                    *delete_datasets,
                    *delete_collections,
                    mongo_del,
                    cache_del,
                    return_exceptions=True
                )

        try:
            # Create handlers for each invocation (limited by semaphore internally)
            invocation_handlers = [handle_invocation(invocation_id) for invocation_id in invocation_ids]
            
            # Gather initial deletions and invocation handlers concurrently
            results = await asyncio.gather(*initial_deletions, *invocation_handlers, return_exceptions=True)
            
            exceptions = [r for r in results if isinstance(r, Exception)]

            if exceptions:
                for exc in exceptions:
                    self.log.error(
                        "Invocation deletion failure.",
                        exc_info=exc
                    )
            else:
             self.log.info(f"Invocation {invocation_ids} deletion complete.")

        except Exception as e:
            self.log.error(f"Error while deleting invocation: {e}")

    async def cancel_invocation(
        self,
        invocation_id: str,
        workflow_manager: WorkflowManager,
        username: str,
        mongo_client: MongoStore
    ):
        """Cancel a running invocation without deleting any data."""
        _invocation = await asyncio.to_thread(
            workflow_manager.gi_object.invocations.get,
            id_=invocation_id
        )

        terminal_states = {'cancelled', 'failed', 'scheduled', 'error'}
        if _invocation.state not in terminal_states:
            await asyncio.to_thread(
                workflow_manager.gi_object.gi.invocations.cancel_invocation,
                invocation_id=invocation_id
            )

        await asyncio.gather(
            self.cache.set_invocation_state(username, invocation_id, InvocationStates.FAILED.value),
            mongo_client.set_element(
                collection_name=CollectionNames.INVOCATION_STATES.value,
                key=username,
                field=invocation_id,
                value=InvocationStates.FAILED.value
            )
        )