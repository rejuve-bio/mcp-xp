"""Galaxy invocation state -> InvocationStates, on both /api/invocation paths (issue #56)."""
from sys import path

path.append(".")

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.enumerations import InvocationStates
from app.GX_integration.invocations.invocation_service import InvocationService
from app.GX_integration.invocations.tracker import InvocationTracker

INVOCATION_ID = "inv1"
USERNAME = "alice"

PENDING = InvocationStates.PENDING.value
COMPLETE = InvocationStates.COMPLETE.value
FAILED = InvocationStates.FAILED.value

# (raw Galaxy invocation state, job state counts, expected InvocationStates value)
CASES = [
    ("completed", {"ok": 3, "skipped": 1}, COMPLETE),
    ("completed", {}, COMPLETE),
    ("completed", {"ok": 2, "error": 1}, FAILED),
    ("completed", {"ok": 2, "failed": 1}, FAILED),
    ("completed", {"ok": 2, "error": 0}, COMPLETE),
    ("failed", {}, FAILED),
    ("cancelled", {}, FAILED),
    ("cancelling", {}, FAILED),
    ("bogus", {}, PENDING),
    (None, {}, PENDING),
]


def make_workflow_manager(raw_state, job_counts):
    workflow_manager = MagicMock()
    workflow_manager.gi_object.invocations.get.return_value = MagicMock()
    workflow_manager.gi_object.gi.invocations.show_invocation.return_value = {
        "id": INVOCATION_ID,
        "state": raw_state,
        "history_id": "hist1",
        "create_time": "2026-01-01T00:00:00",
        "update_time": "2026-01-01T00:01:00",
    }
    workflow_manager.gi_object.gi.invocations.get_invocation_summary.return_value = {
        "states": job_counts, "model": "WorkflowInvocation", "id": INVOCATION_ID, "populated_state": "ok",
    }
    return workflow_manager


def make_cache():
    cache = MagicMock()
    cache.get_deleted_invocations = AsyncMock(return_value=[])
    cache.get_invocation_result = AsyncMock(return_value=None)
    cache.set_invocation_result = AsyncMock()
    cache.get_invocation_workflow_mapping = AsyncMock(return_value={INVOCATION_ID: {"workflow_id": "wf1"}})
    cache.get_workflows_cache = AsyncMock(return_value=[{"id": "wf1", "name": "My workflow"}])
    cache.get_invocation_state = AsyncMock(return_value=None)
    cache.set_invocation_state = AsyncMock()
    cache.redis.set.return_value = False  # already tracked: don't start a background task
    return cache


def make_mongo():
    mongo = MagicMock()
    mongo.get = AsyncMock(return_value=None)
    mongo.set = AsyncMock()
    mongo.get_element = AsyncMock(return_value=None)
    mongo.set_element = AsyncMock()
    return mongo


@pytest.mark.parametrize("raw_state, job_counts, expected", CASES)
async def test_get_invocation_result_maps_state(raw_state, job_counts, expected):
    service = InvocationService(cache=make_cache(), background_tasks=MagicMock(), mongo_client=make_mongo())
    service.inv_data_manager.structure_inputs = AsyncMock(return_value={})
    service.inv_data_manager.report_invocation_failure = AsyncMock(return_value=None)

    result = await service.get_invocation_result(
        invocation_id=INVOCATION_ID,
        username=USERNAME,
        api_key="key",
        galaxy_client=MagicMock(),
        workflow_manager=make_workflow_manager(raw_state, job_counts),
    )

    assert result.state == expected


@pytest.mark.parametrize("job_counts", [{"ok": 1}, {"error": 1}])
async def test_get_invocation_result_treats_completed_as_final(job_counts):
    """A completed invocation is stored, not tracked in the background, whichever way it resolves."""
    mongo = make_mongo()
    service = InvocationService(cache=make_cache(), background_tasks=MagicMock(), mongo_client=mongo)
    service.inv_data_manager.structure_inputs = AsyncMock(return_value={})
    service.inv_data_manager.report_invocation_failure = AsyncMock(return_value=None)

    await service.get_invocation_result(
        invocation_id=INVOCATION_ID,
        username=USERNAME,
        api_key="key",
        galaxy_client=MagicMock(),
        workflow_manager=make_workflow_manager("completed", job_counts),
    )

    service.cache.redis.set.assert_not_called()
    mongo.set.assert_called_once()


@pytest.mark.parametrize("raw_state, job_counts, expected", CASES)
async def test_map_invocation_state_maps_state(raw_state, job_counts, expected):
    tracker = InvocationTracker(cache=make_cache(), redis_client=MagicMock(), inv_data_manager=MagicMock())

    with patch.object(tracker, "compute_deep_invocation_state", new_callable=AsyncMock) as deep:
        state = await tracker.map_invocation_state(
            username=USERNAME,
            raw_state=raw_state,
            invocation_id=INVOCATION_ID,
            workflow_manager=make_workflow_manager(raw_state, job_counts),
            mongo_client=make_mongo(),
        )

    assert state == expected
    deep.assert_not_called()


@pytest.mark.parametrize("raw_state", ["new", "requires_materialization", "ready", "scheduled"])
async def test_map_invocation_state_deep_checks_pending_states(raw_state):
    tracker = InvocationTracker(cache=make_cache(), redis_client=MagicMock(), inv_data_manager=MagicMock())

    with patch.object(tracker, "compute_deep_invocation_state", new_callable=AsyncMock, return_value=COMPLETE) as deep:
        state = await tracker.map_invocation_state(
            username=USERNAME,
            raw_state=raw_state,
            invocation_id=INVOCATION_ID,
            workflow_manager=make_workflow_manager(raw_state, {}),
            mongo_client=make_mongo(),
        )

    assert state == COMPLETE
    deep.assert_awaited_once()


@pytest.mark.parametrize("raw_state", ["new", "requires_materialization", "ready", "scheduled"])
async def test_get_invocation_result_pending_states(raw_state):
    service = InvocationService(cache=make_cache(), background_tasks=MagicMock(), mongo_client=make_mongo())
    service.inv_data_manager.structure_inputs = AsyncMock(return_value={})

    result = await service.get_invocation_result(
        invocation_id=INVOCATION_ID,
        username=USERNAME,
        api_key="key",
        galaxy_client=MagicMock(),
        workflow_manager=make_workflow_manager(raw_state, {}),
    )

    assert result.state == PENDING


OUTPUT_DATASET = {"type": "dataset", "id": "ds1", "name": "out.txt", "visible": True, "data_type": "txt"}


async def test_get_invocation_result_complete_carries_outputs_and_no_failure_report():
    """A successful completed invocation gets its real outputs and is indexed, like the background tracker does."""
    mongo = make_mongo()
    service = InvocationService(cache=make_cache(), background_tasks=MagicMock(), mongo_client=mongo)
    service.inv_data_manager.structure_inputs = AsyncMock(return_value={})
    service.inv_data_manager.report_invocation_failure = AsyncMock(return_value="failure report")
    service.inv_data_manager.structure_outputs = AsyncMock(return_value=([OUTPUT_DATASET], "### Invocation report"))
    workflow_manager = make_workflow_manager("completed", {"ok": 2})
    details = workflow_manager.gi_object.gi.invocations.show_invocation.return_value
    details["outputs"] = {"out": {"id": "ds1", "src": "hda"}}
    details["output_collections"] = {"col": {"id": "hdca1", "src": "hdca"}}

    with patch("app.GX_integration.invocations.invocation_service.OutputIndexer") as indexer_cls:
        indexer_cls.return_value.index_datasets_and_register = AsyncMock()
        result = await service.get_invocation_result(
            invocation_id=INVOCATION_ID,
            username=USERNAME,
            api_key="key",
            galaxy_client=MagicMock(),
            workflow_manager=workflow_manager,
        )
        await asyncio.sleep(0)  # let the indexing task run

    assert result.state == COMPLETE
    assert [r.id for r in result.result] == ["ds1"]
    assert result.report == "### Invocation report"
    service.inv_data_manager.report_invocation_failure.assert_not_called()
    outputs = service.inv_data_manager.structure_outputs.await_args.kwargs["outputs"]
    assert outputs == {"output_datasets": ["ds1"], "collection_datasets": ["hdca1"]}
    stored = mongo.set.await_args.kwargs["value"]
    assert stored["result"] == [OUTPUT_DATASET] and stored["report"] == "### Invocation report"
    indexer_cls.return_value.index_datasets_and_register.assert_awaited_once()


@pytest.mark.parametrize("job_counts, expected", [({"ok": 1}, COMPLETE), ({"error": 1}, FAILED)])
async def test_map_invocation_state_stores_resolved_completed_state(job_counts, expected):
    """The list path asks Galaxy for a completed invocation's jobs once, then serves the stored state."""
    cache, mongo = make_cache(), make_mongo()
    tracker = InvocationTracker(cache=cache, redis_client=MagicMock(), inv_data_manager=MagicMock())

    state = await tracker.map_invocation_state(
        username=USERNAME,
        raw_state="completed",
        invocation_id=INVOCATION_ID,
        workflow_manager=make_workflow_manager("completed", job_counts),
        mongo_client=mongo,
    )
    await asyncio.sleep(0)

    assert state == expected
    cache.set_invocation_state.assert_awaited_once_with(USERNAME, INVOCATION_ID, expected)
    mongo.set_element.assert_awaited_once()
    assert mongo.set_element.await_args.kwargs["value"] == expected


async def test_map_invocation_state_degrades_to_pending_when_summary_fails():
    """One failed jobs-summary call gives Pending for that invocation instead of failing the whole list."""
    cache, mongo = make_cache(), make_mongo()
    tracker = InvocationTracker(cache=cache, redis_client=MagicMock(), inv_data_manager=MagicMock())
    workflow_manager = make_workflow_manager("completed", {})
    workflow_manager.gi_object.gi.invocations.get_invocation_summary.side_effect = RuntimeError("galaxy down")

    state = await tracker.map_invocation_state(
        username=USERNAME,
        raw_state="completed",
        invocation_id=INVOCATION_ID,
        workflow_manager=workflow_manager,
        mongo_client=mongo,
    )
    await asyncio.sleep(0)

    assert state == PENDING
    cache.set_invocation_state.assert_not_called()
    mongo.set_element.assert_not_called()


async def test_map_invocation_state_does_not_store_unknown_state():
    cache, mongo = make_cache(), make_mongo()
    tracker = InvocationTracker(cache=cache, redis_client=MagicMock(), inv_data_manager=MagicMock())

    state = await tracker.map_invocation_state(
        username=USERNAME,
        raw_state="bogus",
        invocation_id=INVOCATION_ID,
        workflow_manager=make_workflow_manager("bogus", {}),
        mongo_client=mongo,
    )
    await asyncio.sleep(0)

    assert state == PENDING
    cache.set_invocation_state.assert_not_called()
    mongo.set_element.assert_not_called()
