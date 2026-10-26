"""Map Galaxy job and invocation states to the five UI statuses.

queued -> running -> completed | failed | cancelled. The three last ones are final.
"""

from typing import Iterable, Mapping, Optional, Tuple

QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"

FINAL = frozenset({COMPLETED, FAILED, CANCELLED})
_ORDER = {QUEUED: 0, RUNNING: 1, COMPLETED: 2, FAILED: 2, CANCELLED: 2}

_JOB_STATUS = {
    "new": QUEUED, "upload": QUEUED, "waiting": QUEUED, "queued": QUEUED, "resubmitted": QUEUED,
    "running": RUNNING, "paused": RUNNING,
    "ok": COMPLETED, "skipped": COMPLETED,
    "error": FAILED, "failed": FAILED,
    "deleted": CANCELLED, "deleting": CANCELLED, "deleted_new": CANCELLED, "stop": CANCELLED, "stopped": CANCELLED,
}

# Which Galaxy state represents a group of jobs: the one most worth showing.
_ACTIVE_PRIORITY = ["paused", "running", "queued", "waiting", "upload", "resubmitted", "new"]
_FINAL_PRIORITY = ["error", "failed", "stopped", "stop", "deleted", "deleting", "deleted_new", "skipped", "ok"]


def job_status(galaxy_state: Optional[str]) -> str:
    """UI status for one Galaxy job state. Unknown states count as queued."""
    return _JOB_STATUS.get(galaxy_state or "", QUEUED)


def is_final(status: str) -> bool:
    return status in FINAL


def tool_run_status(job_states: Iterable[str]) -> Tuple[str, str]:
    """(status, galaxy_state) for a tool run: one job, or one per element when mapped over a collection.

    A run is final only once every job is. Then any failed job makes it failed, any stopped
    one cancelled, and otherwise it completed.
    """
    states = list(job_states)
    if not states:
        return QUEUED, "new"
    statuses = [job_status(s) for s in states]
    if all(is_final(s) for s in statuses):
        status = FAILED if FAILED in statuses else CANCELLED if CANCELLED in statuses else COMPLETED
        return status, _pick(states, _FINAL_PRIORITY)
    if any(s != QUEUED for s in statuses):
        return RUNNING, _pick([s for s in states if not is_final(job_status(s))], _ACTIVE_PRIORITY)
    return QUEUED, _pick(states, _ACTIVE_PRIORITY)


def invocation_run_status(invocation_state: Optional[str], job_state_counts: Mapping[str, int]) -> Tuple[str, str]:
    """(status, galaxy_state) for a workflow run, from the invocation state and its jobs' states.

    galaxy_state is the invocation's own state, except that a paused job shows as paused.
    """
    state = invocation_state or "new"
    present = {s for s, n in job_state_counts.items() if n}
    running = present & {"running", "paused"}
    galaxy_state = "paused" if "paused" in present else state

    if state in ("cancelled", "cancelling"):
        return CANCELLED, galaxy_state
    if state == "failed":
        return FAILED, galaxy_state
    if state == "completed":
        return (FAILED if present & {"error", "failed"} else COMPLETED), galaxy_state
    if state == "scheduled" and running:
        return RUNNING, galaxy_state
    # new, requires_materialization, ready, or scheduled with no job running
    return QUEUED, galaxy_state


def no_going_back(previous: Optional[str], current: str) -> str:
    """A run never goes back: once running, a momentary 'queued' (e.g. between workflow steps) stays running."""
    if previous and _ORDER.get(current, 0) < _ORDER.get(previous, 0):
        return previous
    return current


def _pick(states, priority):
    for p in priority:
        if p in states:
            return p
    return states[0]
