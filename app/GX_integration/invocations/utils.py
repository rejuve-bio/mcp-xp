import hashlib
import shutil
import asyncio
import pathlib
import logging

from enum import Enum
from typing import Mapping, Optional

from app.enumerations import InvocationStates


logger = logging.getLogger("Invocation")


# NOTE: Here are the list of tools used by bioblend to index invocation dataset outputs for visualiztion via JBrowse.
class FASTAIndexerTools(Enum):
    INDEXER_TOOL = "CONVERTER_fasta_to_fai"

class VCFIndexerTools(Enum):
    COMPRESSER_TOOL = "CONVERTER_uncompressed_to_gz"
    INDEXER_TOOL = "CONVERTER_vcf_bgzip_to_tabix_0"

class BAMIndexerTools(Enum):
    INDEXER_TOOL = "CONVERTER_Bam_Bai_0"

class GTFIndexerTools(Enum):
    COMPRESSER_TOOL = "CONVERTER_uncompressed_to_gz"
    INDEXER_TOOL = "CONVERTER_interval_to_tabix_0"
    

def generate_request_hash(*args) -> str:
    """Generate hash for request deduplication"""
    request_string = "|".join(str(arg) if arg is not None else "None" for arg in args)
    return hashlib.md5(request_string.encode()).hexdigest()

def _rmtree_sync(path: pathlib.Path):
    shutil.rmtree(path, ignore_errors=True)
    
def log_task_error(task: asyncio.Task, *, task_name: str) -> None:
    """Log errors from completed background tasks."""
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        logger.info(f"Background task '{task_name}' was cancelled.")
        return
    
    if exc is not None:
        logger.error(
            f"Background task '{task_name}' failed",
            exc_info=exc,
        )

# Galaxy (26.1) invocation states that are still being scheduled or running.
GALAXY_PENDING_INVOCATION_STATES = frozenset({"new", "requires_materialization", "ready", "scheduled"})
GALAXY_FAILED_INVOCATION_STATES = frozenset({"failed", "cancelled", "cancelling"})


def map_galaxy_invocation_state(raw_state: Optional[str], job_state_counts: Optional[Mapping[str, int]] = None) -> str:
    """Map a raw Galaxy invocation state to an InvocationStates value.

    Mirrors invocation_run_status in app/galaxy_runs/status.py. Galaxy calls an invocation
    "completed" once every job is terminal, errored ones included, so a completed invocation
    is Failed if any job ended in error/failed. Unknown or missing states stay Pending.
    """
    if raw_state in GALAXY_FAILED_INVOCATION_STATES:
        return InvocationStates.FAILED.value
    if raw_state == "completed":
        present = {s for s, n in (job_state_counts or {}).items() if n}
        return InvocationStates.FAILED.value if present & {"error", "failed"} else InvocationStates.COMPLETE.value
    if raw_state not in GALAXY_PENDING_INVOCATION_STATES:
        logger.warning(f"Unknown invocation state: {raw_state}")
    return InvocationStates.PENDING.value


async def resolve_galaxy_invocation_state(raw_state: Optional[str], invocation_id: str, gi) -> str:
    """map_galaxy_invocation_state, fetching the invocation's job-state counts only when it is completed."""
    job_state_counts = None
    if raw_state == "completed":
        summary = await asyncio.to_thread(gi.invocations.get_invocation_summary, invocation_id=invocation_id)
        job_state_counts = (summary or {}).get("states") or {}
    return map_galaxy_invocation_state(raw_state, job_state_counts)
