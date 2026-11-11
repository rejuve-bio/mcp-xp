"""Mongo records for Galaxy runs: one history per thread, the runs, and run requests by client_request_id.

Keys start with the Galaxy username, so one user's records are never found from another's.
"""

from typing import Any, Dict, List, Optional

from app.enumerations import CollectionNames
from app.persistence import MongoStore

HISTORIES = CollectionNames.GALAXY_THREAD_HISTORIES.value
RUNS = CollectionNames.GALAXY_RUNS.value
REQUESTS = CollectionNames.GALAXY_RUN_REQUESTS.value


class RunStore:

    def __init__(self, mongo: MongoStore):
        self.mongo = mongo

    # ---- thread histories ----

    async def get_thread_history(self, username: str, thread_id: str) -> Optional[str]:
        record = await self.mongo.get(HISTORIES, f"{username}:{thread_id}")
        return record.get("history_id") if record else None

    async def claim_thread_history(self, username: str, thread_id: str, history_id: str) -> str:
        """Record the thread's history unless another request already did. Returns the one that counts."""
        key = f"{username}:{thread_id}"
        if await self.mongo.set_if_absent(HISTORIES, key, {"history_id": history_id, "thread_id": thread_id}):
            return history_id
        record = await self.mongo.get(HISTORIES, key)
        return record["history_id"]

    # ---- runs ----

    async def get_run(self, username: str, run_id: str) -> Optional[Dict[str, Any]]:
        return await self.mongo.get(RUNS, f"{username}:{run_id}")

    async def save_run(self, username: str, run: Dict[str, Any]) -> None:
        await self.mongo.set(RUNS, f"{username}:{run['id']}", run)

    async def list_runs(self, username: str, thread_id: str) -> List[Dict[str, Any]]:
        docs = await self.mongo.find_by_value_field(RUNS, "thread_id", thread_id, key_prefix=username)
        runs = [d["value"] for d in docs]
        return sorted(runs, key=lambda r: r.get("created_at", ""), reverse=True)

    # ---- run requests (retries) ----

    async def claim_request(self, username: str, client_request_id: str, run_id: str) -> bool:
        return await self.mongo.set_if_absent(REQUESTS, f"{username}:{client_request_id}", {"run_id": run_id})

    async def get_request(self, username: str, client_request_id: str) -> Optional[str]:
        record = await self.mongo.get(REQUESTS, f"{username}:{client_request_id}")
        return record.get("run_id") if record else None

    async def release_request(self, username: str, client_request_id: str) -> None:
        """Forget a request whose start failed, so a retry can start it."""
        await self.mongo.delete(REQUESTS, f"{username}:{client_request_id}")
