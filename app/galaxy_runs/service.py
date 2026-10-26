"""Galaxy side of runs started from chat threads.

Starting a run returns at once. Reading a run that hasn't ended asks Galaxy for its state;
a run that has ended is answered from Mongo, since it never changes again.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from bioblend import ConnectionError as GalaxyConnectionError

from app.galaxy import GalaxyClient
from app.galaxy_runs import links, status as st
from app.galaxy_runs.store import RunStore
from app.galaxy_runs.tool_model import describe_inputs, input_types, shape_inputs, type_of

log = logging.getLogger("GalaxyRuns")

CACHE_SECONDS = 5
STDERR_LINES = 20
SEARCH_LIMIT = 15
REMOVED = "Removed in Galaxy"
WORKFLOW_INPUT_STEPS = {"data_input", "data_collection_input", "parameter_input"}
PUBLIC_RUN_FIELDS = ("id", "thread_id", "proposal_id", "client_request_id", "target", "status", "galaxy_state",
                     "progress", "history_id", "outputs", "error", "created_at", "updated_at")


class GalaxyRunsError(Exception):
    """An error with the HTTP status the route should answer with."""

    def __init__(self, status_code: int, detail: Any):
        super().__init__(str(detail))
        self.status_code = status_code
        self.detail = detail


class _NotInGalaxy(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def public_run(run: Dict[str, Any]) -> Dict[str, Any]:
    return {k: run.get(k) for k in PUBLIC_RUN_FIELDS}


class GalaxyRuns:

    def __init__(self, galaxy_client: GalaxyClient, username: str, store: RunStore, cache: Any = None):
        self.client = galaxy_client
        self.gi = galaxy_client.gi_client
        self.galaxy_url = galaxy_client.galaxy_url.rstrip("/")
        self.username = username
        self.store = store
        self.cache = cache  # redis.asyncio client, or None

    # ------------------------------------------------------------------ Galaxy calls

    async def _call(self, fn: Callable, *args, **kwargs):
        """Run a blocking bioblend call. Galaxy's 404 becomes _NotInGalaxy, 400/422 a 422, the rest a 502."""
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except GalaxyConnectionError as e:
            code = getattr(e, "status_code", None)
            message = _galaxy_message(e)
            if code == 404:
                raise _NotInGalaxy(message) from e
            if code in (400, 422):
                raise GalaxyRunsError(422, message) from e
            if code in (401, 403):
                raise GalaxyRunsError(403, message) from e
            log.error("Galaxy call failed (%s): %s", code, message)
            raise GalaxyRunsError(502, "Galaxy failed. Safe to retry.") from e

    def _url(self, path: str) -> str:
        return f"{self.gi.url}/{path.lstrip('/')}"

    # ------------------------------------------------------------------ thread history

    async def thread_history(self, thread_id: str, create: bool = True) -> Optional[str]:
        """The thread's Galaxy history, created and tagged with the thread on first use."""
        history_id = await self.store.get_thread_history(self.username, thread_id)
        if history_id or not create:
            return history_id
        history = await self._call(self.gi.histories.create_history, name=f"Chat thread {thread_id}")
        await self._call(self.gi.histories.create_history_tag, history["id"], f"thread:{thread_id}")
        winner = await self.store.claim_thread_history(self.username, thread_id, history["id"])
        if winner != history["id"]:
            # Another request created one at the same time; keep theirs.
            try:
                await self._call(self.gi.histories.delete_history, history["id"], purge=True)
            except Exception as e:
                log.warning("Could not remove duplicate thread history %s: %s", history["id"], e)
        return winner

    # ------------------------------------------------------------------ describe / validate

    async def _tool_model(self, target: Dict[str, Any], history_id: Optional[str],
                          inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            return await self._call(self.gi.tools.build, target["id"], inputs=inputs,
                                    tool_version=target.get("version") or None, history_id=history_id)
        except _NotInGalaxy:
            raise GalaxyRunsError(400, f"Unknown tool: {target['id']}")

    async def _workflow(self, target: Dict[str, Any]) -> Dict[str, Any]:
        version = _int_or_none(target.get("version"))
        try:
            return await self._call(self.gi.workflows.show_workflow, target["id"], version=version)
        except _NotInGalaxy:
            raise GalaxyRunsError(400, f"Unknown workflow: {target['id']}")

    async def describe(self, thread_id: str, target: Dict[str, Any], full: bool = False) -> Dict[str, Any]:
        history_id = await self.thread_history(thread_id)
        if target["type"] == "tool":
            model = await self._tool_model(target, history_id)
            described = describe_inputs(model, full=full)
            return {"target": {"type": "tool", "id": model.get("id", target["id"]),
                               "version": model.get("version"), "name": model.get("name")},
                    **described}
        workflow = await self._workflow(target)
        files = await self.thread_files(thread_id)
        entries = _workflow_inputs(workflow, files["items"])
        return {"target": {"type": "workflow", "id": workflow["id"], "version": str(workflow.get("version", "")),
                           "name": workflow.get("name")},
                "inputs": entries, "all_count": len(entries)}

    async def validate(self, thread_id: str, target: Dict[str, Any], inputs: Dict[str, Any],
                       parameters: Dict[str, Any]) -> Dict[str, Any]:
        history_id = await self.thread_history(thread_id)
        if target["type"] == "tool":
            types = input_types(await self._tool_model(target, history_id))
            errors: Dict[str, str] = {p: "Unknown input" for p in inputs if type_of(types, p) is None}
            model = await self._tool_model(target, history_id, inputs=shape_inputs(inputs, types))
            errors.update(_flatten_errors(model.get("errors") or {}))
            errors.update(_missing_required_files(model, inputs))
            return {"valid": not errors, "errors": errors}
        workflow = await self._workflow(target)
        errors = {}
        known = {e["path"]: e for e in _workflow_inputs(workflow, [])}
        for label in inputs:
            if label not in known:
                errors[label] = "Unknown input"
        for label, e in known.items():
            if not e["optional"] and inputs.get(label) in (None, "", [], {}):
                errors[label] = "Required"
        return {"valid": not errors, "errors": errors}

    # ------------------------------------------------------------------ search / files

    async def search(self, q: str, kind: Optional[str]) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        if kind in (None, "tool"):
            response = await self._call(self.gi.make_get_request, self._url("tools"), params={"q": q})
            ids = response.json() if response.status_code == 200 else []
            ids = list(dict.fromkeys(i for i in ids if isinstance(i, str)))[:SEARCH_LIMIT]
            tools = await asyncio.gather(*(self._show_tool(i) for i in ids))
            items.extend(t for t in tools if t)
        if kind in (None, "workflow"):
            needle = q.lower()
            for w in await self._call(self.gi.workflows.get_workflows):
                if needle in (w.get("name") or "").lower() or any(needle in t.lower() for t in w.get("tags") or []):
                    items.append({"type": "workflow", "id": w["id"], "version": None,
                                  "name": w.get("name"), "description": w.get("annotation")})
        return {"items": items, "next_cursor": None}

    async def _show_tool(self, tool_id: str) -> Optional[Dict[str, Any]]:
        try:
            t = await self._call(self.gi.tools.show_tool, tool_id)
        except (_NotInGalaxy, GalaxyRunsError):
            return None
        return {"type": "tool", "id": t.get("id", tool_id), "version": t.get("version"),
                "name": t.get("name"), "description": t.get("description")}

    async def thread_files(self, thread_id: str) -> Dict[str, Any]:
        history_id = await self.thread_history(thread_id, create=False)
        if not history_id:
            return {"items": [], "next_cursor": None}
        contents = await self._call(self.gi.histories.show_history, history_id, contents=True,
                                    deleted=False, visible=True)
        items = []
        for c in sorted(contents, key=lambda c: c.get("hid") or 0):
            collection = c.get("history_content_type") == "dataset_collection"
            item = {"src": "hdca" if collection else "hda", "id": c["id"], "name": c.get("name"),
                    "hid": c.get("hid"), "state": c.get("state") or c.get("populated_state")}
            if collection:
                item["collection_type"] = c.get("collection_type")
            else:
                item["format"] = c.get("extension")
            items.append(item)
        return {"items": items, "next_cursor": None}

    # ------------------------------------------------------------------ start a run

    async def start_run(self, thread_id: str, proposal_id: str, client_request_id: str, target: Dict[str, Any],
                        inputs: Dict[str, Any], parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Returns (run, repeat). A repeated client_request_id returns the stored run and starts nothing."""
        run_id = f"run_{uuid.uuid4().hex[:16]}"
        if not await self.store.claim_request(self.username, client_request_id, run_id):
            return await self._repeat(client_request_id), True
        try:
            history_id = await self.thread_history(thread_id)
            if target["type"] == "tool":
                started = await self._start_tool(target, history_id, inputs)
            else:
                started = await self._start_workflow(target, history_id, inputs, parameters)
        except Exception:
            await self.store.release_request(self.username, client_request_id)
            raise
        now = _now()
        run = {
            "id": run_id, "thread_id": thread_id, "proposal_id": proposal_id,
            "client_request_id": client_request_id, "history_id": history_id,
            "status": st.QUEUED, "galaxy_state": "new", "outputs": [], "error": None,
            "created_at": now, "updated_at": now, **started,
        }
        await self.store.save_run(self.username, run)
        return public_run(run), False

    async def _repeat(self, client_request_id: str) -> Dict[str, Any]:
        run_id = await self.store.get_request(self.username, client_request_id)
        # The first request may still be starting the run; give it a few seconds.
        for _ in range(20):
            if run_id:
                run = await self.store.get_run(self.username, run_id)
                if run:
                    return await self.read_run(run_id)
            await asyncio.sleep(0.5)
            run_id = run_id or await self.store.get_request(self.username, client_request_id)
        raise GalaxyRunsError(502, "The first request for this run is still starting. Safe to retry.")

    async def _start_tool(self, target: Dict[str, Any], history_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model = await self._tool_model(target, history_id)
        payload = {"tool_id": model.get("id", target["id"]), "tool_version": model.get("version"),
                   "history_id": history_id, "inputs": shape_inputs(inputs, input_types(model))}
        response = await self._call(self.gi.make_post_request, self._url("tools"), payload=payload)
        jobs = response.get("jobs") or []
        if not jobs:
            raise GalaxyRunsError(422, response.get("errors") or "Galaxy started no job for these inputs.")
        outputs_meta = {o["id"]: {"id": o["id"], "name": o.get("name"), "format": o.get("file_ext")}
                        for o in response.get("outputs") or []}
        return {
            "target": {"type": "tool", "id": payload["tool_id"], "version": model.get("version"), "name": model.get("name")},
            "job_ids": [j["id"] for j in jobs],
            "outputs_meta": outputs_meta,
            "progress": {"jobs_done": 0, "jobs_total": len(jobs), "steps_done": 0, "steps_total": 1},
        }

    async def _start_workflow(self, target: Dict[str, Any], history_id: str, inputs: Dict[str, Any],
                              parameters: Dict[str, Any]) -> Dict[str, Any]:
        workflow = await self._workflow(target)
        steps_total = sum(1 for s in (workflow.get("steps") or {}).values() if s.get("type") not in WORKFLOW_INPUT_STEPS)
        invocation = await self._call(
            self.gi.workflows.invoke_workflow, workflow["id"], inputs=inputs, params=parameters or None,
            history_id=history_id, inputs_by="name", version=_int_or_none(target.get("version")))
        return {
            "target": {"type": "workflow", "id": workflow["id"], "version": str(workflow.get("version", "")),
                       "name": workflow.get("name")},
            "invocation_id": invocation["id"],
            "steps_total": steps_total,
            "outputs_meta": {},
            "progress": {"jobs_done": 0, "jobs_total": 0, "steps_done": 0, "steps_total": steps_total},
        }

    # ------------------------------------------------------------------ read runs

    async def read_run(self, run_id: str) -> Dict[str, Any]:
        run = await self.store.get_run(self.username, run_id)
        if not run:
            raise GalaxyRunsError(404, "No such run")
        return await self._fresh(run)

    async def list_runs(self, thread_id: str) -> Dict[str, Any]:
        runs = await self.store.list_runs(self.username, thread_id)
        items = await asyncio.gather(*(self._fresh(r) for r in runs))
        return {"items": list(items), "next_cursor": None}

    async def _fresh(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """Ask Galaxy about a run that hasn't ended. Several readers within 5 s share one answer."""
        if st.is_final(run.get("status", "")):
            return public_run(run)
        cached = await self._cache_get(run["id"])
        if cached:
            return cached
        try:
            snapshot = await (self._read_tool(run) if run.get("job_ids") else self._read_workflow(run))
        except _NotInGalaxy:
            snapshot = {"status": st.CANCELLED, "galaxy_state": "deleted", "progress": run.get("progress"),
                        "outputs": run.get("outputs") or [], "error": {"step": None, "message": REMOVED}}
        snapshot["status"] = st.no_going_back(run.get("status"), snapshot["status"])
        changed = any(run.get(k) != snapshot.get(k) for k in ("status", "galaxy_state", "progress", "outputs", "error"))
        run.update(snapshot)
        if changed:
            run["updated_at"] = _now()
            await self.store.save_run(self.username, run)
        view = public_run(run)
        await self._cache_set(run["id"], view)
        return view

    async def _read_tool(self, run: Dict[str, Any]) -> Dict[str, Any]:
        jobs = await asyncio.gather(*(self._call(self.gi.jobs.show_job, j) for j in run["job_ids"]))
        states = [j.get("state") for j in jobs]
        status, galaxy_state = st.tool_run_status(states)
        done = sum(1 for s in states if st.is_final(st.job_status(s)))
        meta = run.get("outputs_meta") or {}
        outputs = []
        for job in jobs:
            if job.get("state") != "ok":
                continue
            for name, out in (job.get("outputs") or {}).items():
                if out.get("src", "hda") == "hda":
                    outputs.append(meta.get(out["id"]) or {"id": out["id"], "name": name, "format": None})
        error = None
        if status == st.FAILED:
            failed = next(j for j in jobs if st.job_status(j.get("state")) == st.FAILED)
            error = {"step": None, "message": await self._job_error(failed["id"])}
        return {"status": status, "galaxy_state": galaxy_state, "outputs": outputs, "error": error,
                "progress": {"jobs_done": done, "jobs_total": len(jobs),
                             "steps_done": 1 if st.is_final(status) else 0, "steps_total": 1}}

    async def _read_workflow(self, run: Dict[str, Any]) -> Dict[str, Any]:
        invocation_id = run["invocation_id"]
        invocation, steps = await asyncio.gather(
            self._call(self.gi.invocations.show_invocation, invocation_id),
            self._call(self.gi.invocations.get_invocation_step_jobs_summary, invocation_id))
        counts: Dict[str, int] = {}
        steps_done = 0
        for step in steps:
            step_states = step.get("states") or {}
            for s, n in step_states.items():
                counts[s] = counts.get(s, 0) + n
            if step_states and all(st.is_final(st.job_status(s)) for s, n in step_states.items() if n):
                steps_done += 1
        status, galaxy_state = st.invocation_run_status(invocation.get("state"), counts)
        jobs_total = sum(counts.values())
        jobs_done = sum(n for s, n in counts.items() if st.is_final(st.job_status(s)))

        meta = run.setdefault("outputs_meta", {})
        outputs = []
        for out in (invocation.get("outputs") or {}).values():
            if out.get("src") != "hda":
                continue
            known = meta.get(out["id"])
            if not known:
                dataset = await self._call(self.gi.datasets.show_dataset, out["id"])
                if dataset.get("state") != "ok":
                    continue
                known = meta[out["id"]] = {"id": out["id"], "name": dataset.get("name"),
                                           "format": dataset.get("extension") or dataset.get("file_ext")}
            outputs.append(known)

        error = None
        if status == st.FAILED:
            error = await self._workflow_error(invocation)
        steps_total = run.get("steps_total") or len(steps)
        return {"status": status, "galaxy_state": galaxy_state, "outputs": outputs, "error": error,
                "outputs_meta": meta,
                "progress": {"jobs_done": jobs_done, "jobs_total": jobs_total,
                             "steps_done": min(steps_done, steps_total), "steps_total": steps_total}}

    async def _job_error(self, job_id: str) -> str:
        try:
            job = await self._call(self.gi.jobs.show_job, job_id, full_details=True)
        except (_NotInGalaxy, GalaxyRunsError):
            return "The job failed."
        text = job.get("tool_stderr") or job.get("stderr") or job.get("job_stderr") or job.get("info") or ""
        lines = [line for line in str(text).strip().splitlines() if line.strip()]
        return "\n".join(lines[-STDERR_LINES:]) or "The job failed."

    async def _workflow_error(self, invocation: Dict[str, Any]) -> Dict[str, Any]:
        jobs = await self._call(self.gi.jobs.get_jobs, invocation_id=invocation["id"], state="error", limit=1)
        if not jobs:
            messages = invocation.get("messages") or []
            reason = messages[0].get("reason") if messages and isinstance(messages[0], dict) else None
            return {"step": None, "message": reason or "The workflow failed."}
        job_id = jobs[0]["id"]
        step_name = None
        for step in invocation.get("steps") or []:
            if step.get("job_id") == job_id:
                step_name = step.get("workflow_step_label") or f"Step {int(step.get('order_index', 0)) + 1}"
                break
        return {"step": step_name, "message": await self._job_error(job_id)}

    # ------------------------------------------------------------------ cancel

    async def cancel_run(self, run_id: str) -> Dict[str, Any]:
        run = await self.store.get_run(self.username, run_id)
        if not run:
            raise GalaxyRunsError(404, "No such run")
        if st.is_final(run.get("status", "")):
            return public_run(run)
        try:
            if run.get("job_ids"):
                await asyncio.gather(*(self._call(self.gi.jobs.cancel_job, j) for j in run["job_ids"]))
            else:
                await self._call(self.gi.invocations.cancel_invocation, run["invocation_id"])
        except _NotInGalaxy:
            pass  # already gone; the read below ends it as removed
        await self._cache_clear(run_id)
        # Still running until Galaxy confirms; a later read shows cancelled.
        return await self._fresh(run)

    # ------------------------------------------------------------------ Open in Galaxy

    async def link(self, thread_id: str, proposal_id: str, target: Dict[str, Any], inputs: Dict[str, Any],
                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        if target["type"] == "workflow":
            payload = {"workflow_id": target["id"], "workflow_target_type": "stored_workflow",
                       "request_state": inputs, "public": False}
            landing = await self._call(self.gi.make_post_request, self._url("workflow_landings"), payload=payload)
            return {"url": links.workflow_landing_url(self.galaxy_url, landing["uuid"]),
                    "kind": "workflow_landing", "not_carried": sorted(parameters or {})}

        runs = [r for r in await self.store.list_runs(self.username, thread_id)
                if r.get("proposal_id") == proposal_id and r.get("job_ids")]
        if runs:
            return {"url": links.rerun_url(self.galaxy_url, runs[0]["job_ids"][-1]), "kind": "rerun", "not_carried": []}
        history_id = await self.thread_history(thread_id)
        url, not_carried = links.tool_form_url(self.galaxy_url, target["id"], target.get("version") or "",
                                               history_id, inputs)
        return {"url": url, "kind": "tool_form", "not_carried": not_carried}

    # ------------------------------------------------------------------ cache

    def _cache_key(self, run_id: str) -> str:
        return f"galaxy_run:{self.username}:{run_id}"

    async def _cache_get(self, run_id: str) -> Optional[Dict[str, Any]]:
        if not self.cache:
            return None
        try:
            raw = await self.cache.get(self._cache_key(run_id))
            return json.loads(raw) if raw else None
        except Exception as e:
            log.debug("Run cache read failed: %s", e)
            return None

    async def _cache_set(self, run_id: str, view: Dict[str, Any]) -> None:
        if not self.cache:
            return
        try:
            await self.cache.set(self._cache_key(run_id), json.dumps(view), ex=CACHE_SECONDS)
        except Exception as e:
            log.debug("Run cache write failed: %s", e)

    async def _cache_clear(self, run_id: str) -> None:
        if not self.cache:
            return
        try:
            await self.cache.delete(self._cache_key(run_id))
        except Exception as e:
            log.debug("Run cache clear failed: %s", e)


# ---------------------------------------------------------------------- helpers

def _galaxy_message(e: GalaxyConnectionError) -> Any:
    body = getattr(e, "body", None)
    try:
        parsed = json.loads(body) if isinstance(body, (str, bytes)) else body
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed.get("err_msg") or parsed.get("detail") or parsed
    return str(body or e)


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _flatten_errors(errors: Any, prefix: str = "") -> Dict[str, str]:
    """Galaxy's build errors, nested or flat, as {path: message}."""
    flat: Dict[str, str] = {}
    if isinstance(errors, dict):
        for k, v in errors.items():
            flat.update(_flatten_errors(v, f"{prefix}{k}|" if isinstance(v, dict) else f"{prefix}{k}"))
    elif isinstance(errors, list):
        for i, v in enumerate(errors):
            flat.update(_flatten_errors(v, f"{prefix}{i}|"))
    elif errors:
        flat[prefix.rstrip("|")] = str(errors)
    return flat


def _missing_required_files(model: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, str]:
    """Required file inputs, in the branches the given inputs select, that have no value."""
    entries = describe_inputs(model, full=True)["inputs"]
    chosen = {e["path"]: inputs.get(e["path"], e.get("default")) for e in entries if e.get("selector")}
    missing = {}
    for e in entries:
        if e["type"] not in ("data", "data_collection") or e.get("optional"):
            continue
        applies = all(_matches_when(chosen.get(p), v) for p, v in (e.get("when") or {}).items())
        if applies and inputs.get(e["path"]) in (None, "", [], {}):
            missing[e["path"]] = "Required"
    return missing


def _matches_when(value: Any, expected: Any) -> bool:
    expected_values = expected if isinstance(expected, list) else [expected]
    return str(value) in {str(v) for v in expected_values}


def _workflow_inputs(workflow: Dict[str, Any], files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A workflow's inputs by label, with the thread's matching files for file inputs."""
    entries = []
    for step in sorted((workflow.get("steps") or {}).values(), key=lambda s: s.get("id", 0)):
        kind = step.get("type")
        if kind not in WORKFLOW_INPUT_STEPS:
            continue
        tool_inputs = step.get("tool_inputs") or {}
        label = step.get("label") or (step.get("annotation") or f"Input {step.get('id')}")
        entry: Dict[str, Any] = {"path": label, "label": label, "optional": bool(tool_inputs.get("optional", False))}
        if kind == "parameter_input":
            entry["type"] = tool_inputs.get("parameter_type", "text")
            if tool_inputs.get("default") is not None:
                entry["default"] = tool_inputs["default"]
        else:
            entry["type"] = "data" if kind == "data_input" else "data_collection"
            formats = tool_inputs.get("format") or []
            entry["formats"] = formats if isinstance(formats, list) else [formats]
            if kind == "data_collection_input" and tool_inputs.get("collection_type"):
                entry["collection_types"] = [tool_inputs["collection_type"]]
            src = "hda" if kind == "data_input" else "hdca"
            matches = [{"src": f["src"], "id": f["id"], "name": f.get("name"), "hid": f.get("hid")}
                       for f in files if f.get("src") == src
                       and (not entry["formats"] or src == "hdca" or f.get("format") in entry["formats"])]
            if matches:
                entry["matches"] = matches
        entries.append(entry)
    return entries
