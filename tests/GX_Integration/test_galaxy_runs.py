from sys import path
path.append(".")

from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import unquote

import pytest
from bioblend import ConnectionError as GalaxyConnectionError

from app.galaxy_runs import links, status as st
from app.galaxy_runs.service import GalaxyRuns, GalaxyRunsError
from app.galaxy_runs.tool_model import describe_inputs, input_types, shape_inputs, type_of


# ---------------------------------------------------------------- a fastp-like tool model

def _data(name, label, optional=False, options=None):
    return {"type": "data", "name": name, "label": label, "optional": optional,
            "extensions": ["fastqsanger.gz"], "options": options or {"hda": [], "hdca": []}}


TOOL_MODEL = {
    "id": "toolshed.g2.bx.psu.edu/repos/iuc/fastp/fastp/1.3.7+galaxy0", "version": "1.3.7+galaxy0", "name": "fastp",
    "errors": {},
    "inputs": [
        {"type": "conditional", "name": "single_paired",
         "test_param": {"type": "select", "name": "single_paired_selector", "label": "Single-end or paired reads",
                        "value": "single", "options": [["Single", "single", True], ["Paired", "paired", False]]},
         "cases": [
             {"value": "single", "inputs": [_data("in1", "Input 1", options={"hda": [{"id": "f2db", "name": "reads_R1.fastq.gz", "hid": 1}], "hdca": []})]},
             {"value": "paired", "inputs": [_data("in1", "Input 1"), _data("in2", "Input 2")]},
         ]},
        {"type": "section", "name": "filter_options", "title": "Filter options", "inputs": [
            {"type": "integer", "name": "length_required", "label": "Length required", "optional": False, "value": "15"},
            {"type": "text", "name": "adapter", "label": "Adapter", "optional": True, "value": ""},
        ]},
        {"type": "repeat", "name": "rep", "title": "Extra", "min": 0, "max": None,
         "inputs": [{"type": "text", "name": "x", "label": "X", "optional": False, "value": ""}], "cache": []},
        {"type": "hidden", "name": "secret", "value": "1"},
    ],
}


def test_describe_trimmed_keeps_required_choices_and_files():
    d = describe_inputs(TOOL_MODEL)
    paths = {e["path"]: e for e in d["inputs"]}
    assert "single_paired|single_paired_selector" in paths
    assert paths["single_paired|single_paired_selector"]["options"] == ["single", "paired"]
    # in1 is in every branch, in2 only in "paired"
    assert "when" not in paths["single_paired|in1"]
    assert paths["single_paired|in2"]["when"] == {"single_paired|single_paired_selector": "paired"}
    assert paths["single_paired|in1"]["matches"] == [{"src": "hda", "id": "f2db", "name": "reads_R1.fastq.gz", "hid": 1}]
    # settings with a default are left out of the trimmed description
    assert "filter_options|length_required" not in paths
    assert "secret" not in paths
    assert paths["rep"]["type"] == "repeat"


def test_describe_full_keeps_every_setting_with_its_section():
    d = describe_inputs(TOOL_MODEL, full=True)
    paths = {e["path"]: e for e in d["inputs"]}
    assert paths["filter_options|length_required"]["section"] == "Filter options"
    assert paths["filter_options|adapter"]["optional"] is True
    assert d["all_count"] == 5  # selector, in1, in2 and two filter options; hidden and repeat not counted


def test_input_types_cover_repeat_blocks_and_batch_shaping():
    types = input_types(TOOL_MODEL)
    assert type_of(types, "rep_3|x")["type"] == "text"
    shaped = shape_inputs({
        "single_paired|in1": {"src": "hdca", "id": "c1"},
        "single_paired|in2": {"src": "hda", "id": "d2"},
        "filter_options|length_required": 30,
    }, types)
    assert shaped["single_paired|in1"] == {"batch": True, "values": [{"src": "hdca", "id": "c1"}]}
    assert shaped["single_paired|in2"] == {"values": [{"src": "hda", "id": "d2"}]}
    assert shaped["filter_options|length_required"] == 30


# ---------------------------------------------------------------- status

@pytest.mark.parametrize("states,expected", [
    (["new"], ("queued", "new")),
    (["running"], ("running", "running")),
    (["paused"], ("running", "paused")),
    (["ok"], ("completed", "ok")),
    (["ok", "running", "queued"], ("running", "running")),
    (["ok", "queued"], ("running", "queued")),
    (["ok", "error"], ("failed", "error")),
    (["ok", "error", "running"], ("running", "running")),  # not final until every job is
    (["ok", "deleted"], ("cancelled", "deleted")),
    (["skipped"], ("completed", "skipped")),
])
def test_tool_run_status(states, expected):
    assert st.tool_run_status(states) == expected


@pytest.mark.parametrize("state,counts,expected", [
    ("new", {}, "queued"),
    ("ready", {"ok": 2}, "queued"),
    ("scheduled", {"ok": 1, "queued": 1}, "queued"),
    ("scheduled", {"running": 1}, "running"),
    ("completed", {"ok": 3}, "completed"),
    ("completed", {"ok": 2, "error": 1}, "failed"),
    ("failed", {}, "failed"),
    ("cancelling", {"running": 1}, "cancelled"),
])
def test_invocation_run_status(state, counts, expected):
    assert st.invocation_run_status(state, counts)[0] == expected


def test_paused_job_shows_as_paused():
    assert st.invocation_run_status("scheduled", {"paused": 1}) == ("running", "paused")


def test_a_run_never_goes_back():
    assert st.no_going_back("running", "queued") == "running"
    assert st.no_going_back("queued", "running") == "running"
    assert st.no_going_back(None, "queued") == "queued"


# ---------------------------------------------------------------- links

def test_tool_form_link_encodes_everything_and_lists_what_it_cannot_carry():
    url, not_carried = links.tool_form_url(
        "https://galaxy.example.org/", "toolshed.g2.bx.psu.edu/repos/iuc/fastp/fastp/1.3.7+galaxy0", "1.3.7+galaxy0", "h1",
        {"single_paired|in1": {"src": "hda", "id": "f2db"},
         "reads": {"src": "hdca", "id": "c1"},
         "many": [{"src": "hda", "id": "a"}, {"src": "hda", "id": "b"}],
         "rep_0|x": "first", "rep_1|x": "second",
         "filter_options|length_required": 30, "title": "a b+c"})
    assert url.startswith("https://galaxy.example.org/?tool_id=toolshed.g2.bx.psu.edu%2Frepos")
    assert "%2Bgalaxy0" in url and "single_paired%7Cin1=f2db" in url
    assert "rep_0%7Cx=first" in url and "title=a%20b%2Bc" in url
    assert "+" not in url.split("?", 1)[1]
    assert sorted(not_carried) == ["many", "reads", "rep_1|x"]
    assert unquote(url).count("history_id=h1") == 1


# ---------------------------------------------------------------- service with a fake Galaxy

class MemoryStore:
    def __init__(self):
        self.histories, self.runs, self.requests = {}, {}, {}

    async def get_thread_history(self, u, t):
        return self.histories.get((u, t))

    async def claim_thread_history(self, u, t, h):
        return self.histories.setdefault((u, t), h)

    async def get_run(self, u, r):
        run = self.runs.get((u, r))
        return dict(run) if run else None

    async def save_run(self, u, run):
        self.runs[(u, run["id"])] = dict(run)

    async def list_runs(self, u, t):
        runs = [dict(r) for (owner, _), r in self.runs.items() if owner == u and r["thread_id"] == t]
        return sorted(runs, key=lambda r: r["created_at"], reverse=True)

    async def claim_request(self, u, c, r):
        if (u, c) in self.requests:
            return False
        self.requests[(u, c)] = r
        return True

    async def get_request(self, u, c):
        return self.requests.get((u, c))

    async def release_request(self, u, c):
        self.requests.pop((u, c), None)


def _fake_gi():
    gi = MagicMock()
    gi.url = "https://galaxy.example.org/api"
    gi.histories.create_history.return_value = {"id": "hist1"}
    gi.tools.build.return_value = TOOL_MODEL
    gi.make_post_request.return_value = {
        "jobs": [{"id": "j1"}, {"id": "j2"}],
        "outputs": [{"id": "o1", "name": "fastp on S1", "file_ext": "fastqsanger.gz"},
                    {"id": "o2", "name": "fastp on S2", "file_ext": "fastqsanger.gz"}],
    }
    gi.jobs.show_job.side_effect = lambda job_id, full_details=False: {
        "j1": {"id": "j1", "state": "ok", "outputs": {"out1": {"id": "o1", "src": "hda"}}},
        "j2": {"id": "j2", "state": "running", "outputs": {"out1": {"id": "o2", "src": "hda"}}},
    }[job_id]
    return gi


def _service(gi, store=None, user="alice"):
    client = SimpleNamespace(gi_client=gi, galaxy_url="https://galaxy.example.org")
    return GalaxyRuns(client, user, store or MemoryStore(), cache=None)


TARGET = {"type": "tool", "id": TOOL_MODEL["id"], "version": "1.3.7+galaxy0"}
INPUTS = {"single_paired|single_paired_selector": "single", "single_paired|in1": {"src": "hdca", "id": "c1"}}


async def test_start_run_returns_at_once_and_a_retry_starts_nothing():
    gi = _fake_gi()
    svc = _service(gi)
    run, repeat = await svc.start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})
    assert repeat is False
    assert run["status"] == "queued" and run["history_id"] == "hist1"
    assert run["progress"] == {"jobs_done": 0, "jobs_total": 2, "steps_done": 0, "steps_total": 1}
    payload = gi.make_post_request.call_args.kwargs["payload"]
    assert payload["inputs"]["single_paired|in1"] == {"batch": True, "values": [{"src": "hdca", "id": "c1"}]}
    assert payload["history_id"] == "hist1" and payload["tool_version"] == "1.3.7+galaxy0"

    again, repeat = await svc.start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})
    assert repeat is True and again["id"] == run["id"]
    assert gi.make_post_request.call_count == 1


async def test_failed_start_can_be_retried():
    gi = _fake_gi()
    gi.make_post_request.side_effect = GalaxyConnectionError("bad", body='{"err_msg": "Invalid input", "err_code": 400008}', status_code=400)
    svc = _service(gi)
    with pytest.raises(GalaxyRunsError) as e:
        await svc.start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})
    assert e.value.status_code == 422 and e.value.detail == "Invalid input"
    gi.make_post_request.side_effect = None
    gi.make_post_request.return_value = {"jobs": [{"id": "j1"}], "outputs": []}
    run, repeat = await svc.start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})
    assert repeat is False and run["status"] == "queued"


async def test_read_run_reports_progress_outputs_and_failure():
    gi = _fake_gi()
    svc = _service(gi)
    run, _ = await svc.start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})

    running = await svc.read_run(run["id"])
    assert running["status"] == "running"
    assert running["progress"]["jobs_done"] == 1
    assert running["outputs"] == [{"id": "o1", "name": "fastp on S1", "format": "fastqsanger.gz"}]

    jobs = {"j1": {"state": "ok", "outputs": {"out1": {"id": "o1"}}},
            "j2": {"state": "error", "outputs": {}, "tool_stderr": "line 1\nEXITING because of fatal ERROR"}}
    gi.jobs.show_job.side_effect = lambda job_id, full_details=False: {"id": job_id, **jobs[job_id]}
    failed = await svc.read_run(run["id"])
    assert failed["status"] == "failed"
    assert failed["error"] == {"step": None, "message": "line 1\nEXITING because of fatal ERROR"}

    calls = gi.jobs.show_job.call_count
    assert (await svc.read_run(run["id"]))["status"] == "failed"
    assert gi.jobs.show_job.call_count == calls  # a final run is not read from Galaxy again


async def test_a_job_galaxy_no_longer_has_ends_the_run_as_cancelled():
    gi = _fake_gi()
    svc = _service(gi)
    run, _ = await svc.start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})
    gi.jobs.show_job.side_effect = GalaxyConnectionError("gone", body="{}", status_code=404)
    removed = await svc.read_run(run["id"])
    assert removed["status"] == "cancelled"
    assert removed["error"] == {"step": None, "message": "Removed in Galaxy"}


async def test_another_users_run_is_not_found():
    store = MemoryStore()
    run, _ = await _service(_fake_gi(), store, "alice").start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})
    with pytest.raises(GalaxyRunsError) as e:
        await _service(_fake_gi(), store, "bob").read_run(run["id"])
    assert e.value.status_code == 404


async def test_cancel_asks_galaxy_and_returns_the_run():
    gi = _fake_gi()
    svc = _service(gi)
    run, _ = await svc.start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})
    cancelled = await svc.cancel_run(run["id"])
    assert gi.jobs.cancel_job.call_count == 2
    assert cancelled["id"] == run["id"]


async def test_link_opens_the_form_before_a_run_and_rerun_after():
    gi = _fake_gi()
    svc = _service(gi)
    first = await svc.link("t1", "prop_8", TARGET, INPUTS, {})
    assert first["kind"] == "tool_form" and first["not_carried"] == ["single_paired|in1"]
    await svc.start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})
    second = await svc.link("t1", "prop_8", TARGET, INPUTS, {})
    assert second == {"url": "https://galaxy.example.org/?job_id=j2", "kind": "rerun", "not_carried": []}


async def test_validate_reports_unknown_and_missing_inputs():
    gi = _fake_gi()
    svc = _service(gi)
    result = await svc.validate("t1", TARGET, {"single_paired|single_paired_selector": "paired",
                                                "single_paired|in1": {"src": "hda", "id": "a"}, "nope": 1}, {})
    assert result["valid"] is False
    assert result["errors"] == {"nope": "Unknown input", "single_paired|in2": "Required"}


async def test_thread_files_without_a_history_is_empty():
    svc = _service(_fake_gi())
    assert await svc.thread_files("t-new") == {"items": [], "next_cursor": None}


# ---------------------------------------------------------------- routes

def _client(svc):
    from fastapi import FastAPI
    from fastapi.exceptions import RequestValidationError
    from fastapi.testclient import TestClient
    from app.api.endpoints import galaxy_runs as routes
    from app.main import validation_exception_handler

    app = FastAPI()
    app.include_router(routes.router, prefix="/v1/galaxy")
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.dependency_overrides[routes.galaxy_runs] = lambda: svc
    return TestClient(app)


def test_invalid_body_is_400_and_a_repeat_is_200():
    client = _client(_service(_fake_gi()))
    body = {"thread_id": "t1", "proposal_id": "prop_8", "client_request_id": "req-1", "target": TARGET, "inputs": INPUTS}

    assert client.post("/v1/galaxy/runs", json={**body, "target": {"type": "dataset", "id": "x"}}).status_code == 400
    first = client.post("/v1/galaxy/runs", json=body)
    assert first.status_code == 202
    again = client.post("/v1/galaxy/runs", json=body)
    assert again.status_code == 200 and again.json()["id"] == first.json()["id"]
    assert client.get(f"/v1/galaxy/runs/{first.json()['id']}").status_code == 200
    assert client.get("/v1/galaxy/runs/run_missing").status_code == 404
    listed = client.get("/v1/galaxy/runs", params={"thread_id": "t1"}).json()
    assert [r["id"] for r in listed["items"]] == [first.json()["id"]] and listed["next_cursor"] is None
