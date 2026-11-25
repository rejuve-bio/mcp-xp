from sys import path
path.append(".")

from types import SimpleNamespace
from unittest.mock import DEFAULT, MagicMock
from urllib.parse import unquote

import asyncio
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


# ---------------------------------------------------------------- an RNA-STAR-like model (#100)
# The same names (GTFconditional, GTFselect, genomeDir, sjdbGTFfile) sit in several branches with
# different content; genomeDir is filled from a data table, filtered differently in each branch.

def _select(name, values, dynamic=False, value=None):
    return {"type": "select", "name": name, "label": name, "optional": False, "is_dynamic": dynamic,
            "value": value if value is not None else (values[0] if values else None),
            "options": [[v.upper(), v, i == 0] for i, v in enumerate(values)]}


def _gtf_file():
    return {"type": "data", "name": "sjdbGTFfile", "label": "Gene model", "optional": False,
            "extensions": ["gtf", "gff3"], "options": {"hda": [], "hdca": []}}


def _overhang():
    return {"type": "integer", "name": "sjdbOverhang", "label": "Overhang", "optional": False, "value": "100"}


def _gtf_conditional(cases):
    values = [v for v, _ in cases]
    return {"type": "conditional", "name": "GTFconditional", "test_param": _select("GTFselect", values),
            "cases": [{"value": v, "inputs": inputs} for v, inputs in cases]}


def _star_model(no_gtf=("hg38", "mm10"), with_gtf=("hg38_gencode",)):
    return {"id": "rna_star", "version": "2.7.11b+galaxy1", "name": "RNA STAR", "errors": {}, "inputs": [
        {"type": "conditional", "name": "refGenomeSource",
         "test_param": _select("geneSource", ["indexed", "history"]),
         "cases": [
             {"value": "indexed", "inputs": [_gtf_conditional([
                 ("without-gtf-with-gtf", [_select("genomeDir", list(no_gtf), dynamic=True), _gtf_file(), _overhang()]),
                 ("without-gtf", [_select("genomeDir", list(no_gtf), dynamic=True)]),
                 ("with-gtf", [_select("genomeDir", list(with_gtf), dynamic=True)]),
             ])]},
             {"value": "history", "inputs": [
                 {"type": "data", "name": "genomeFastaFiles", "label": "Reference", "optional": False,
                  "extensions": ["fasta"], "options": {"hda": [], "hdca": []}},
                 _gtf_conditional([("without-gtf", []), ("with-gtf", [_gtf_file(), _overhang()])]),
             ]},
         ]},
    ]}


SOURCE = "refGenomeSource|geneSource"
GTF = "refGenomeSource|GTFconditional|GTFselect"


def _variants(d, path):
    return [e for e in d["inputs"] if e["path"] == path]


def test_a_data_table_select_keeps_each_branchs_options():
    d = describe_inputs(_star_model(), full=True)
    genome = _variants(d, "refGenomeSource|GTFconditional|genomeDir")
    assert [(e["options"], e["when"]) for e in genome] == [
        (["hg38", "mm10"], {SOURCE: "indexed", GTF: ["without-gtf-with-gtf", "without-gtf"]}),
        (["hg38_gencode"], {SOURCE: "indexed", GTF: "with-gtf"}),
    ]
    assert not any(e.get("no_options") for e in genome)


def test_a_selector_in_two_branches_keeps_each_branchs_choices():
    d = describe_inputs(_star_model())
    assert [(e["options"], e["when"]) for e in _variants(d, GTF)] == [
        (["without-gtf-with-gtf", "without-gtf", "with-gtf"], {SOURCE: "indexed"}),
        (["without-gtf", "with-gtf"], {SOURCE: "history"}),
    ]
    assert all(e["selector"] for e in _variants(d, GTF))


def test_a_nested_input_keeps_its_inner_condition_in_every_outer_branch():
    d = describe_inputs(_star_model(), full=True)
    for path in ("refGenomeSource|GTFconditional|sjdbGTFfile", "refGenomeSource|GTFconditional|sjdbOverhang"):
        assert [e["when"] for e in _variants(d, path)] == [
            {GTF: "without-gtf-with-gtf", SOURCE: "indexed"},
            {GTF: "with-gtf", SOURCE: "history"},
        ]


def test_the_same_input_in_every_branch_stays_one_entry_without_a_condition():
    # fastp's in1 is in both branches; only the datasets Galaxy offers differ, and those are kept
    d = describe_inputs(TOOL_MODEL)
    in1 = _variants(d, "single_paired|in1")
    assert len(in1) == 1 and "when" not in in1[0] and in1[0]["matches"]


def test_a_data_table_select_galaxy_offers_nothing_for_says_so():
    d = describe_inputs(_star_model(no_gtf=()), full=True)
    genome = _variants(d, "refGenomeSource|GTFconditional|genomeDir")
    assert [(e["options"], e.get("no_options"), e["when"][GTF]) for e in genome] == [
        ([], True, ["without-gtf-with-gtf", "without-gtf"]),
        (["hg38_gencode"], None, "with-gtf"),
    ]
    # a static select is never flagged, even an empty one
    static = describe_inputs({"inputs": [_select("mode", [])]}, full=True)["inputs"][0]
    assert static["options"] == [] and "no_options" not in static


def test_any_tool_with_a_built_in_index_lists_it_hisat2_shape():
    model = {"inputs": [{"type": "conditional", "name": "reference_genome",
                         "test_param": _select("source", ["indexed", "history"]),
                         "cases": [
                             {"value": "indexed", "inputs": [_select("index", ["hg38", "mm39"], dynamic=True)]},
                             {"value": "history", "inputs": [
                                 {"type": "data", "name": "history_item", "label": "Reference", "optional": False,
                                  "extensions": ["fasta"], "options": {"hda": [], "hdca": []}}]},
                         ]}]}
    index = _variants(describe_inputs(model), "reference_genome|index")
    assert [(e["options"], e["when"]) for e in index] == [(["hg38", "mm39"], {"reference_genome|source": "indexed"})]


async def test_describe_route_keeps_no_options_and_every_branch():
    gi = _fake_gi()
    gi.tools.build.return_value = _star_model(no_gtf=())
    d = await _service(gi).describe("t1", {"type": "tool", "id": "rna_star", "version": "2.7.11b+galaxy1"}, full=True)
    from app.api.schemas.galaxy_runs import Description
    out = Description(**d).model_dump(exclude_none=True)
    genome = [e for e in out["inputs"] if e["path"] == "refGenomeSource|GTFconditional|genomeDir"]
    assert [(e["options"], e.get("no_options")) for e in genome] == [([], True), (["hg38_gencode"], None)]


def test_rna_star_on_dev_lists_the_genome_indexes():
    # Dev's real build model (refGenomeSource only, no datasets). "with-gtf" is the first case
    # and has no index on dev; the old first-wins merge returned its empty list for every case.
    import json
    from pathlib import Path
    model = json.loads((Path(__file__).parent / "fixtures" / "rna_star_build_refgenome.json").read_text())
    genome = _variants(describe_inputs(model), "refGenomeSource|GTFconditional|genomeDir")
    assert [(len(e["options"]), e.get("no_options"), e["when"][GTF]) for e in genome] == [
        (0, True, "with-gtf"), (18, None, ["without-gtf-with-gtf", "without-gtf"])]
    assert {"hg38", "mm10", "dm6"} <= set(genome[1]["options"])
    quant = _variants(describe_inputs(model, full=True), "refGenomeSource|GTFconditional|quantmode_output|quantMode")
    assert {len(e["options"]) for e in quant} == {1, 4}


def test_all_count_counts_each_setting_once_whatever_its_branches():
    d = describe_inputs(_star_model(), full=True)
    assert d["all_count"] == len({e["path"] for e in d["inputs"]}) == 6


def test_input_types_and_required_files_read_the_branch_chosen():
    types = input_types(_star_model())
    assert type_of(types, "refGenomeSource|GTFconditional|genomeDir")["type"] == "select"
    from app.galaxy_runs.service import _missing_required_files
    model = _star_model()
    # defaults: indexed + without-gtf-with-gtf, which needs a GTF
    assert _missing_required_files(model, {}) == {"refGenomeSource|GTFconditional|sjdbGTFfile": "Required"}
    # history + without-gtf: the FASTA only; the indexed branch's default GTFselect must not leak in
    assert _missing_required_files(model, {SOURCE: "history", GTF: "without-gtf"}) == {
        "refGenomeSource|genomeFastaFiles": "Required"}
    assert _missing_required_files(model, {SOURCE: "history"}) == {
        "refGenomeSource|genomeFastaFiles": "Required"}  # history's own default is without-gtf


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
    (["deleting"], ("cancelled", "deleted")),  # a final run keeps where Galaxy ends up, not a transitional state
    (["stop"], ("cancelled", "stopped")),
    (["ok", "deleting", "running"], ("running", "running")),
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


def test_a_cancelling_workflow_settles_as_cancelled():
    assert st.invocation_run_status("cancelling", {"paused": 1}) == ("cancelled", "cancelled")


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

    async def claim_import(self, u, c, claimed_at):
        if ("import", u, c) in self.requests:
            return False
        self.requests[("import", u, c)] = {"dataset": None, "claimed_at": claimed_at}
        return True

    async def get_import(self, u, c):
        record = self.requests.get(("import", u, c))
        return dict(record) if record is not None else None

    async def save_import(self, u, c, dataset):
        self.requests[("import", u, c)] = {"dataset": dict(dataset)}

    async def release_import(self, u, c):
        self.requests.pop(("import", u, c), None)


OWNERS = {"hist1": "alice", "hist-bob": "bob"}  # who owns each history in the fake Galaxy


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
    gi.datasets.show_dataset.return_value = {"id": "f2db", "name": "reads_R1", "extension": "fastqsanger.gz", "state": "ok",
                                             "history_id": "hist1"}
    gi.dataset_collections.show_dataset_collection.return_value = {"id": "c1", "history_id": "hist1",
                                                                    "populated_state": "ok", "elements": []}
    gi.histories.show_history.side_effect = lambda history_id, **kw: {
        "id": history_id, "username": OWNERS.get(history_id)}
    gi.jobs.show_job.side_effect = lambda job_id, full_details=False: {
        "j1": {"id": "j1", "state": "ok", "outputs": {"out1": {"id": "o1", "src": "hda"}}},
        "j2": {"id": "j2", "state": "running", "outputs": {"out1": {"id": "o2", "src": "hda"}}},
    }[job_id]
    return gi


def _service(gi, store=None, user="alice"):
    client = SimpleNamespace(gi_client=gi, galaxy_url="https://galaxy.example.org", user_api_key="user-key")
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


@pytest.mark.parametrize("job_state,settled", [("deleting", "deleted"), ("stop", "stopped")])
async def test_a_run_stored_as_final_keeps_a_settled_galaxy_state(job_state, settled):
    store = MemoryStore()
    gi = _fake_gi()
    svc = _service(gi, store)
    run, _ = await svc.start_run("t1", "prop_8", "req-1", TARGET, INPUTS, {})
    gi.jobs.show_job.side_effect = lambda job_id, full_details=False: {"id": job_id, "state": job_state, "outputs": {}}
    first = await svc.read_run(run["id"])
    assert (first["status"], first["galaxy_state"]) == ("cancelled", settled)
    stored = await _service(_fake_gi(), store).read_run(run["id"])
    assert (stored["status"], stored["galaxy_state"]) == ("cancelled", settled)


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


def _history_contents(gi, contents):
    """show_history(contents=True) lists these; other show_history calls keep whatever the fake already does."""
    previous = gi.histories.show_history.side_effect
    gi.histories.show_history.side_effect = lambda history_id, **kw: (
        contents if kw.get("contents") else previous(history_id, **kw) if previous else DEFAULT)
    return gi


async def test_thread_files_say_auto_while_galaxy_is_still_detecting_the_format():
    gi = _history_contents(_fake_gi(), [
        {"id": "d1", "hid": 1, "state": "queued", "extension": "data"},
        {"id": "d2", "hid": 2, "state": "running", "extension": "auto"},
        {"id": "d3", "hid": 3, "state": "ok", "extension": "fastqsanger"},
        {"id": "d4", "hid": 4, "state": "ok", "extension": "data"},
        {"id": "d5", "hid": 5, "state": "queued", "extension": "fastqsanger"},
    ])
    svc = _service(gi)
    await svc.thread_history("t1")
    files = await svc.thread_files("t1")
    assert [f["format"] for f in files["items"]] == ["auto", "auto", "fastqsanger", "data", "fastqsanger"]


GENOME_BUILD_CONTENTS = [
    {"id": "d1", "hid": 1, "state": "ok", "extension": "bam", "genome_build": "hg38"},
    {"id": "d2", "hid": 2, "state": "ok", "extension": "txt", "genome_build": "?"},  # Galaxy's "not set"
    {"id": "d3", "hid": 3, "state": "ok", "extension": "vcf"},
    {"id": "c1", "hid": 4, "history_content_type": "dataset_collection", "collection_type": "paired",
     "populated_state": "ok"},
]


async def test_thread_files_carry_the_genome_build():
    # The file service records a Galaxy output's build from here (#95).
    svc = _service(_history_contents(_fake_gi(), GENOME_BUILD_CONTENTS))
    await svc.thread_history("t1")
    files = (await svc.thread_files("t1"))["items"]
    assert [f.get("genome_build") for f in files] == ["hg38", None, None, None]


def test_thread_files_route_returns_the_genome_build():
    # The response model must carry the field, or FastAPI drops it.
    svc = _service(_history_contents(_fake_gi(), GENOME_BUILD_CONTENTS))
    asyncio.run(svc.thread_history("t1"))
    items = _client(svc).get("/v1/galaxy/threads/t1/files").json()["items"]
    assert items[0]["genome_build"] == "hg38" and items[1]["genome_build"] is None


# ---------------------------------------------------------------- workflow inputs by Galaxy label

WORKFLOW_TARGET = {"type": "workflow", "id": "wf1"}


def _workflow(*labels, formats=None):
    """Galaxy 26.1 show_workflow: steps carry no label; labels are in the top-level inputs map.

    Galaxy fills an unlabelled data input's label with "Input Dataset"."""
    labels = labels or ("reads",)
    steps = {str(i): {"id": i, "type": "data_input", "annotation": "",
                      "tool_inputs": {"optional": False, "format": formats or []}} for i in range(len(labels))}
    steps[str(len(labels))] = {"id": len(labels), "type": "tool", "tool_inputs": {}}
    return {"id": "wf1", "name": "QC", "version": 1, "steps": steps,
            "inputs": {str(i): {"label": label, "value": "", "uuid": f"u{i}"} for i, label in enumerate(labels)}}


def _workflow_gi(workflow, contents=()):
    gi = _history_contents(_fake_gi(), list(contents))
    gi.workflows.show_workflow.return_value = workflow
    return gi


async def test_workflow_describe_names_inputs_by_their_galaxy_label():
    described = await _service(_workflow_gi(_workflow())).describe("t1", WORKFLOW_TARGET)
    assert [(e["path"], e["label"]) for e in described["inputs"]] == [("reads", "reads")]


async def test_workflow_validate_accepts_the_label_runs_accept():
    svc = _service(_workflow_gi(_workflow()))
    ok = await svc.validate("t1", WORKFLOW_TARGET, {"reads": {"src": "hda", "id": "f2db"}}, {})
    assert ok == {"valid": True, "errors": {}}
    old = await svc.validate("t1", WORKFLOW_TARGET, {"Input 0": {"src": "hda", "id": "f2db"}}, {})
    assert old["errors"] == {"Input 0": "Unknown input", "reads": "Required"}


async def test_unlabelled_workflow_inputs_get_unique_stable_fallback_names():
    svc = _service(_workflow_gi(_workflow("Input Dataset", "reads", "Input Dataset")))
    first = await svc.describe("t1", WORKFLOW_TARGET)
    again = await svc.describe("t1", WORKFLOW_TARGET)
    assert [e["path"] for e in first["inputs"]] == [e["path"] for e in again["inputs"]] == ["Input 0", "reads",
                                                                                           "Input 2"]
    missing = await svc.validate("t1", WORKFLOW_TARGET, {"reads": {"src": "hda", "id": "f2db"},
                                                         "Input 0": {"src": "hda", "id": "f2db"}}, {})
    assert missing["errors"] == {"Input 2": "Required"}


async def test_a_run_accepts_the_fallback_name_of_an_unlabelled_input():
    gi = _workflow_gi(_workflow("Input Dataset", "reads"))
    gi.workflows.invoke_workflow.return_value = {"id": "inv1"}
    reads = {"src": "hda", "id": "f2db"}
    await _service(gi).start_run("t1", "prop_8", "req-1", WORKFLOW_TARGET, {"Input 0": reads, "reads": reads}, {})
    kwargs = gi.workflows.invoke_workflow.call_args.kwargs
    assert kwargs["inputs"] == {"0": reads, "reads": reads}
    assert kwargs["inputs_by"] == "step_index|name"


async def _invoked(workflow, inputs):
    gi = _workflow_gi(workflow)
    gi.workflows.invoke_workflow.return_value = {"id": "inv1"}
    await _service(gi).start_run("t1", "prop_8", "req-1", WORKFLOW_TARGET, inputs, {})
    return gi.workflows.invoke_workflow.call_args.kwargs["inputs"]


async def test_labelled_workflow_inputs_run_by_their_names_unchanged():
    reads, adapters = {"src": "hda", "id": "f2db"}, {"src": "hda", "id": "a1"}
    assert await _invoked(_workflow("reads", "adapters"), {"reads": reads, "adapters": adapters}) == {
        "reads": reads, "adapters": adapters}


async def test_a_real_label_that_looks_like_a_fallback_name_stays_a_name():
    workflow = _workflow("Input Dataset", "Input 0")
    described = await _service(_workflow_gi(workflow)).describe("t1", WORKFLOW_TARGET)
    assert [e["path"] for e in described["inputs"]] == ["Input 0 (unlabelled)", "Input 0"]
    reads, adapters = {"src": "hda", "id": "f2db"}, {"src": "hda", "id": "a1"}
    assert await _invoked(workflow, {"Input 0 (unlabelled)": reads, "Input 0": adapters}) == {
        "0": reads, "Input 0": adapters}


async def test_a_label_that_is_another_inputs_step_number_is_refused():
    gi = _workflow_gi(_workflow("1", "reads"))
    reads = {"src": "hda", "id": "f2db"}
    svc = _service(gi)
    result = await svc.validate("t1", WORKFLOW_TARGET, {"1": reads, "reads": reads}, {})
    assert result["errors"] == {"1": "Rename this input in Galaxy: its label is also a step number"}
    with pytest.raises(GalaxyRunsError) as e:
        await svc.start_run("t1", "prop_8", "req-1", WORKFLOW_TARGET, {"1": reads, "reads": reads}, {})
    assert e.value.status_code == 422 and e.value.detail == result["errors"]
    gi.workflows.invoke_workflow.assert_not_called()


async def test_workflow_describe_matches_a_file_by_its_galaxy_extension_while_detecting():
    gi = _workflow_gi(_workflow(formats=["data"]),
                      [{"id": "d1", "hid": 1, "name": "reads", "state": "queued", "extension": "data"}])
    described = await _service(gi).describe("t1", WORKFLOW_TARGET)
    assert described["inputs"][0]["matches"] == [{"src": "hda", "id": "d1", "name": "reads", "hid": 1}]


# ---------------------------------------------------------------- routes

def _client(svc):
    from fastapi import FastAPI
    from fastapi.exceptions import RequestValidationError
    from fastapi.testclient import TestClient
    from app.api.endpoints import galaxy_runs as routes
    from app.exceptions import validation_exception_handler

    app = FastAPI()
    app.include_router(routes.router, prefix="/v1/galaxy")
    app.include_router(routes.datasets_router, prefix="/v1/datasets")
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


# ---------------------------------------------------------------- 401/403 and file readiness

async def test_galaxy_403_is_not_found_and_401_stays_a_token_problem():
    gi = _fake_gi()
    svc = _service(gi)
    gi.tools.build.side_effect = GalaxyConnectionError("no", body='{"err_msg": "denied"}', status_code=403)
    with pytest.raises(GalaxyRunsError) as e:
        await svc.describe("t1", TARGET)
    assert e.value.status_code == 404
    gi.tools.build.side_effect = GalaxyConnectionError("no", body='{"err_msg": "bad key"}', status_code=401)
    with pytest.raises(GalaxyRunsError) as e:
        await svc.describe("t1", TARGET)
    assert e.value.status_code == 401


@pytest.mark.parametrize("state,expected", [("queued", "Still uploading"), ("running", "Still uploading"),
                                            ("error", "Upload failed"), ("ok", None), ("deferred", None)])
async def test_validate_says_still_uploading_until_the_file_is_ready(state, expected):
    gi = _fake_gi()
    gi.datasets.show_dataset.return_value = {"id": "f2db", "state": state, "history_id": "hist1"}
    result = await _service(gi).validate("t1", TARGET, INPUTS_HDA, {})
    assert result["errors"].get("single_paired|in1") == expected


INPUTS_HDA = {"single_paired|single_paired_selector": "single", "single_paired|in1": {"src": "hda", "id": "f2db"}}


# ---------------------------------------------------------------- another user's files (#60)

def _bobs_files(gi):
    """Galaxy lets anyone read a dataset by id by default; these ones are in bob's history."""
    gi.datasets.show_dataset.return_value = {"id": "bobs", "name": "bob.fastq", "extension": "fastqsanger",
                                             "state": "ok", "history_id": "hist-bob"}
    gi.dataset_collections.show_dataset_collection.return_value = {"id": "bobsc", "history_id": "hist-bob",
                                                                    "populated_state": "ok", "elements": []}


@pytest.mark.parametrize("ref", [{"src": "hda", "id": "bobs"}, {"src": "hdca", "id": "bobsc"}])
async def test_validate_treats_another_users_file_as_missing(ref):
    gi = _fake_gi()
    _bobs_files(gi)
    result = await _service(gi).validate("t1", TARGET, {**INPUTS_HDA, "single_paired|in1": ref}, {})
    assert result["errors"].get("single_paired|in1") == "No such file for this user"


@pytest.mark.parametrize("target,inputs", [
    (TARGET, {**INPUTS_HDA, "single_paired|in1": {"src": "hda", "id": "bobs"}}),
    (TARGET, {**INPUTS_HDA, "single_paired|in1": [{"src": "hda", "id": "f2db"}, {"src": "hdca", "id": "bobsc"}]}),
    ({"type": "workflow", "id": "wf1"}, {"reads": {"src": "hda", "id": "bobs"}}),
])
async def test_a_run_on_another_users_file_starts_nothing(target, inputs):
    gi = _fake_gi()
    _bobs_files(gi)
    _fake_workflow(gi)
    gi.datasets.show_dataset.side_effect = lambda i: {"id": i, "state": "ok",
                                                      "history_id": "hist1" if i == "f2db" else "hist-bob"}
    svc = _service(gi)
    with pytest.raises(GalaxyRunsError) as e:
        await svc.start_run("t1", "prop_8", "req-1", target, inputs, {})
    assert e.value.status_code == 422
    assert list(e.value.detail.values()) == ["No such file for this user"]
    gi.make_post_request.assert_not_called()
    gi.workflows.invoke_workflow.assert_not_called()
    assert svc.store.requests == {}  # a retry with the same client_request_id may start it later


async def test_a_run_on_own_files_checks_each_history_once():
    gi = _fake_gi()
    svc = _service(gi)
    inputs = {**INPUTS_HDA, "single_paired|in1": [{"src": "hda", "id": "f2db"}, {"src": "hda", "id": "f2db"}]}
    await svc.start_run("t1", "prop_8", "req-1", TARGET, inputs, {})
    assert gi.histories.show_history.call_count == 1


async def test_a_link_with_another_users_file_is_refused():
    gi = _fake_gi()
    _bobs_files(gi)
    with pytest.raises(GalaxyRunsError) as e:
        await _service(gi).link("t1", "prop_8", {"type": "workflow", "id": "wf1"},
                                {"reads": {"src": "hda", "id": "bobs"}}, {})
    assert e.value.status_code == 422
    gi.make_post_request.assert_not_called()


# Every other way Galaxy accepts to name a file: only {src: hda|hdca, id} is checked, so the rest are refused.
OTHER_WAYS = [
    "bobs",                                   # a bare encoded id
    1234, "1234", "1,2,3",                    # database ids
    {"src": "dce", "id": "x1"}, {"src": "ldda", "id": "x1"},
    "__collection_reduce__|bobsc",
    "dce:5", "hdca:5",                        # collection inputs
    ["bobs"], [{"src": "hda", "id": "f2db"}, "bobs"],
    {"values": [{"src": "dce", "id": "x1"}]}, {"batch": True, "values": ["bobs"]},
    {"src": "hda", "id": 1234},
    {"__class__": "RuntimeValue"},
]
IN1 = "single_paired|in1"
WF = {"type": "workflow", "id": "wf1"}


def _fake_workflow(gi):
    gi.workflows.show_workflow.return_value = {"id": "wf1", "version": 1, "name": "wf", "steps": {
        "0": {"id": 0, "type": "data_input", "label": "reads", "tool_inputs": {"optional": False}},
        "1": {"id": 1, "type": "data_collection_input", "label": None, "tool_inputs": {"name": "samples", "optional": True}},
        "2": {"id": 2, "type": "tool", "tool_id": TOOL_MODEL["id"], "tool_version": "1.3.7+galaxy0",
              "uuid": "u-2", "label": None, "tool_inputs": {}},
        "3": {"id": 3, "type": "parameter_input", "label": "min_len", "tool_inputs": {"parameter_type": "integer",
                                                                                    "optional": True}},
    }, "inputs": {  # as Galaxy builds it: the step label, else tool_inputs' name
        "0": {"label": "reads"}, "1": {"label": "samples"}, "3": {"label": "min_len"}}}
    gi.workflows.invoke_workflow.return_value = {"id": "inv1"}


def _with_bob(gi):
    _bobs_files(gi)
    _fake_workflow(gi)
    gi.datasets.show_dataset.side_effect = lambda i: {"id": i, "state": "ok",
                                                      "history_id": "hist1" if i == "f2db" else "hist-bob"}
    return gi


def _refused(e, path=IN1, message="No such file for this user"):
    assert e.value.status_code == 422
    assert e.value.detail == {path: message}


@pytest.mark.parametrize("value", OTHER_WAYS)
async def test_a_tool_run_with_a_file_named_another_way_starts_nothing(value):
    gi = _with_bob(_fake_gi())
    with pytest.raises(GalaxyRunsError) as e:
        await _service(gi).start_run("t1", "prop_8", "req-1", TARGET, {**INPUTS_HDA, IN1: value}, {})
    _refused(e)
    gi.make_post_request.assert_not_called()


@pytest.mark.parametrize("value", OTHER_WAYS)
async def test_a_workflow_run_with_a_file_named_another_way_starts_nothing(value):
    for label in ("reads", "samples"):
        gi = _with_bob(_fake_gi())
        with pytest.raises(GalaxyRunsError) as e:
            await _service(gi).start_run("t1", "prop_8", "req-1", WF, {label: value}, {})
        _refused(e, label)
        gi.workflows.invoke_workflow.assert_not_called()


@pytest.mark.parametrize("value", OTHER_WAYS)
async def test_validate_treats_a_file_named_another_way_as_missing_and_never_sends_it(value):
    gi = _with_bob(_fake_gi())
    result = await _service(gi).validate("t1", TARGET, {**INPUTS_HDA, IN1: value}, {})
    assert result["errors"].get(IN1) == "No such file for this user"
    for call in gi.tools.build.call_args_list:
        assert IN1 not in (call.kwargs.get("inputs") or {})
    result = await _service(gi).validate("t1", WF, {"reads": value}, {})
    assert result["errors"].get("reads") == "No such file for this user"


@pytest.mark.parametrize("value", OTHER_WAYS)
async def test_a_link_with_a_file_named_another_way_is_refused(value):
    for target, path in ((TARGET, IN1), (WF, "reads")):
        gi = _with_bob(_fake_gi())
        inputs = {**INPUTS_HDA, IN1: value} if target is TARGET else {"reads": value}
        with pytest.raises(GalaxyRunsError) as e:
            await _service(gi).link("t1", "prop_8", target, inputs, {})
        _refused(e, path)
        gi.make_post_request.assert_not_called()


async def test_validate_never_sends_another_users_file_to_galaxys_build():
    gi = _with_bob(_fake_gi())
    gi.tools.build.side_effect = lambda *a, **kw: {**TOOL_MODEL, "errors": (
        {IN1: "Dataset bobs has been deleted"} if IN1 in (kw.get("inputs") or {}) else {})}
    result = await _service(gi).validate("t1", TARGET, {**INPUTS_HDA, IN1: {"src": "hda", "id": "bobs"}}, {})
    assert result["errors"] == {IN1: "No such file for this user"}


@pytest.mark.parametrize("inputs", [
    {"single_paired": {"single_paired_selector": "single", "in1": "bobs"}},  # Galaxy's nested form
    {"secret": "bobs"},                                                       # a hidden input
])
async def test_a_tool_run_with_an_input_the_tool_does_not_show_starts_nothing(inputs):
    gi = _with_bob(_fake_gi())
    with pytest.raises(GalaxyRunsError) as e:
        await _service(gi).start_run("t1", "prop_8", "req-1", TARGET, inputs, {})
    _refused(e, next(iter(inputs)), "Unknown input")
    gi.make_post_request.assert_not_called()


@pytest.mark.parametrize("parameters,path", [
    ({"2": {IN1: {"src": "hda", "id": "bobs"}}}, f"2|{IN1}"),
    ({"2": {"single_paired": {"in1": {"src": "hdca", "id": "bobsc"}}}}, f"2|{IN1}"),  # nested, as Galaxy flattens it
    ({"2": {IN1: "bobs"}}, f"2|{IN1}"),
    ({"2": {IN1: {"src": "dce", "id": "x1"}}}, f"2|{IN1}"),
    ({TOOL_MODEL["id"]: {IN1: "1234"}}, f"2|{IN1}"),                   # by tool id
    ({"u-2": {"param": IN1, "value": "bobs"}}, f"2|{IN1}"),            # by uuid, one-parameter form
    ({"2": {"secret": "bobs"}}, "2|secret"),                            # an input the tool does not show
    ({"9": {"x": 1}}, "9"),                                             # no such step
    ({"0": {"input": "bobs"}}, "0|input"),                              # an input step
    ({"3": {"x": {"src": "hda", "id": "bobs"}}}, "3|x"),
])
async def test_workflow_parameters_with_another_users_file_start_nothing(parameters, path):
    gi = _with_bob(_fake_gi())
    for call in ("start_run", "link"):
        svc = _service(gi)
        args = ("t1", "prop_8", "req-1") if call == "start_run" else ("t1", "prop_8")
        with pytest.raises(GalaxyRunsError) as e:
            await getattr(svc, call)(*args, WF, {"reads": {"src": "hda", "id": "f2db"}}, parameters)
        assert e.value.status_code == 422 and list(e.value.detail) == [f"parameters|{path}"]
    gi.workflows.invoke_workflow.assert_not_called()
    gi.make_post_request.assert_not_called()
    result = await _service(gi).validate("t1", WF, {"reads": {"src": "hda", "id": "f2db"}}, parameters)
    assert list(result["errors"]) == [f"parameters|{path}"]


async def test_a_workflow_run_with_own_files_and_plain_parameters_starts():
    gi = _with_bob(_fake_gi())
    parameters = {"2": {"filter_options": {"length_required": 30}, IN1: {"src": "hda", "id": "f2db"}}}
    run, _ = await _service(gi).start_run("t1", "prop_8", "req-1", WF,
                                          {"reads": {"src": "hda", "id": "f2db"}, "samples": None, "min_len": 5},
                                          parameters)
    assert run["status"] == "queued"
    assert gi.workflows.invoke_workflow.call_args.kwargs["params"] == parameters


async def test_files_are_checked_at_the_same_time_and_each_history_once():
    import asyncio
    gi = _fake_gi()
    started, release = [], asyncio.Event()

    def show(i):
        started.append(i)
        return {"id": i, "state": "ok", "history_id": "hist1"}
    gi.datasets.show_dataset.side_effect = show
    svc = _service(gi)
    original = svc._call

    async def slow_call(fn, *a, **kw):
        if fn is gi.datasets.show_dataset:
            result = await original(fn, *a, **kw)
            if len(started) == 3:
                release.set()
            await asyncio.wait_for(release.wait(), 2)  # only returns once all three were asked for
            return result
        return await original(fn, *a, **kw)
    svc._call = slow_call
    inputs = {**INPUTS_HDA, IN1: [{"src": "hda", "id": i} for i in ("a", "b", "c")]}
    await svc.start_run("t1", "prop_8", "req-1", TARGET, inputs, {})
    assert sorted(started) == ["a", "b", "c"]
    assert gi.histories.show_history.call_count == 1


# ---------------------------------------------------------------- /v1/datasets

import httpx  # noqa: E402

from app.galaxy_runs import datasets  # noqa: E402


def _multipart(fields_before, file_bytes, fields_after=(), boundary="XyZ"):
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
             for k, v in fields_before]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="reads_R1.fastq.gz"\r\n'
                 f'Content-Type: application/gzip\r\n\r\n'.encode() + file_bytes + b"\r\n")
    parts += [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
              for k, v in fields_after]
    return b"".join(parts) + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


async def _chunks(body, size=7):
    for i in range(0, len(body), size):
        yield body[i:i + size]


def _galaxy_fetch(seen):
    async def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = await request.aread()
        return httpx.Response(200, json={"outputs": [{"id": "ds1", "name": "reads_R1.fastq.gz", "file_ext": "auto",
                                                     "state": "queued", "create_time": "2026-10-09T10:02:00.123"}],
                                         "jobs": [{"id": "j9"}]})
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_upload_streams_the_file_into_the_threads_history():
    gi = _fake_gi()
    svc = _service(gi)
    seen = {}
    payload = b"@r1\nACGT\n+\nFFFF\n" * 50
    body, ctype = _multipart([("thread_id", "t1")], payload, [("format", "fastqsanger.gz")])
    dataset = await datasets.upload(svc, ctype, str(len(body)), _chunks(body), http=_galaxy_fetch(seen))

    assert dataset == {"id": "ds1", "name": "reads_R1.fastq.gz", "format": "auto", "size": len(payload),
                       "state": "queued", "history_id": "hist1", "created_at": "2026-10-09T10:02:00Z"}
    assert seen["url"] == "https://galaxy.example.org/api/tools/fetch"
    assert seen["headers"]["x-api-key"] == "user-key"
    sent = seen["body"]
    assert b'name="files_0|file_data"; filename="reads_R1.fastq.gz"' in sent
    assert payload in sent
    assert b'name="history_id"\r\n\r\nhist1' in sent
    # format arrived after the file and still reaches Galaxy
    assert b'"ext": "fastqsanger.gz"' in sent


async def test_upload_needs_thread_id_first_a_file_and_stays_under_the_cap(monkeypatch):
    svc = _service(_fake_gi())
    body, ctype = _multipart([], b"data", [("thread_id", "t1")])
    with pytest.raises(GalaxyRunsError) as e:
        await datasets.upload(svc, ctype, None, _chunks(body), http=_galaxy_fetch({}))
    assert e.value.status_code == 400

    body = b'--XyZ\r\nContent-Disposition: form-data; name="thread_id"\r\n\r\nt1\r\n--XyZ--\r\n'
    with pytest.raises(GalaxyRunsError) as e:
        await datasets.upload(svc, "multipart/form-data; boundary=XyZ", None, _chunks(body), http=_galaxy_fetch({}))
    assert e.value.status_code == 400

    monkeypatch.setattr(datasets, "UPLOAD_MAX_BYTES", 100)
    body, ctype = _multipart([("thread_id", "t1")], b"x" * 500)
    with pytest.raises(GalaxyRunsError) as e:
        await datasets.upload(svc, ctype, None, _chunks(body), http=_galaxy_fetch({}))
    assert e.value.status_code == 413
    with pytest.raises(GalaxyRunsError) as e:  # refused before reading when Content-Length says so
        await datasets.upload(svc, ctype, str(2_000_000), _chunks(body), http=_galaxy_fetch({}))
    assert e.value.status_code == 413


@pytest.mark.parametrize("chunk_size", [1_000_000, 7], ids=["one-chunk", "many-chunks"])
async def test_upload_rejects_thread_id_after_the_file_however_the_body_arrives(chunk_size):
    svc = _service(_fake_gi())
    seen = {}
    body, ctype = _multipart([("format", "txt")], b"data", [("thread_id", "t1")])
    with pytest.raises(GalaxyRunsError) as e:
        await datasets.upload(svc, ctype, str(len(body)), _chunks(body, chunk_size), http=_galaxy_fetch(seen))
    assert (e.value.status_code, e.value.detail) == (400, "thread_id must come before the file")
    assert seen == {}  # nothing reached Galaxy


@pytest.mark.parametrize("chunk_size", [1_000_000, 7], ids=["one-chunk", "many-chunks"])
async def test_upload_takes_format_before_or_after_the_file(chunk_size):
    for before, after in [([("thread_id", "t1"), ("format", "txt")], []), ([("thread_id", "t1")], [("format", "txt")])]:
        seen = {}
        body, ctype = _multipart(before, b"data", after)
        dataset = await datasets.upload(_service(_fake_gi()), ctype, str(len(body)), _chunks(body, chunk_size),
                                        http=_galaxy_fetch(seen))
        assert dataset["id"] == "ds1"
        assert b'"ext": "txt"' in seen["body"]


def _galaxy_display(seen, status=206, body=b"ACGT"):
    async def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["range"] = request.headers.get("range")
        headers = {"content-type": "application/octet-stream", "content-length": str(len(body))}
        if status == 206:
            headers["content-range"] = f"bytes 0-{len(body) - 1}/1000"
        return httpx.Response(status, headers=headers, content=_chunks(body, 2))  # streamed, as Galaxy does
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _read(response):
    return b"".join([chunk async for chunk in response.body_iterator])


async def test_download_streams_with_range_and_names_the_file():
    gi = _fake_gi()
    svc = _service(gi)
    seen = {}
    response = await datasets.download(svc, "f2db", "bytes=0-3", None, http=_galaxy_display(seen))
    assert response.status_code == 206
    assert await _read(response) == b"ACGT"
    assert seen["range"] == "bytes=0-3"
    assert seen["url"] == "https://galaxy.example.org/api/datasets/f2db/display?to_ext=fastqsanger.gz"
    assert response.headers["content-range"] == "bytes 0-3/1000"
    assert response.headers["content-disposition"].startswith('attachment; filename="reads_R1"')


@pytest.mark.parametrize("asked, sent", [
    ("bytes=0-1048575", "bytes=0-19"),  # a 1 MB preview of a 20-byte file gets the whole file (#91)
    ("bytes=5-9", "bytes=5-9"),         # a range inside the file is left alone
    ("bytes=5-", "bytes=5-"),           # open-ended and suffix ranges already fit
    ("bytes=-4", "bytes=-4"),
    ("bytes=20-30", "bytes=20-30"),     # starts past the end: Galaxy's 416 is right
])
async def test_download_shortens_a_range_that_runs_past_the_end(asked, sent):
    gi = _fake_gi()
    gi.datasets.show_dataset.return_value = {**gi.datasets.show_dataset.return_value, "file_size": 20}
    seen = {}
    response = await datasets.download(_service(gi), "f2db", asked, None, http=_galaxy_display(seen))
    assert seen["range"] == sent
    await _read(response)


async def test_download_html_gets_the_page_not_a_zip():
    # With to_ext, Galaxy zips an HTML dataset with its extra files (#90).
    gi = _fake_gi()
    gi.datasets.show_dataset.return_value = {**gi.datasets.show_dataset.return_value, "name": "fastp report",
                                             "extension": "html"}
    seen = {}
    response = await datasets.download(_service(gi), "f2db", None, None, http=_galaxy_display(seen, 200))
    assert seen["url"] == "https://galaxy.example.org/api/datasets/f2db/display"
    assert 'filename="fastp report"' in response.headers["content-disposition"]
    await _read(response)


async def test_download_index_files_and_errors():
    gi = _fake_gi()
    svc = _service(gi)
    seen = {}
    response = await datasets.download(svc, "f2db", None, "bam_index", http=_galaxy_display(seen, 200))
    assert seen["url"].endswith("/api/datasets/f2db/metadata_file?metadata_file=bam_index")
    assert 'filename="reads_R1.bai"' in response.headers["content-disposition"]
    await _read(response)

    with pytest.raises(GalaxyRunsError) as e:
        await datasets.download(svc, "f2db", None, "../etc", http=_galaxy_display({}))
    assert e.value.status_code == 400

    _bobs_files(gi)  # Galaxy shows another user's dataset by id, so mcp-xp must refuse it
    with pytest.raises(GalaxyRunsError) as e:
        await datasets.download(svc, "bobs", None, None, http=_galaxy_display({}))
    assert e.value.status_code == 404

    for galaxy_code in (403, 400):  # another user's private dataset; an id Galaxy can't decode
        gi.datasets.show_dataset.side_effect = GalaxyConnectionError("no", body="{}", status_code=galaxy_code)
        with pytest.raises(GalaxyRunsError) as e:
            await datasets.download(svc, "other-users", None, None, http=_galaxy_display({}))
        assert e.value.status_code == 404


def test_upload_and_download_routes(monkeypatch):
    seen = {}
    client = _client(_service(_fake_gi()))
    monkeypatch.setattr(datasets, "_default_http", _galaxy_fetch(seen))
    up = client.post("/v1/datasets/upload", data={"thread_id": "t1", "format": "fastqsanger.gz"},
                     files={"file": ("reads_R1.fastq.gz", b"@r1\nACGT\n+\nFFFF\n", "application/gzip")})
    assert up.status_code == 201 and up.json()["id"] == "ds1" and up.json()["size"] == 16

    monkeypatch.setattr(datasets, "_default_http", _galaxy_display({}))
    down = client.get("/v1/datasets/f2db/download", headers={"Range": "bytes=0-3"})
    assert down.status_code == 206 and down.content == b"ACGT"
    assert client.get("/v1/datasets/f2db/download", params={"metadata_file": "BAD!"}).status_code == 400


# ---------------------------------------------------------------- /v1/datasets/import (#74)

LINK = "http://127.0.0.1:18800/v1/links/SECRETTOKEN123/sample.tsv"
SHA = "a" * 64


def _galaxy_fetch_json(gi, state="deferred", ext="tabular"):
    gi.make_post_request.return_value = {
        "outputs": [{"id": "ds7", "name": "sample.tsv", "state": state, "extension": ext}], "jobs": [{"id": "j7"}]}


async def test_import_makes_a_deferred_dataset_from_the_link():
    gi = _fake_gi()
    _galaxy_fetch_json(gi)
    svc = _service(gi)
    dataset, repeat = await svc.import_dataset("t1", LINK, "sample.tsv", True, "d_1", SHA, "tabular")

    assert repeat is False
    assert dataset == {"id": "ds7", "name": "sample.tsv", "state": "deferred", "format": "tabular",
                       "history_id": "hist1"}
    url, = gi.make_post_request.call_args.args
    payload = gi.make_post_request.call_args.kwargs["payload"]
    assert url == "https://galaxy.example.org/api/tools/fetch"
    assert payload["history_id"] == "hist1"
    element, = payload["targets"][0]["elements"]
    assert payload["targets"][0]["destination"] == {"type": "hdas"}
    assert element == {"src": "url", "url": LINK, "name": "sample.tsv", "ext": "tabular", "deferred": True,
                       "hashes": [{"hash_function": "SHA-256", "hash_value": SHA}]}


async def test_a_repeated_import_returns_the_first_dataset_and_makes_no_second():
    gi = _fake_gi()
    _galaxy_fetch_json(gi)
    store = MemoryStore()
    svc = _service(gi, store)
    first, _ = await svc.import_dataset("t1", LINK, "sample.tsv", True, "d_1", SHA, None)
    again, repeat = await svc.import_dataset("t1", LINK, "sample.tsv", True, "d_1", SHA, None)
    assert repeat is True and again == first
    assert gi.make_post_request.call_count == 1

    # the same request id from another user is that user's own request
    gi.histories.create_history.return_value = {"id": "hist-bob"}
    _, repeat = await _service(gi, store, "bob").import_dataset("t1", LINK, "sample.tsv", True, "d_1", SHA, None)
    assert repeat is False and gi.make_post_request.call_count == 2


@pytest.mark.parametrize("galaxy_code,status", [(400, 422), (403, 422), (404, 422), (500, 502)])
async def test_a_refused_import_never_shows_the_link_and_can_be_retried(galaxy_code, status, caplog):
    gi = _fake_gi()
    gi.make_post_request.side_effect = GalaxyConnectionError(
        "no", body=f'{{"err_msg": "Invalid URL {LINK} for this server"}}', status_code=galaxy_code)
    svc = _service(gi)
    with pytest.raises(GalaxyRunsError) as e:
        await svc.import_dataset("t1", LINK, "sample.tsv", True, "d_1", None, None)
    assert e.value.status_code == status
    assert "SECRETTOKEN123" not in str(e.value.detail) and "SECRETTOKEN123" not in caplog.text
    if galaxy_code == 400:
        assert "Invalid URL" in e.value.detail  # Galaxy's own reason still reaches the user

    gi.make_post_request.side_effect = None
    _galaxy_fetch_json(gi)
    _, repeat = await svc.import_dataset("t1", LINK, "sample.tsv", True, "d_1", None, None)
    assert repeat is False  # nothing was made, so the same request id may try again


def test_import_route_answers_201_then_200_and_never_echoes_the_link():
    gi = _fake_gi()
    _galaxy_fetch_json(gi)
    client = _client(_service(gi))
    body = {"thread_id": "t1", "url": LINK, "name": "sample.tsv", "deferred": True,
            "client_request_id": "d_" + "0" * 32, "sha256": SHA, "format": "tabular"}

    first = client.post("/v1/datasets/import", json=body)
    assert first.status_code == 201
    assert first.json() == {"id": "ds7", "name": "sample.tsv", "state": "deferred", "format": "tabular",
                            "history_id": "hist1"}
    again = client.post("/v1/datasets/import", json=body)
    assert again.status_code == 200 and again.json() == first.json()

    # sha256 and format may be null, as the file service sends them
    assert client.post("/v1/datasets/import",
                       json={**body, "client_request_id": "d_2", "sha256": None}).status_code == 201

    for bad in ({**body, "url": LINK.replace("http://", "ftp://")}, {**body, "sha256": "nothex"},
                {k: v for k, v in body.items() if k != "client_request_id"}, {**body, "format": "a b"}):
        refused = client.post("/v1/datasets/import", json=bad)
        assert refused.status_code == 400
        assert "SECRETTOKEN123" not in refused.text


async def test_galaxy_making_no_dataset_is_502_and_lets_the_request_try_again():
    gi = _fake_gi()
    gi.make_post_request.return_value = {"outputs": [], "jobs": []}
    svc = _service(gi)
    with pytest.raises(GalaxyRunsError) as e:
        await svc.import_dataset("t1", LINK, "sample.tsv", True, "d_1", None, None)
    assert e.value.status_code == 502
    _galaxy_fetch_json(gi)
    _, repeat = await svc.import_dataset("t1", LINK, "sample.tsv", True, "d_1", None, None)
    assert repeat is False


async def test_an_import_left_pending_by_a_crash_is_taken_over_after_a_while(monkeypatch):
    from app.galaxy_runs import service as service_module
    gi = _fake_gi()
    _galaxy_fetch_json(gi)
    svc = _service(gi)
    await svc.store.claim_import("alice", "d_1", 1000.0)  # a request that died before saving
    monkeypatch.setattr(service_module.time, "time", lambda: 1000.0 + service_module.IMPORT_STALE_SECONDS + 1)
    dataset, repeat = await svc.import_dataset("t1", LINK, "sample.tsv", True, "d_1", None, None)
    assert repeat is False and dataset["id"] == "ds7"


@pytest.mark.parametrize("galaxy_code", [403, 404])
async def test_import_never_answers_404_even_when_the_thread_history_fails(galaxy_code):
    gi = _fake_gi()
    gi.histories.create_history.side_effect = GalaxyConnectionError("no", body="{}", status_code=galaxy_code)
    with pytest.raises(GalaxyRunsError) as e:
        await _service(gi).import_dataset("t1", LINK, "sample.tsv", True, "d_1", None, None)
    assert e.value.status_code == 422


async def test_links_are_redacted_inside_lists_and_when_url_encoded(caplog):
    gi = _fake_gi()
    encoded = LINK.replace("/", "%2F")
    gi.make_post_request.side_effect = GalaxyConnectionError(
        "no", body=f'{{"err_msg": ["bad {encoded}", "and {LINK}"]}}', status_code=500)
    with pytest.raises(GalaxyRunsError):
        await _service(gi).import_dataset("t1", LINK, "sample.tsv", True, "d_1", None, None)
    assert "SECRETTOKEN123" not in caplog.text
