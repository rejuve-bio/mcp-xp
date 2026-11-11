from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Id = Annotated[str, Field(min_length=1, max_length=256)]

THREAD = "8d2f0c1e-4b7a-4c39-9e21-5f6a7b8c9d0e"
FASTP = "toolshed.g2.bx.psu.edu/repos/iuc/fastp/fastp/1.3.7+galaxy0"


# ---------------------------------------------------------------- requests

class Target(BaseModel):
    type: Literal["tool", "workflow"]
    id: str = Field(..., min_length=1, max_length=512, description="A tool's full ToolShed id, or a stored workflow id")
    version: Optional[str] = None

    model_config = ConfigDict(json_schema_extra={"example": {"type": "tool", "id": FASTP, "version": "1.3.7+galaxy0"}})


class DescribeRequest(BaseModel):
    thread_id: Id
    target: Target

    model_config = ConfigDict(json_schema_extra={"example": {
        "thread_id": THREAD, "target": {"type": "tool", "id": FASTP, "version": "1.3.7+galaxy0"}}})


class ValidateRequest(BaseModel):
    thread_id: Id
    target: Target
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Galaxy's run format: '|'-joined paths, files as {src, id}")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="A workflow's runtime settings by step; {} for a tool")

    model_config = ConfigDict(json_schema_extra={"example": {
        "thread_id": THREAD, "target": {"type": "tool", "id": FASTP, "version": "1.3.7+galaxy0"},
        "inputs": {"single_paired|single_paired_selector": "paired",
                   "single_paired|in1": {"src": "hda", "id": "f2db41e1fa331b3e"},
                   "single_paired|in2": {"src": "hda", "id": "a7c3c7d2b8a64e1f"},
                   "filter_options|length_required": 30},
        "parameters": {}}})


class RunRequest(ValidateRequest):
    proposal_id: Id
    client_request_id: str = Field(..., min_length=1, max_length=128, description="Same id on a retry returns the stored run")

    model_config = ConfigDict(json_schema_extra={"example": {
        "thread_id": THREAD, "proposal_id": "prop_8", "client_request_id": "c2a8f4e1-7b3d-4d6a-8e5f-1a9b0c3d7e62",
        "target": {"type": "tool", "id": FASTP, "version": "1.3.7+galaxy0"},
        "inputs": {"single_paired|single_paired_selector": "paired",
                   "single_paired|in1": {"src": "hda", "id": "f2db41e1fa331b3e"},
                   "single_paired|in2": {"src": "hda", "id": "a7c3c7d2b8a64e1f"},
                   "filter_options|length_required": 30},
        "parameters": {}}})


class LinkRequest(ValidateRequest):
    proposal_id: Id

    model_config = ConfigDict(json_schema_extra={"example": {
        "thread_id": THREAD, "proposal_id": "prop_8",
        "target": {"type": "tool", "id": FASTP, "version": "1.3.7+galaxy0"},
        "inputs": {"single_paired|in1": {"src": "hda", "id": "f2db41e1fa331b3e"},
                   "filter_options|length_required": 30},
        "parameters": {}}})


# ---------------------------------------------------------------- responses (documentation of the contract)

class Problem(BaseModel):
    detail: Any = Field(..., description="FastAPI's error body; for 422, Galaxy's message per path")


class TargetOut(BaseModel):
    type: Literal["tool", "workflow"]
    id: str
    version: Optional[str] = None
    name: Optional[str] = None


class SearchItem(BaseModel):
    type: Literal["tool", "workflow"]
    id: str
    version: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None


class SearchList(BaseModel):
    items: List[SearchItem]
    next_cursor: Optional[str] = None


class FileMatch(BaseModel):
    src: Literal["hda", "hdca"]
    id: str
    name: Optional[str] = None
    hid: Optional[int] = None


class InputDescription(BaseModel):
    path: str = Field(..., description="Galaxy's run-format key, e.g. single_paired|in1")
    label: str
    type: str = Field(..., description="Galaxy's input type: data, data_collection, select, integer, text, boolean, repeat …")
    optional: Optional[bool] = None
    default: Any = None
    options: Optional[List[Any]] = None
    formats: Optional[List[str]] = None
    collection_types: Optional[List[str]] = None
    matches: Optional[List[FileMatch]] = Field(None, description="The thread's files Galaxy offers for this input")
    when: Optional[Dict[str, Any]] = Field(None, description="Applies only when these choices are made")
    selector: Optional[bool] = Field(None, description="This input is a conditional's choice")
    section: Optional[str] = Field(None, description="The section Galaxy groups it under (full description)")
    min: Optional[int] = None
    max: Optional[int] = None
    multiple: Optional[bool] = None

    model_config = ConfigDict(extra="allow")


class Description(BaseModel):
    target: TargetOut
    inputs: List[InputDescription]
    all_count: int = Field(..., description="Every setting the tool or workflow has")

    model_config = ConfigDict(json_schema_extra={"example": {
        "target": {"type": "tool", "id": FASTP, "version": "1.3.7+galaxy0", "name": "fastp"},
        "inputs": [
            {"path": "single_paired|single_paired_selector", "label": "Single-end or paired reads", "type": "select",
             "options": ["single", "paired", "paired_collection"], "default": "single", "selector": True},
            {"path": "single_paired|in1", "label": "Input 1", "type": "data", "formats": ["fastqsanger", "fastqsanger.gz"],
             "matches": [{"src": "hda", "id": "f2db41e1fa331b3e", "name": "reads_R1.fastq.gz", "hid": 1}]},
            {"path": "single_paired|in2", "label": "Input 2", "type": "data", "formats": ["fastqsanger", "fastqsanger.gz"],
             "when": {"single_paired|single_paired_selector": "paired"}}],
        "all_count": 41}})


class Validation(BaseModel):
    valid: bool
    errors: Dict[str, str] = Field(..., description="Per input path; includes 'Still uploading' for files not ready yet")

    model_config = ConfigDict(json_schema_extra={"example": {"valid": False, "errors": {"single_paired|in2": "Still uploading"}}})


class ThreadFile(BaseModel):
    src: Literal["hda", "hdca"]
    id: str
    name: Optional[str] = None
    hid: Optional[int] = None
    state: Optional[str] = None
    format: Optional[str] = None
    collection_type: Optional[str] = None


class ThreadFiles(BaseModel):
    items: List[ThreadFile]
    next_cursor: Optional[str] = None


class Progress(BaseModel):
    jobs_done: int
    jobs_total: int
    steps_done: int
    steps_total: int


class RunOutput(BaseModel):
    id: str
    name: Optional[str] = None
    format: Optional[str] = None


class RunError(BaseModel):
    step: Optional[str] = Field(None, description="null for a tool")
    message: str


class Run(BaseModel):
    id: str
    thread_id: str
    proposal_id: str
    client_request_id: str
    target: TargetOut
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    galaxy_state: Optional[str] = Field(None, description="Galaxy's own state, e.g. new, running, ok, paused")
    progress: Optional[Progress] = None
    history_id: Optional[str] = None
    outputs: List[RunOutput] = []
    error: Optional[RunError] = None
    created_at: str
    updated_at: str

    model_config = ConfigDict(json_schema_extra={"example": {
        "id": "run_31", "thread_id": THREAD, "proposal_id": "prop_8",
        "client_request_id": "c2a8f4e1-7b3d-4d6a-8e5f-1a9b0c3d7e62",
        "target": {"type": "tool", "id": FASTP, "version": "1.3.7+galaxy0", "name": "fastp"},
        "status": "completed", "galaxy_state": "ok",
        "progress": {"jobs_done": 1, "jobs_total": 1, "steps_done": 1, "steps_total": 1},
        "history_id": "1cd8e2f6b131e891",
        "outputs": [{"id": "d2a6e7f1b0c94a58", "name": "fastp report (HTML)", "format": "html"}],
        "error": None, "created_at": "2026-10-09T10:14:00Z", "updated_at": "2026-10-09T10:21:40Z"}})


class RunList(BaseModel):
    items: List[Run]
    next_cursor: Optional[str] = None


class Link(BaseModel):
    url: str
    kind: Literal["tool_form", "rerun", "workflow_landing"]
    not_carried: List[str] = Field(..., description="Paths the link can't hold; the details panel marks them 'Set in Galaxy'")


class Dataset(BaseModel):
    id: str
    name: str
    format: str = Field(..., description="'auto' until Galaxy detects the type")
    size: int
    state: str = Field(..., description="queued, running, ok or error: the upload is a Galaxy job")
    history_id: str
    created_at: str

    model_config = ConfigDict(json_schema_extra={"example": {
        "id": "f2db41e1fa331b3e", "name": "reads_R1.fastq.gz", "format": "fastqsanger.gz", "size": 1288490188,
        "state": "queued", "history_id": "1cd8e2f6b131e891", "created_at": "2026-10-09T10:02:00Z"}})
