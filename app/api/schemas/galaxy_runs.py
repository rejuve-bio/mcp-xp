from typing import Annotated, Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

Id = Annotated[str, Field(min_length=1, max_length=256)]


class Target(BaseModel):
    type: Literal["tool", "workflow"]
    id: str = Field(..., min_length=1, max_length=512, description="A tool's full ToolShed id, or a stored workflow id")
    version: Optional[str] = None


class DescribeRequest(BaseModel):
    thread_id: Id
    target: Target


class ValidateRequest(BaseModel):
    thread_id: Id
    target: Target
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Galaxy's run format: '|'-joined paths, files as {src, id}")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="A workflow's runtime settings by step; {} for a tool")


class RunRequest(ValidateRequest):
    proposal_id: Id
    client_request_id: str = Field(..., min_length=1, max_length=128, description="Same id on a retry returns the stored run")


class LinkRequest(ValidateRequest):
    proposal_id: Id
