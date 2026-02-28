from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Tuple


# Graph Schema Constants

NODE_LABELS: List[str] = [
    "Workflow", "Step", "Tool", "ToolInput", "ToolOutput",
    "Input", "Output", "Category",
]

NODE_ATTRIBUTES: Dict[str, List[str]] = {
    "Workflow": [
        "category", "workflow_id", "file_name", "number_of_steps",
        "workflow_repository", "readme_content", "raw_download_url",
    ],
    "Step": [
        "name", "file_name", "workflow_repository", "step_uid",
        "annotation", "tool_version", "tool_id", "type", "step_id",
    ],
    "Tool": [
        "name", "tool_id", "owner", "tool_category", "id",
        "tool_shed_url", "version", "help", "description",
    ],
    "ToolInput": ["id", "tool_input_uid", "input_name", "input_type"],
    "ToolOutput": ["id", "tool_output_uid", "output_name", "output_format"],
    "Input": [
        "name", "file_name", "workflow_repository", "step_id",
        "description", "input_uid",
    ],
    "Output": [
        "name", "file_name", "workflow_repository", "step_id",
        "description", "output_uid",
    ],
    "Category": ["category_id", "name", "category"],
}

# (source_label, relationship_type, target_label)
SCHEMA_RELATIONSHIPS: List[Tuple[str, str, str]] = [
    ("Category",  "HAS_WORKFLOW",       "Workflow"),
    ("Workflow",   "HAS_STEP",          "Step"),
    ("Step",       "NEXT_STEP",         "Step"),
    ("Step",       "STEP_REQUIRES",     "Input"),
    ("Step",       "STEP_GENERATES",    "Output"),
    ("Step",       "STEP_FEEDS_INTO",   "Step"),
    ("Workflow",   "WORKFLOW_USES_TOOL", "Tool"),
    ("Step",       "STEP_USES_TOOL",    "Tool"),
    ("Category",   "HAS_TOOL",         "Tool"),
    ("ToolInput",  "TOOL_HAS_INPUT",   "Tool"),
    ("Tool",       "TOOL_HAS_OUTPUT",  "ToolOutput"),
    ("Tool",       "SIMILAR_TO",       "Tool"),
]

ALL_RELATIONSHIP_TYPES: List[str] = sorted(
    {rel for _, rel, _ in SCHEMA_RELATIONSHIPS}
)

# Mapping from node label to the primary ID field used for canonical IDs
NODE_ID_FIELDS: Dict[str, List[str]] = {
    "Workflow":   ["workflow_id", "file_name"],
    "Step":       ["step_uid", "step_id", "name"],
    "Tool":       ["tool_id", "id", "name"],
    "ToolInput":  ["tool_input_uid", "id"],
    "ToolOutput": ["tool_output_uid", "id"],
    "Input":      ["input_uid", "name"],
    "Output":     ["output_uid", "name"],
    "Category":   ["category_id", "name"],
}


# Pipeline Configuration

@dataclass
class GraphRAGConfig:
    """Configuration for the Graph-RAG pipeline."""

    # Semantic retrieval
    k_seed: int = 5

    # Graph expansion
    max_depth: int = 2
    max_hops: int = 3
    max_nodes: int = 30
    max_subgraph_nodes: int = 50

    # Relationship priority for traversal (most important first)
    relation_priority: List[str] = field(
        default_factory=lambda: [
            "HAS_STEP", "STEP_USES_TOOL", "NEXT_STEP",
            "STEP_REQUIRES", "STEP_GENERATES", "STEP_FEEDS_INTO",
            "WORKFLOW_USES_TOOL", "HAS_WORKFLOW", "HAS_TOOL",
            "TOOL_HAS_INPUT", "TOOL_HAS_OUTPUT", "SIMILAR_TO",
        ]
    )

    # Hybrid ranking weights
    weight_semantic: float = 0.6
    weight_graph: float = 0.3
    weight_type: float = 0.1

    node_type_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "Workflow": 1.0,
            "Step": 0.9,
            "Tool": 0.8,
            "Input": 0.7,
            "Output": 0.7,
            "Category": 0.6,
            "ToolInput": 0.6,
            "ToolOutput": 0.6,
        }
    )

    # Informer entity types to query (maps to Informer's entity_type param)
    entity_types: List[str] = field(
        default_factory=lambda: ["tool", "workflow"]
    )

    # Context assembly
    context_budget_chars: int = 2000

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, overrides: Dict[str, Any] | None) -> "GraphRAGConfig":
        if not overrides:
            return cls()
        allowed_keys = {field_info.name for field_info in fields(cls)}
        filtered = {key: value for key, value in overrides.items() if key in allowed_keys}
        return cls(**filtered)
