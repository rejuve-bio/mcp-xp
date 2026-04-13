"""Galaxy Knowledge Graph schema constants.

Single source of truth for all node labels, relationship types, property
mappings, and ID fields used across the GraphRAG module.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Node labels
# ---------------------------------------------------------------------------

NODE_LABELS: set[str] = {
    "Workflow",
    "Step",
    "Tool",
    "ToolInput",
    "ToolOutput",
    "Input",
    "Output",
    "Category",
}

# ---------------------------------------------------------------------------
# Relationship types
# ---------------------------------------------------------------------------

EDGE_TYPES: set[str] = {
    "HAS_STEP",
    "STEP_USES_TOOL",
    "STEP_REQUIRES",
    "STEP_GENERATES",
    "TOOL_HAS_INPUT",
    "TOOL_HAS_OUTPUT",
    "WORKFLOW_USES_TOOL",
    "HAS_TOOL",
}

# Source → Target direction for each relationship
EDGE_DIRECTIONS: dict[str, tuple[str, str]] = {
    "HAS_STEP":           ("Workflow", "Step"),
    "STEP_USES_TOOL":     ("Step", "Tool"),
    "STEP_REQUIRES":      ("Step", "Input"),
    "STEP_GENERATES":     ("Step", "Output"),
    "TOOL_HAS_INPUT":     ("Tool", "ToolInput"),
    "TOOL_HAS_OUTPUT":    ("Tool", "ToolOutput"),
    "WORKFLOW_USES_TOOL": ("Workflow", "Tool"),
    "HAS_TOOL":           ("Category", "Tool"),
}

# ---------------------------------------------------------------------------
# Node ID fields (primary identifiers per label)
# ---------------------------------------------------------------------------

NODE_ID_FIELDS: dict[str, list[str]] = {
    "Workflow":   ["workflow_id", "file_name"],
    "Step":       ["step_uid", "step_id", "name"],
    "Tool":       ["tool_id", "id", "name"],
    "ToolInput":  ["tool_input_uid", "id"],
    "ToolOutput": ["tool_output_uid", "id"],
    "Input":      ["input_uid", "name"],
    "Output":     ["output_uid", "name"],
    "Category":   ["category_id", "name"],
}

# ---------------------------------------------------------------------------
# Property mappings for entity resolution
# ---------------------------------------------------------------------------

# Ordered (label, property) pairs for exact-match lookups
PROPERTY_MAPPING_ORDER: list[tuple[str, str]] = [
    ("Workflow", "workflow_id"),
    ("Workflow", "file_name"),
    ("Workflow", "name"),
    ("Step", "step_id"),
    ("Step", "step_uid"),
    ("Step", "file_name"),
    ("Step", "name"),
    ("Tool", "tool_id"),
    ("Tool", "id"),
    ("Tool", "name"),
    ("Input", "input_uid"),
    ("Input", "name"),
    ("Output", "output_uid"),
    ("Output", "name"),
    ("Category", "category_id"),
    ("Category", "name"),
    ("ToolInput", "tool_input_uid"),
    ("ToolInput", "input_name"),
    ("ToolOutput", "tool_output_uid"),
    ("ToolOutput", "output_name"),
]

# (label, property) pairs for fuzzy (CONTAINS) matching
FUZZY_MAPPING_FIELDS: list[tuple[str, str]] = [
    ("Workflow", "name"),
    ("Workflow", "file_name"),
    ("Step", "name"),
    ("Step", "file_name"),
    ("Tool", "name"),
]

# Metadata keys to extract from semantic hit results
SEMANTIC_META_KEYS: list[str] = [
    "id",
    "workflow_id",
    "step_id",
    "step_uid",
    "tool_id",
    "file_name",
    "name",
    "input_uid",
    "output_uid",
    "category_id",
    "tool_input_uid",
    "tool_output_uid",
]

# ---------------------------------------------------------------------------
# Known properties per label (used in LLM planner prompt)
# ---------------------------------------------------------------------------

NODE_PROPERTIES: dict[str, list[str]] = {
    "Workflow": [
        "workflow_id", "file_name", "name", "description",
        "annotation", "readme_content",
    ],
    "Step": [
        "step_uid", "step_id", "name", "file_name", "annotation",
    ],
    "Tool": [
        "tool_id", "id", "name", "description", "version",
        "help", "annotation", "readme_content",
        "tool_shed_url", "owner", "pagerank", "community",
    ],
    "ToolInput": [
        "tool_input_uid", "id", "input_name", "input_type",
    ],
    "ToolOutput": [
        "tool_output_uid", "id", "output_name", "output_format",
    ],
    "Input": [
        "input_uid", "name", "description",
    ],
    "Output": [
        "output_uid", "name", "description",
    ],
    "Category": [
        "category_id", "name",
    ],
}

# ---------------------------------------------------------------------------
# Filter operators supported by the Cypher builder
# ---------------------------------------------------------------------------

FILTER_OPERATORS: set[str] = {
    "eq",
    "neq",
    "contains",
    "starts_with",
    "in",
    "gt",
    "lt",
}
