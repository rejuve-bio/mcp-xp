from __future__ import annotations

import os
from dotenv import load_dotenv
from dataclasses import dataclass, fields
from typing import Any, Dict, List

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

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

    # Informer entity types to query (maps to Informer's entity_type param)
    entity_types: List[str] = None

    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = ["tool", "workflow"]

    @classmethod
    def from_dict(cls, overrides: Dict[str, Any] | None) -> "GraphRAGConfig":
        if not overrides:
            return cls()
        allowed_keys = {field_info.name for field_info in fields(cls)}
        filtered = {key: value for key, value in overrides.items() if key in allowed_keys}
        return cls(**filtered)
