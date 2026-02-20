from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List


@dataclass
class GraphRAGConfig:
    """Small config surface for a minimal Graph-RAG pipeline."""

    k_seed: int = 5
    max_depth: int = 2
    max_nodes: int = 30
    max_subgraph_nodes: int = 50
    relation_priority: List[str] = field(
        default_factory=lambda: ["HAS_STEP", "STEP_USES_TOOL", "NEXT_STEP", "STEP_REQUIRES"]
    )
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
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, overrides: Dict[str, Any] | None) -> "GraphRAGConfig":
        if not overrides:
            return cls()
        allowed_keys = {field_info.name for field_info in fields(cls)}
        filtered = {key: value for key, value in overrides.items() if key in allowed_keys}
        return cls(**filtered)
