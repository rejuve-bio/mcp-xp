"""GraphRAG pipeline configuration.

Environment variables for Neo4j connection and Pydantic-based pipeline config.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Neo4j connection (from environment)
# ---------------------------------------------------------------------------

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ---------------------------------------------------------------------------
# Budget configuration
# ---------------------------------------------------------------------------


class BudgetConfig(BaseModel):
    """Execution budget defaults for the GraphRAG pipeline."""

    semantic_top_k: int = Field(default=8, ge=1, le=15)
    max_seeds: int = Field(default=5, ge=1, le=15)
    max_schemas_per_query: int = Field(default=5, ge=1, le=10)
    max_query_limit: int = Field(default=100, ge=1, le=500)
    path_max_hops: int = Field(default=4, ge=1, le=6)
    compare_max_hops: int = Field(default=3, ge=1, le=4)
    format_max_chars: int = Field(default=300, ge=50)
    planner_max_retries: int = Field(default=1, ge=0, le=3)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


class GraphRAGConfig(BaseModel):
    """Top-level configuration for the GraphRAG pipeline."""

    entity_types: list[str] = Field(default=["tool", "workflow"])
    budget: BudgetConfig = Field(default_factory=BudgetConfig)

    @classmethod
    def from_dict(cls, overrides: dict[str, Any] | None) -> "GraphRAGConfig":
        if not overrides:
            return cls()
        return cls(**{
            k: v for k, v in overrides.items()
            if k in {f.alias or name for name, f in cls.model_fields.items()}
        })
