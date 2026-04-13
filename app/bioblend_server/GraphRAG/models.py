"""Pydantic v2 typed contracts for the GraphRAG pipeline.

Defines the intermediate representation (CypherQuerySchema) between the LLM
planner and the deterministic Cypher builder, plus all pipeline data models.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.bioblend_server.GraphRAG.schema import (
    EDGE_TYPES,
    NODE_LABELS,
)


# ---------------------------------------------------------------------------
# CypherQuerySchema components
# ---------------------------------------------------------------------------


class PropertyFilter(BaseModel):
    """A single property condition for a WHERE clause."""

    property: str
    operator: Literal["eq", "neq", "contains", "starts_with", "in", "gt", "lt"]
    value: Any


class NodeMatch(BaseModel):
    """Describes which node(s) to anchor a query on."""

    label: str = Field(description="Node label, must be in NODE_LABELS")
    filters: list[PropertyFilter] = Field(default_factory=list)
    element_id: str | None = Field(
        default=None,
        description="Neo4j elementId() for pre-resolved entities",
    )
    alias: str = Field(
        default="n",
        description="Cypher variable name for referencing in expansions/returns",
    )

    @model_validator(mode="after")
    def _validate_label(self) -> "NodeMatch":
        if self.label not in NODE_LABELS:
            raise ValueError(
                f"Unknown node label {self.label!r}. "
                f"Allowed: {sorted(NODE_LABELS)}"
            )
        return self


class Expansion(BaseModel):
    """A single relationship traversal from a matched node."""

    from_alias: str = Field(description="Alias of the source node")
    relationship: str = Field(description="Relationship type, must be in EDGE_TYPES")
    direction: Literal["out", "in", "both"] = "out"
    target_label: str = Field(description="Target node label, must be in NODE_LABELS")
    target_alias: str = Field(description="Alias for the target node")
    optional: bool = Field(
        default=True,
        description="Use OPTIONAL MATCH (True) or MATCH (False)",
    )

    @model_validator(mode="after")
    def _validate_types(self) -> "Expansion":
        if self.relationship not in EDGE_TYPES:
            raise ValueError(
                f"Unknown relationship {self.relationship!r}. "
                f"Allowed: {sorted(EDGE_TYPES)}"
            )
        if self.target_label not in NODE_LABELS:
            raise ValueError(
                f"Unknown target label {self.target_label!r}. "
                f"Allowed: {sorted(NODE_LABELS)}"
            )
        return self


class PathSpec(BaseModel):
    """Shortest-path search between two nodes."""

    from_node: NodeMatch
    to_node: NodeMatch
    max_hops: int = Field(default=4, ge=1, le=6)
    relationship_types: list[str] = Field(
        default_factory=list,
        description="Empty = any relationship type. Each validated against EDGE_TYPES.",
    )

    @model_validator(mode="after")
    def _validate_relationship_types(self) -> "PathSpec":
        for rt in self.relationship_types:
            if rt not in EDGE_TYPES:
                raise ValueError(
                    f"Unknown relationship type {rt!r} in path spec. "
                    f"Allowed: {sorted(EDGE_TYPES)}"
                )
        return self

    @model_validator(mode="after")
    def _validate_distinct_aliases(self) -> "PathSpec":
        if self.from_node.alias == self.to_node.alias:
            raise ValueError(
                f"PathSpec from_node and to_node must have distinct aliases, "
                f"both are {self.from_node.alias!r}. "
                f"Use different alias values (e.g. 'a' and 'b')."
            )
        return self


class AggregateSpec(BaseModel):
    """Aggregation over matched/traversed results."""

    function: Literal["count", "collect"]
    input_alias: str = Field(description="Alias of the node to aggregate")
    group_by: str | None = Field(
        default=None,
        description="Property to group by, format: alias.property",
    )
    order_by: Literal["asc", "desc"] = "desc"


class CompareSpec(BaseModel):
    """Compare two entities via their connected nodes."""

    entity_a: NodeMatch
    entity_b: NodeMatch
    via_relationship: str = Field(description="Relationship to traverse for comparison")
    via_target_label: str = Field(description="Label of the nodes being compared")
    hops: int = Field(default=2, ge=1, le=3)

    @model_validator(mode="after")
    def _validate_types(self) -> "CompareSpec":
        if self.via_relationship not in EDGE_TYPES:
            raise ValueError(
                f"Unknown relationship {self.via_relationship!r}. "
                f"Allowed: {sorted(EDGE_TYPES)}"
            )
        if self.via_target_label not in NODE_LABELS:
            raise ValueError(
                f"Unknown target label {self.via_target_label!r}. "
                f"Allowed: {sorted(NODE_LABELS)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_distinct_aliases(self) -> "CompareSpec":
        if self.entity_a.alias == self.entity_b.alias:
            raise ValueError(
                f"CompareSpec entity_a and entity_b must have distinct aliases, "
                f"both are {self.entity_a.alias!r}. "
                f"Use different alias values (e.g. 'a' and 'b')."
            )
        return self


class CypherQuerySchema(BaseModel):
    """Single query specification generated by the LLM planner.

    The Cypher builder reads this and produces a parameterized Cypher string.
    Multiple schemas can be generated per user query to collect different
    facets of information from the knowledge graph.
    """

    description: str = Field(description="Human-readable intent of this query")
    anchor: NodeMatch | None = Field(
        default=None,
        description="Primary MATCH clause",
    )
    expansions: list[Expansion] = Field(
        default_factory=list,
        description="Relationship traversals from the anchor",
    )
    path: PathSpec | None = Field(
        default=None,
        description="Shortest-path search between two nodes",
    )
    aggregate: AggregateSpec | None = Field(
        default=None,
        description="Aggregation (count/collect) over results",
    )
    compare: CompareSpec | None = Field(
        default=None,
        description="Set comparison between two entities",
    )
    limit: int = Field(default=25, ge=1, le=100)

    @model_validator(mode="after")
    def _validate_has_operation(self) -> "CypherQuerySchema":
        operations = [
            self.anchor is not None,
            self.path is not None,
            self.compare is not None,
        ]
        active = sum(operations)
        if active == 0:
            raise ValueError(
                "CypherQuerySchema must have exactly one of: anchor, path, or compare"
            )
        if active > 1:
            raise ValueError(
                "CypherQuerySchema must have only one of anchor, path, or compare — "
                "not multiple. Split into separate schemas instead."
            )
        return self

    @model_validator(mode="after")
    def _validate_expansion_aliases(self) -> "CypherQuerySchema":
        if not self.expansions:
            return self
        known_aliases = set()
        if self.anchor:
            known_aliases.add(self.anchor.alias)
        for exp in self.expansions:
            if exp.from_alias not in known_aliases:
                raise ValueError(
                    f"Expansion from_alias {exp.from_alias!r} not found in "
                    f"known aliases: {sorted(known_aliases)}"
                )
            known_aliases.add(exp.target_alias)
        return self

    @model_validator(mode="after")
    def _validate_aggregate_aliases(self) -> "CypherQuerySchema":
        if not self.aggregate:
            return self
        # Collect all aliases defined in this schema
        known_aliases = set()
        if self.anchor:
            known_aliases.add(self.anchor.alias)
        for exp in self.expansions:
            known_aliases.add(exp.target_alias)

        # Check input_alias
        if self.aggregate.input_alias not in known_aliases:
            raise ValueError(
                f"Aggregate input_alias {self.aggregate.input_alias!r} not found "
                f"in known aliases: {sorted(known_aliases)}"
            )
        # Check group_by alias part (format: "alias.property")
        if self.aggregate.group_by:
            group_alias = self.aggregate.group_by.split(".", 1)[0]
            if group_alias not in known_aliases:
                raise ValueError(
                    f"Aggregate group_by alias {group_alias!r} not found "
                    f"in known aliases: {sorted(known_aliases)}"
                )
        return self


# ---------------------------------------------------------------------------
# Planner output
# ---------------------------------------------------------------------------


class PlannerOutput(BaseModel):
    """Output of the LLM planner: reasoning + validated query schemas."""

    reasoning: str = Field(description="LLM's reasoning about the query")
    query_schemas: list[CypherQuerySchema] = Field(
        min_length=1,
        description="One or more query schemas to execute",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations for this query",
    )


# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------


class ResolvedEntity(BaseModel):
    """A Neo4j node resolved from semantic search results."""

    canonical_id: str = Field(description="Format: Label:primary_id_value")
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)
    element_id: str = Field(
        default="",
        description="Neo4j elementId() string",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Execution results
# ---------------------------------------------------------------------------


class QueryResult(BaseModel):
    """Result of executing one CypherQuerySchema."""

    schema_description: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    node_count: int = 0
    truncated: bool = False


class ExecutionResult(BaseModel):
    """Final pipeline output returned to the MCP tool."""

    answer: str = Field(
        default="",
        description="LLM-synthesized natural-language answer grounded in the evidence",
    )
    raw_evidence: str = Field(
        default="",
        description="Raw Markdown evidence rendered by ContextBuilder",
    )
    matched_entities: list[ResolvedEntity] = Field(default_factory=list)
    query_results: list[QueryResult] = Field(default_factory=list)
    plan_summary: str = Field(
        default="",
        description="Planner reasoning about the query",
    )
    limitations: list[str] = Field(default_factory=list)
    debug_trace: dict[str, Any] | None = Field(
        default=None,
        description="Per-query timing and counts, populated when debug=True",
    )


# ---------------------------------------------------------------------------
# Planner errors
# ---------------------------------------------------------------------------


class PlannerValidationError(Exception):
    """Raised when the LLM planner output fails validation after retries."""

    def __init__(self, message: str, raw_output: Any = None):
        super().__init__(message)
        self.raw_output = raw_output
