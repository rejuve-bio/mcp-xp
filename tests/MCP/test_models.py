"""Tests for GraphRAG Pydantic models and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.bioblend_server.GraphRAG.models import (
    CypherQuerySchema,
    Expansion,
    NodeMatch,
    PathSpec,
    PlannerOutput,
    PropertyFilter,
    QueryResult,
    ResolvedEntity,
    ExecutionResult,
    CompareSpec,
)


class TestNodeMatch:
    def test_valid_label(self):
        nm = NodeMatch(label="Tool", alias="t")
        assert nm.label == "Tool"

    def test_invalid_label_rejected(self):
        with pytest.raises(ValidationError):
            NodeMatch(label="InvalidLabel", alias="t")


class TestExpansion:
    def test_valid_expansion(self):
        exp = Expansion(
            from_alias="t",
            relationship="TOOL_HAS_INPUT",
            direction="out",
            target_label="ToolInput",
            target_alias="ti",
        )
        assert exp.relationship == "TOOL_HAS_INPUT"

    def test_invalid_relationship_rejected(self):
        with pytest.raises(ValidationError):
            Expansion(
                from_alias="t",
                relationship="FAKE_REL",
                direction="out",
                target_label="Tool",
                target_alias="x",
            )

    def test_invalid_target_label_rejected(self):
        with pytest.raises(ValidationError):
            Expansion(
                from_alias="t",
                relationship="HAS_STEP",
                direction="out",
                target_label="FakeLabel",
                target_alias="x",
            )


class TestPathSpec:
    def test_valid_path(self):
        ps = PathSpec(
            from_node=NodeMatch(label="Tool", alias="a"),
            to_node=NodeMatch(label="Tool", alias="b"),
            max_hops=4,
        )
        assert ps.max_hops == 4

    def test_invalid_relationship_types(self):
        with pytest.raises(ValidationError):
            PathSpec(
                from_node=NodeMatch(label="Tool", alias="a"),
                to_node=NodeMatch(label="Tool", alias="b"),
                relationship_types=["FAKE_TYPE"],
            )

    def test_max_hops_capped(self):
        with pytest.raises(ValidationError):
            PathSpec(
                from_node=NodeMatch(label="Tool", alias="a"),
                to_node=NodeMatch(label="Tool", alias="b"),
                max_hops=10,
            )


    def test_same_alias_rejected(self):
        """Path endpoints with same alias should be rejected."""
        with pytest.raises(ValidationError, match="distinct aliases"):
            PathSpec(
                from_node=NodeMatch(label="Tool", alias="n"),
                to_node=NodeMatch(label="Tool", alias="n"),
            )


class TestCompareSpec:
    def test_invalid_via_relationship(self):
        with pytest.raises(ValidationError):
            CompareSpec(
                entity_a=NodeMatch(label="Workflow", alias="a"),
                entity_b=NodeMatch(label="Workflow", alias="b"),
                via_relationship="FAKE",
                via_target_label="Tool",
            )

    def test_same_alias_rejected(self):
        with pytest.raises(ValidationError, match="distinct aliases"):
            CompareSpec(
                entity_a=NodeMatch(label="Workflow", alias="n"),
                entity_b=NodeMatch(label="Workflow", alias="n"),
                via_relationship="WORKFLOW_USES_TOOL",
                via_target_label="Tool",
            )


class TestCypherQuerySchema:
    def test_valid_anchor_only(self):
        schema = CypherQuerySchema(
            description="Test",
            anchor=NodeMatch(label="Tool", alias="t"),
        )
        assert schema.anchor.label == "Tool"

    def test_must_have_operation(self):
        with pytest.raises(ValidationError):
            CypherQuerySchema(description="Empty")

    def test_mixed_anchor_and_path_rejected(self):
        """Schema with both anchor and path should be rejected."""
        with pytest.raises(ValidationError, match="only one"):
            CypherQuerySchema(
                description="Mixed",
                anchor=NodeMatch(label="Tool", alias="t"),
                path=PathSpec(
                    from_node=NodeMatch(label="Tool", alias="a"),
                    to_node=NodeMatch(label="Tool", alias="b"),
                ),
            )

    def test_mixed_anchor_and_compare_rejected(self):
        with pytest.raises(ValidationError, match="only one"):
            CypherQuerySchema(
                description="Mixed",
                anchor=NodeMatch(label="Workflow", alias="w"),
                compare=CompareSpec(
                    entity_a=NodeMatch(label="Workflow", alias="a"),
                    entity_b=NodeMatch(label="Workflow", alias="b"),
                    via_relationship="WORKFLOW_USES_TOOL",
                    via_target_label="Tool",
                ),
            )

    def test_expansion_alias_validation(self):
        with pytest.raises(ValidationError):
            CypherQuerySchema(
                description="Bad alias",
                anchor=NodeMatch(label="Tool", alias="t"),
                expansions=[
                    Expansion(
                        from_alias="nonexistent",
                        relationship="TOOL_HAS_INPUT",
                        direction="out",
                        target_label="ToolInput",
                        target_alias="ti",
                    )
                ],
            )

    def test_chained_expansions(self):
        schema = CypherQuerySchema(
            description="Chained",
            anchor=NodeMatch(label="Workflow", alias="w"),
            expansions=[
                Expansion(
                    from_alias="w",
                    relationship="HAS_STEP",
                    direction="out",
                    target_label="Step",
                    target_alias="s",
                ),
                Expansion(
                    from_alias="s",
                    relationship="STEP_USES_TOOL",
                    direction="out",
                    target_label="Tool",
                    target_alias="t",
                ),
            ],
        )
        assert len(schema.expansions) == 2

    def test_limit_capped(self):
        with pytest.raises(ValidationError):
            CypherQuerySchema(
                description="Over limit",
                anchor=NodeMatch(label="Tool", alias="t"),
                limit=200,
            )

    def test_aggregate_input_alias_validated(self):
        """aggregate.input_alias must reference a known alias."""
        from app.bioblend_server.GraphRAG.models import AggregateSpec

        with pytest.raises(ValidationError):
            CypherQuerySchema(
                description="Bad aggregate alias",
                anchor=NodeMatch(label="Tool", alias="t"),
                aggregate=AggregateSpec(
                    function="count",
                    input_alias="nonexistent",
                ),
            )

    def test_aggregate_group_by_alias_validated(self):
        """aggregate.group_by alias part must reference a known alias."""
        from app.bioblend_server.GraphRAG.models import AggregateSpec

        with pytest.raises(ValidationError):
            CypherQuerySchema(
                description="Bad group_by alias",
                anchor=NodeMatch(label="Tool", alias="t"),
                expansions=[
                    Expansion(
                        from_alias="t",
                        relationship="STEP_USES_TOOL",
                        direction="in",
                        target_label="Step",
                        target_alias="s",
                    ),
                ],
                aggregate=AggregateSpec(
                    function="count",
                    input_alias="s",
                    group_by="nonexistent.prop",
                ),
            )

    def test_valid_aggregate_aliases_pass(self):
        from app.bioblend_server.GraphRAG.models import AggregateSpec

        schema = CypherQuerySchema(
            description="Valid aggregate",
            anchor=NodeMatch(label="Tool", alias="t"),
            expansions=[
                Expansion(
                    from_alias="t",
                    relationship="STEP_USES_TOOL",
                    direction="in",
                    target_label="Step",
                    target_alias="s",
                ),
            ],
            aggregate=AggregateSpec(
                function="count",
                input_alias="s",
                group_by="t.community",
            ),
        )
        assert schema.aggregate.input_alias == "s"


class TestPlannerOutput:
    def test_requires_at_least_one_schema(self):
        with pytest.raises(ValidationError):
            PlannerOutput(reasoning="test", query_schemas=[])

    def test_valid_output(self):
        po = PlannerOutput(
            reasoning="Looking up a tool",
            query_schemas=[
                CypherQuerySchema(
                    description="Fetch tool",
                    anchor=NodeMatch(label="Tool", alias="t"),
                )
            ],
        )
        assert len(po.query_schemas) == 1


class TestResolvedEntity:
    def test_construction(self):
        e = ResolvedEntity(
            canonical_id="Tool:bowtie2",
            label="Tool",
            properties={"name": "Bowtie2"},
            element_id="4:abc:123",
            confidence=0.9,
        )
        assert e.canonical_id == "Tool:bowtie2"
        assert e.confidence == 0.9

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            ResolvedEntity(
                canonical_id="Tool:x",
                label="Tool",
                confidence=1.5,
            )


class TestExecutionResult:
    def test_serialization_roundtrip(self):
        er = ExecutionResult(
            answer="Bowtie2 is a fast alignment tool.",
            raw_evidence="[TOOL] Bowtie2\n  Summary: Fast read alignment",
            plan_summary="Looked up a tool",
            limitations=["partial"],
        )
        data = er.model_dump()
        restored = ExecutionResult.model_validate(data)
        assert restored.answer == "Bowtie2 is a fast alignment tool."
        assert restored.raw_evidence.startswith("[TOOL]")
        assert restored.limitations == ["partial"]
