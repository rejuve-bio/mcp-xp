"""Tests for the deterministic CypherBuilder."""

from __future__ import annotations

import pytest

from app.bioblend_server.GraphRAG.cypher_builder import CypherBuilder
from app.bioblend_server.GraphRAG.models import (
    AggregateSpec,
    CompareSpec,
    CypherQuerySchema,
    Expansion,
    NodeMatch,
    PathSpec,
    PropertyFilter,
)


@pytest.fixture
def builder():
    return CypherBuilder()


class TestAnchorQueries:

    def test_basic_anchor(self, builder):
        schema = CypherQuerySchema(
            description="Basic tool lookup",
            anchor=NodeMatch(label="Tool", alias="t", element_id="4:abc:100"),
        )
        cypher, params = builder.build(schema)
        assert "MATCH (t:Tool)" in cypher
        assert "elementId(t) = $anchor_eid" in cypher
        assert params["anchor_eid"] == "4:abc:100"

    def test_anchor_with_filter(self, builder):
        schema = CypherQuerySchema(
            description="Filter by name",
            anchor=NodeMatch(
                label="Tool",
                alias="t",
                filters=[
                    PropertyFilter(
                        property="name", operator="contains", value="bowtie"
                    )
                ],
            ),
        )
        cypher, params = builder.build(schema)
        assert "CONTAINS" in cypher
        assert "anchor_f0" in params

    def test_anchor_with_expansion(self, builder):
        schema = CypherQuerySchema(
            description="Tool with inputs",
            anchor=NodeMatch(label="Tool", alias="t", element_id="4:abc:100"),
            expansions=[
                Expansion(
                    from_alias="t",
                    relationship="TOOL_HAS_INPUT",
                    direction="out",
                    target_label="ToolInput",
                    target_alias="ti",
                )
            ],
        )
        cypher, params = builder.build(schema)
        assert "OPTIONAL MATCH (t)-[:TOOL_HAS_INPUT]->(ti:ToolInput)" in cypher
        assert "collect(DISTINCT properties(ti)) AS ti_list" in cypher

    def test_mandatory_match_expansion(self, builder):
        schema = CypherQuerySchema(
            description="Required match",
            anchor=NodeMatch(label="Workflow", alias="w"),
            expansions=[
                Expansion(
                    from_alias="w",
                    relationship="HAS_STEP",
                    direction="out",
                    target_label="Step",
                    target_alias="s",
                    optional=False,
                )
            ],
        )
        cypher, _ = builder.build(schema)
        # Should use MATCH not OPTIONAL MATCH
        assert "MATCH (w)-[:HAS_STEP]->(s:Step)" in cypher

    def test_incoming_direction(self, builder):
        schema = CypherQuerySchema(
            description="Incoming rel",
            anchor=NodeMatch(label="Tool", alias="t"),
            expansions=[
                Expansion(
                    from_alias="t",
                    relationship="STEP_USES_TOOL",
                    direction="in",
                    target_label="Step",
                    target_alias="s",
                )
            ],
        )
        cypher, _ = builder.build(schema)
        assert "<-[:STEP_USES_TOOL]-" in cypher


    def test_chained_expansion_nesting(self, builder):
        """Workflow->Step->Tool should produce nested structure, not flat lists."""
        schema = CypherQuerySchema(
            description="Workflow topology",
            anchor=NodeMatch(label="Workflow", alias="w", element_id="4:abc:1"),
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
        cypher, _ = builder.build(schema)
        # Should have a WITH that aggregates t per s
        assert "collect(DISTINCT properties(t)) AS t_list" in cypher
        # Should produce nested RETURN with parent props + child list
        assert "s_nested" in cypher
        assert "props: properties(s)" in cypher
        assert "t_list: t_list" in cypher
        # The variable t should NOT be re-collected after being consumed
        # (only one WITH should reference properties(t))
        assert cypher.count("collect(DISTINCT properties(t))") == 1


class TestPathQueries:

    def test_basic_path(self, builder):
        schema = CypherQuerySchema(
            description="Path between tools",
            path=PathSpec(
                from_node=NodeMatch(label="Tool", alias="a", element_id="4:abc:1"),
                to_node=NodeMatch(label="Tool", alias="b", element_id="4:abc:2"),
                max_hops=4,
            ),
            limit=5,
        )
        cypher, params = builder.build(schema)
        assert "shortestPath" in cypher
        assert "*..4" in cypher
        assert "path_nodes" in cypher
        assert "path_rels" in cypher
        assert params["path_from_eid"] == "4:abc:1"
        assert params["path_to_eid"] == "4:abc:2"

    def test_path_with_relationship_filter(self, builder):
        schema = CypherQuerySchema(
            description="Filtered path",
            path=PathSpec(
                from_node=NodeMatch(label="Tool", alias="a", element_id="4:abc:1"),
                to_node=NodeMatch(label="Workflow", alias="b", element_id="4:abc:2"),
                relationship_types=["STEP_USES_TOOL", "HAS_STEP"],
            ),
        )
        cypher, params = builder.build(schema)
        # Relationship types should be embedded in the path pattern directly
        assert "STEP_USES_TOOL|HAS_STEP" in cypher
        assert "shortestPath" in cypher


class TestAggregateQueries:

    def test_count_aggregate(self, builder):
        schema = CypherQuerySchema(
            description="Most used tools",
            anchor=NodeMatch(label="Tool", alias="t"),
            expansions=[
                Expansion(
                    from_alias="t",
                    relationship="STEP_USES_TOOL",
                    direction="in",
                    target_label="Step",
                    target_alias="s",
                )
            ],
            aggregate=AggregateSpec(
                function="count", input_alias="s", order_by="desc"
            ),
            limit=10,
        )
        cypher, _ = builder.build(schema)
        assert "count(s) AS agg_result" in cypher
        assert "ORDER BY agg_result DESC" in cypher
        assert "LIMIT 10" in cypher

    def test_grouped_aggregate(self, builder):
        schema = CypherQuerySchema(
            description="Tools by community",
            anchor=NodeMatch(label="Tool", alias="t"),
            expansions=[
                Expansion(
                    from_alias="t",
                    relationship="STEP_USES_TOOL",
                    direction="in",
                    target_label="Step",
                    target_alias="s",
                )
            ],
            aggregate=AggregateSpec(
                function="count",
                input_alias="s",
                group_by="t.community",
                order_by="desc",
            ),
            limit=20,
        )
        cypher, _ = builder.build(schema)
        assert "group_key" in cypher


class TestCompareQueries:

    def test_basic_compare(self, builder):
        schema = CypherQuerySchema(
            description="Compare workflows",
            compare=CompareSpec(
                entity_a=NodeMatch(label="Workflow", alias="a", element_id="4:abc:10"),
                entity_b=NodeMatch(label="Workflow", alias="b", element_id="4:abc:20"),
                via_relationship="HAS_STEP",
                via_target_label="Step",
                hops=2,
            ),
        )
        cypher, params = builder.build(schema)
        assert "shared" in cypher
        assert "unique_a" in cypher
        assert "unique_b" in cypher
        assert "shared_count" in cypher
        assert params["cmp_a_eid"] == "4:abc:10"


    def test_compare_respects_limit(self, builder):
        schema = CypherQuerySchema(
            description="Limited compare",
            compare=CompareSpec(
                entity_a=NodeMatch(label="Workflow", alias="a", element_id="4:abc:10"),
                entity_b=NodeMatch(label="Workflow", alias="b", element_id="4:abc:20"),
                via_relationship="WORKFLOW_USES_TOOL",
                via_target_label="Tool",
                hops=1,
            ),
            limit=10,
        )
        cypher, _ = builder.build(schema)
        # shared/unique arrays should be sliced to limit
        assert "shared[0..10]" in cypher
        assert "unique_a[0..10]" in cypher
        assert "unique_b[0..10]" in cypher


class TestSafety:

    def test_values_are_parameterized(self, builder):
        schema = CypherQuerySchema(
            description="Test injection safety",
            anchor=NodeMatch(
                label="Tool",
                alias="t",
                filters=[
                    PropertyFilter(
                        property="name",
                        operator="eq",
                        value="'; DROP TABLE tools; --",
                    )
                ],
            ),
        )
        cypher, params = builder.build(schema)
        # The dangerous value must be in params, not in the Cypher string
        assert "DROP TABLE" not in cypher
        assert any("DROP TABLE" in str(v) for v in params.values())

    def test_group_by_injection_blocked(self, builder):
        """group_by with invalid format should raise."""
        schema = CypherQuerySchema(
            description="Bad group_by",
            anchor=NodeMatch(label="Tool", alias="t"),
            expansions=[
                Expansion(
                    from_alias="t",
                    relationship="STEP_USES_TOOL",
                    direction="in",
                    target_label="Step",
                    target_alias="s",
                )
            ],
            aggregate=AggregateSpec(
                function="count",
                input_alias="s",
                group_by="t.community; MATCH (x)",
                order_by="desc",
            ),
            limit=10,
        )
        with pytest.raises(ValueError):
            builder.build(schema)

    def test_group_by_valid(self, builder):
        """Valid group_by should be sanitized and pass."""
        schema = CypherQuerySchema(
            description="Valid group_by",
            anchor=NodeMatch(label="Tool", alias="t"),
            expansions=[
                Expansion(
                    from_alias="t",
                    relationship="STEP_USES_TOOL",
                    direction="in",
                    target_label="Step",
                    target_alias="s",
                )
            ],
            aggregate=AggregateSpec(
                function="count",
                input_alias="s",
                group_by="t.community",
                order_by="desc",
            ),
            limit=10,
        )
        cypher, _ = builder.build(schema)
        assert "t.community AS group_key" in cypher

    def test_aggregate_self_count_omits_anchor(self, builder):
        """When counting the anchor itself, anchor should not be in WITH grouping."""
        schema = CypherQuerySchema(
            description="Count all workflows",
            anchor=NodeMatch(label="Workflow", alias="w"),
            aggregate=AggregateSpec(
                function="count",
                input_alias="w",
                order_by="desc",
            ),
            limit=1,
        )
        cypher, _ = builder.build(schema)
        # WITH should NOT include 'w' — just the count
        assert "WITH count(w) AS agg_result" in cypher
        # RETURN should not reference properties(w) since we're not grouping by it
        assert "properties(w)" not in cypher

    def test_aggregate_expansion_count_keeps_anchor(self, builder):
        """When counting an expansion target, anchor should remain in WITH."""
        schema = CypherQuerySchema(
            description="Steps per tool",
            anchor=NodeMatch(label="Tool", alias="t"),
            expansions=[
                Expansion(
                    from_alias="t",
                    relationship="STEP_USES_TOOL",
                    direction="in",
                    target_label="Step",
                    target_alias="s",
                )
            ],
            aggregate=AggregateSpec(function="count", input_alias="s"),
            limit=10,
        )
        cypher, _ = builder.build(schema)
        # WITH should include 't' for per-tool grouping
        assert "WITH t, count(s) AS agg_result" in cypher
        assert "properties(t) AS props" in cypher

    def test_3_hop_expansion(self, builder):
        """Workflow->Step->Tool->ToolInput should preserve all 3 levels."""
        schema = CypherQuerySchema(
            description="Full topology",
            anchor=NodeMatch(label="Workflow", alias="w", element_id="4:abc:1"),
            expansions=[
                Expansion(
                    from_alias="w", relationship="HAS_STEP",
                    direction="out", target_label="Step", target_alias="s",
                ),
                Expansion(
                    from_alias="s", relationship="STEP_USES_TOOL",
                    direction="out", target_label="Tool", target_alias="t",
                ),
                Expansion(
                    from_alias="t", relationship="TOOL_HAS_INPUT",
                    direction="out", target_label="ToolInput", target_alias="ti",
                ),
            ],
        )
        cypher, _ = builder.build(schema)
        # Should have WITH clauses that aggregate bottom-up
        assert "ti_list" in cypher
        assert "s_nested" in cypher
        # All aliases should appear somewhere in the Cypher
        assert ":Step" in cypher
        assert ":Tool" in cypher
        assert ":ToolInput" in cypher

    def test_chained_plus_flat_sibling(self, builder):
        """w->s->t (chained) + w->wt (flat) should keep wt in scope for RETURN."""
        schema = CypherQuerySchema(
            description="Mixed chained + flat",
            anchor=NodeMatch(label="Workflow", alias="w", element_id="4:abc:1"),
            expansions=[
                Expansion(
                    from_alias="w", relationship="HAS_STEP",
                    direction="out", target_label="Step", target_alias="s",
                ),
                Expansion(
                    from_alias="s", relationship="STEP_USES_TOOL",
                    direction="out", target_label="Tool", target_alias="t",
                ),
                Expansion(
                    from_alias="w", relationship="WORKFLOW_USES_TOOL",
                    direction="out", target_label="Tool", target_alias="wt",
                ),
            ],
        )
        cypher, _ = builder.build(schema)
        # wt must appear in the final RETURN (not dropped from scope)
        assert "properties(wt)" in cypher
        # s_nested for the chained branch
        assert "s_nested" in cypher

    def test_collect_uses_properties(self, builder):
        """collect() aggregate should use properties() not raw nodes."""
        schema = CypherQuerySchema(
            description="Collect workflows",
            anchor=NodeMatch(label="Tool", alias="t"),
            expansions=[
                Expansion(
                    from_alias="t", relationship="WORKFLOW_USES_TOOL",
                    direction="in", target_label="Workflow", target_alias="w",
                ),
            ],
            aggregate=AggregateSpec(function="collect", input_alias="w"),
            limit=10,
        )
        cypher, _ = builder.build(schema)
        assert "collect(DISTINCT properties(w)) AS agg_result" in cypher
