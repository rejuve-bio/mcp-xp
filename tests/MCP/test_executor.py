"""Tests for the QueryExecutor."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.bioblend_server.GraphRAG.config import GraphRAGConfig
from app.bioblend_server.GraphRAG.executor import QueryExecutor
from app.bioblend_server.GraphRAG.models import (
    CypherQuerySchema,
    NodeMatch,
    PathSpec,
    PlannerOutput,
)


class TestQueryExecutor:

    @pytest.fixture
    def executor(self, mock_neo4j_connector):
        return QueryExecutor(mock_neo4j_connector, GraphRAGConfig())

    def _make_planner_output(self, schemas: list[CypherQuerySchema]) -> PlannerOutput:
        return PlannerOutput(
            reasoning="Test",
            query_schemas=schemas,
            limitations=[],
        )

    @pytest.mark.asyncio
    async def test_executes_all_schemas(self, executor, mock_neo4j_connector):
        mock_neo4j_connector.run_query = AsyncMock(
            return_value=[{"props": {"name": "Tool1"}, "agg_result": 5}]
        )

        schemas = [
            CypherQuerySchema(
                description=f"Query {i}",
                anchor=NodeMatch(label="Tool", alias="t"),
                limit=10,
            )
            for i in range(3)
        ]
        output = self._make_planner_output(schemas)
        results, trace = await executor.execute(output)

        assert len(results) == 3
        assert all(r.node_count == 1 for r in results)

    @pytest.mark.asyncio
    async def test_empty_schemas_returns_empty(self, executor):
        output = PlannerOutput(
            reasoning="Test",
            query_schemas=[
                CypherQuerySchema(
                    description="Minimal",
                    anchor=NodeMatch(label="Tool", alias="t"),
                )
            ],
        )
        results, _ = await executor.execute(output)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_debug_trace_populated(self, executor, mock_neo4j_connector):
        mock_neo4j_connector.run_query = AsyncMock(return_value=[])

        output = self._make_planner_output([
            CypherQuerySchema(
                description="Debug test",
                anchor=NodeMatch(label="Tool", alias="t"),
            )
        ])
        results, trace = await executor.execute(output, debug=True)

        assert trace is not None
        assert "queries" in trace
        assert len(trace["queries"]) == 1
        assert "exec_time_ms" in trace["queries"][0]

    @pytest.mark.asyncio
    async def test_debug_off_no_trace(self, executor, mock_neo4j_connector):
        mock_neo4j_connector.run_query = AsyncMock(return_value=[])

        output = self._make_planner_output([
            CypherQuerySchema(
                description="No debug",
                anchor=NodeMatch(label="Tool", alias="t"),
            )
        ])
        results, trace = await executor.execute(output, debug=False)
        assert trace is None

    @pytest.mark.asyncio
    async def test_connector_error_graceful(self, executor, mock_neo4j_connector):
        mock_neo4j_connector.run_query = AsyncMock(
            side_effect=Exception("Connection lost")
        )

        output = self._make_planner_output([
            CypherQuerySchema(
                description="Will fail",
                anchor=NodeMatch(label="Tool", alias="t"),
            )
        ])
        results, _ = await executor.execute(output)

        assert len(results) == 1
        assert results[0].node_count == 0

    @pytest.mark.asyncio
    async def test_budget_caps_limit(self, executor, mock_neo4j_connector):
        """Schema limit is capped to config.budget.max_query_limit."""
        mock_neo4j_connector.run_query = AsyncMock(return_value=[])

        # Create schema with limit exceeding budget
        schema = CypherQuerySchema(
            description="Big limit",
            anchor=NodeMatch(label="Tool", alias="t"),
            limit=100,  # max in model
        )
        # Set budget lower
        executor.config.budget.max_query_limit = 50

        output = self._make_planner_output([schema])
        results, trace = await executor.execute(output, debug=True)

        # Check the cypher used the capped limit
        assert trace is not None
        cypher = trace["queries"][0]["cypher"]
        assert "LIMIT 50" in cypher

    @pytest.mark.asyncio
    async def test_budget_caps_path_hops(self, executor, mock_neo4j_connector):
        mock_neo4j_connector.run_query = AsyncMock(return_value=[])

        schema = CypherQuerySchema(
            description="Long path",
            path=PathSpec(
                from_node=NodeMatch(label="Tool", alias="a", element_id="4:x:1"),
                to_node=NodeMatch(label="Tool", alias="b", element_id="4:x:2"),
                max_hops=6,
            ),
        )
        executor.config.budget.path_max_hops = 3

        output = self._make_planner_output([schema])
        results, trace = await executor.execute(output, debug=True)

        cypher = trace["queries"][0]["cypher"]
        assert "*..3" in cypher
