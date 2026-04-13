"""End-to-end tests for the GraphRAGPipeline (mocked dependencies)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.bioblend_server.GraphRAG.models import ResolvedEntity
from app.bioblend_server.GraphRAG.pipeline import GraphRAGPipeline


def _valid_planner_response():
    return {
        "reasoning": "Looking up tool info",
        "query_schemas": [
            {
                "description": "Fetch tool details",
                "anchor": {"label": "Tool", "element_id": "4:abc:100", "alias": "t"},
                "limit": 25,
            }
        ],
        "limitations": [],
    }


def _patch_both_llm_calls(planner_return, synthesis_return="Bowtie2 is an alignment tool."):
    """Patch get_llm_response for both planner and synthesis calls."""
    call_count = {"n": 0}

    async def _side_effect(messages, **kwargs):
        call_count["n"] += 1
        # First call is the planner, subsequent calls are synthesis
        if call_count["n"] == 1:
            return planner_return
        return synthesis_return

    return patch(
        "app.bioblend_server.GraphRAG.planner.get_llm_response",
        side_effect=_side_effect,
    ), patch(
        "app.bioblend_server.GraphRAG.pipeline.get_llm_response",
        new_callable=AsyncMock,
        return_value=synthesis_return,
    )


class TestGraphRAGPipeline:

    @pytest.fixture
    def pipeline(self, mock_neo4j_connector, mock_semantic_adapter):
        return GraphRAGPipeline(
            connector=mock_neo4j_connector,
            semantic_adapter=mock_semantic_adapter,
        )

    @pytest.mark.asyncio
    async def test_end_to_end(
        self, pipeline, mock_neo4j_connector, mock_semantic_adapter
    ):
        mock_semantic_adapter.search = AsyncMock(
            return_value=[
                {
                    "id": "bowtie2",
                    "score": 0.9,
                    "meta": {"name": "Bowtie2", "tool_id": "bowtie2"},
                }
            ]
        )
        mock_neo4j_connector.find_nodes_by_values = AsyncMock(
            return_value=[
                {
                    "id": "Tool:bowtie2",
                    "label": "Tool",
                    "properties": {"name": "Bowtie2"},
                    "element_id": "4:abc:100",
                }
            ]
        )
        mock_neo4j_connector.run_query = AsyncMock(
            return_value=[{"t_props": {"name": "Bowtie2", "description": "Alignment"}}]
        )

        with patch(
            "app.bioblend_server.GraphRAG.planner.get_llm_response",
            new_callable=AsyncMock,
            return_value=_valid_planner_response(),
        ), patch(
            "app.bioblend_server.GraphRAG.pipeline.get_llm_response",
            new_callable=AsyncMock,
            return_value="Bowtie2 is a fast read alignment tool.",
        ):
            result = await pipeline.run("Tell me about Bowtie2")

        assert result.answer
        assert "Bowtie2" in result.answer
        assert result.raw_evidence  # raw evidence also populated
        assert result.plan_summary == "Looking up tool info"
        assert len(result.matched_entities) >= 1

    @pytest.mark.asyncio
    async def test_empty_semantic_hits(
        self, pipeline, mock_neo4j_connector, mock_semantic_adapter
    ):
        """Pipeline still works when semantic search returns nothing."""
        mock_semantic_adapter.search = AsyncMock(return_value=[])
        mock_neo4j_connector.find_nodes_by_values = AsyncMock(return_value=[])
        mock_neo4j_connector.run_query = AsyncMock(return_value=[])

        with patch(
            "app.bioblend_server.GraphRAG.planner.get_llm_response",
            new_callable=AsyncMock,
            return_value=_valid_planner_response(),
        ), patch(
            "app.bioblend_server.GraphRAG.pipeline.get_llm_response",
            new_callable=AsyncMock,
            return_value="No relevant information found.",
        ):
            result = await pipeline.run("Something obscure")

        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_planner_failure_returns_error(
        self, pipeline, mock_neo4j_connector, mock_semantic_adapter
    ):
        """If planner fails after retries, pipeline returns error result."""
        mock_semantic_adapter.search = AsyncMock(return_value=[])
        mock_neo4j_connector.find_nodes_by_values = AsyncMock(return_value=[])

        with patch(
            "app.bioblend_server.GraphRAG.planner.get_llm_response",
            new_callable=AsyncMock,
            return_value="invalid garbage",
        ):
            result = await pipeline.run("Test query")

        assert "couldn't plan" in result.answer.lower() or "error" in result.answer.lower()
        assert "planner_validation_failed" in result.limitations

    @pytest.mark.asyncio
    async def test_debug_mode(
        self, pipeline, mock_neo4j_connector, mock_semantic_adapter
    ):
        mock_semantic_adapter.search = AsyncMock(return_value=[])
        mock_neo4j_connector.find_nodes_by_values = AsyncMock(return_value=[])
        mock_neo4j_connector.run_query = AsyncMock(return_value=[])

        with patch(
            "app.bioblend_server.GraphRAG.planner.get_llm_response",
            new_callable=AsyncMock,
            return_value=_valid_planner_response(),
        ), patch(
            "app.bioblend_server.GraphRAG.pipeline.get_llm_response",
            new_callable=AsyncMock,
            return_value="Answer text.",
        ):
            result = await pipeline.run("Test", debug=True)

        assert result.debug_trace is not None

    @pytest.mark.asyncio
    async def test_semantic_search_called_before_planner(
        self, pipeline, mock_neo4j_connector, mock_semantic_adapter
    ):
        """Verify semantic search runs before the planner receives seeds."""
        call_order = []

        original_search = AsyncMock(return_value=[])

        async def tracking_search(*args, **kwargs):
            call_order.append("semantic_search")
            return await original_search(*args, **kwargs)

        mock_semantic_adapter.search = tracking_search
        mock_neo4j_connector.find_nodes_by_values = AsyncMock(return_value=[])
        mock_neo4j_connector.run_query = AsyncMock(return_value=[])

        async def tracking_planner_llm(messages, **kwargs):
            call_order.append("planner")
            return _valid_planner_response()

        with patch(
            "app.bioblend_server.GraphRAG.planner.get_llm_response",
            side_effect=tracking_planner_llm,
        ), patch(
            "app.bioblend_server.GraphRAG.pipeline.get_llm_response",
            new_callable=AsyncMock,
            return_value="Answer.",
        ):
            await pipeline.run("Test")

        assert call_order.index("semantic_search") < call_order.index("planner")

    @pytest.mark.asyncio
    async def test_synthesis_failure_falls_back_to_evidence(
        self, pipeline, mock_neo4j_connector, mock_semantic_adapter
    ):
        """If synthesis LLM fails, pipeline returns raw evidence."""
        mock_semantic_adapter.search = AsyncMock(return_value=[])
        mock_neo4j_connector.find_nodes_by_values = AsyncMock(return_value=[])
        mock_neo4j_connector.run_query = AsyncMock(
            return_value=[{"t_props": {"name": "Bowtie2"}}]
        )

        with patch(
            "app.bioblend_server.GraphRAG.planner.get_llm_response",
            new_callable=AsyncMock,
            return_value=_valid_planner_response(),
        ), patch(
            "app.bioblend_server.GraphRAG.pipeline.get_llm_response",
            new_callable=AsyncMock,
            side_effect=Exception("LLM unavailable"),
        ):
            result = await pipeline.run("Tell me about Bowtie2")

        # Should fall back to raw evidence, not crash
        assert result.answer
        assert result.raw_evidence
