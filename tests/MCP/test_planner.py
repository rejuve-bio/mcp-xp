"""Tests for the LLM-based GraphRAGPlanner (with mocked LLM)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.bioblend_server.GraphRAG.config import GraphRAGConfig
from app.bioblend_server.GraphRAG.models import PlannerValidationError, ResolvedEntity
from app.bioblend_server.GraphRAG.planner import GraphRAGPlanner


def _make_seeds() -> list[ResolvedEntity]:
    return [
        ResolvedEntity(
            canonical_id="Tool:bowtie2",
            label="Tool",
            properties={"name": "Bowtie2", "tool_id": "bowtie2"},
            element_id="4:abc:100",
            confidence=0.9,
        ),
    ]


def _valid_planner_json() -> dict:
    return {
        "reasoning": "Looking up Bowtie2 tool details",
        "query_schemas": [
            {
                "description": "Fetch Bowtie2 tool with inputs",
                "anchor": {
                    "label": "Tool",
                    "element_id": "4:abc:100",
                    "alias": "t",
                },
                "expansions": [
                    {
                        "from_alias": "t",
                        "relationship": "TOOL_HAS_INPUT",
                        "direction": "out",
                        "target_label": "ToolInput",
                        "target_alias": "ti",
                        "optional": True,
                    }
                ],
                "limit": 25,
            }
        ],
        "limitations": [],
    }


class TestGraphRAGPlanner:

    @pytest.fixture
    def planner(self):
        return GraphRAGPlanner(GraphRAGConfig())

    @pytest.mark.asyncio
    async def test_valid_response_parsed(self, planner):
        with patch(
            "app.bioblend_server.GraphRAG.planner.get_llm_response",
            new_callable=AsyncMock,
            return_value=_valid_planner_json(),
        ):
            output = await planner.plan("Tell me about Bowtie2", _make_seeds())
            assert len(output.query_schemas) == 1
            assert output.reasoning == "Looking up Bowtie2 tool details"
            assert output.query_schemas[0].anchor.label == "Tool"

    @pytest.mark.asyncio
    async def test_string_json_response_parsed(self, planner):
        """get_llm_response may return a raw JSON string."""
        import json

        with patch(
            "app.bioblend_server.GraphRAG.planner.get_llm_response",
            new_callable=AsyncMock,
            return_value=json.dumps(_valid_planner_json()),
        ):
            output = await planner.plan("Tell me about Bowtie2", _make_seeds())
            assert len(output.query_schemas) == 1

    @pytest.mark.asyncio
    async def test_invalid_json_retries(self, planner):
        """First call returns invalid JSON, second returns valid."""
        call_count = 0

        async def mock_llm(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "not valid json at all"
            return _valid_planner_json()

        with patch(
            "app.bioblend_server.GraphRAG.planner.get_llm_response",
            side_effect=mock_llm,
        ):
            output = await planner.plan("Test", _make_seeds())
            assert call_count == 2
            assert len(output.query_schemas) == 1

    @pytest.mark.asyncio
    async def test_repeated_failure_raises(self, planner):
        """Both attempts fail → PlannerValidationError."""
        with patch(
            "app.bioblend_server.GraphRAG.planner.get_llm_response",
            new_callable=AsyncMock,
            return_value="garbage",
        ):
            with pytest.raises(PlannerValidationError):
                await planner.plan("Test", _make_seeds())

    @pytest.mark.asyncio
    async def test_schema_count_truncated(self, planner):
        """Excess schemas are truncated to budget.max_schemas_per_query."""
        many_schemas = {
            "reasoning": "Many schemas",
            "query_schemas": [
                {
                    "description": f"Schema {i}",
                    "anchor": {"label": "Tool", "alias": f"t{i}"},
                    "limit": 10,
                }
                for i in range(20)
            ],
            "limitations": [],
        }
        with patch(
            "app.bioblend_server.GraphRAG.planner.get_llm_response",
            new_callable=AsyncMock,
            return_value=many_schemas,
        ):
            output = await planner.plan("Test", [])
            assert len(output.query_schemas) <= planner.config.budget.max_schemas_per_query

    @pytest.mark.asyncio
    async def test_prompt_includes_seeds(self, planner):
        """Verify the prompt sent to the LLM includes seed entity info."""
        captured_messages = []

        async def capture_llm(messages):
            captured_messages.extend(messages)
            return _valid_planner_json()

        with patch(
            "app.bioblend_server.GraphRAG.planner.get_llm_response",
            side_effect=capture_llm,
        ):
            await planner.plan("Test", _make_seeds())
            prompt_text = captured_messages[0]["content"]
            assert "Bowtie2" in prompt_text
            assert "4:abc:100" in prompt_text

    @pytest.mark.asyncio
    async def test_prompt_includes_kg_schema(self, planner):
        """Verify the prompt includes node labels and edge types."""
        captured_messages = []

        async def capture_llm(messages):
            captured_messages.extend(messages)
            return _valid_planner_json()

        with patch(
            "app.bioblend_server.GraphRAG.planner.get_llm_response",
            side_effect=capture_llm,
        ):
            await planner.plan("Test", [])
            prompt_text = captured_messages[0]["content"]
            assert "Workflow" in prompt_text
            assert "HAS_STEP" in prompt_text
            assert "STEP_USES_TOOL" in prompt_text

    @pytest.mark.asyncio
    async def test_prompt_includes_user_query(self, planner):
        """Verify the user's actual query appears in the prompt."""
        captured_messages = []

        async def capture_llm(messages):
            captured_messages.extend(messages)
            return _valid_planner_json()

        with patch(
            "app.bioblend_server.GraphRAG.planner.get_llm_response",
            side_effect=capture_llm,
        ):
            await planner.plan("Which workflows use Bowtie2 for alignment?", [])
            prompt_text = captured_messages[0]["content"]
            assert "Which workflows use Bowtie2 for alignment?" in prompt_text
