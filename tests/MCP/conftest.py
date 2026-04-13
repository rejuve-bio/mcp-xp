"""Shared fixtures for GraphRAG tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.bioblend_server.GraphRAG.models import ResolvedEntity


@pytest.fixture
def mock_neo4j_connector():
    """AsyncMock Neo4j connector with find_nodes_by_values and run_query."""
    connector = AsyncMock()
    connector.find_nodes_by_values = AsyncMock(return_value=[])
    connector.run_query = AsyncMock(return_value=[])
    connector.close = AsyncMock()
    return connector


@pytest.fixture
def sample_resolved_entities() -> list[ResolvedEntity]:
    """Sample resolved entities for Tool, Workflow, and Step."""
    return [
        ResolvedEntity(
            canonical_id="Tool:bowtie2",
            label="Tool",
            properties={
                "tool_id": "toolshed.g2.bx.psu.edu/repos/devteam/bowtie2/bowtie2/2.5.3",
                "name": "Bowtie2",
                "description": "Fast and sensitive read alignment",
                "version": "2.5.3",
            },
            element_id="4:abc:100",
            confidence=0.92,
        ),
        ResolvedEntity(
            canonical_id="Workflow:rna-seq-pipeline",
            label="Workflow",
            properties={
                "workflow_id": "wf-001",
                "name": "RNA-seq Pipeline",
                "description": "Standard RNA-seq analysis workflow",
            },
            element_id="4:abc:200",
            confidence=0.85,
        ),
        ResolvedEntity(
            canonical_id="Step:alignment-step",
            label="Step",
            properties={
                "step_uid": "step-42",
                "name": "Alignment Step",
            },
            element_id="4:abc:300",
            confidence=0.78,
        ),
    ]


@pytest.fixture
def sample_semantic_hits() -> list[dict]:
    """Typical semantic adapter output."""
    return [
        {
            "id": "bowtie2",
            "text": "Bowtie2 - fast read alignment tool",
            "score": 0.92,
            "meta": {
                "name": "Bowtie2",
                "entity_type": "tool",
                "tool_id": "toolshed.g2.bx.psu.edu/repos/devteam/bowtie2/bowtie2/2.5.3",
                "source": "global",
            },
        },
        {
            "id": "rna-seq",
            "text": "RNA-seq analysis workflow",
            "score": 0.85,
            "meta": {
                "name": "RNA-seq Pipeline",
                "entity_type": "workflow",
                "workflow_id": "wf-001",
                "source": "global",
            },
        },
    ]


@pytest.fixture
def mock_semantic_adapter():
    """AsyncMock semantic adapter."""
    adapter = AsyncMock()
    adapter.search = AsyncMock(return_value=[])
    return adapter


@pytest.fixture
def mock_llm_response():
    """Returns a function that patches get_llm_response to return controlled JSON."""
    from unittest.mock import patch

    def _make_mock(return_value):
        return patch(
            "app.bioblend_server.GraphRAG.planner.get_llm_response",
            new_callable=AsyncMock,
            return_value=return_value,
        )

    return _make_mock
