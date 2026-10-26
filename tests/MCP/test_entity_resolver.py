"""Tests for the EntityResolver."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.bioblend_server.GraphRAG.entity_resolver import EntityResolver


class TestEntityResolver:

    @pytest.fixture
    def resolver(self, mock_neo4j_connector):
        return EntityResolver(mock_neo4j_connector)

    @pytest.mark.asyncio
    async def test_exact_match(self, resolver, mock_neo4j_connector):
        mock_neo4j_connector.find_nodes_by_values = AsyncMock(
            return_value=[
                {
                    "id": "Tool:bowtie2",
                    "label": "Tool",
                    "properties": {"name": "Bowtie2", "tool_id": "bowtie2"},
                    "element_id": "4:abc:100",
                }
            ]
        )

        hits = [
            {
                "id": "bowtie2",
                "score": 0.9,
                "meta": {"tool_id": "bowtie2", "name": "Bowtie2"},
            }
        ]

        entities = await resolver.resolve(hits, "Bowtie2")
        assert len(entities) >= 1
        assert entities[0].label == "Tool"

    @pytest.mark.asyncio
    async def test_empty_hits_returns_empty(self, resolver):
        entities = await resolver.resolve([], "")
        assert entities == []

    @pytest.mark.asyncio
    async def test_fuzzy_fallback(self, resolver, mock_neo4j_connector):
        """When exact match returns nothing, fuzzy text fallback is tried."""
        call_count = 0

        async def side_effect(label, exact_props, fuzzy_props, values):
            nonlocal call_count
            call_count += 1
            # Return results only for fuzzy lookups (which include fuzzy_props)
            if fuzzy_props:
                return [
                    {
                        "id": "Tool:samtools",
                        "label": "Tool",
                        "properties": {"name": "samtools"},
                        "element_id": "4:abc:50",
                    }
                ]
            return []

        mock_neo4j_connector.find_nodes_by_values = AsyncMock(side_effect=side_effect)

        hits = [{"id": "samtools", "text": "samtools", "score": 0.8, "meta": {}}]
        entities = await resolver.resolve(hits, "samtools")

        # Should have found something via fuzzy
        assert len(entities) >= 1

    @pytest.mark.asyncio
    async def test_query_fallback(self, resolver, mock_neo4j_connector):
        """When no hits provided, falls back to query text."""
        mock_neo4j_connector.find_nodes_by_values = AsyncMock(
            return_value=[
                {
                    "id": "Tool:bwa",
                    "label": "Tool",
                    "properties": {"name": "BWA"},
                    "element_id": "4:abc:99",
                }
            ]
        )

        entities = await resolver.resolve([], "BWA alignment tool")
        assert len(entities) >= 1

    @pytest.mark.asyncio
    async def test_dedup_by_canonical_id(self, resolver, mock_neo4j_connector):
        """Same node returned from multiple labels is deduped."""
        mock_neo4j_connector.find_nodes_by_values = AsyncMock(
            return_value=[
                {
                    "id": "Tool:fastqc",
                    "label": "Tool",
                    "properties": {"name": "FastQC"},
                    "element_id": "4:abc:10",
                }
            ]
        )

        hits = [
            {"id": "fastqc", "score": 0.9, "meta": {"tool_id": "fastqc"}},
            {"id": "fastqc", "score": 0.85, "meta": {"name": "FastQC"}},
        ]
        entities = await resolver.resolve(hits, "FastQC")

        # Should be deduped to 1
        ids = [e.canonical_id for e in entities]
        assert len(set(ids)) == len(ids)

    @pytest.mark.asyncio
    async def test_max_seeds_limit(self, resolver, mock_neo4j_connector):
        """Truncates to max_seeds."""
        nodes = [
            {
                "id": f"Tool:tool{i}",
                "label": "Tool",
                "properties": {"name": f"tool{i}"},
                "element_id": f"4:abc:{i}",
            }
            for i in range(10)
        ]
        mock_neo4j_connector.find_nodes_by_values = AsyncMock(return_value=nodes)

        hits = [
            {"id": f"tool{i}", "score": 0.9 - i * 0.05, "meta": {"name": f"tool{i}"}}
            for i in range(10)
        ]
        entities = await resolver.resolve(hits, "tools", max_seeds=3)
        assert len(entities) <= 3
