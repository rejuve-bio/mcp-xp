"""Tests for the ContextBuilder."""

from __future__ import annotations

import pytest

from app.bioblend_server.GraphRAG.config import GraphRAGConfig
from app.bioblend_server.GraphRAG.context_builder import ContextBuilder
from app.bioblend_server.GraphRAG.models import QueryResult


@pytest.fixture
def builder():
    return ContextBuilder(GraphRAGConfig())


class TestContextBuilder:

    def test_empty_results_message(self, builder):
        context = builder.build_context_from_results([], [])
        assert context == "No matching context found."

    def test_entity_context_formatting(self, builder):
        result = QueryResult(
            schema_description="Fetch Bowtie2",
            rows=[
                {
                    "t_props": {
                        "name": "Bowtie2",
                        "description": "Fast read alignment",
                        "version": "2.5.3",
                    },
                    "ti_list": [
                        {"input_name": "reads", "input_type": "fastq"},
                        {"input_name": "reference", "input_type": "fasta"},
                    ],
                }
            ],
            node_count=1,
        )
        context = builder.build_context_from_results([result], [])
        assert "Bowtie2" in context
        assert "reads" in context
        assert "reference" in context

    def test_ranked_aggregate_formatting(self, builder):
        result = QueryResult(
            schema_description="Most used tools",
            rows=[
                {"props": {"name": "FastQC", "version": "0.11"}, "agg_result": 150},
                {"props": {"name": "Trimmomatic"}, "agg_result": 120},
            ],
            node_count=2,
        )
        context = builder.build_context_from_results([result], [])
        assert "[ANALYTICS]" in context
        assert "FastQC" in context
        assert "150" in context
        assert "Trimmomatic" in context

    def test_multiple_analytics_not_deduped(self, builder):
        """Regression: both analytics results should appear (old bug dropped second)."""
        result_a = QueryResult(
            schema_description="Top tools by usage",
            rows=[{"props": {"name": "FastQC"}, "agg_result": 100}],
            node_count=1,
        )
        result_b = QueryResult(
            schema_description="Tools by community",
            rows=[
                {"props": {"name": "BWA"}, "agg_result": 50, "group_key": "alignment"}
            ],
            node_count=1,
        )
        context = builder.build_context_from_results([result_a, result_b], [])
        assert "FastQC" in context
        assert "BWA" in context

    def test_compare_formatting(self, builder):
        result = QueryResult(
            schema_description="Compare workflows",
            rows=[
                {
                    "entity_a": {"name": "RNA-seq"},
                    "entity_b": {"name": "ChIP-seq"},
                    "shared": [{"name": "FastQC"}, {"name": "Trimmomatic"}],
                    "unique_to_a": [{"name": "HTSeq"}],
                    "unique_to_b": [{"name": "MACS2"}],
                    "shared_count": 2,
                }
            ],
            node_count=1,
        )
        context = builder.build_context_from_results([result], [])
        assert "[COMPARISON]" in context
        assert "RNA-seq" in context
        assert "ChIP-seq" in context
        assert "FastQC" in context
        assert "HTSeq" in context
        assert "MACS2" in context

    def test_path_formatting(self, builder):
        result = QueryResult(
            schema_description="Path between tools",
            rows=[
                {
                    "path_nodes": [
                        {"label": "Tool", "props": {"name": "FastQC"}, "element_id": "1"},
                        {"label": "Step", "props": {"name": "QC Step"}, "element_id": "2"},
                        {"label": "Tool", "props": {"name": "MultiQC"}, "element_id": "3"},
                    ],
                    "path_rels": [
                        {"type": "STEP_USES_TOOL", "start_id": "2", "end_id": "1"},
                        {"type": "STEP_USES_TOOL", "start_id": "2", "end_id": "3"},
                    ],
                }
            ],
            node_count=1,
        )
        context = builder.build_context_from_results([result], [])
        assert "[PATH]" in context
        assert "FastQC" in context
        assert "MultiQC" in context

    def test_generic_fallback(self, builder):
        result = QueryResult(
            schema_description="Unknown shape",
            rows=[{"some_field": "some_value", "count": 42}],
            node_count=1,
        )
        context = builder.build_context_from_results([result], [])
        assert "[RESULT]" in context
        assert "some_value" in context

    def test_identical_content_deduped(self, builder):
        """Identical rendered content should appear only once."""
        result = QueryResult(
            schema_description="Same query",
            rows=[{"t_props": {"name": "UniqueToolName", "description": "A tool"}}],
            node_count=1,
        )
        context = builder.build_context_from_results([result, result], [])
        # Content-based dedup: identical sections appear once
        assert context.count("[ENTITY] UniqueToolName") == 1

    def test_same_description_different_content_kept(self, builder):
        """Two results with same description but different data should both appear."""
        result_a = QueryResult(
            schema_description="Fetch tool details",
            rows=[{"t_props": {"name": "Bowtie2", "tool_id": "bt2"}}],
            node_count=1,
        )
        result_b = QueryResult(
            schema_description="Fetch tool details",
            rows=[{"t_props": {"name": "samtools", "tool_id": "st"}}],
            node_count=1,
        )
        context = builder.build_context_from_results([result_a, result_b], [])
        assert "Bowtie2" in context
        assert "samtools" in context

    def test_collect_aggregate_formatting(self, builder):
        """When agg_result is a list (from collect), render as items not count."""
        result = QueryResult(
            schema_description="Workflows per tool",
            rows=[
                {
                    "props": {"name": "Bowtie2"},
                    "agg_result": [
                        {"name": "RNA-seq Pipeline"},
                        {"name": "ChIP-seq Pipeline"},
                    ],
                },
            ],
            node_count=1,
        )
        context = builder.build_context_from_results([result], [])
        assert "2 items" in context
        assert "RNA-seq Pipeline" in context
        assert "ChIP-seq Pipeline" in context
        assert "occurrences" not in context

    def test_nested_expansion_no_parent_duplication(self, builder):
        """Nested results should show each parent once with its children."""
        result = QueryResult(
            schema_description="Workflow steps with tools",
            rows=[
                {
                    "w_props": {"name": "RNA-seq", "workflow_id": "wf-1"},
                    "s_nested": [
                        {
                            "props": {"name": "Alignment", "step_uid": "s1"},
                            "t_list": [
                                {"name": "Bowtie2"},
                                {"name": "BWA"},
                            ],
                        },
                        {
                            "props": {"name": "QC", "step_uid": "s2"},
                            "t_list": [
                                {"name": "FastQC"},
                            ],
                        },
                    ],
                }
            ],
            node_count=1,
        )
        context = builder.build_context_from_results([result], [])
        # The section header [S] should list each step once with its children
        lines = context.split("\n")
        # Count lines that start with "    - Alignment" (the step entry)
        step_lines = [l for l in lines if l.strip().startswith("- Alignment")]
        assert len(step_lines) == 1, f"Alignment step should appear once, got: {step_lines}"
        step_lines_qc = [l for l in lines if l.strip().startswith("- QC")]
        assert len(step_lines_qc) == 1
        # Tools nested under their steps
        assert "Bowtie2" in context
        assert "BWA" in context
        assert "FastQC" in context

    def test_3_hop_nested_rendering(self, builder):
        """3-hop: Workflow->Step->Tool->ToolInput should render all levels."""
        result = QueryResult(
            schema_description="Full workflow topology",
            rows=[
                {
                    "w_props": {"name": "RNA-seq", "workflow_id": "wf-1"},
                    "s_nested": [
                        {
                            "props": {"name": "Alignment", "step_uid": "s1"},
                            "t_nested": [
                                {
                                    "props": {"name": "Bowtie2", "tool_id": "bt2"},
                                    "ti_list": [
                                        {"input_name": "reads"},
                                        {"input_name": "reference"},
                                    ],
                                },
                            ],
                        },
                    ],
                }
            ],
            node_count=1,
        )
        context = builder.build_context_from_results([result], [])
        assert "Alignment" in context
        assert "Bowtie2" in context
        assert "reads" in context
        assert "reference" in context
