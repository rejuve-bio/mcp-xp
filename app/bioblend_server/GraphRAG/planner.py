"""LLM-based query planner for the GraphRAG pipeline.

Generates validated ``CypherQuerySchema`` objects by prompting the configured
LLM with the KG schema, resolved seed entities, and the user's query.  Output
is free-form JSON validated with Pydantic; invalid output triggers one retry
with error feedback before raising ``PlannerValidationError``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from app.bioblend_server.GraphRAG.config import GraphRAGConfig
from app.bioblend_server.GraphRAG.models import (
    PlannerOutput,
    PlannerValidationError,
    ResolvedEntity,
)
from app.bioblend_server.GraphRAG.schema import (
    EDGE_DIRECTIONS,
    EDGE_TYPES,
    FILTER_OPERATORS,
    NODE_LABELS,
    NODE_PROPERTIES,
)
from app.bioblend_server.utils import get_llm_response

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a query planner for a Galaxy bioinformatics knowledge graph stored in
Neo4j.  Your job is to translate a user's natural-language question into one or
more structured query specifications that a Cypher builder will execute.

## Knowledge Graph Schema

### Node types and their properties

{node_schema}

### Relationship types (source → target)

{edge_schema}

## User Query

{query}

## Resolved Entities (from semantic search)

These entities have been matched in the graph and can be referenced by their
element_id for exact lookups:

{seed_context}

## CypherQuerySchema Format

Each query schema is a JSON object with these fields:

- **description** (str, required): Human-readable intent of this query.
- **anchor** (object | null): Primary node to match.
  - label (str): Node type from the schema above.
  - filters (list): Property conditions, each with "property", "operator" (one of: {filter_ops}), "value".
  - element_id (str | null): Use this for resolved entities (exact match).
  - alias (str): Variable name for Cypher (default "n").
- **expansions** (list): Relationship traversals from anchor or prior targets.
  - from_alias (str): Alias of the source node.
  - relationship (str): Relationship type from the schema above.
  - direction ("out" | "in" | "both"): Traversal direction.
  - target_label (str): Target node type.
  - target_alias (str): Variable name for the target.
  - optional (bool): true for OPTIONAL MATCH (default true).
- **path** (object | null): Shortest-path search between two nodes.
  - from_node, to_node: Same structure as anchor.
  - max_hops (int): Maximum path length (default 4, max 6).
  - relationship_types (list[str]): Filter to these types (empty = any).
- **aggregate** (object | null): Aggregation over results.
  - function ("count" | "collect").
  - input_alias (str): What to aggregate.
  - group_by (str | null): Property for grouping (format: "alias.property").
  - order_by ("asc" | "desc", default "desc").
- **compare** (object | null): Set comparison between two entities.
  - entity_a, entity_b: Same structure as anchor.
  - via_relationship (str): Relationship to traverse.
  - via_target_label (str): Label of compared nodes.
  - hops (int): Traversal depth (default 2, max 3).
- **limit** (int): Max results (default 25, max 100).

Each schema must have at least one of: anchor, path, or compare.

## Rules

1. Only use node labels from the schema above.
2. Only use relationship types from the schema above.
3. Use element_id from resolved entities when available for exact matching.
4. Use property filters for entities not in the resolved set.
5. Prefer multiple simple schemas over one complex one.
6. For comparison queries: use the compare field.
7. For path/connection queries: use the path field.
8. For analytics (most used, ranking): use anchor + expansions + aggregate.
9. Set reasonable limits (default 25).

## Examples

### Entity lookup: "Tell me about Bowtie2"
```json
{{
  "reasoning": "Look up the Bowtie2 tool and its inputs/outputs",
  "query_schemas": [
    {{
      "description": "Fetch Bowtie2 tool with its inputs and outputs",
      "anchor": {{"label": "Tool", "element_id": "4:xxx:123", "alias": "t"}},
      "expansions": [
        {{"from_alias": "t", "relationship": "TOOL_HAS_INPUT", "direction": "out", "target_label": "ToolInput", "target_alias": "ti", "optional": true}},
        {{"from_alias": "t", "relationship": "TOOL_HAS_OUTPUT", "direction": "out", "target_label": "ToolOutput", "target_alias": "to", "optional": true}}
      ],
      "limit": 25
    }},
    {{
      "description": "Find workflows that use Bowtie2",
      "anchor": {{"label": "Tool", "element_id": "4:xxx:123", "alias": "t"}},
      "expansions": [
        {{"from_alias": "t", "relationship": "WORKFLOW_USES_TOOL", "direction": "in", "target_label": "Workflow", "target_alias": "w", "optional": true}}
      ],
      "limit": 5
    }}
  ],
  "limitations": []
}}
```

### Analytics: "What are the most popular tools?"
```json
{{
  "reasoning": "Rank tools by how many steps use them",
  "query_schemas": [
    {{
      "description": "Rank tools by step usage count",
      "anchor": {{"label": "Tool", "alias": "t"}},
      "expansions": [
        {{"from_alias": "t", "relationship": "STEP_USES_TOOL", "direction": "in", "target_label": "Step", "target_alias": "s", "optional": true}}
      ],
      "aggregate": {{"function": "count", "input_alias": "s", "order_by": "desc"}},
      "limit": 10
    }}
  ],
  "limitations": []
}}
```

### Comparison: "Compare RNA-seq and ChIP-seq workflows"
Use WORKFLOW_USES_TOOL to compare by tools (direct edge), or HAS_STEP to compare by steps.
```json
{{
  "reasoning": "Compare the two workflows by their shared and unique tools via the direct WORKFLOW_USES_TOOL relationship",
  "query_schemas": [
    {{
      "description": "Compare RNA-seq and ChIP-seq workflows by tools used",
      "compare": {{
        "entity_a": {{"label": "Workflow", "element_id": "4:xxx:10", "alias": "a"}},
        "entity_b": {{"label": "Workflow", "element_id": "4:xxx:20", "alias": "b"}},
        "via_relationship": "WORKFLOW_USES_TOOL",
        "via_target_label": "Tool",
        "hops": 1
      }},
      "limit": 50
    }}
  ],
  "limitations": []
}}
```

### Path finding: "How are fastqc and multiqc connected?"
```json
{{
  "reasoning": "Find the shortest path between the two tools in the graph",
  "query_schemas": [
    {{
      "description": "Find shortest path between fastqc and multiqc",
      "path": {{
        "from_node": {{"label": "Tool", "element_id": "4:xxx:5", "alias": "a"}},
        "to_node": {{"label": "Tool", "element_id": "4:xxx:8", "alias": "b"}},
        "max_hops": 4
      }},
      "limit": 5
    }}
  ],
  "limitations": []
}}
```

## Output Format

Return ONLY a JSON object (no markdown fences, no explanation outside the JSON):
{{
  "reasoning": "Your brief reasoning about the query",
  "query_schemas": [ ... ],
  "limitations": ["list any known limitations or unsupported aspects"]
}}
"""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class GraphRAGPlanner:
    """LLM-based planner that generates ``PlannerOutput`` from a natural-language query."""

    def __init__(self, config: GraphRAGConfig) -> None:
        self.config = config
        self.log = logging.getLogger(self.__class__.__name__)

    async def plan(
        self,
        query: str,
        seeds: list[ResolvedEntity],
    ) -> PlannerOutput:
        """Generate and validate query schemas via LLM.

        Raises ``PlannerValidationError`` if validation fails after retries.
        """
        prompt = self._build_prompt(query, seeds)

        last_error: str | None = None
        max_attempts = 1 + self.config.budget.planner_max_retries

        for attempt in range(max_attempts):
            try:
                messages = self._build_messages(prompt, last_error)
                raw = await get_llm_response(messages)
                output = self._parse_and_validate(raw, seeds)
                self.log.info(
                    f"Planner succeeded on attempt {attempt + 1}: "
                    f"{len(output.query_schemas)} schemas"
                )
                return output
            except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                self.log.warning(
                    f"Planner attempt {attempt + 1} failed: {last_error}"
                )

        raise PlannerValidationError(
            f"Planner failed after {max_attempts} attempts. Last error: {last_error}",
            raw_output=last_error,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, seeds: list[ResolvedEntity]) -> str:
        node_schema = self._format_node_schema()
        edge_schema = self._format_edge_schema()
        seed_context = self._format_seeds(seeds) if seeds else "(no entities resolved)"
        filter_ops = ", ".join(sorted(FILTER_OPERATORS))

        return _SYSTEM_PROMPT.format(
            query=query,
            node_schema=node_schema,
            edge_schema=edge_schema,
            seed_context=seed_context,
            filter_ops=filter_ops,
        )

    @staticmethod
    def _build_messages(
        system_prompt: str, last_error: str | None
    ) -> list[dict[str, str]]:
        messages = [
            {"role": "user", "content": system_prompt},
        ]
        if last_error:
            messages.append({
                "role": "user",
                "content": (
                    f"Your previous response was invalid: {last_error}\n"
                    "Please fix the JSON and try again."
                ),
            })
        return messages

    # ------------------------------------------------------------------
    # Response parsing & validation
    # ------------------------------------------------------------------

    def _parse_and_validate(
        self,
        raw: Any,
        seeds: list[ResolvedEntity],
    ) -> PlannerOutput:
        # get_llm_response may return parsed dict or raw string
        if isinstance(raw, str):
            data = json.loads(raw)
        elif isinstance(raw, dict):
            data = raw
        else:
            raise TypeError(f"Unexpected LLM response type: {type(raw)}")

        output = PlannerOutput.model_validate(data)

        # Enforce schema count budget
        max_schemas = self.config.budget.max_schemas_per_query
        if len(output.query_schemas) > max_schemas:
            output.query_schemas = output.query_schemas[:max_schemas]
            output.limitations.append(
                f"Schema count truncated to {max_schemas}"
            )

        return output

    # ------------------------------------------------------------------
    # Schema formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_node_schema() -> str:
        lines = []
        for label in sorted(NODE_LABELS):
            props = NODE_PROPERTIES.get(label, [])
            lines.append(f"- **{label}**: {', '.join(props)}")
        return "\n".join(lines)

    @staticmethod
    def _format_edge_schema() -> str:
        lines = []
        for edge_type in sorted(EDGE_TYPES):
            src, tgt = EDGE_DIRECTIONS.get(edge_type, ("?", "?"))
            lines.append(f"- {src} -[:{edge_type}]-> {tgt}")
        return "\n".join(lines)

    @staticmethod
    def _format_seeds(seeds: list[ResolvedEntity]) -> str:
        lines = []
        for s in seeds:
            props_summary = {
                k: v
                for k, v in list(s.properties.items())[:6]
                if v is not None
            }
            lines.append(
                f"- [{s.label}] {s.canonical_id} "
                f"(element_id: {s.element_id!r}, "
                f"confidence: {s.confidence:.2f}, "
                f"properties: {props_summary})"
            )
        return "\n".join(lines) if lines else "(no entities resolved)"
