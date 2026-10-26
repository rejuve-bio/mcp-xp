"""Query executor for the GraphRAG pipeline.

Takes a ``PlannerOutput``, builds Cypher for each schema via ``CypherBuilder``,
executes against Neo4j, and collects results with budget enforcement.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.bioblend_server.GraphRAG.config import GraphRAGConfig
from app.bioblend_server.GraphRAG.cypher_builder import CypherBuilder
from app.bioblend_server.GraphRAG.models import PlannerOutput, QueryResult
from app.bioblend_server.GraphRAG.neo4j_connector import Neo4jGraphConnector


class QueryExecutor:
    """Executes planned query schemas and collects results."""

    def __init__(
        self,
        connector: Neo4jGraphConnector,
        config: GraphRAGConfig,
    ) -> None:
        self.connector = connector
        self.builder = CypherBuilder()
        self.config = config
        self.log = logging.getLogger(self.__class__.__name__)

    async def execute(
        self,
        planner_output: PlannerOutput,
        debug: bool = False,
    ) -> tuple[list[QueryResult], dict[str, Any] | None]:
        """Execute all schemas from the planner output.

        Returns:
            A tuple of (query_results, debug_trace_or_None).
        """
        results: list[QueryResult] = []
        trace_entries: list[dict[str, Any]] = [] if debug else []

        for i, schema in enumerate(planner_output.query_schemas):
            try:
                # Enforce budget caps on the schema
                schema = self._enforce_budgets(schema)

                # Build Cypher
                start = time.monotonic()
                cypher, params = self.builder.build(schema)
                build_time = time.monotonic() - start

                # Execute
                start = time.monotonic()
                rows = await self.connector.run_query(cypher, params)
                exec_time = time.monotonic() - start

                node_count = len(rows)
                truncated = node_count >= schema.limit

                result = QueryResult(
                    schema_description=schema.description,
                    rows=rows,
                    node_count=node_count,
                    truncated=truncated,
                )
                results.append(result)

                self.log.info(
                    f"Schema {i + 1}/{len(planner_output.query_schemas)} "
                    f"'{schema.description}': {node_count} rows "
                    f"({exec_time:.3f}s)"
                )

                if debug:
                    trace_entries.append({
                        "schema_index": i,
                        "description": schema.description,
                        "cypher": cypher,
                        "params": _sanitize_params(params),
                        "row_count": node_count,
                        "truncated": truncated,
                        "build_time_ms": round(build_time * 1000, 2),
                        "exec_time_ms": round(exec_time * 1000, 2),
                    })

            except Exception as error:
                self.log.error(
                    f"Error executing schema {i}: {error}", exc_info=True
                )
                results.append(
                    QueryResult(
                        schema_description=schema.description,
                        rows=[],
                        node_count=0,
                        truncated=False,
                    )
                )
                if debug:
                    trace_entries.append({
                        "schema_index": i,
                        "description": schema.description,
                        "error": str(error),
                    })

        debug_trace = {"queries": trace_entries} if debug else None
        return results, debug_trace

    def _enforce_budgets(self, schema):
        """Clamp schema limits to configured budget maximums.

        Returns a modified copy if changes are needed (schemas are Pydantic
        models so we use model_copy).
        """
        budget = self.config.budget
        updates: dict[str, Any] = {}

        if schema.limit > budget.max_query_limit:
            updates["limit"] = budget.max_query_limit

        if schema.path and schema.path.max_hops > budget.path_max_hops:
            new_path = schema.path.model_copy(
                update={"max_hops": budget.path_max_hops}
            )
            updates["path"] = new_path

        if schema.compare and schema.compare.hops > budget.compare_max_hops:
            new_compare = schema.compare.model_copy(
                update={"hops": budget.compare_max_hops}
            )
            updates["compare"] = new_compare

        if updates:
            return schema.model_copy(update=updates)
        return schema


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Ensure params are JSON-serializable for debug trace."""
    sanitized = {}
    for k, v in params.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            sanitized[k] = v
        elif isinstance(v, list):
            sanitized[k] = [str(x) for x in v]
        else:
            sanitized[k] = str(v)
    return sanitized
