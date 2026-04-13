"""End-to-end async GraphRAG pipeline.

Orchestrates the full flow: query preprocessing → semantic search →
entity resolution → LLM planning → Cypher execution → context rendering →
answer synthesis.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.bioblend_server.GraphRAG.config import GraphRAGConfig
from app.bioblend_server.GraphRAG.context_builder import ContextBuilder
from app.bioblend_server.GraphRAG.entity_resolver import EntityResolver
from app.bioblend_server.GraphRAG.executor import QueryExecutor
from app.bioblend_server.GraphRAG.models import ExecutionResult, PlannerValidationError
from app.bioblend_server.GraphRAG.neo4j_connector import Neo4jGraphConnector
from app.bioblend_server.GraphRAG.planner import GraphRAGPlanner
from app.bioblend_server.GraphRAG.semantic_adapter import InformerSemanticAdapter
from app.bioblend_server.utils import get_llm_response


# ---------------------------------------------------------------------------
# Answer synthesis prompt
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
You are a knowledgeable Galaxy bioinformatics assistant. Your task is to answer
the user's question using ONLY the graph evidence provided below. Do not
fabricate information that is not present in the evidence.

## Instructions

1. Read the user's question carefully.
2. Read the graph evidence — it contains structured facts retrieved from a
   Galaxy knowledge graph of bioinformatics tools, workflows, steps, and their
   relationships.
3. Synthesize a clear, accurate, and helpful answer grounded strictly in the
   evidence.
4. If the evidence is incomplete or does not fully answer the question,
   acknowledge what is known and what is missing.
5. Use specific names, versions, and relationships from the evidence.
6. Structure your answer with clear sections or bullet points when appropriate.
7. Keep the answer concise but thorough — prefer precision over length.

## User Question

{query}

## Graph Evidence

{evidence}

## Known Limitations

{limitations}

## Your Answer
"""


class GraphRAGPipeline:
    """Orchestrates the planner-driven GraphRAG pipeline.

    Stages:
        1. Query preprocessing
        2. Semantic search (Qdrant via InformerSemanticAdapter)
        3. Entity resolution (semantic hits → Neo4j nodes)
        4. LLM planning (query + seeds → CypherQuerySchemas)
        5. Cypher execution (schemas → parameterized queries → results)
        6. Context rendering (results → Markdown evidence)
        7. Answer synthesis (query + evidence → LLM-generated answer)
        8. Output packaging
    """

    def __init__(
        self,
        connector: Neo4jGraphConnector,
        semantic_adapter: InformerSemanticAdapter,
        config: dict[str, Any] | GraphRAGConfig | None = None,
    ) -> None:
        if isinstance(config, dict):
            self.config = GraphRAGConfig.from_dict(config)
        elif isinstance(config, GraphRAGConfig):
            self.config = config
        else:
            self.config = GraphRAGConfig()

        self.log = logging.getLogger(self.__class__.__name__)

        self.connector = connector
        self.adapter = semantic_adapter
        self.resolver = EntityResolver(connector)
        self.planner = GraphRAGPlanner(self.config)
        self.executor = QueryExecutor(connector, self.config)
        self.context_builder = ContextBuilder(self.config)

    async def run(
        self,
        query: str,
        debug: bool = False,
    ) -> ExecutionResult:
        """Run the full GraphRAG pipeline.

        Args:
            query: Natural-language user query.
            debug: If True, include per-query timing in the result.

        Returns:
            ``ExecutionResult`` with synthesized answer, raw evidence,
            matched entities, plan summary, limitations, and optional debug trace.
        """
        try:
            self.log.info(f"Pipeline starting for query: '{query}'")

            # 1. Preprocess
            cleaned = _preprocess_query(query)
            self.log.debug(f"Cleaned query: '{cleaned}'")

            # 2. Semantic search
            self.log.info("Running semantic search")
            semantic_hits = await self.adapter.search(
                cleaned,
                top_k=self.config.budget.semantic_top_k,
            )
            self.log.info(f"Semantic search returned {len(semantic_hits)} hits")

            # 3. Entity resolution
            self.log.info("Resolving entities")
            seeds = await self.resolver.resolve(
                semantic_hits,
                cleaned,
                max_seeds=self.config.budget.max_seeds,
            )
            self.log.info(f"Resolved {len(seeds)} seed entities")

            # 4. LLM planning
            self.log.info("Running LLM planner")
            planner_output = await self.planner.plan(cleaned, seeds)
            self.log.info(
                f"Planner generated {len(planner_output.query_schemas)} schemas"
            )

            # 5. Execute queries
            self.log.info("Executing Cypher queries")
            query_results, trace = await self.executor.execute(
                planner_output, debug
            )
            self.log.info(
                f"Executed {len(query_results)} queries, "
                f"total rows: {sum(r.node_count for r in query_results)}"
            )

            # 6. Render evidence context
            raw_evidence = self.context_builder.build_context_from_results(
                query_results, seeds
            )

            # 7. Synthesize answer
            limitations = list(planner_output.limitations)
            self.log.info("Synthesizing answer from evidence")
            answer = await self._synthesize_answer(cleaned, raw_evidence, limitations)

            # 8. Package
            return ExecutionResult(
                answer=answer,
                raw_evidence=raw_evidence,
                matched_entities=seeds,
                query_results=query_results,
                plan_summary=planner_output.reasoning,
                limitations=limitations,
                debug_trace=trace,
            )

        except PlannerValidationError as e:
            self.log.error(f"Planner validation failed: {e}", exc_info=True)
            return ExecutionResult(
                answer=f"I couldn't plan a query for your question: {e}",
                raw_evidence="",
                limitations=["planner_validation_failed"],
            )
        except Exception as e:
            self.log.error(f"Pipeline error: {e}", exc_info=True)
            return ExecutionResult(
                answer=f"An error occurred while processing your question: {e}",
                raw_evidence="",
                limitations=["pipeline_error"],
            )

    async def _synthesize_answer(
        self,
        query: str,
        evidence: str,
        limitations: list[str],
    ) -> str:
        """Use the LLM to synthesize a natural-language answer from evidence.

        Falls back to returning raw evidence if the LLM call fails.
        """
        if not evidence or evidence == "No matching context found.":
            return (
                "I couldn't find relevant information in the Galaxy knowledge graph "
                "to answer your question. Try rephrasing or asking about specific "
                "tools, workflows, or their relationships."
            )

        limitations_text = (
            "\n".join(f"- {l}" for l in limitations)
            if limitations
            else "None"
        )

        prompt = _SYNTHESIS_PROMPT.format(
            query=query,
            evidence=evidence,
            limitations=limitations_text,
        )

        try:
            response = await get_llm_response(prompt)
            if isinstance(response, str) and response.strip():
                return response.strip()
            if isinstance(response, dict):
                # Some LLM providers return parsed JSON
                return str(response.get("answer", response.get("content", str(response))))
            return evidence
        except Exception as e:
            self.log.warning(
                f"Answer synthesis failed, returning raw evidence: {e}",
                exc_info=True,
            )
            return evidence


def _preprocess_query(query: str) -> str:
    """Normalise and clean the raw user query."""
    text = query.strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    return text
