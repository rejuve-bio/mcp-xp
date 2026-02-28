"""GraphRAG Pipeline – async, schema-aware retrieval for Galaxy knowledge graph.

Orchestrates query preprocessing, semantic retrieval (via Informer adapter),
schema-aware graph expansion, hybrid ranking, context assembly, and output
packaging.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from app.bioblend_server.GraphRAG.config import GraphRAGConfig
from app.bioblend_server.GraphRAG.context_builder import ContextBuilder
from app.bioblend_server.GraphRAG.graph_retriever import GraphRetriever
from app.bioblend_server.GraphRAG.hybrid_ranker import HybridRanker


class GraphRAGPipeline:
    """End-to-end async GraphRAG pipeline.

    Stages:
      1. Query intake & preprocessing
      2. Semantic retrieval (Informer adapter)
      3. Schema-aware graph expansion
      4. Hybrid ranking
      5. Context assembly
      6. Output packaging
    """

    def __init__(
        self,
        graph_connector: Any,
        semantic_adapter: Any,
        config: Dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            graph_connector: A ``Neo4jGraphConnector`` instance.
            semantic_adapter: An ``InformerSemanticAdapter`` instance.
            config: Optional dict of overrides for ``GraphRAGConfig``.
        """
        self.config = GraphRAGConfig.from_dict(config)
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper(), logging.INFO)
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        self.graph_connector = graph_connector
        self.semantic_adapter = semantic_adapter

        self.retriever = GraphRetriever(
            graph_connector, semantic_adapter, self.config
        )
        self.hybrid_ranker = HybridRanker(self.config)
        self.context_builder = ContextBuilder(self.config)

    # Public API

    async def retrieve_context(
        self,
        query: str,
        context_budget_chars: int | None = None,
        top_k: int = 10,
        entities_by_type: Dict[str, List[Dict[str, Any]]] | None = None,
    ) -> Dict[str, Any]:
        """Run the full GraphRAG pipeline.

        Args:
            query: Raw user query.
            context_budget_chars: Max chars for the assembled context
                (defaults to ``config.context_budget_chars``).
            top_k: Number of top-ranked candidates to keep.
            entities_by_type: Optional entity lists passed to the
                semantic adapter.

        Returns:
            A dict containing ``context``, ``extracted_subgraph``,
            ``ranked_nodes``, ``semantic_matches``, ``graph_expanded``,
            and ``metadata``.
        """
        budget = context_budget_chars or self.config.context_budget_chars

        # 1. Preprocess query
        cleaned_query = self._preprocess_query(query)
        self.logger.info("Pipeline query: %r → %r", query, cleaned_query)

        # 2–3. Semantic retrieval + graph expansion (async)
        retrieval_payload = await self.retriever.retrieve(
            query=cleaned_query,
            top_k=top_k,
            entities_by_type=entities_by_type,
        )

        # 4. Hybrid ranking
        ranked_nodes = self.hybrid_ranker.rank(
            candidate_nodes=retrieval_payload["candidate_nodes"],
            semantic_scores=retrieval_payload["semantic_scores"],
            graph_distances=retrieval_payload["graph_distances"],
            top_k=top_k,
            base_reasons=retrieval_payload["reasons"],
        )

        # 5. Subgraph extraction for context
        selected_ids = [item["node"]["id"] for item in ranked_nodes]
        if not selected_ids:
            selected_ids = [
                seed["id"] for seed in retrieval_payload["seed_nodes"]
            ]

        subgraph_nodes, subgraph_edges = self.graph_connector.extract_subgraph(
            seed_node_ids=selected_ids,
            max_nodes=self.config.max_subgraph_nodes,
            rel_filter=None,
        )
        subgraph = {"nodes": subgraph_nodes, "edges": subgraph_edges}

        # 6. Context assembly (schema-aware)
        context = self.context_builder.build_context(
            ranked_nodes=ranked_nodes,
            subgraph=subgraph,
            max_chars=budget,
            expanded_nodes=retrieval_payload.get("expanded_nodes"),
            expanded_edges=retrieval_payload.get("expanded_edges"),
        )

        # 7. Output packaging
        ranked_payload = [
            {
                "node": item["node"],
                "score": float(item["score"]),
                "reason": item["reason"],
            }
            for item in ranked_nodes
        ]

        metadata = {
            "query": query,
            "cleaned_query": cleaned_query,
            "seed_count": len(retrieval_payload["seed_nodes"]),
            "candidate_count": len(retrieval_payload["candidate_nodes"]),
            "semantic_hit_count": len(retrieval_payload["semantic_hits"]),
            "graph_expanded_count": len(
                retrieval_payload.get("expanded_nodes", {})
            ),
            "ranked_count": len(ranked_payload),
            "context_chars": len(context),
            "weights": {
                "semantic": self.config.weight_semantic,
                "graph": self.config.weight_graph,
                "type": self.config.weight_type,
            },
            "config": {
                "max_hops": self.config.max_hops,
                "max_depth": self.config.max_depth,
                "k_seed": self.config.k_seed,
            },
        }

        return {
            "context": context,
            "extracted_subgraph": subgraph,
            "ranked_nodes": ranked_payload,
            "semantic_matches": retrieval_payload["semantic_hits"],
            "graph_expanded": {
                "nodes": retrieval_payload.get("expanded_nodes", {}),
                "edges": retrieval_payload.get("expanded_edges", []),
            },
            "metadata": metadata,
        }

    # Query preprocessing

    @staticmethod
    def _preprocess_query(query: str) -> str:
        """Normalise and clean the raw user query.

        - Strip leading/trailing whitespace
        - Collapse multiple whitespace characters
        - Strip surrounding quotes
        """
        text = query.strip()
        text = re.sub(r"\s+", " ", text)
        # Remove surrounding quotes if present
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
            text = text[1:-1].strip()
        return text
