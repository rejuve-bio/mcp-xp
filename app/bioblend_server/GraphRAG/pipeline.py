from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from app.bioblend_server.GraphRAG.config import GraphRAGConfig, GraphRAGEnum
from app.bioblend_server.GraphRAG.context_builder import ContextBuilder
from app.bioblend_server.GraphRAG.graph_retriever import GraphRetriever


class GraphRAGPipeline:
    """End-to-end async GraphRAG pipeline using targeted Cypher contexts.

    Stages:
      1. Query intake & preprocessing
      2. Semantic retrieval (Informer schema-aware wrapper)
      3. Payload fetching (Targeted Cypher contexts)
      4. Markdown Context assembly
      5. Output packaging
      
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
        self.context_builder = ContextBuilder(self.config)

    # Public API

    async def retrieve_context(
        self,
        query: str,
        query_type: str = "local",
        top_k: int = 10,
        entities_by_type: Dict[str, List[Dict[str, Any]]] | None = None,
        compare_workflows: tuple[str, str] | None = None,
        connect_tools: tuple[str, str] | None = None,
        category: str | None = None,
    ) -> Dict[str, Any]:
        """Run the full Cypher-Native GraphRAG pipeline.

        Args:
            query: Raw user query.
            query_type: "local" (semantic search), "global" (GDS analytics),
                or "complex" (multi-hop relationship queries).
            top_k: Number of top-ranked candidates to keep.
            entities_by_type: Optional entity lists passed to the
                semantic adapter.
            compare_workflows: Tuple of (workflow_name_a, workflow_name_b)
                for workflow comparison queries.
            connect_tools: Tuple of (tool_name_a, tool_name_b) for
                tool connection path queries.
            category: Category name for category drill-down queries.

        Returns:
            A dict containing ``context``, ``semantic_matches``, 
            ``context_payloads``, and ``metadata``.
        """

        # 1. Preprocess query
        cleaned_query = self._preprocess_query(query)
        self.logger.info("Pipeline query: %r → %r", query, cleaned_query)

        # 2. Dual-Mode Routing
        is_global = query_type.lower() == "global"
        
        if is_global:
            self.logger.info("Executing GLOBAL analytics query route.")
            context_payloads = [
                self.graph_connector.get_most_used_tools(limit=GraphRAGEnum.MOST_USED_TOOL.value),
                self.graph_connector.get_tools_by_community(limit=GraphRAGEnum.TOOl_IN_COMMUNITY.value)
            ]
            seed_nodes = []
            semantic_hits = []
        elif query_type.lower() == "complex":
            self.logger.info("Executing COMPLEX query route.")
            # First, do the standard local semantic search for base context
            retrieval_payload = await self.retriever.retrieve(
                query=cleaned_query,
                top_k=top_k,
                entities_by_type=entities_by_type,
            )
            context_payloads = list(retrieval_payload["context_payloads"])
            seed_nodes = retrieval_payload["seed_nodes"]
            semantic_hits = retrieval_payload["semantic_hits"]
            
            # Then, layer on the complex Cypher fetchers
            if compare_workflows:
                payload = self.graph_connector.get_workflow_comparison(
                    compare_workflows[0], compare_workflows[1]
                )
                if payload:
                    context_payloads.append(payload)
                    
            if connect_tools:
                payload = self.graph_connector.get_tool_connection_path(
                    connect_tools[0], connect_tools[1]
                )
                if payload:
                    context_payloads.append(payload)
                    
            if category:
                payload = self.graph_connector.get_category_tools(category)
                if payload:
                    context_payloads.append(payload)
        else:
            self.logger.info("Executing LOCAL semantic query route.")
            retrieval_payload = await self.retriever.retrieve(
                query=cleaned_query,
                top_k=top_k,
                entities_by_type=entities_by_type,
            )
            context_payloads = retrieval_payload["context_payloads"]
            seed_nodes = retrieval_payload["seed_nodes"]
            semantic_hits = retrieval_payload["semantic_hits"]

        # 3. Context assembly (schema-aware payload formatting)
        context = self.context_builder.build_context(
            context_payloads=context_payloads,
        )

        # 4. Output packaging
        metadata = {
            "query": query,
            "cleaned_query": cleaned_query,
            "query_type": query_type,
            "seed_count": len(seed_nodes),
            "semantic_hit_count": len(semantic_hits),
            "context_payloads_fetched": len(context_payloads),
            "context_chars": len(context),
        }

        return {
            "context": context,
            "semantic_matches": semantic_hits,
            "seed_nodes": seed_nodes,
            "context_payloads": context_payloads,
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
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
            text = text[1:-1].strip()
        return text
