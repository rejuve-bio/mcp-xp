from __future__ import annotations

import logging
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Tuple,
)

from app.bioblend_server.GraphRAG.config import GraphRAGConfig


PROPERTY_MAPPING_ORDER: List[Tuple[str, str]] = [
    ("Workflow", "workflow_id"),
    ("Workflow", "file_name"),
    ("Workflow", "name"),
    ("Step", "step_id"),
    ("Step", "step_uid"),
    ("Step", "file_name"),
    ("Step", "name"),
    ("Tool", "tool_id"),
    ("Tool", "id"),
    ("Tool", "name"),
    ("Input", "input_uid"),
    ("Input", "name"),
    ("Output", "output_uid"),
    ("Output", "name"),
    ("Category", "category_id"),
    ("Category", "name"),
    ("ToolInput", "tool_input_uid"),
    ("ToolInput", "input_name"),
    ("ToolOutput", "tool_output_uid"),
    ("ToolOutput", "output_name"),
]

FUZZY_MAPPING_FIELDS: List[Tuple[str, str]] = [
    ("Workflow", "name"),
    ("Workflow", "file_name"),
    ("Step", "name"),
    ("Step", "file_name"),
    ("Tool", "name"),
]

SEMANTIC_META_KEYS = [
    "id",
    "workflow_id",
    "step_id",
    "step_uid",
    "tool_id",
    "file_name",
    "name",
    "input_uid",
    "output_uid",
    "category_id",
    "tool_input_uid",
    "tool_output_uid",
]


class GraphRetriever:
    """
    Maps semantic hits to specific Graph Schema endpoints.

    Delegates vector search to the async `InformerSemanticAdapter`, maps
    results to Neo4j nodes, and fetches pre-aggregated contextual structures
    based strictly on entity type (Workflow, Tool, Step).
    """

    def __init__(
        self,
        graph_connector: Any,
        semantic_adapter: Any,
        config: GraphRAGConfig,
    ) -> None:
        self.graph_connector = graph_connector
        self.semantic_adapter = semantic_adapter
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    async def retrieve(
        self,
        query: str,
        top_k: int,
        entities_by_type: dict | None = None,
    ) -> Dict[str, Any]:
        """
        Run semantic search and targeted context retrieval.

        Args:
            query: Natural-language query string.
            top_k: Maximum seed nodes to process.
            entities_by_type: Optional entity lists forwarded to adapter.

        Returns:
            A payload dict containing ``semantic_hits``, ``seed_nodes``,
            ``semantic_scores``, and ``context_payloads``.
        """
        # 1. Semantic retrieval via adapter
        semantic_hits = await self._safe_semantic_search(
            query=query,
            top_k=top_k,
            entities_by_type=entities_by_type,
        )

        # 2. Map hits to exact DB nodes
        seed_nodes, semantic_scores = self.map_semantic_hits_to_seed_nodes(
            semantic_hits, query
        )
        if not seed_nodes:
            seed_nodes = self._fallback_query_seeds(query)[:top_k]

        seed_nodes = seed_nodes[:top_k]
        
        # 3. Targeted routing: Dispatch to corresponding Cypher patterns
        context_payloads: List[Dict[str, Any]] = []
        
        for seed in seed_nodes:
            label = seed.get("label")
            internal_id_raw = seed.get("internal_id")
            if internal_id_raw is None or internal_id_raw == "":
                self.logger.warning(f"Seed {seed['id']} missing internal_id. Skipping context fetch.")
                continue
                
            internal_id = int(str(internal_id_raw))
            payload = {}
            
            if label == "Workflow":
                payload = self.graph_connector.get_workflow_context(internal_id)
            elif label == "Tool":
                payload = self.graph_connector.get_tool_context(internal_id)
            elif label == "Step":
                payload = self.graph_connector.get_step_context(internal_id)
            else:
                self.logger.debug(f"Label {label} does not have a dedicated fetcher yet.")
                
            if payload:
                payload["_source_seed"] = seed
                context_payloads.append(payload)

        return {
            "semantic_hits": semantic_hits,
            "seed_nodes": seed_nodes,
            "semantic_scores": semantic_scores,
            "context_payloads": context_payloads,
        }

    # Mapping
    def map_semantic_hits_to_seed_nodes(
        self,
        semantic_hits: List[Dict[str, Any]],
        query: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """ Map semantic hits to seed nodes. """

        seed_map: Dict[str, Dict[str, Any]] = {}
        semantic_scores: Dict[str, float] = {}

        for rank, hit in enumerate(semantic_hits):
            hit_score = self._semantic_hit_score(hit, rank)
            lookup_values = self._extract_lookup_values(hit)
            matched_nodes = self._lookup_nodes(lookup_values)

            if not matched_nodes:
                text_value = hit.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    matched_nodes = self._lookup_nodes([text_value], fuzzy_only=True)

            for node in matched_nodes:
                node_id = node["id"]
                if node_id not in seed_map:
                    seed_map[node_id] = node
                semantic_scores[node_id] = max(
                    semantic_scores.get(node_id, 0.0), hit_score
                )

        if not seed_map and query.strip():
            for node in self._fallback_query_seeds(query):
                seed_map[node["id"]] = node

        ordered_nodes = sorted(
            seed_map.values(),
            key=lambda node: (-semantic_scores.get(node["id"], 0.0), node["id"]),
        )
        return ordered_nodes, semantic_scores

    def _lookup_nodes(
        self,
        lookup_values: Iterable[str],
        fuzzy_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """ Lookup nodes in the graph based on a list of lookup values. """

        found: Dict[str, Dict[str, Any]] = {}
        seen_values: set[str] = set()
        values_to_try: List[str] = []

        for raw_value in lookup_values:
            value = str(raw_value).strip()
            if not value or value in seen_values:
                continue
            seen_values.add(value)
            values_to_try.append(value)
            
            if ":" in value:
                part = value.split(":", 1)[1]
                if part not in seen_values:
                    seen_values.add(part)
                    values_to_try.append(part)

        if not values_to_try:
            return []

        exact_props_by_label: Dict[str, set[str]] = {}
        fuzzy_props_by_label: Dict[str, set[str]] = {}

        if not fuzzy_only:
            for label, prop in PROPERTY_MAPPING_ORDER:
                exact_props_by_label.setdefault(label, set()).add(prop)

        for label, prop in FUZZY_MAPPING_FIELDS:
            fuzzy_props_by_label.setdefault(label, set()).add(prop)

        labels = set(exact_props_by_label.keys()) | set(fuzzy_props_by_label.keys())
        
        for label in labels:
            exact_props = list(exact_props_by_label.get(label, []))
            fuzzy_props = list(fuzzy_props_by_label.get(label, []))
            
            nodes = self.graph_connector.find_nodes_by_values(
                label=label,
                exact_props=exact_props,
                fuzzy_props=fuzzy_props,
                values=values_to_try
            )
            for node in nodes:
                found[node["id"]] = node

        return list(found.values())

    def _fallback_query_seeds(self, query: str) -> List[Dict[str, Any]]:
        
        if not query.strip():
            return []
        return self._lookup_nodes([query], fuzzy_only=True)

    def _extract_lookup_values(self, hit: Dict[str, Any]) -> List[str]:

        values: List[str] = []
        hit_id = hit.get("id")
        if hit_id is not None:
            values.append(str(hit_id))

        meta = hit.get("meta", {})
        if isinstance(meta, dict):
            for key in SEMANTIC_META_KEYS:
                if key in meta and meta[key] is not None:
                    values.append(str(meta[key]))
        return values

    def _semantic_hit_score(self, hit: Dict[str, Any], rank: int) -> float:

        for key in ("score", "relevance", "similarity"):
            score_value = hit.get(key)
            if isinstance(score_value, (int, float)):
                return float(score_value)
        meta = hit.get("meta", {})
        if isinstance(meta, dict):
            for key in ("score", "relevance", "similarity"):
                score_value = meta.get(key)
                if isinstance(score_value, (int, float)):
                    return float(score_value)
        return 1.0 / float(rank + 1)

    async def _safe_semantic_search(
        self,
        query: str,
        top_k: int,
        entities_by_type: dict | None = None,
    ) -> List[Dict[str, Any]]:

        try:
            return await self.semantic_adapter.search(
                query,
                top_k=top_k,
                entities_by_type=entities_by_type,
            )
        except Exception as error:
            self.logger.warning("Semantic search failed: %s", error)
            return []