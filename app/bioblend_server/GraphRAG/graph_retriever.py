from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Tuple

from graph_rag.config import GraphRAGConfig


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
    """Maps semantic hits to graph seeds and expands their neighborhoods."""

    def __init__(self, graph_connector: Any, semantic_retriever: Any, config: GraphRAGConfig) -> None:
        self.graph_connector = graph_connector
        self.semantic_retriever = semantic_retriever
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def retrieve(self, query: str, top_k: int) -> Dict[str, Any]:
        semantic_hits = self._safe_semantic_search(query=query, top_k=self.config.k_seed)
        seed_nodes, semantic_scores = self.map_semantic_hits_to_seed_nodes(semantic_hits, query)

        if not seed_nodes:
            seed_nodes = self._fallback_query_seeds(query)

        candidate_nodes: Dict[str, Dict[str, Any]] = {}
        graph_distances: Dict[str, int] = {}
        reasons: Dict[str, str] = {}

        for seed in seed_nodes:
            seed_id = seed["id"]
            candidate_nodes[seed_id] = seed
            graph_distances[seed_id] = 0
            reasons[seed_id] = "semantic_hit" if semantic_scores.get(seed_id, 0.0) > 0 else "graph_candidate"

        for seed in seed_nodes:
            if len(candidate_nodes) >= self.config.max_nodes:
                break
            neighbors = self.graph_connector.neighbors(
                seed["id"], depth=self.config.max_depth, rel_types=self.config.relation_priority
            )
            ordered_neighbors = sorted(
                neighbors.values(), key=lambda node: (int(node.get("distance", 10**6)), node.get("id", ""))
            )
            for node in ordered_neighbors:
                node_id = node["id"]
                distance = int(node.get("distance", self.config.max_depth + 1))
                if node_id not in graph_distances or distance < graph_distances[node_id]:
                    graph_distances[node_id] = distance

                if node_id not in candidate_nodes and len(candidate_nodes) >= self.config.max_nodes:
                    continue
                candidate_nodes[node_id] = node

                if semantic_scores.get(node_id, 0.0) > 0:
                    reasons[node_id] = "semantic_hit"
                elif graph_distances.get(node_id, 99999) > 0:
                    reasons[node_id] = "neighbor_of_seed"
                elif int(node.get("degree", 0)) >= 4:
                    reasons[node_id] = "high_degree"
                else:
                    reasons[node_id] = "graph_candidate"

        if not candidate_nodes:
            return {
                "candidate_nodes": {},
                "seed_nodes": [],
                "semantic_scores": {},
                "graph_distances": {},
                "reasons": {},
                "semantic_hits": semantic_hits,
            }

        ordered_ids = sorted(
            candidate_nodes.keys(),
            key=lambda node_id: (
                -semantic_scores.get(node_id, 0.0),
                graph_distances.get(node_id, 10**6),
                node_id,
            ),
        )[: max(top_k, 1, self.config.max_nodes)]

        candidate_nodes = {node_id: candidate_nodes[node_id] for node_id in ordered_ids}
        graph_distances = {node_id: graph_distances.get(node_id, 10**6) for node_id in ordered_ids}
        reasons = {node_id: reasons.get(node_id, "graph_candidate") for node_id in ordered_ids}
        seed_set = {node["id"] for node in seed_nodes}
        seed_nodes = [candidate_nodes[node_id] for node_id in ordered_ids if node_id in seed_set]

        return {
            "candidate_nodes": candidate_nodes,
            "seed_nodes": seed_nodes,
            "semantic_scores": semantic_scores,
            "graph_distances": graph_distances,
            "reasons": reasons,
            "semantic_hits": semantic_hits,
        }

    def map_semantic_hits_to_seed_nodes(
        self, semantic_hits: List[Dict[str, Any]], query: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
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
                semantic_scores[node_id] = max(semantic_scores.get(node_id, 0.0), hit_score)

        if not seed_map and query.strip():
            for node in self._fallback_query_seeds(query):
                seed_map[node["id"]] = node

        ordered_nodes = sorted(
            seed_map.values(), key=lambda node: (-semantic_scores.get(node["id"], 0.0), node["id"])
        )
        return ordered_nodes, semantic_scores

    def _lookup_nodes(self, lookup_values: Iterable[str], fuzzy_only: bool = False) -> List[Dict[str, Any]]:
        found: Dict[str, Dict[str, Any]] = {}
        seen_values: set[str] = set()

        for raw_value in lookup_values:
            value = str(raw_value).strip()
            if not value:
                continue
            if value in seen_values:
                continue
            seen_values.add(value)
            if ":" in value:
                seen_values.add(value.split(":", 1)[1])

            values_to_try = [value]
            if ":" in value:
                values_to_try.append(value.split(":", 1)[1])

            for candidate_value in values_to_try:
                if not fuzzy_only:
                    for label, prop in PROPERTY_MAPPING_ORDER:
                        for node in self.graph_connector.find_nodes_by_property(
                            label, prop, candidate_value, fuzzy=False
                        ):
                            found[node["id"]] = node

                for label, prop in FUZZY_MAPPING_FIELDS:
                    for node in self.graph_connector.find_nodes_by_property(
                        label, prop, candidate_value, fuzzy=True
                    ):
                        found[node["id"]] = node
        return list(found.values())

    def _fallback_query_seeds(self, query: str) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        fallback_nodes = self._lookup_nodes([query], fuzzy_only=True)
        fallback_nodes.sort(key=lambda node: (-int(node.get("degree", 0)), node["id"]))
        return fallback_nodes[: self.config.k_seed]

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

    def _safe_semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        try:
            return self.semantic_retriever.search(query, top_k=top_k)
        except Exception as error:  # pragma: no cover - defensive path
            self.logger.warning("Semantic search failed: %s", error)
            return []
