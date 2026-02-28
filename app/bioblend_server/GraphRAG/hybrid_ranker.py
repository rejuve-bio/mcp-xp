from __future__ import annotations

from typing import Any, Dict, List

from app.bioblend_server.GraphRAG.config import GraphRAGConfig


def hybrid_score(
    semantic_score: float,
    graph_proximity: float,
    node_type_weight: float,
    w_sem: float = 0.6,
    w_graph: float = 0.3,
    w_type: float = 0.1,
) -> float:
    """
    Linear hybrid score:
      score = w_sem * semantic_score + w_graph * graph_proximity + w_type * node_type_weight
    """

    return (w_sem * semantic_score) + (w_graph * graph_proximity) + (w_type * node_type_weight)


class HybridRanker:
    def __init__(self, config: GraphRAGConfig) -> None:
        self.config = config

    def rank(
        self,
        candidate_nodes: Dict[str, Dict[str, Any]],
        semantic_scores: Dict[str, float],
        graph_distances: Dict[str, int],
        top_k: int,
        base_reasons: Dict[str, str] | None = None,
    ) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        reasons = base_reasons or {}

        for node_id, node in candidate_nodes.items():
            semantic_value = float(semantic_scores.get(node_id, 0.0))
            distance = graph_distances.get(node_id)
            graph_proximity = 0.0 if distance is None else 1.0 / (1.0 + float(distance))
            type_weight = float(self.config.node_type_weights.get(node.get("label", ""), 0.5))
            score = hybrid_score(
                semantic_score=semantic_value,
                graph_proximity=graph_proximity,
                node_type_weight=type_weight,
                w_sem=self.config.weight_semantic,
                w_graph=self.config.weight_graph,
                w_type=self.config.weight_type,
            )

            if semantic_value > 0:
                reason = "semantic_hit"
            elif graph_proximity > 0:
                reason = "neighbor_of_seed"
            elif int(node.get("degree", 0)) >= 4:
                reason = "high_degree"
            else:
                reason = reasons.get(node_id, "graph_candidate")

            ranked.append(
                {
                    "node": node,
                    "score": score,
                    "reason": reason,
                    "components": {
                        "semantic_score": semantic_value,
                        "graph_proximity": graph_proximity,
                        "node_type_weight": type_weight,
                    },
                }
            )

        ranked.sort(key=lambda item: (-item["score"], item["node"]["id"]))
        return ranked[: max(top_k, 1)]
