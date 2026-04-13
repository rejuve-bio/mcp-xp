"""Deterministic entity resolution: semantic hits → Neo4j nodes.

Maps raw semantic search results to exact Neo4j nodes via property matching,
producing ``ResolvedEntity`` instances with ``element_id`` for downstream use
by the LLM planner and Cypher builder.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from app.bioblend_server.GraphRAG.models import ResolvedEntity
from app.bioblend_server.GraphRAG.neo4j_connector import Neo4jGraphConnector
from app.bioblend_server.GraphRAG.schema import (
    FUZZY_MAPPING_FIELDS,
    PROPERTY_MAPPING_ORDER,
    SEMANTIC_META_KEYS,
)


class EntityResolver:
    """Maps semantic search hits to exact Neo4j nodes.

    This runs *before* the LLM planner so that the planner receives concrete
    graph entities (with ``element_id``) instead of fuzzy text hits.
    """

    def __init__(self, connector: Neo4jGraphConnector) -> None:
        self.connector = connector
        self.log = logging.getLogger(self.__class__.__name__)

    async def resolve(
        self,
        semantic_hits: list[dict[str, Any]],
        query: str,
        max_seeds: int = 5,
    ) -> list[ResolvedEntity]:
        """Map semantic hits to resolved Neo4j entities.

        Steps:
            1. For each hit, extract lookup values from IDs and metadata.
            2. Multi-label property matching via ``find_nodes_by_values``.
            3. Fuzzy fallback on hit text if exact match fails.
            4. Fallback on raw query string if nothing mapped.
            5. Deduplicate, score-sort, truncate to *max_seeds*.
        """
        self.log.info(
            f"Resolving {len(semantic_hits)} semantic hits for query: '{query}'"
        )

        seed_map: dict[str, dict[str, Any]] = {}
        scores: dict[str, float] = {}

        for rank, hit in enumerate(semantic_hits):
            try:
                hit_score = self._semantic_hit_score(hit, rank)
                lookup_values = self._extract_lookup_values(hit)
                matched_nodes = await self._lookup_nodes(lookup_values)

                # Fuzzy fallback: try entity name from metadata first,
                # then hit text as last resort
                if not matched_nodes:
                    fuzzy_values: list[str] = []
                    meta = hit.get("meta", {})
                    if isinstance(meta, dict):
                        meta_name = meta.get("name")
                        if isinstance(meta_name, str) and meta_name.strip():
                            fuzzy_values.append(meta_name.strip())
                    text_value = hit.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        fuzzy_values.append(text_value.strip())
                    if fuzzy_values:
                        matched_nodes = await self._lookup_nodes(
                            fuzzy_values, fuzzy_only=True
                        )

                for node in matched_nodes:
                    node_id = node["id"]
                    if node_id not in seed_map:
                        seed_map[node_id] = node
                    scores[node_id] = max(scores.get(node_id, 0.0), hit_score)
            except Exception as error:
                self.log.error(
                    f"Error processing hit rank {rank}: {error}", exc_info=True
                )

        # Fallback: try the raw query as a fuzzy lookup
        if not seed_map and query.strip():
            self.log.debug("No seeds from hits, falling back to query text")
            for node in await self._lookup_nodes([query], fuzzy_only=True):
                seed_map[node["id"]] = node

        # Sort by score descending, truncate
        ordered = sorted(
            seed_map.values(),
            key=lambda n: (-scores.get(n["id"], 0.0), n["id"]),
        )[:max_seeds]

        # Convert to ResolvedEntity
        entities = []
        for node in ordered:
            entities.append(
                ResolvedEntity(
                    canonical_id=node["id"],
                    label=node["label"],
                    properties=node.get("properties", {}),
                    element_id=node.get("element_id", ""),
                    confidence=min(scores.get(node["id"], 0.0), 1.0),
                )
            )

        self.log.info(f"Resolved {len(entities)} entities")
        return entities

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _lookup_nodes(
        self,
        lookup_values: Iterable[str],
        fuzzy_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Look up nodes across all labels using exact and/or fuzzy matching."""
        found: dict[str, dict[str, Any]] = {}
        seen_values: set[str] = set()
        values_to_try: list[str] = []

        for raw_value in lookup_values:
            value = str(raw_value).strip()
            if not value or value in seen_values:
                continue
            seen_values.add(value)
            values_to_try.append(value)

            # Also try the part after ":" for canonical IDs
            if ":" in value:
                part = value.split(":", 1)[1]
                if part and part not in seen_values:
                    seen_values.add(part)
                    values_to_try.append(part)

        if not values_to_try:
            return []

        exact_props_by_label: dict[str, set[str]] = {}
        fuzzy_props_by_label: dict[str, set[str]] = {}

        if fuzzy_only:
            # Fuzzy-only mode: only CONTAINS matching on name/file_name fields
            for label, prop in FUZZY_MAPPING_FIELDS:
                fuzzy_props_by_label.setdefault(label, set()).add(prop)
        else:
            # Exact-only mode: IN-list matching on ID and name fields
            for label, prop in PROPERTY_MAPPING_ORDER:
                exact_props_by_label.setdefault(label, set()).add(prop)

        labels = set(exact_props_by_label.keys()) | set(fuzzy_props_by_label.keys())

        for label in labels:
            try:
                exact_props = list(exact_props_by_label.get(label, []))
                fuzzy_props = list(fuzzy_props_by_label.get(label, []))

                nodes = await self.connector.find_nodes_by_values(
                    label=label,
                    exact_props=exact_props,
                    fuzzy_props=fuzzy_props,
                    values=values_to_try,
                )
                for node in nodes:
                    found[node["id"]] = node
            except Exception as error:
                self.log.error(
                    f"Error looking up nodes for label {label}: {error}",
                    exc_info=True,
                )

        return list(found.values())

    @staticmethod
    def _extract_lookup_values(hit: dict[str, Any]) -> list[str]:
        """Extract candidate lookup values from a semantic hit."""
        values: list[str] = []
        hit_id = hit.get("id")
        if hit_id is not None:
            values.append(str(hit_id))

        meta = hit.get("meta", {})
        if isinstance(meta, dict):
            for key in SEMANTIC_META_KEYS:
                if key in meta and meta[key] is not None:
                    values.append(str(meta[key]))
        return values

    @staticmethod
    def _semantic_hit_score(hit: dict[str, Any], rank: int) -> float:
        """Extract a relevance score from a semantic hit."""
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
