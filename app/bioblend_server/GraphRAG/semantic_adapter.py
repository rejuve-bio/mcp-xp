"""Adapter wrapping Informer's SemanticSearcher for the GraphRAG pipeline.

This module exposes :class:`InformerSemanticAdapter`, a thin async wrapper
that delegates vector search to the Informer's :class:`SemanticSearcher`
and normalises results into the format expected by :class:`GraphRetriever`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.bioblend_server.informer.search.semantic_searcher import SemanticSearcher


class InformerSemanticAdapter:
    """Wraps Informer's ``SemanticSearcher`` for GraphRAG consumption.

    GraphRAG expects semantic hits in the shape::

        [
          {
            "id":    "<entity_id>",
            "text":  "<content snippet>",
            "score": 0.87,
            "meta":  { "name": "...", "tool_id": "...", ... }
          },
          ...
        ]

    The Informer's searcher returns dicts whose keys vary by entity type
    (``tool_id`` / ``workflow_id`` / ``dataset_id``).  This adapter maps
    those results into a uniform shape.
    """

    # Entity-specific ID field names in Informer results
    _ID_FIELDS = {
        "tool": "tool_id",
        "workflow": "workflow_id",
        "dataset": "dataset_id",
    }

    def __init__(
        self,
        semantic_searcher: SemanticSearcher,
        entity_types: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            semantic_searcher: A fully-initialised Informer ``SemanticSearcher``.
            entity_types: Entity types to search over. Defaults to
                ``["tool", "workflow"]``.
        """
        self.searcher = semantic_searcher
        self.entity_types = entity_types or ["tool", "workflow"]
        self.logger = logging.getLogger(self.__class__.__name__)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        entities_by_type: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """Run semantic search across configured entity types.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of results to return.
            entities_by_type: Optional mapping of ``entity_type`` →
                ``list[entity_dict]`` as required by the Informer searcher.
                When omitted an empty list is passed (Informer will use
                its own Qdrant-indexed data).

        Returns:
            A list of normalised hit dicts sorted by score descending,
            capped at *top_k*.
        """
        if entities_by_type is None:
            entities_by_type = {}

        all_hits: List[Dict[str, Any]] = []

        for entity_type in self.entity_types:
            entities = entities_by_type.get(entity_type, [])
            try:
                raw_results = await self.searcher.search(
                    query=query,
                    entity_type=entity_type,
                    entities=entities,
                )
                normalised = self._normalise_results(raw_results, entity_type)
                all_hits.extend(normalised)
            except Exception as exc:
                self.logger.warning(
                    "Semantic search failed for entity_type=%s: %s",
                    entity_type,
                    exc,
                )

        # Sort by score descending, then truncate
        all_hits.sort(key=lambda h: -h.get("score", 0.0))
        return all_hits[:top_k]

    def _normalise_results(
        self,
        raw_results: List[Dict[str, Any]],
        entity_type: str,
    ) -> List[Dict[str, Any]]:
        """Map Informer result dicts to GraphRAG hit format."""
        id_field = self._ID_FIELDS.get(entity_type, "id")
        hits: List[Dict[str, Any]] = []

        for item in raw_results:
            entity_id = str(item.get(id_field, item.get("name", "")))
            score = float(item.get("score", 0.0))
            text = str(item.get("content", item.get("description", "")))

            meta: Dict[str, Any] = {
                "name": item.get("name", ""),
                "entity_type": entity_type,
                "source": item.get("source", "unknown"),
            }
            # Preserve original ID fields for downstream mapping
            if id_field in item:
                meta[id_field] = item[id_field]
            for extra_key in ("description", "owner", "version", "tool_shed_url"):
                if extra_key in item:
                    meta[extra_key] = item[extra_key]

            hits.append({
                "id": entity_id,
                "text": text,
                "score": score,
                "meta": meta,
            })

        return hits
