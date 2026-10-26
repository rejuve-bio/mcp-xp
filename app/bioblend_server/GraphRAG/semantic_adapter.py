from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.bioblend_server.informer.search.semantic_searcher import SemanticSearcher


class InformerSemanticAdapter:
    """ Wraps Informer's ``SemanticSearcher`` for GraphRAG consumption. """

    # Entity-specific ID field names in Informer results
    _ID_FIELDS = {
        "tool": "tool_id",
        "workflow": "workflow_id",
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
        self.log = logging.getLogger(self.__class__.__name__)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        entities_by_type: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run semantic search across configured entity types.

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
        try:
            self.log.info(f"Starting semantic search for query: '{query}', top_k: {top_k}, entity_types: {self.entity_types}")

            if entities_by_type is None:
                entities_by_type = {}

            all_hits: List[Dict[str, Any]] = []

            for entity_type in self.entity_types:
                try:
                    self.log.debug(f"Searching for entity_type: {entity_type}")
                    entities = entities_by_type.get(entity_type, [])
                    raw_results = await self.searcher.search(
                        query=query,
                        entity_type=entity_type,
                        entities=entities,
                    )
                    normalised = self._normalise_results(raw_results, entity_type)
                    all_hits.extend(normalised)
                except Exception as exc:
                    self.log.warning(
                        f"Semantic search failed for entity_type={entity_type}: {exc}",
                        exc_info=True,
                    )

            # Sort by score descending, then truncate
            all_hits.sort(key=lambda h: -h.get("score", 0.0))
            result = all_hits[:top_k]
            self.log.info(f"Semantic search completed with {len(result)} hits")
            return result
        except Exception as error:
            self.log.error(f"Error in search: {error}", exc_info=True)
            return []

    def _normalise_results(
        self,
        raw_results: List[Dict[str, Any]],
        entity_type: str,
    ) -> List[Dict[str, Any]]:
        """Map Informer result dicts to GraphRAG hit format."""
        try:
            self.log.info(f"Normalising {len(raw_results)} results for entity_type: {entity_type}")
            id_field = self._ID_FIELDS.get(entity_type, "id")
            hits: List[Dict[str, Any]] = []

            for item in raw_results:
                try:
                    self.log.debug(f"Processing item: {item}")
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
                except Exception as error:
                    self.log.error(f"Error processing item {item}: {error}", exc_info=True)
                    continue

            self.log.info(f"Normalised {len(hits)} hits")
            return hits
        except Exception as error:
            self.log.error(f"Error in _normalise_results: {error}", exc_info=True)
            return []