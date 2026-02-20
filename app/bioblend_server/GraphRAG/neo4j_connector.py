from __future__ import annotations

import logging
import re
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
from neo4j import GraphDatabase


SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_cypher_name(name: str) -> str:
    if not SAFE_NAME.match(name):
        raise ValueError(f"Unsafe Cypher name: {name!r}")
    return name


class Neo4jGraphConnector:
    """Minimal Neo4j connector with the same API as the in-memory connector."""

    def __init__(self, uri: str, user: str, password: str, database: str | None = None) -> None:

        self.logger = logging.getLogger(self.__class__.__name__)
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def get_node_by_id(self, node_label: str, node_id: str) -> Optional[Dict[str, Any]]:
        if not node_id:
            return None
        safe_label = _validate_cypher_name(node_label)
        field_names = NODE_ID_FIELDS.get(node_label, ["id"])
        clauses: List[str] = []
        for field_name in field_names:
            safe_field = _validate_cypher_name(field_name)
            clauses.append(f"toString(n.{safe_field}) = $node_id")

        query = (
            f"MATCH (n:{safe_label}) "
            f"WHERE {' OR '.join(clauses)} "
            "RETURN labels(n)[0] AS label, properties(n) AS properties, id(n) AS internal_id "
            "LIMIT 1"
        )
        rows = self._run(query, {"node_id": node_id})
        if not rows and ":" in node_id:
            rows = self._run(query, {"node_id": node_id.split(":", 1)[1]})
        if not rows:
            return None
        return self._row_to_node(rows[0], snapshot=None)

    def find_nodes_by_property(
        self, label: str, prop: str, value: str, fuzzy: bool = False
    ) -> List[Dict[str, Any]]:
        if value is None:
            return []
        value = str(value).strip()
        if not value:
            return []

        safe_label = _validate_cypher_name(label)
        safe_prop = _validate_cypher_name(prop)
        if fuzzy:
            query = (
                f"MATCH (n:{safe_label}) "
                f"WHERE toLower(toString(n.{safe_prop})) CONTAINS toLower($value) "
                "RETURN labels(n)[0] AS label, properties(n) AS properties, id(n) AS internal_id "
                "LIMIT 100"
            )
        else:
            query = (
                f"MATCH (n:{safe_label}) "
                f"WHERE toString(n.{safe_prop}) = $value "
                "RETURN labels(n)[0] AS label, properties(n) AS properties, id(n) AS internal_id "
                "LIMIT 100"
            )
        rows = self._run(query, {"value": value})
        return [self._row_to_node(row, snapshot=None) for row in rows]

    def neighbors(
        self, node_id: str, depth: int, rel_types: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        snapshot = self._snapshot_graph()
        resolved = self._resolve_node_id(snapshot, node_id)
        if not resolved:
            return {}

        allowed = _normalize_rel_types(rel_types)
        visited: Dict[str, int] = {resolved: 0}
        queue: deque[Tuple[str, int]] = deque([(resolved, 0)])

        while queue:
            current, current_depth = queue.popleft()
            if current_depth >= depth:
                continue
            for neighbor, _ in self._iter_adjacent(snapshot, current, allowed):
                next_depth = current_depth + 1
                if neighbor not in visited or next_depth < visited[neighbor]:
                    visited[neighbor] = next_depth
                    queue.append((neighbor, next_depth))

        ordered = sorted(visited.items(), key=lambda item: (item[1], item[0]))
        return {
            current_id: self._node_to_dict(snapshot, current_id, distance=distance)
            for current_id, distance in ordered
        }

    def extract_subgraph(
        self, seed_node_ids: List[str], max_nodes: int, rel_filter: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        snapshot = self._snapshot_graph()
        allowed = _normalize_rel_types(rel_filter)
        queue: deque[str] = deque()
        selected: set[str] = set()

        for seed in seed_node_ids:
            resolved = self._resolve_node_id(snapshot, seed)
            if resolved and resolved not in selected:
                selected.add(resolved)
                queue.append(resolved)
            if len(selected) >= max_nodes:
                break

        while queue and len(selected) < max_nodes:
            current = queue.popleft()
            for neighbor, _ in self._iter_adjacent(snapshot, current, allowed):
                if neighbor in selected:
                    continue
                selected.add(neighbor)
                queue.append(neighbor)
                if len(selected) >= max_nodes:
                    break

        nodes = [self._node_to_dict(snapshot, node_id) for node_id in sorted(selected)]
        edges: List[Dict[str, Any]] = []
        for source, target, _, edge_data in snapshot.edges(keys=True, data=True):
            rel_type = str(edge_data.get("type", "RELATED_TO"))
            if allowed and rel_type not in allowed:
                continue
            if source in selected and target in selected:
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "type": rel_type,
                        "properties": edge_data.get("properties", {}),
                    }
                )
        edges.sort(key=lambda item: (item["source"], item["target"], item["type"]))
        return nodes, edges

    def _snapshot_graph(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        node_rows = self._run(
            "MATCH (n) RETURN labels(n)[0] AS label, properties(n) AS properties, id(n) AS internal_id"
        )
        for row in node_rows:
            node = self._row_to_node(row, snapshot=None)
            graph.add_node(
                node["id"],
                label=node["label"],
                properties=node["properties"],
                internal_id=row.get("internal_id"),
            )

        edge_rows = self._run(
            "MATCH (a)-[r]->(b) "
            "RETURN "
            "labels(a)[0] AS source_label, properties(a) AS source_properties, id(a) AS source_internal_id, "
            "labels(b)[0] AS target_label, properties(b) AS target_properties, id(b) AS target_internal_id, "
            "type(r) AS rel_type, properties(r) AS rel_properties"
        )

        for row in edge_rows:
            source_id = _canonical_id(
                str(row.get("source_label", "Unknown")),
                row.get("source_properties", {}) or {},
                fallback_id=str(row.get("source_internal_id", "")),
            )
            target_id = _canonical_id(
                str(row.get("target_label", "Unknown")),
                row.get("target_properties", {}) or {},
                fallback_id=str(row.get("target_internal_id", "")),
            )
            if source_id not in graph:
                graph.add_node(
                    source_id,
                    label=row.get("source_label", "Unknown"),
                    properties=row.get("source_properties", {}) or {},
                    internal_id=row.get("source_internal_id"),
                )
            if target_id not in graph:
                graph.add_node(
                    target_id,
                    label=row.get("target_label", "Unknown"),
                    properties=row.get("target_properties", {}) or {},
                    internal_id=row.get("target_internal_id"),
                )
            graph.add_edge(
                source_id,
                target_id,
                type=str(row.get("rel_type", "RELATED_TO")),
                properties=row.get("rel_properties", {}) or {},
            )
        return graph

    def _resolve_node_id(self, snapshot: nx.MultiDiGraph, node_id: str) -> str:
        if node_id in snapshot:
            return node_id
        for existing_id in snapshot.nodes:
            if existing_id.endswith(f":{node_id}"):
                return existing_id
        return ""

    def _iter_adjacent(
        self, snapshot: nx.MultiDiGraph, node_id: str, rel_filter: Optional[set[str]]
    ) -> Iterable[Tuple[str, str]]:
        for _, neighbor, _, edge_data in snapshot.out_edges(node_id, keys=True, data=True):
            rel_type = str(edge_data.get("type", "RELATED_TO"))
            if rel_filter and rel_type not in rel_filter:
                continue
            yield neighbor, rel_type
        for neighbor, _, _, edge_data in snapshot.in_edges(node_id, keys=True, data=True):
            rel_type = str(edge_data.get("type", "RELATED_TO"))
            if rel_filter and rel_type not in rel_filter:
                continue
            yield neighbor, rel_type

    def _node_to_dict(
        self, snapshot: nx.MultiDiGraph, node_id: str, distance: Optional[int] = None
    ) -> Dict[str, Any]:
        attrs = snapshot.nodes[node_id]
        payload: Dict[str, Any] = {
            "id": node_id,
            "label": attrs.get("label", "Unknown"),
            "properties": attrs.get("properties", {}),
            "degree": int(snapshot.degree(node_id)),
        }
        if distance is not None:
            payload["distance"] = distance
        return payload

    def _row_to_node(
        self, row: Dict[str, Any], snapshot: Optional[nx.MultiDiGraph]
    ) -> Dict[str, Any]:
        label = str(row.get("label", "Unknown"))
        properties = row.get("properties", {}) or {}
        internal_id = str(row.get("internal_id", ""))
        node_id = _canonical_id(label, properties, fallback_id=internal_id)
        degree = int(snapshot.degree(node_id)) if snapshot is not None and node_id in snapshot else 0
        return {"id": node_id, "label": label, "properties": properties, "degree": degree}

    def _run(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
