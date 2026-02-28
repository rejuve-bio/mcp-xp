"""Schema-aware context builder for the GraphRAG pipeline.

Assembles a structured, token-efficient context string from ranked nodes
and graph-expanded entities, respecting the Galaxy knowledge-graph schema.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from app.bioblend_server.GraphRAG.config import GraphRAGConfig, SCHEMA_RELATIONSHIPS


# Fields whose values are used to produce a per-node summary line
SUMMARY_FIELDS = ["readme_content", "description", "help", "annotation", "name"]

# Order in which entity-type sections are emitted
_SECTION_ORDER = [
    "Category", "Workflow", "Step", "Tool",
    "Input", "Output", "ToolInput", "ToolOutput",
]


class ContextBuilder:
    """Char-budgeted, schema-aware context serialiser."""

    def __init__(self, config: GraphRAGConfig) -> None:
        self.config = config


    def build_context(
        self,
        ranked_nodes: List[Dict[str, Any]],
        subgraph: Dict[str, Any],
        max_chars: int,
        expanded_nodes: Dict[str, Dict[str, Any]] | None = None,
        expanded_edges: List[Dict[str, Any]] | None = None,
    ) -> str:
        """Build a structured context payload.

        The output groups entities by type (Workflows → Steps → Tools →
        Inputs/Outputs → Categories), orders steps by ``NEXT_STEP``
        edges, includes tool metadata, and deduplicates entities.

        Args:
            ranked_nodes: Ranked candidate list from the hybrid ranker.
            subgraph: ``{"nodes": [...], "edges": [...]}`` from the
                connector's ``extract_subgraph``.
            max_chars: Hard character budget for the final string.
            expanded_nodes: Nodes from schema-aware expansion (optional).
            expanded_edges: Edges from schema-aware expansion (optional).

        Returns:
            A deterministic, token-efficient context string.
        """
        # Merge all available nodes/edges (dedup by id)
        all_nodes, all_edges = self._merge_graph_data(
            subgraph, expanded_nodes, expanded_edges
        )
        node_lookup: Dict[str, Dict[str, Any]] = {n["id"]: n for n in all_nodes}

        # Build ranked-node id set for priority inclusion
        ranked_ids: Set[str] = {
            entry["node"]["id"] for entry in ranked_nodes
        }

        # Group nodes by label
        grouped = self._group_by_label(all_nodes)

        # Order steps using NEXT_STEP edges
        if "Step" in grouped:
            grouped["Step"] = self._order_steps(grouped["Step"], all_edges)

        # Assemble sections
        sections: List[str] = []

        # 1. Ranked-node highlights
        ranked_section = self._ranked_section(ranked_nodes, node_lookup, all_edges)
        if ranked_section:
            sections.append(ranked_section)

        # 2. Entity-type sections
        for label in _SECTION_ORDER:
            nodes = grouped.get(label, [])
            if not nodes:
                continue
            section = self._entity_section(label, nodes, all_edges, node_lookup)
            if section:
                sections.append(section)

        # 3. Relationship summary
        rel_section = self._relationship_section(all_edges, node_lookup)
        if rel_section:
            sections.append(rel_section)

        # Combine & truncate
        context = "\n\n".join(sections)
        if len(context) > max_chars:
            context = context[: max(0, max_chars - 3)].rstrip() + "..."
        return context

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _ranked_section(
        self,
        ranked_nodes: List[Dict[str, Any]],
        node_lookup: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> str:
        """Top-ranked nodes with scores."""
        if not ranked_nodes:
            return ""
        lines = ["[Top Ranked Matches]"]
        for entry in ranked_nodes[:8]:
            node = entry["node"]
            label = node.get("label", "Unknown")
            display_id = self._display_id(node["id"])
            summary = self._summary(node.get("properties", {}))
            score = entry.get("score", 0.0)
            reason = entry.get("reason", "graph_candidate")
            lines.append(
                f"- {label}|{display_id}: {summary} "
                f"(score={score:.4f}, reason={reason})"
            )
        return "\n".join(lines)

    def _entity_section(
        self,
        label: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        node_lookup: Dict[str, Dict[str, Any]],
    ) -> str:
        """Emit a section for a single entity type."""
        lines = [f"[{label}s]"]
        for node in nodes[:12]:  # cap per section
            block = self._node_block(node, edges, node_lookup)
            lines.append(block)
        return "\n".join(lines)

    def _node_block(
        self,
        node: Dict[str, Any],
        edges: List[Dict[str, Any]],
        node_lookup: Dict[str, Dict[str, Any]],
    ) -> str:
        """Compact representation of a single node."""
        label = node.get("label", "Unknown")
        display_id = self._display_id(node["id"])
        props = node.get("properties", {})
        summary = self._summary(props)

        parts = [f"  {label}|{display_id}: {summary}"]

        # Key properties (exclude summary fields to avoid repetition)
        prop_lines: List[str] = []
        for key in sorted(props.keys()):
            if key in SUMMARY_FIELDS:
                continue
            prop_lines.append(f"    {key}: {self._format_value(props[key], 80)}")
            if len(prop_lines) >= 6:
                break
        if prop_lines:
            parts.extend(prop_lines)

        # Outgoing relationships (compact)
        outgoing = [
            e for e in edges
            if e.get("source") == node["id"] and e.get("target") in node_lookup
        ]
        for edge in outgoing[:3]:
            target = node_lookup[edge["target"]]
            parts.append(
                f"    → {edge['type']} → {target.get('label','?')}|"
                f"{self._display_id(edge['target'])}"
            )

        return "\n".join(parts)

    def _relationship_section(
        self,
        edges: List[Dict[str, Any]],
        node_lookup: Dict[str, Dict[str, Any]],
    ) -> str:
        """Compact relationship listing."""
        if not edges:
            return ""
        lines = ["[Relationships]"]
        seen: Set[Tuple[str, str, str]] = set()
        for edge in edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            rel = edge.get("type", "RELATED_TO")
            if src not in node_lookup or tgt not in node_lookup:
                continue
            key = (src, rel, tgt)
            if key in seen:
                continue
            seen.add(key)
            src_label = node_lookup[src].get("label", "?")
            tgt_label = node_lookup[tgt].get("label", "?")
            lines.append(
                f"- {src_label}|{self._display_id(src)} "
                f"-[{rel}]→ "
                f"{tgt_label}|{self._display_id(tgt)}"
            )
            if len(lines) > 20:
                lines.append("- ... (truncated)")
                break
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_graph_data(
        subgraph: Dict[str, Any],
        expanded_nodes: Dict[str, Dict[str, Any]] | None,
        expanded_edges: List[Dict[str, Any]] | None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Merge subgraph + expansion data, deduplicating by node id."""
        node_map: Dict[str, Dict[str, Any]] = {}
        for n in subgraph.get("nodes", []):
            node_map[n["id"]] = n
        if expanded_nodes:
            for nid, ndata in expanded_nodes.items():
                if nid not in node_map:
                    node_map[nid] = ndata

        edge_set: Set[Tuple[str, str, str]] = set()
        edges: List[Dict[str, Any]] = []
        for e in subgraph.get("edges", []):
            key = (e["source"], e["target"], e.get("type", ""))
            if key not in edge_set:
                edge_set.add(key)
                edges.append(e)
        if expanded_edges:
            for e in expanded_edges:
                key = (e["source"], e["target"], e.get("type", ""))
                if key not in edge_set:
                    edge_set.add(key)
                    edges.append(e)

        return list(node_map.values()), edges

    @staticmethod
    def _group_by_label(
        nodes: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for node in nodes:
            groups[node.get("label", "Unknown")].append(node)
        return dict(groups)

    @staticmethod
    def _order_steps(
        steps: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Topologically order step nodes using NEXT_STEP edges."""
        step_ids = {s["id"] for s in steps}
        id_to_step = {s["id"]: s for s in steps}

        # Build adjacency from NEXT_STEP edges
        successors: Dict[str, str] = {}
        has_predecessor: Set[str] = set()
        for edge in edges:
            if edge.get("type") != "NEXT_STEP":
                continue
            src, tgt = edge["source"], edge["target"]
            if src in step_ids and tgt in step_ids:
                successors[src] = tgt
                has_predecessor.add(tgt)

        # Find roots (steps without predecessor in chain)
        roots = [sid for sid in step_ids if sid not in has_predecessor]
        roots.sort()

        ordered: List[Dict[str, Any]] = []
        visited: Set[str] = set()

        for root in roots:
            current = root
            while current and current not in visited:
                visited.add(current)
                if current in id_to_step:
                    ordered.append(id_to_step[current])
                current = successors.get(current)

        # Append any steps not reachable from roots
        for step in steps:
            if step["id"] not in visited:
                ordered.append(step)

        return ordered

    def _summary(self, properties: Dict[str, Any]) -> str:
        for key in SUMMARY_FIELDS:
            value = properties.get(key)
            if value is None:
                continue
            value_string = " ".join(str(value).split())
            if value_string:
                return self._format_value(value_string, 140)
        return "No summary available."

    @staticmethod
    def _display_id(node_id: str) -> str:
        return node_id.split(":", 1)[1] if ":" in node_id else node_id

    @staticmethod
    def _format_value(value: Any, max_len: int) -> str:
        text = " ".join(str(value).split())
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."