"""Context builder: structures query results into human/LLM-readable Markdown.

Supports both the new dynamic ``QueryResult`` format (from the Cypher builder
pipeline) and generic dict payloads.  Existing formatting templates are
preserved for workflow, tool, step, analytics, comparison, path, and category
result shapes.
"""

from __future__ import annotations

import logging
from typing import Any

from app.bioblend_server.GraphRAG.config import GraphRAGConfig
from app.bioblend_server.GraphRAG.models import QueryResult, ResolvedEntity

SUMMARY_FIELDS = ["readme_content", "description", "help", "annotation", "name"]


class ContextBuilder:
    """Formats query results into clean, hierarchical Markdown contexts."""

    def __init__(self, config: GraphRAGConfig) -> None:
        self.config = config
        self.log = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Primary entry point for the new pipeline
    # ------------------------------------------------------------------

    def build_context_from_results(
        self,
        query_results: list[QueryResult],
        seeds: list[ResolvedEntity],
    ) -> str:
        """Render a list of ``QueryResult`` objects into Markdown evidence."""
        self.log.info(f"Building context from {len(query_results)} query results")

        if not query_results:
            return "No matching context found."

        sections: list[str] = []

        for result in query_results:
            try:
                if not result.rows:
                    continue

                section = self._dispatch_result(result)
                if section and section not in sections:
                    sections.append(section)
            except Exception as e:
                self.log.error(f"Error processing result: {e}", exc_info=True)

        if not sections:
            return "No matching context found."

        context = "\n\n---\n\n".join(sections)
        self.log.info(f"Built {len(sections)} context sections")
        return context

    # ------------------------------------------------------------------
    # Dispatch: detect result shape and route to formatter
    # ------------------------------------------------------------------

    def _dispatch_result(self, result: QueryResult) -> str:
        """Detect the result shape and route to the appropriate formatter."""
        rows = result.rows
        if not rows:
            return ""

        first = rows[0]

        # Path results
        if "path_nodes" in first:
            return self._format_path_result(result)

        # Compare results
        if "shared" in first and "entity_a" in first:
            return self._format_compare_result(result)

        # Aggregate results
        if "agg_result" in first and "props" in first:
            if "group_key" in first:
                return self._format_grouped_aggregate(result)
            return self._format_ranked_aggregate(result)

        # Entity context with expansions (flat _list or nested _nested)
        props_keys = [k for k in first if k.endswith("_props")]
        list_keys = [k for k in first if k.endswith("_list") or k.endswith("_nested")]
        if props_keys:
            return self._format_entity_context(result, props_keys, list_keys)

        # Generic fallback
        return self._format_generic(result)

    # ------------------------------------------------------------------
    # Entity context (anchor + expansions)
    # ------------------------------------------------------------------

    def _format_entity_context(
        self,
        result: QueryResult,
        props_keys: list[str],
        list_keys: list[str],
    ) -> str:
        try:
            lines: list[str] = []
            seen_names: set[str] = set()

            for row in result.rows:
                for pk in props_keys:
                    entity = row.get(pk, {})
                    if not entity:
                        continue

                    name = self._get_entity_name(entity)
                    if name in seen_names:
                        continue
                    seen_names.add(name)

                    # Determine label from alias prefix
                    label = self._guess_label(entity)
                    lines.append(f"[{label.upper()}] {name}")

                    if summary := self._get_summary(entity):
                        lines.append(f"  Summary: {summary}")

                    if ver := entity.get("version"):
                        lines.append(f"  Version: {ver}")

                # Render expansion lists (flat _list or nested _nested)
                for lk in list_keys:
                    items = row.get(lk, [])
                    if not items:
                        continue
                    list_label = (
                        lk.replace("_nested", "").replace("_list", "")
                        .replace("_", " ").title()
                    )
                    lines.append(f"\n  - [{list_label}]:")
                    for item in items:
                        if not item:
                            continue
                        # Nested items have {props: {...}, child_list: [...]}
                        if "props" in item and isinstance(item["props"], dict):
                            parent_props = item["props"]
                            item_name = self._get_entity_name(parent_props)
                            detail = self._get_summary(parent_props)
                            line = f"    - {item_name}"
                            if detail:
                                line += f": {detail}"
                            lines.append(line)
                            # Render nested children recursively
                            self._render_nested_children(item, lines, depth=3)
                        else:
                            # Flat item (simple _list)
                            item_name = self._get_entity_name(item)
                            detail = self._get_summary(item)
                            line = f"    - {item_name}"
                            if detail:
                                line += f": {detail}"
                            lines.append(line)

            return "\n".join(lines) if lines else ""
        except Exception as e:
            self.log.error(f"Error formatting entity context: {e}", exc_info=True)
            return ""

    # ------------------------------------------------------------------
    # Ranked aggregate (e.g. most used tools)
    # ------------------------------------------------------------------

    def _format_ranked_aggregate(self, result: QueryResult) -> str:
        try:
            lines = [f"[ANALYTICS] {result.schema_description}\n"]
            for i, row in enumerate(result.rows, 1):
                props = row.get("props", {})
                agg = row.get("agg_result", 0)
                name = self._get_entity_name(props)
                ver = props.get("version", "")
                ver_str = f" (v{ver})" if ver else ""

                if isinstance(agg, list):
                    # collect() result — render as a list of items
                    lines.append(f"{i}. {name}{ver_str}: {len(agg)} items")
                    for item in agg[:10]:
                        if isinstance(item, dict):
                            lines.append(f"   - {self._get_entity_name(item)}")
                        elif item is not None:
                            lines.append(f"   - {self._format_value(item)}")
                else:
                    # count() result — render as numeric
                    lines.append(f"{i}. {name}{ver_str}: {agg} occurrences")

                if text := self._get_summary(props):
                    lines.append(f"   - {text}")
            return "\n".join(lines)
        except Exception as e:
            self.log.error(f"Error formatting ranked aggregate: {e}", exc_info=True)
            return ""

    # ------------------------------------------------------------------
    # Grouped aggregate (e.g. tools by community)
    # ------------------------------------------------------------------

    def _format_grouped_aggregate(self, result: QueryResult) -> str:
        try:
            lines = [f"[ANALYTICS] {result.schema_description}\n"]
            groups: dict[str, list[dict]] = {}
            for row in result.rows:
                gk = str(row.get("group_key", "Unknown"))
                groups.setdefault(gk, []).append(row)

            for group_key, members in groups.items():
                lines.append(f"Group [{group_key}]: {len(members)} items")
                for m in members:
                    props = m.get("props", {})
                    agg = m.get("agg_result", 0)
                    name = self._get_entity_name(props)
                    if isinstance(agg, list):
                        lines.append(f"  - {name} ({len(agg)} items)")
                    else:
                        lines.append(f"  - {name} ({agg})")
                lines.append("")

            return "\n".join(lines)
        except Exception as e:
            self.log.error(f"Error formatting grouped aggregate: {e}", exc_info=True)
            return ""

    # ------------------------------------------------------------------
    # Compare results
    # ------------------------------------------------------------------

    def _format_compare_result(self, result: QueryResult) -> str:
        try:
            row = result.rows[0]
            entity_a = row.get("entity_a", {})
            entity_b = row.get("entity_b", {})
            shared = row.get("shared", [])
            unique_a = row.get("unique_to_a", [])
            unique_b = row.get("unique_to_b", [])

            name_a = self._get_entity_name(entity_a)
            name_b = self._get_entity_name(entity_b)

            lines = [f"[COMPARISON] {name_a} vs {name_b}\n"]

            if shared:
                lines.append(f"Shared ({len(shared)}):")
                for item in shared:
                    lines.append(f"  - {self._get_entity_name(item)}")
            else:
                lines.append("No shared elements found.")

            if unique_a:
                lines.append(f"\nUnique to {name_a} ({len(unique_a)}):")
                for item in unique_a:
                    lines.append(f"  - {self._get_entity_name(item)}")

            if unique_b:
                lines.append(f"\nUnique to {name_b} ({len(unique_b)}):")
                for item in unique_b:
                    lines.append(f"  - {self._get_entity_name(item)}")

            return "\n".join(lines)
        except Exception as e:
            self.log.error(f"Error formatting compare result: {e}", exc_info=True)
            return ""

    # ------------------------------------------------------------------
    # Path results
    # ------------------------------------------------------------------

    def _format_path_result(self, result: QueryResult) -> str:
        try:
            lines = [f"[PATH] {result.schema_description}\n"]

            for row in result.rows:
                path_nodes = row.get("path_nodes", [])
                path_rels = row.get("path_rels", [])

                if not path_nodes:
                    lines.append("No path found.")
                    continue

                lines.append(f"Path ({len(path_nodes)} nodes, {len(path_rels)} relationships):")
                for j, node in enumerate(path_nodes):
                    props = node.get("props", {})
                    label = node.get("label", "?")
                    name = self._get_entity_name(props)
                    prefix = "  " if j == 0 else "    → "
                    lines.append(f"{prefix}[{label}] {name}")

            return "\n".join(lines)
        except Exception as e:
            self.log.error(f"Error formatting path result: {e}", exc_info=True)
            return ""

    # ------------------------------------------------------------------
    # Generic fallback
    # ------------------------------------------------------------------

    def _format_generic(self, result: QueryResult) -> str:
        try:
            lines = [f"[RESULT] {result.schema_description}\n"]
            for i, row in enumerate(result.rows[:20], 1):
                parts = []
                for key, value in row.items():
                    if isinstance(value, dict):
                        name = self._get_entity_name(value)
                        if name != "Unknown":
                            parts.append(f"{key}: {name}")
                    elif isinstance(value, list):
                        parts.append(f"{key}: [{len(value)} items]")
                    elif value is not None:
                        parts.append(f"{key}: {self._format_value(value)}")
                if parts:
                    lines.append(f"  {i}. {' | '.join(parts)}")
            if result.truncated:
                lines.append(f"  ... (truncated at {result.node_count} rows)")
            return "\n".join(lines)
        except Exception as e:
            self.log.error(f"Error formatting generic result: {e}", exc_info=True)
            return ""

    # ------------------------------------------------------------------
    # Recursive nested renderer
    # ------------------------------------------------------------------

    def _render_nested_children(
        self,
        item: dict[str, Any],
        lines: list[str],
        depth: int,
        max_depth: int = 6,
    ) -> None:
        """Recursively render nested children from WITH-aggregated maps.

        Handles arbitrary nesting depth: each child value can be:
        - list of dicts with "props" key → nested entity with its own children
        - list of plain dicts → leaf entities (render name directly)
        - single dict → render name
        """
        if depth > max_depth:
            return
        indent = "  " * depth
        for ck, cv in item.items():
            if ck == "props":
                continue
            if isinstance(cv, list):
                for child in cv:
                    if not isinstance(child, dict):
                        continue
                    if "props" in child and isinstance(child["props"], dict):
                        # Nested entity with its own children — recurse
                        child_name = self._get_entity_name(child["props"])
                        if child_name != "Unknown":
                            lines.append(f"{indent}- {child_name}")
                            self._render_nested_children(
                                child, lines, depth + 1, max_depth
                            )
                    else:
                        # Leaf entity — render name
                        child_name = self._get_entity_name(child)
                        if child_name != "Unknown":
                            lines.append(f"{indent}- {child_name}")
            elif isinstance(cv, dict):
                child_name = self._get_entity_name(cv)
                if child_name != "Unknown":
                    lines.append(f"{indent}- {child_name}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _get_entity_name(props: dict[str, Any]) -> str:
        if not props:
            return "Unknown"
        for key in (
            "name", "input_name", "output_name", "file_name",
            "workflow_id", "tool_id", "step_uid",
        ):
            val = props.get(key)
            if val and str(val).strip():
                return str(val).strip()
        return "Unknown"

    @staticmethod
    def _guess_label(props: dict[str, Any]) -> str:
        if "workflow_id" in props or "file_name" in props:
            return "Workflow"
        if "tool_id" in props:
            return "Tool"
        if "step_uid" in props or "step_id" in props:
            return "Step"
        if "category_id" in props:
            return "Category"
        if "input_name" in props:
            return "ToolInput"
        if "output_name" in props:
            return "ToolOutput"
        return "Entity"

    def _get_summary(self, properties: dict[str, Any]) -> str:
        try:
            for key in SUMMARY_FIELDS:
                value = properties.get(key)
                if value is None:
                    continue
                text = " ".join(str(value).split())
                if text:
                    return self._format_value(text)
            return ""
        except Exception as e:
            self.log.error(f"Error getting summary: {e}", exc_info=True)
            return ""

    def _format_value(self, value: Any) -> str:
        try:
            max_len = self.config.budget.format_max_chars
            text = " ".join(str(value).split())
            if len(text) <= max_len:
                return text
            return text[: max_len - 3] + "..."
        except Exception:
            return ""
