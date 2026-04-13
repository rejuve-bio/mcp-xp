"""Deterministic Cypher builder: CypherQuerySchema → parameterized Cypher.

Converts validated schemas produced by the LLM planner into safe, parameterized
Cypher queries.  No LLM calls.  All node labels and relationship types come
from pre-validated schema objects — values are always passed as ``$parameters``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.bioblend_server.GraphRAG.models import (
    AggregateSpec,
    CypherQuerySchema,
    Expansion,
    NodeMatch,
    PropertyFilter,
)
from app.bioblend_server.GraphRAG.schema import EDGE_DIRECTIONS

SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

log = logging.getLogger(__name__)


def _safe(name: str) -> str:
    """Validate and return a Cypher-safe identifier."""
    if not SAFE_NAME.match(name):
        raise ValueError(f"Unsafe Cypher identifier: {name!r}")
    return name


# ---------------------------------------------------------------------------
# Filter → WHERE clause fragment
# ---------------------------------------------------------------------------

_OP_MAP = {
    "eq": "=",
    "neq": "<>",
    "contains": "CONTAINS",
    "starts_with": "STARTS WITH",
    "gt": ">",
    "lt": "<",
}


def _build_filter_clause(
    alias: str,
    filt: PropertyFilter,
    param_key: str,
    params: dict[str, Any],
) -> str:
    safe_prop = _safe(filt.property)

    if filt.operator == "in":
        params[param_key] = filt.value if isinstance(filt.value, list) else [filt.value]
        return f"{alias}.{safe_prop} IN ${param_key}"

    if filt.operator == "contains":
        params[param_key] = str(filt.value).lower()
        return f"toLower(toString({alias}.{safe_prop})) CONTAINS ${param_key}"

    if filt.operator == "starts_with":
        params[param_key] = str(filt.value).lower()
        return f"toLower(toString({alias}.{safe_prop})) STARTS WITH ${param_key}"

    cypher_op = _OP_MAP.get(filt.operator, "=")
    params[param_key] = filt.value
    return f"{alias}.{safe_prop} {cypher_op} ${param_key}"


# ---------------------------------------------------------------------------
# NodeMatch → MATCH clause + WHERE conditions
# ---------------------------------------------------------------------------


def _build_node_match(
    node: NodeMatch,
    params: dict[str, Any],
    param_prefix: str,
) -> tuple[str, list[str]]:
    """Return (match_pattern, where_conditions)."""
    alias = _safe(node.alias)
    label = _safe(node.label)
    pattern = f"({alias}:{label})"

    conditions: list[str] = []
    if node.element_id:
        pk = f"{param_prefix}_eid"
        params[pk] = node.element_id
        conditions.append(f"elementId({alias}) = ${pk}")

    for i, filt in enumerate(node.filters):
        pk = f"{param_prefix}_f{i}"
        conditions.append(_build_filter_clause(alias, filt, pk, params))

    return pattern, conditions


# ---------------------------------------------------------------------------
# Expansion → OPTIONAL MATCH / MATCH clause
# ---------------------------------------------------------------------------


def _build_expansion(
    exp: Expansion,
    params: dict[str, Any],
    param_prefix: str,
) -> str:
    from_a = _safe(exp.from_alias)
    rel = _safe(exp.relationship)
    target_a = _safe(exp.target_alias)
    target_l = _safe(exp.target_label)

    # Determine arrow direction
    src, _ = EDGE_DIRECTIONS.get(exp.relationship, ("", ""))
    if exp.direction == "out":
        pattern = f"({from_a})-[:{rel}]->({target_a}:{target_l})"
    elif exp.direction == "in":
        pattern = f"({from_a})<-[:{rel}]-({target_a}:{target_l})"
    else:
        pattern = f"({from_a})-[:{rel}]-({target_a}:{target_l})"

    keyword = "OPTIONAL MATCH" if exp.optional else "MATCH"
    return f"{keyword} {pattern}"


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


class CypherBuilder:
    """Converts a validated ``CypherQuerySchema`` into a ``(cypher, params)`` tuple."""

    def build(self, schema: CypherQuerySchema) -> tuple[str, dict[str, Any]]:
        """Dispatch to the appropriate builder method based on schema content."""
        if schema.compare:
            return self._build_compare(schema)
        if schema.path:
            return self._build_path(schema)
        if schema.anchor:
            return self._build_anchor(schema)
        raise ValueError("Schema has no anchor, path, or compare — nothing to build")

    # ------------------------------------------------------------------
    # anchor + expansions [+ aggregate]
    # ------------------------------------------------------------------

    def _build_anchor(self, schema: CypherQuerySchema) -> tuple[str, dict[str, Any]]:
        params: dict[str, Any] = {}
        lines: list[str] = []

        # MATCH anchor
        pattern, conditions = _build_node_match(schema.anchor, params, "anchor")
        lines.append(f"MATCH {pattern}")
        if conditions:
            lines.append(f"WHERE {' AND '.join(conditions)}")

        # Expansions
        for i, exp in enumerate(schema.expansions):
            lines.append(_build_expansion(exp, params, f"exp{i}"))

        # Aggregate
        if schema.aggregate:
            return self._append_aggregate(
                lines, params, schema.anchor, schema.expansions, schema.aggregate, schema.limit
            )

        # Default RETURN: anchor props + collected expansion targets.
        # For chained expansions (A->B->C), nest children under parents
        # so the result preserves which C belongs to which B.
        anchor_alias = _safe(schema.anchor.alias)
        return_parts = [f"properties({anchor_alias}) AS {anchor_alias}_props"]

        if self._has_chained_expansions(schema):
            # Chained: insert WITH to aggregate children per parent first,
            # then collect parent+children maps in RETURN.
            # E.g. Workflow(w)->Step(s)->Tool(t) becomes:
            #   WITH w, s, collect(DISTINCT properties(t)) AS t_list
            #   RETURN properties(w) AS w_props,
            #          collect(DISTINCT {props: properties(s), t_list: t_list}) AS s_nested
            with_clause, nested_return = self._build_nested_with_return(
                schema.anchor.alias, schema.expansions
            )
            lines.append(with_clause)
            return_parts.extend(nested_return)
        else:
            # Flat: single-hop expansions, collect each independently.
            seen_aliases: set[str] = set()
            for exp in schema.expansions:
                ta = _safe(exp.target_alias)
                if ta not in seen_aliases:
                    seen_aliases.add(ta)
                    return_parts.append(
                        f"collect(DISTINCT properties({ta})) AS {ta}_list"
                    )

        lines.append(f"RETURN {', '.join(return_parts)}")
        lines.append(f"LIMIT {schema.limit}")

        cypher = "\n".join(lines)
        log.debug(f"Built anchor query: {cypher}")
        return cypher, params

    @staticmethod
    def _has_chained_expansions(schema: CypherQuerySchema) -> bool:
        """Return True if any expansion's from_alias is another expansion's target."""
        if len(schema.expansions) < 2:
            return False
        target_aliases = set()
        for exp in schema.expansions:
            if exp.from_alias in target_aliases:
                return True
            target_aliases.add(exp.target_alias)
        return False

    @staticmethod
    def _build_nested_with_return(
        anchor_alias: str, expansions: list[Expansion]
    ) -> tuple[str, list[str]]:
        """Build WITH clauses + RETURN fragments that group children per parent.

        Works bottom-up: aggregates the deepest children first, then works up.
        Each WITH only references variables that are currently in scope.

        For w->s->t (2-hop):
          WITH w, s, collect(DISTINCT properties(t)) AS t_list
        RETURN: collect(DISTINCT {props: properties(s), t_list: t_list}) AS s_nested

        For w->s->t->ti (3-hop):
          WITH w, s, t, collect(DISTINCT properties(ti)) AS ti_list
          WITH w, s, collect(DISTINCT {props: properties(t), ti_list: ti_list}) AS t_nested
        RETURN: collect(DISTINCT {props: properties(s), t_nested: t_nested}) AS s_nested
        """
        safe_anchor = _safe(anchor_alias)

        # Build parent→children map
        children_of: dict[str, list[str]] = {}
        all_targets: set[str] = set()
        for exp in expansions:
            children_of.setdefault(exp.from_alias, []).append(exp.target_alias)
            all_targets.add(exp.target_alias)

        # Find leaves (targets with no children of their own)
        leaves = {a for a in all_targets if a not in children_of}

        # Process bottom-up: each round aggregates current leaf-level nodes
        # into collect variables, then removes them from the working set.
        # After processing, their parent references the collect var name.
        with_clauses: list[str] = []
        collect_vars: dict[str, str] = {}  # alias → its collected variable name
        remaining = set(all_targets)

        while remaining:
            # Current round's leaves: nodes whose children (if any) are all processed
            round_leaves = [
                a for a in remaining
                if all(c not in remaining for c in children_of.get(a, []))
            ]
            if not round_leaves:
                break

            for leaf in round_leaves:
                remaining.discard(leaf)

            # True leaves (no children): mark for raw-properties collection
            # Intermediate nodes (have already-processed children): emit a WITH
            for node in round_leaves:
                node_children = children_of.get(node, [])
                if not node_children:
                    # True leaf — will be collected as properties in parent's WITH
                    collect_vars[node] = f"{_safe(node)}_list"
                else:
                    # Intermediate node — emit WITH to aggregate its children
                    sn = _safe(node)

                    # Carry: anchor + remaining aliases + sibling top-level
                    # targets that are still raw (not yet consumed by a WITH).
                    # This ensures flat siblings like 'wt' stay in scope for RETURN.
                    top_targets = {
                        exp.target_alias
                        for exp in expansions
                        if exp.from_alias == anchor_alias
                    }
                    carry = [safe_anchor]
                    # Add remaining (unprocessed) aliases
                    for r in sorted(remaining):
                        carry.append(_safe(r))
                    # Add sibling top-level targets that were processed as leaves
                    # (they're out of `remaining` but still needed for RETURN)
                    for tt in sorted(top_targets):
                        safe_tt = _safe(tt)
                        if tt not in remaining and tt != node and safe_tt not in carry:
                            carry.append(safe_tt)
                    carry.append(sn)

                    # Collect each child (leaf children → properties, nested children → var)
                    for child in node_children:
                        sc = _safe(child)
                        if child in leaves:
                            carry.append(
                                f"collect(DISTINCT properties({sc})) AS {collect_vars[child]}"
                            )
                        else:
                            # Already nested by a previous WITH — carry the var
                            carry.append(collect_vars[child])

                    with_clauses.append(f"WITH {', '.join(carry)}")
                    collect_vars[node] = f"{_safe(node)}_nested"

        # RETURN fragments: for each top-level expansion (child of anchor),
        # build a map with {props + child collect vars}
        top_level = [exp for exp in expansions if exp.from_alias == anchor_alias]
        return_parts: list[str] = []
        for exp in top_level:
            ta = _safe(exp.target_alias)
            map_fields = [f"props: properties({ta})"]
            for child in children_of.get(exp.target_alias, []):
                cvar = collect_vars.get(child, f"{_safe(child)}_list")
                map_fields.append(f"{cvar}: {cvar}")
            map_expr = ", ".join(map_fields)
            return_parts.append(
                f"collect(DISTINCT {{{map_expr}}}) AS {ta}_nested"
            )

        combined_with = "\n".join(with_clauses)
        return combined_with, return_parts

    # ------------------------------------------------------------------
    # aggregate extension
    # ------------------------------------------------------------------

    def _append_aggregate(
        self,
        lines: list[str],
        params: dict[str, Any],
        anchor: NodeMatch,
        expansions: list[Expansion],
        agg: AggregateSpec,
        limit: int,
    ) -> tuple[str, dict[str, Any]]:
        anchor_alias = _safe(anchor.alias)
        input_alias = _safe(agg.input_alias)
        func = agg.function  # "count" or "collect"

        # Determine whether the anchor should be in the grouping.
        # If input_alias IS the anchor (e.g. "count all workflows"),
        # grouping by anchor produces count=1 per row — wrong.
        # Only group by anchor when aggregating over a different alias
        # (e.g. count steps per tool).
        aggregating_anchor = input_alias == anchor_alias

        with_parts: list[str] = []
        if not aggregating_anchor:
            with_parts.append(anchor_alias)

        if agg.group_by:
            group_parts = agg.group_by.split(".", 1)
            if len(group_parts) != 2:
                raise ValueError(
                    f"group_by must be 'alias.property', got: {agg.group_by!r}"
                )
            safe_group = f"{_safe(group_parts[0])}.{_safe(group_parts[1])}"
            with_parts.append(f"{safe_group} AS group_key")

        if func == "collect":
            with_parts.append(f"collect(DISTINCT properties({input_alias})) AS agg_result")
        else:
            with_parts.append(f"{func}({input_alias}) AS agg_result")
        lines.append(f"WITH {', '.join(with_parts)}")

        order_dir = "DESC" if agg.order_by == "desc" else "ASC"
        lines.append(f"ORDER BY agg_result {order_dir}")
        lines.append(f"LIMIT {limit}")

        return_parts: list[str] = []
        if not aggregating_anchor:
            return_parts.append(f"properties({anchor_alias}) AS props")
        return_parts.append("agg_result")
        if agg.group_by:
            return_parts.append("group_key")

        lines.append(f"RETURN {', '.join(return_parts)}")

        cypher = "\n".join(lines)
        log.debug(f"Built aggregate query: {cypher}")
        return cypher, params

    # ------------------------------------------------------------------
    # path (shortestPath)
    # ------------------------------------------------------------------

    def _build_path(self, schema: CypherQuerySchema) -> tuple[str, dict[str, Any]]:
        path = schema.path
        params: dict[str, Any] = {}
        lines: list[str] = []

        # Match from_node
        from_pattern, from_conds = _build_node_match(
            path.from_node, params, "path_from"
        )
        lines.append(f"MATCH {from_pattern}")
        if from_conds:
            lines.append(f"WHERE {' AND '.join(from_conds)}")

        # Match to_node
        to_pattern, to_conds = _build_node_match(path.to_node, params, "path_to")
        lines.append(f"MATCH {to_pattern}")
        if to_conds:
            lines.append(f"WHERE {' AND '.join(to_conds)}")

        from_alias = _safe(path.from_node.alias)
        to_alias = _safe(path.to_node.alias)
        max_hops = path.max_hops

        # Build the path pattern — embed relationship types directly so
        # shortestPath is constrained during traversal, not filtered after.
        if path.relationship_types:
            safe_rels = [_safe(r) for r in path.relationship_types]
            rel_pattern = "|".join(safe_rels)
            path_expr = f"(({from_alias})-[:{rel_pattern}*..{max_hops}]-({to_alias}))"
        else:
            path_expr = f"(({from_alias})-[*..{max_hops}]-({to_alias}))"

        lines.append(f"MATCH p = shortestPath{path_expr}")

        lines.append(
            "RETURN "
            "[n IN nodes(p) | {element_id: elementId(n), label: labels(n)[0], "
            "props: properties(n)}] AS path_nodes, "
            "[r IN relationships(p) | {type: type(r), "
            "start_id: elementId(startNode(r)), "
            "end_id: elementId(endNode(r))}] AS path_rels"
        )
        lines.append(f"LIMIT {schema.limit}")

        cypher = "\n".join(lines)
        log.debug(f"Built path query: {cypher}")
        return cypher, params

    # ------------------------------------------------------------------
    # compare (set intersection / difference)
    # ------------------------------------------------------------------

    def _build_compare(self, schema: CypherQuerySchema) -> tuple[str, dict[str, Any]]:
        cmp = schema.compare
        params: dict[str, Any] = {}
        lines: list[str] = []

        # Match entity A
        a_pattern, a_conds = _build_node_match(cmp.entity_a, params, "cmp_a")
        lines.append(f"MATCH {a_pattern}")
        if a_conds:
            lines.append(f"WHERE {' AND '.join(a_conds)}")

        # Match entity B
        b_pattern, b_conds = _build_node_match(cmp.entity_b, params, "cmp_b")
        lines.append(f"MATCH {b_pattern}")
        if b_conds:
            lines.append(f"WHERE {' AND '.join(b_conds)}")

        a_alias = _safe(cmp.entity_a.alias)
        b_alias = _safe(cmp.entity_b.alias)
        via_rel = _safe(cmp.via_relationship)
        via_label = _safe(cmp.via_target_label)
        hops = cmp.hops

        # Determine traversal direction from schema
        src, _ = EDGE_DIRECTIONS.get(cmp.via_relationship, ("", ""))

        # Collect set A
        lines.append(
            f"OPTIONAL MATCH ({a_alias})-[:{via_rel}*1..{hops}]-(ta:{via_label})"
        )
        lines.append(f"WITH {a_alias}, {b_alias}, collect(DISTINCT ta) AS set_a")

        # Collect set B
        lines.append(
            f"OPTIONAL MATCH ({b_alias})-[:{via_rel}*1..{hops}]-(tb:{via_label})"
        )
        lines.append(f"WITH {a_alias}, {b_alias}, set_a, collect(DISTINCT tb) AS set_b")

        # Compute intersection / difference
        lines.append(
            f"WITH {a_alias}, {b_alias}, set_a, set_b, "
            "[x IN set_a WHERE x IN set_b] AS shared, "
            "[x IN set_a WHERE NOT x IN set_b] AS unique_a, "
            "[x IN set_b WHERE NOT x IN set_a] AS unique_b"
        )

        lim = schema.limit
        lines.append(
            f"RETURN properties({a_alias}) AS entity_a, "
            f"properties({b_alias}) AS entity_b, "
            f"[x IN shared[0..{lim}] | properties(x)] AS shared, "
            f"[x IN unique_a[0..{lim}] | properties(x)] AS unique_to_a, "
            f"[x IN unique_b[0..{lim}] | properties(x)] AS unique_to_b, "
            "size(shared) AS shared_count"
        )

        cypher = "\n".join(lines)
        log.debug(f"Built compare query: {cypher}")
        return cypher, params
