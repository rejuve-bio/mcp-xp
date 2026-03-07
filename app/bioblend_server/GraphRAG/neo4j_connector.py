from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase

from app.bioblend_server.GraphRAG.config import NODE_ID_FIELDS


SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_cypher_name(name: str) -> str:
    if not SAFE_NAME.match(name):
        raise ValueError(f"Unsafe Cypher name: {name!r}")
    return name


def _canonical_id(
    label: str,
    properties: Dict[str, Any],
    fallback_id: str = "",
) -> str:
    """ Generate a canonical ID for a node based on its label and properties. """

    id_fields = NODE_ID_FIELDS.get(label, ["id"])
    for field_name in id_fields:
        value = properties.get(field_name)
        if value is not None and str(value).strip():
            return f"{label}:{value}"
    if fallback_id:
        return f"{label}:{fallback_id}"
    return f"{label}:unknown"




class Neo4jGraphConnector:
    """Neo4j connector targeting Cypher-native subgraph contexts.

    Instead of pulling tens of thousands of nodes into Python via generic BFS,
    this connector executes highly specific 1-hop and 2-hop pattern matching queries
    based on the Knowledge Graph's exact schema to return structured dictionaries.
    """

    def __init__(self, uri: str, user: str, password: str, database: str | None = None) -> None:

        self.logger = logging.getLogger(self.__class__.__name__)
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    # Node lookup
    def find_nodes_by_values(
        self, label: str, exact_props: List[str], fuzzy_props: List[str], values: List[str]
    ) -> List[Dict[str, Any]]:
        """Batch finding nodes by multiple exact/fuzzy properties to minimize DB roundtrips."""

        if not values or (not exact_props and not fuzzy_props):
            return []

        safe_label = _validate_cypher_name(label)
        clauses = []
        params: Dict[str, Any] = {}

        if exact_props:
            typed_values = []
            for v in values:
                typed_values.append(v)
                if v.isdigit():
                    typed_values.append(int(v))
                elif v.replace('.', '', 1).isdigit() and v.count('.') == 1:
                    typed_values.append(float(v))
            
            params["exact_values"] = typed_values
            
            for prop in exact_props:
                safe_prop = _validate_cypher_name(prop)
                clauses.append(f"n.{safe_prop} IN $exact_values")

        if fuzzy_props:
            for i, val in enumerate(values):
                params[f"fuzz_val_{i}"] = str(val).lower()
                for prop in fuzzy_props:
                    safe_prop = _validate_cypher_name(prop)
                    clauses.append(f"toLower(toString(n.{safe_prop})) CONTAINS $fuzz_val_{i}")

        if not clauses:
            return []

        where_clause = " OR ".join(clauses)
        query = (
            f"MATCH (n:{safe_label}) "
            f"WHERE {where_clause} "
            "RETURN labels(n)[0] AS label, properties(n) AS properties, id(n) AS internal_id "
            "LIMIT 100"
        )

        rows = self._run(query, params)
        return [self._row_to_node(row) for row in rows]

    # Target Context Fetchers (Cypher-Native Routing)
    def get_workflow_context(self, internal_id: int) -> Dict[str, Any]:
        """Fetches a Workflow and its complete topology: Steps, attached Tools, Inputs, and Outputs."""
        query = """
        MATCH (w:Workflow) WHERE id(w) = $internal_id
        OPTIONAL MATCH (w)-[:HAS_STEP]->(s:Step)
        OPTIONAL MATCH (s)-[:STEP_USES_TOOL]->(t:Tool)
        OPTIONAL MATCH (s)-[:STEP_REQUIRES]->(i:Input)
        OPTIONAL MATCH (s)-[:STEP_GENERATES]->(o:Output)
        
        RETURN 
            properties(w) AS workflow_properties,
            collect(DISTINCT {
                step_id: id(s), 
                properties: properties(s),
                tools: properties(t),
                inputs: properties(i),
                outputs: properties(o)
            }) AS explicit_steps
        """
        rows = self._run(query, {"internal_id": internal_id})

        if not rows:
            return {}
            
        row = rows[0]
        
        # Deduplicate and group steps
        steps_dict = {}
        for s in row.get("explicit_steps", []):
            if not s.get("properties"):
                continue
            step_id = s["step_id"]
            if step_id not in steps_dict:
                steps_dict[step_id] = {
                    "properties": s["properties"],
                    "tools": [],
                    "inputs": [],
                    "outputs": []
                }
            if s.get("tools") and s["tools"] not in steps_dict[step_id]["tools"]:
                steps_dict[step_id]["tools"].append(s["tools"])
            if s.get("inputs") and s["inputs"] not in steps_dict[step_id]["inputs"]:
                steps_dict[step_id]["inputs"].append(s["inputs"])
            if s.get("outputs") and s["outputs"] not in steps_dict[step_id]["outputs"]:
                steps_dict[step_id]["outputs"].append(s["outputs"])
                
        return {
            "workflow": row.get("workflow_properties", {}),
            "steps": list(steps_dict.values())
        }

    def get_tool_context(self, internal_id: int) -> Dict[str, Any]:
        """Fetches a Tool, its inputs/outputs, and top example Workflows using it."""

        query = """
        MATCH (t:Tool) WHERE id(t) = $internal_id
        OPTIONAL MATCH (t)-[:TOOL_HAS_INPUT]->(ti:ToolInput)
        OPTIONAL MATCH (t)-[:TOOL_HAS_OUTPUT]->(to:ToolOutput)
        WITH t, collect(DISTINCT properties(ti)) AS tool_inputs, collect(DISTINCT properties(to)) AS tool_outputs
        
        OPTIONAL MATCH (w:Workflow)-[:WORKFLOW_USES_TOOL]->(t)
        WITH t, tool_inputs, tool_outputs, collect(DISTINCT properties(w))[0..5] AS top_workflows
        
        RETURN 
            properties(t) AS tool_properties,
            tool_inputs,
            tool_outputs,
            top_workflows
        """
        
        rows = self._run(query, {"internal_id": internal_id})
        if not rows:
            return {}
        return {
            "tool": rows[0].get("tool_properties", {}),
            "tool_inputs": rows[0].get("tool_inputs", []),
            "tool_outputs": rows[0].get("tool_outputs", []),
            "workflows": rows[0].get("top_workflows", [])
        }

    def get_step_context(self, internal_id: int) -> Dict[str, Any]:
        """Fetches a Step, the Tool it uses, its Inputs/Outputs, and its parent Workflow."""

        query = """
        MATCH (s:Step) WHERE id(s) = $internal_id
        OPTIONAL MATCH (w:Workflow)-[:HAS_STEP]->(s)
        OPTIONAL MATCH (s)-[:STEP_USES_TOOL]->(t:Tool)
        OPTIONAL MATCH (s)-[:STEP_REQUIRES]->(i:Input)
        OPTIONAL MATCH (s)-[:STEP_GENERATES]->(o:Output)
        
        RETURN 
            properties(s) AS step_properties,
            properties(w) AS parent_workflow,
            properties(t) AS step_tool,
            collect(DISTINCT properties(i)) AS inputs,
            collect(DISTINCT properties(o)) AS outputs
        """
        rows = self._run(query, {"internal_id": internal_id})
        if not rows:
            return {}
        return {
            "step": rows[0].get("step_properties", {}),
            "workflow": rows[0].get("parent_workflow", {}),
            "tool": rows[0].get("step_tool", {}),
            "inputs": rows[0].get("inputs", []),
            "outputs": rows[0].get("outputs", [])
        }

    # Global Analytics Fetchers (GDS & Aggregations)
    def get_most_used_tools(self, limit: int = 10) -> Dict[str, Any]:
        """Finds the most dominant tools across the entire ecosystem."""

        query = """
        MATCH (s:Step)-[:STEP_USES_TOOL]->(t:Tool)
        WITH t, count(s) AS usage_count
        ORDER BY usage_count DESC
        LIMIT $limit
        RETURN 
            properties(t) AS tool_properties,
            usage_count
        """
        rows = self._run(query, {"limit": limit})
        return {
            "analytics_type": "top_tools",
            "results": rows
        }

    def get_tools_by_community(self, limit: int = 20) -> Dict[str, Any]:
        """Retrieves top tools across GDS communities or by PageRank if available."""
        
        # This query assumes standard GDS properties (pagerank, community) might 
        # exist. We fallback to basic centrality if they aren't pre-mutated.
        query = """
        MATCH (t:Tool)
        // Coalesce GDS metrics if they exist, or default to 0
        WITH t, 
             coalesce(t.pagerank, 0) AS pr_score,
             coalesce(t.community, 'Unknown') AS community_id
        OPTIONAL MATCH (s:Step)-[:STEP_USES_TOOL]->(t)
        WITH t, pr_score, community_id, count(s) AS usage_count
        ORDER BY pr_score DESC, usage_count DESC
        LIMIT $limit
        RETURN 
            properties(t) AS tool_properties,
            pr_score,
            community_id,
            usage_count
        """
        rows = self._run(query, {"limit": limit})
        
        # Group by community
        communities = {}
        for row in rows:
            cid = row["community_id"]
            if cid not in communities:
                communities[cid] = []
            communities[cid].append({
                "tool": row.get("tool_properties", {}),
                "pagerank": row["pr_score"],
                "usage_count": row["usage_count"]
            })
            
        return {
            "analytics_type": "tool_communities",
            "communities": communities
        }

    # Complex Query Fetchers (Multi-hop, Comparative, Path-finding)
    def get_workflow_comparison(self, workflow_name_a: str, workflow_name_b: str) -> Dict[str, Any]:
        """Compares two workflows: shared tools, unique tools, shared categories.
        
        Answers questions like: 'What do RNA-seq and ChIP-seq workflows have in common?'
        """
        query = """
        MATCH (wa:Workflow) WHERE toLower(wa.name) CONTAINS toLower($name_a)
        MATCH (wb:Workflow) WHERE toLower(wb.name) CONTAINS toLower($name_b)
        WITH wa, wb LIMIT 1
        
        OPTIONAL MATCH (wa)-[:HAS_STEP]->(:Step)-[:STEP_USES_TOOL]->(ta:Tool)
        WITH wa, wb, collect(DISTINCT ta) AS tools_a
        
        OPTIONAL MATCH (wb)-[:HAS_STEP]->(:Step)-[:STEP_USES_TOOL]->(tb:Tool)
        WITH wa, wb, tools_a, collect(DISTINCT tb) AS tools_b
        
        WITH wa, wb, tools_a, tools_b,
             [t IN tools_a WHERE t IN tools_b] AS shared_tools,
             [t IN tools_a WHERE NOT t IN tools_b] AS unique_a,
             [t IN tools_b WHERE NOT t IN tools_a] AS unique_b
        
        RETURN 
            properties(wa) AS workflow_a,
            properties(wb) AS workflow_b,
            [t IN shared_tools | properties(t)] AS shared_tools,
            [t IN unique_a | properties(t)] AS unique_to_a,
            [t IN unique_b | properties(t)] AS unique_to_b,
            size(shared_tools) AS shared_count
        """

        rows = self._run(query, {"name_a": workflow_name_a, "name_b": workflow_name_b})
        if not rows:
            return {}
        row = rows[0]
        return {
            "analytics_type": "workflow_comparison",
            "workflow_a": row.get("workflow_a", {}),
            "workflow_b": row.get("workflow_b", {}),
            "shared_tools": row.get("shared_tools", []),
            "unique_to_a": row.get("unique_to_a", []),
            "unique_to_b": row.get("unique_to_b", []),
            "shared_count": row.get("shared_count", 0)
        }

    def get_tool_connection_path(self, tool_name_a: str, tool_name_b: str) -> Dict[str, Any]:
        """
        Finds how two tools connect through shared workflows and steps.
        
        Answers: 'How are fastqc and multiqc related?' or
                 'Do Bowtie2 and samtools appear together in any pipeline?'
        """

        query = """
        MATCH (ta:Tool) WHERE toLower(ta.name) CONTAINS toLower($name_a)
        MATCH (tb:Tool) WHERE toLower(tb.name) CONTAINS toLower($name_b)
        WITH ta, tb LIMIT 1
        
        // Find workflows that use BOTH tools
        OPTIONAL MATCH (w:Workflow)-[:HAS_STEP]->(:Step)-[:STEP_USES_TOOL]->(ta)
        WHERE EXISTS { (w)-[:HAS_STEP]->(:Step)-[:STEP_USES_TOOL]->(tb) }
        WITH ta, tb, collect(DISTINCT properties(w))[0..5] AS shared_workflows
        
        // Find if they appear in the same step sequence (tool A's output -> tool B's input)
        OPTIONAL MATCH (sa:Step)-[:STEP_USES_TOOL]->(ta)
        OPTIONAL MATCH (sa)-[:STEP_GENERATES]->(o:Output)
        OPTIONAL MATCH (sb:Step)-[:STEP_USES_TOOL]->(tb)
        OPTIONAL MATCH (sb)-[:STEP_REQUIRES]->(i:Input)
        WITH ta, tb, shared_workflows,
             collect(DISTINCT {output: properties(o), input: properties(i)})[0..5] AS data_flows
        
        RETURN
            properties(ta) AS tool_a,
            properties(tb) AS tool_b,
            shared_workflows,
            data_flows
        """
        rows = self._run(query, {"name_a": tool_name_a, "name_b": tool_name_b})
        if not rows:
            return {}
        row = rows[0]
        return {
            "analytics_type": "tool_connection",
            "tool_a": row.get("tool_a", {}),
            "tool_b": row.get("tool_b", {}),
            "shared_workflows": row.get("shared_workflows", []),
            "data_flows": row.get("data_flows", [])
        }

    def get_category_tools(self, category_name: str, limit: int = 10) -> Dict[str, Any]:
        """
        Drills into a category: lists its tools ranked by workflow usage.
        
        Answers: 'What are the best tools for sequence analysis?'
        """
        query = """
        MATCH (c:Category) WHERE toLower(c.name) CONTAINS toLower($cat_name)
        WITH c LIMIT 1
        
        OPTIONAL MATCH (c)-[:HAS_TOOL]->(t:Tool)
        OPTIONAL MATCH (s:Step)-[:STEP_USES_TOOL]->(t)
        WITH c, t, count(s) AS usage_count
        ORDER BY usage_count DESC
        LIMIT $limit
        
        OPTIONAL MATCH (w:Workflow)-[:HAS_STEP]->(:Step)-[:STEP_USES_TOOL]->(t)
        WITH c, t, usage_count, collect(DISTINCT w.name)[0..3] AS example_workflows
        
        RETURN 
            properties(c) AS category,
            collect({
                tool: properties(t),
                usage_count: usage_count,
                example_workflows: example_workflows
            }) AS tools
        """
        rows = self._run(query, {"cat_name": category_name, "limit": limit})
        if not rows:
            return {}
        row = rows[0]
        return {
            "analytics_type": "category_drilldown",
            "category": row.get("category", {}),
            "tools": row.get("tools", [])
        }

    # Internal helpers
    def _row_to_node(self, row: Dict[str, Any]) -> Dict[str, Any]:
        label = str(row.get("label", "Unknown"))
        properties = row.get("properties", {}) or {}
        internal_id = str(row.get("internal_id", ""))
        node_id = _canonical_id(label, properties, fallback_id=internal_id)
        return {"id": node_id, "label": label, "properties": properties, "internal_id": internal_id}

    def _run(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
