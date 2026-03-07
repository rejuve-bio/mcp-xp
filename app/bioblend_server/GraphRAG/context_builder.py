from __future__ import annotations

import logging
from typing import Any, Dict, List
from app.bioblend_server.GraphRAG.config import GraphRAGConfig, GraphRAGEnum

SUMMARY_FIELDS = ["readme_content", "description", "help", "annotation", "name"]


class ContextBuilder:
    """Formats structured dict payloads into clean, hierarchical Markdown contexts."""

    def __init__(self, config: GraphRAGConfig) -> None:
        self.config = config
        self.log = logging.getLogger(__class__.__name__)

    def build_context(
        self,
        context_payloads: List[Dict[str, Any]],
    ) -> str:
        """Translates targeted Cypher payloads into token-budgeted text.

        Assembles dedicated templates for Workflows, Tools, and Steps according to
        what `neo4j_connector` resolved.

        Args:
            context_payloads: List of dictionaries matching the target fetcher output.

        Returns:
            A string containing human/LLM-readable graph representation.
        """
        self.log.info(f"Starting to build context from {len(context_payloads)} payloads")
        if not context_payloads:
            self.log.debug("No context payloads provided")
            return "No matching context found."

        sections: List[str] = []

        seen_entities = set()  # Dedup by internal_id
        seen_names = set()     # Dedup tools/steps by name to avoid version spam

        for payload in context_payloads:
            try:
                self.log.debug(f"Processing payload: {payload}")
                seed = payload.get("_source_seed", {})
                seed_id = seed.get("id", "Unknown")
                if seed_id in seen_entities:
                    self.log.debug(f"Skipping duplicate entity: {seed_id}")
                    continue
                seen_entities.add(seed_id)
                
                label = seed.get("label")
                section = ""

                # For Tool and Step payloads, dedup by entity name so different
                # versions of the same tool don't flood the context.
                if label == "Tool":
                    tool_name = (payload.get("tool") or {}).get("name")
                    if tool_name and tool_name in seen_names:
                        self.log.debug(f"Skipping duplicate tool name: {tool_name}")
                        continue
                    if tool_name:
                        seen_names.add(tool_name)
                    section = self._format_tool(payload)
                elif label == "Step":
                    step_name = (payload.get("step") or {}).get("name")
                    if step_name and step_name in seen_names:
                        self.log.debug(f"Skipping duplicate step name: {step_name}")
                        continue
                    if step_name:
                        seen_names.add(step_name)
                    section = self._format_step(payload)
                elif label == "Workflow":
                    section = self._format_workflow(payload)
                elif payload.get("analytics_type") == "top_tools":
                    section = self._format_analytics_top_tools(payload)
                elif payload.get("analytics_type") == "tool_communities":
                    section = self._format_analytics_communities(payload)
                elif payload.get("analytics_type") == "workflow_comparison":
                    section = self._format_workflow_comparison(payload)
                elif payload.get("analytics_type") == "tool_connection":
                    section = self._format_tool_connection(payload)
                elif payload.get("analytics_type") == "category_drilldown":
                    section = self._format_category_drilldown(payload)
                else:
                    self.log.warning(f"Unknown label or analytics_type in payload: {payload}")
                    pass # Catch all
                    
                if section and section not in sections:
                    sections.append(section)
            except Exception as e:
                self.log.error(f"Error processing payload: {e}", exc_info=True)
                continue

        context = "\n\n---\n\n".join(sections)
        self.log.info(f"Context building completed with {len(sections)} sections")
            
        return context

    # Formatting Templates
    def _format_workflow(self, payload: Dict[str, Any]) -> str:
        """Format a Workflow and its execution graph."""
        try:
            self.log.info("Formatting workflow payload")
            wf = payload.get("workflow", {})
            steps = payload.get("steps", [])
            
            lines = [f"[WORKFLOW] {wf.get('name', 'Unknown')}"]
            if summary := self._get_summary(wf):
                lines.append(f"  Summary: {summary}")
                
            if not steps:
                self.log.debug("No steps found in workflow")
                lines.append("  (No attached steps defined)")
                return "\n".join(lines)
                
            lines.append("\n  - [Execution Steps]:")
            for i, step_dict in enumerate(steps, 1):
                try:
                    self.log.debug(f"Processing step {i}: {step_dict}")
                    props = step_dict.get("properties", {})
                    lines.append(f"    Step {i}: {props.get('name', 'Unnamed Step')}")
                    if annotation := props.get("annotation"):
                        lines.append(f"      Annotation: {self._format_value(annotation)}")
                    
                    # Tools 
                    for tool in step_dict.get("tools", []):
                        if tool_name := tool.get('name'):
                            lines.append(f"      - Uses Tool: {tool_name}")
                    
                    # Input Needs
                    for inp in step_dict.get("inputs", []):
                        if inp_name := inp.get('name'):
                            lines.append(f"      - Requires Input: {inp_name} - {self._format_value(inp.get('description', ''))}")
                        
                    # Generated Artifacts
                    for out in step_dict.get("outputs", []):
                        if out_name := out.get('name'):
                            lines.append(f"      - Generates Output: {out_name} - {self._format_value(out.get('description', ''))}")
                except Exception as e:
                    self.log.error(f"Error processing step {i}: {e}", exc_info=True)
                    continue

            return "\n".join(lines)
        except Exception as e:
            self.log.error(f"Error formatting workflow: {e}", exc_info=True)
            return ""

    def _format_tool(self, payload: Dict[str, Any]) -> str:
        """Format a Tool, its standard IO, and places it's used."""

        try:
            self.log.info("Formatting tool payload")
            tool = payload.get("tool", {})
            inputs = payload.get("tool_inputs", [])
            outputs = payload.get("tool_outputs", [])
            workflows = payload.get("workflows", [])
            
            lines = [f"[TOOL] {tool.get('name', 'Unknown')}"]
            if summary := self._get_summary(tool):
                lines.append(f"  Summary: {summary}")
            if ver := tool.get("version"):
                lines.append(f"  Version: {ver}")
                
            if inputs or outputs:
                lines.append("\n  - [Tool Specifications]:")
                for inp in inputs:
                    lines.append(f"    - Input: {inp.get('input_name')} [{inp.get('input_type')}]")
                for out in outputs:
                    lines.append(f"    - Output: {out.get('output_name')} [{out.get('output_format')}]")
                    
            if workflows:
                valid_wfs = [wf.get('name') for wf in workflows if wf.get('name')]
                if valid_wfs:
                    lines.append("\n  - [Used in Workflows]:")
                    for wf_name in valid_wfs:
                        lines.append(f"    - Workflow: {wf_name}")

            return "\n".join(lines)
        except Exception as e:
            self.log.error(f"Error formatting tool: {e}", exc_info=True)
            return ""

    def _format_step(self, payload: Dict[str, Any]) -> str:
        """Format a Step in isolation, locating it in its workflow."""
        try:
            self.log.info("Formatting step payload")
            step = payload.get("step", {})
            wf = payload.get("workflow", {})
            tool = payload.get("tool", {})
            inputs = payload.get("inputs", [])
            outputs = payload.get("outputs", [])
            
            lines = [f"[STEP] {step.get('name', 'Unknown')}"]
            
            if wf_name := wf.get('name'):
                lines.append(f"  Part of Workflow: {wf_name}")
                
            if tool_name := tool.get('name'):
                lines.append(f"  Executes Tool: {tool_name}")
                if text := self._get_summary(tool):
                    lines.append(f"    Tool Description: {text}")
                    
            if inputs:
                lines.append("\n  - Required Workflow Inputs:")
                for inp in inputs:
                    if inp_name := inp.get('name'):
                        lines.append(f"    - {inp_name} - {self._format_value(inp.get('description', ''))}")
                    
            if outputs:
                lines.append("\n  - Generated Workflow Outputs:")
                for out in outputs:
                    if out_name := out.get('name'):
                        val = self._format_value(out.get('description', ''))
                        lines.append(f"    - {out_name}" + (f" - {val}" if val else ""))

            return "\n".join(lines)
        except Exception as e:
            self.log.error(f"Error formatting step: {e}", exc_info=True)
            return ""

    def _format_analytics_top_tools(self, payload: Dict[str, Any]) -> str:
        """Formats the result of a global macro-query for most used tools."""

        try:
            self.log.info("Formatting top tools analytics")
            results = payload.get("results", [])
            if not results:
                self.log.debug("No results in top tools analytics")
                return "[GLOBAL ANALYTICS] No tool usage data found."
                
            lines = ["[GLOBAL ANALYTICS] Most Widely Used Tools Across All Workflows\n"]
            for i, row in enumerate(results, 1):
                try:
                    self.log.debug(f"Processing top tool row {i}: {row}")
                    tool = row.get("tool_properties", {})
                    count = row.get("usage_count", 0)
                    name = tool.get("name", "Unknown Tool")
                    ver = tool.get("version", "")
                    ver_str = f" (v{ver})" if ver else ""
                    lines.append(f"{i}. {name}{ver_str}: Used in {count} steps")
                    if text := self._get_summary(tool):
                        lines.append(f"   - {text}")
                except Exception as e:
                    self.log.error(f"Error processing top tool row {i}: {e}", exc_info=True)
                    continue
                    
            return "\n".join(lines)
        except Exception as e:
            self.log.error(f"Error formatting top tools analytics: {e}", exc_info=True)
            return ""

        
    def _format_analytics_communities(self, payload: Dict[str, Any]) -> str:
        """Formats tools grouped by Graph Data Science (GDS) communities."""

        try:
            self.log.info("Formatting tool communities analytics")
            communities = payload.get("communities", {})
            if not communities:
                self.log.debug("No communities data found")
                return "[GLOBAL ANALYTICS] No GDS community data found."
                
            lines = ["[GLOBAL ANALYTICS] Tool Ecosystem Communities & PageRank\n"]
            
            # Sort communities by total usage of tools inside them for display
            sorted_comms = sorted(
                communities.items(), 
                key=lambda x: sum(t.get("usage_count", 0) for t in x[1]),
                reverse=True
            )
            
            for comm_id, tools in sorted_comms:
                try:
                    self.log.debug(f"Processing community {comm_id} with {len(tools)} tools")
                    lines.append(f"Community ID [{comm_id}]: {len(tools)} Core Tools")
                    for t_data in tools:
                        tool = t_data.get("tool", {})
                        name = tool.get("name", "Unknown")
                        pr = t_data.get("pagerank", 0)
                        usage = t_data.get("usage_count", 0)
                        
                        metrics = []
                        if pr > 0:
                            metrics.append(f"PageRank: {pr:.4f}")
                        metrics.append(f"Plays: {usage}")
                        
                        lines.append(f"  - {name} ({', '.join(metrics)})")
                    lines.append("") # spacer
                except Exception as e:
                    self.log.error(f"Error processing community {comm_id}: {e}", exc_info=True)
                    continue
                    
            return "\n".join(lines)
        except Exception as e:
            self.log.error("Error formatting communities analytics: {e}", e, exc_info=True)
            return ""

    def _format_workflow_comparison(self, payload: Dict[str, Any]) -> str:
        """Formats comparison between two workflows."""

        try:
            self.log.info("Formatting workflow comparison")
            wf_a = payload.get("workflow_a", {})
            wf_b = payload.get("workflow_b", {})
            shared = payload.get("shared_tools", [])
            unique_a = payload.get("unique_to_a", [])
            unique_b = payload.get("unique_to_b", [])
            
            name_a = self._get_entity_name(wf_a)
            name_b = self._get_entity_name(wf_b)
            
            lines = [f"[WORKFLOW COMPARISON] {name_a} vs {name_b}\n"]
            
            if shared:
                lines.append(f"Shared Tools ({len(shared)}):")
                for t in shared:
                    lines.append(f"  - {self._get_entity_name(t)}")
            else:
                self.log.debug("No shared tools in comparison")
                lines.append("No shared tools found.")
                
            if unique_a:
                lines.append(f"\nUnique to {name_a} ({len(unique_a)}):")
                for t in unique_a:
                    lines.append(f"  - {self._get_entity_name(t)}")
                    
            if unique_b:
                lines.append(f"\nUnique to {name_b} ({len(unique_b)}):")
                for t in unique_b:
                    lines.append(f"  - {self._get_entity_name(t)}")
                    
            return "\n".join(lines)
        except Exception as e:
            self.log.error(f"Error formatting workflow comparison: {e}", exc_info=True)
            return ""

    def _format_tool_connection(self, payload: Dict[str, Any]) -> str:
        """Formats the relationship path between two tools."""

        try:
            self.log.info("Formatting tool connection")
            tool_a = payload.get("tool_a", {})
            tool_b = payload.get("tool_b", {})
            shared_wfs = payload.get("shared_workflows", [])
            data_flows = payload.get("data_flows", [])
            
            name_a = self._get_entity_name(tool_a)
            name_b = self._get_entity_name(tool_b)
            
            lines = [f"[TOOL CONNECTION] {name_a} ↔ {name_b}\n"]
            
            if shared_wfs:
                lines.append(f"Appear Together in {len(shared_wfs)} Workflow(s):")
                for wf in shared_wfs:
                    lines.append(f"  - {self._get_entity_name(wf)}")
            else:
                self.log.debug("No shared workflows in tool connection")
                lines.append("These tools do not appear in any shared workflow.")
                
            # Show data flow connections
            valid_flows = [f for f in data_flows if f.get("output") or f.get("input")]
            if valid_flows:
                lines.append(f"\nPotential Data Flow Connections:")
                for flow in valid_flows[:5]:
                    out = flow.get("output", {})
                    inp = flow.get("input", {})
                    if out and inp:
                        lines.append(f"  {name_a} Output: {out.get('name', '?')} → {name_b} Input: {inp.get('name', '?')}")
                        
            return "\n".join(lines)
        except Exception as e:
            self.log.error(f"Error formatting tool connection: {e}", exc_info=True)
            return ""

    def _format_category_drilldown(self, payload: Dict[str, Any]) -> str:
        """Formats a category with its ranked tools."""

        try:
            self.log.info("Formatting category drilldown")
            category = payload.get("category", {})
            tools = payload.get("tools", [])
            
            cat_name = category.get("name", "Unknown Category")
            lines = [f"[CATEGORY] {cat_name}\n"]
            
            if not tools:
                self.log.debug(f"No tools in category: {cat_name}")
                lines.append("No tools found in this category.")
                return "\n".join(lines)
                
            lines.append(f"Top Tools (ranked by usage):")
            for i, t_data in enumerate(tools, 1):
                try:
                    self.log.debug(f"Processing category tool {i}: {t_data}")
                    tool = t_data.get("tool") or {}
                    name = tool.get("name")
                    if not name:
                        self.log.warning(f"Missing name in tool data: {t_data}")
                        continue
                    usage = t_data.get("usage_count", 0)
                    examples = t_data.get("example_workflows", [])
                    
                    lines.append(f"{i}. {name}: {usage} uses")
                    if summary := self._get_summary(tool):
                        lines.append(f"   - {summary}")
                    if examples:
                        valid_examples = [str(e) for e in examples if e]
                        if valid_examples:
                            lines.append(f"   - Example Workflows: {', '.join(valid_examples[:3])}")
                except Exception as e:
                    self.log.error(f"Error processing category tool {i}: {e}", exc_info=True)
                    continue
                    
            return "\n".join(lines)
        except Exception as e:
            self.log.error(f"Error formatting category drilldown: {e}", exc_info=True)
            return ""

    # Utilities
    @staticmethod
    def _get_entity_name(props: Dict[str, Any]) -> str:
        """Resolve a display name from entity properties, falling back through common fields."""
        if not props:
            return "Unknown"
        for key in ("name", "file_name", "workflow_id", "tool_id", "step_uid"):
            val = props.get(key)
            if val and str(val).strip():
                return str(val).strip()
        return "Unknown"

    def _get_summary(self, properties: Dict[str, Any]) -> str:
        try:
            self.log.debug(f"Getting summary from properties: {properties}")
            for key in SUMMARY_FIELDS:
                value = properties.get(key)
                if value is None:
                    continue
                value_string = " ".join(str(value).split())
                if value_string:
                    return self._format_value(value_string)
            self.log.debug("No summary found")
            return ""
        except Exception as e:
            self.log.error(f"Error getting summary: {e}", exc_info=True)
            return ""

    @staticmethod
    def _format_value(value: Any) -> str:
        
        try:
            max_len = GraphRAGEnum.FORMAT_SIZE.value
            text = " ".join(str(value).split())
            if len(text) <= max_len:
                return text
            return text[: max_len - 3] + "..."
        except Exception as e:
            logging.getLogger(__class__.__name__).error(f"Error formatting value: {e}", exc_info=True)
            return ""