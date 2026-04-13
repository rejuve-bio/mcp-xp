# Galaxy GraphRAG

LLM-planned graph retrieval-augmented generation over the Galaxy knowledge graph.

## Architecture

```
query
  → semantic_adapter.py     → Qdrant (vector search)
  → entity_resolver.py      → Neo4j (seed resolution)
  → planner.py              → LLM (generates CypherQuerySchema JSON)
  → cypher_builder.py       → parameterized Cypher (deterministic)
  → executor.py             → Neo4j (query execution)
  → context_builder.py      → Markdown evidence
  → pipeline.py             → LLM (answer synthesis)
  → server.py               → MCP tool response
```

## Files

| File | Purpose |
|---|---|
| `schema.py` | KG schema constants: node labels, edge types, property mappings |
| `models.py` | Pydantic v2 contracts: `CypherQuerySchema`, `PlannerOutput`, `ExecutionResult` |
| `config.py` | Neo4j env vars, `BudgetConfig`, `GraphRAGConfig` |
| `neo4j_connector.py` | Async Neo4j driver: `find_nodes_by_values`, `run_query` |
| `entity_resolver.py` | Semantic hits → `ResolvedEntity` with `element_id` |
| `semantic_adapter.py` | Wraps Informer's `SemanticSearcher` for GraphRAG |
| `planner.py` | LLM-based planner: query + seeds → validated query schemas |
| `cypher_builder.py` | Schema → parameterized Cypher (anchor, path, aggregate, compare) |
| `executor.py` | Executes built queries with budget enforcement |
| `context_builder.py` | Renders query results to structured Markdown evidence |
| `pipeline.py` | 8-stage orchestration + LLM answer synthesis |

## MCP Tool

```python
graph_rag_query(query: str, debug: bool = False) -> DefaultTextResponses
```

Single tool, natural-language input. The planner determines the retrieval strategy automatically.

## Requirements

- Neo4j with Galaxy KG loaded (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`)
- Qdrant with indexed tool/workflow embeddings
- LLM API configured via `CURRENT_LLM` (Gemini or OpenAI)
