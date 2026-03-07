# Galaxy GraphRAG: Cypher-Native Graph Retrieval-Augmented Generation

Graph-aware context retrieval for the Galaxy knowledge graph. Performs semantic search via Qdrant, expands results through targeted Cypher queries on Neo4j, and returns structured context for LLM consumption.

## Architecture

```
query → pipeline.py → semantic_adapter.py → Qdrant (vector search)
                     → graph_retriever.py  → neo4j_connector.py → Neo4j (Cypher expansion)
                     → context_builder.py  → structured markdown context
```

## Files

| File | Purpose |
|---|---|
| `config.py` | `NODE_ID_FIELDS`, `GraphRAGConfig` |
| `neo4j_connector.py` | Cypher fetchers: workflow/tool/step context, global analytics, complex queries |
| `semantic_adapter.py` | Wraps Informer's `SemanticSearcher` for GraphRAG |
| `graph_retriever.py` | Maps semantic hits → Neo4j seed nodes → targeted Cypher payloads |
| `context_builder.py` | Formats payloads into LLM-readable markdown context |
| `pipeline.py` | Orchestrates retrieval with 3-mode routing |
| `demo_graph_rag.py` | Standalone demo with sample queries |

## Query Modes

| Mode | Use Case |
|---|---|
| `local` | Semantic search + graph expansion for specific entities |
| `global` | Ecosystem-wide analytics (most used tools, communities) |
| `complex` | Multi-hop queries: workflow comparison, tool connections, category drill-down |

## Usage

Exposed as the `graph_rag_query` MCP tool in `server.py`. Can also be used directly:

```python
from app.bioblend_server.GraphRAG.pipeline import GraphRAGPipeline

result = await pipeline.retrieve_context(
    query="Which tools are used in RNA-seq?",
    query_type="local",   # or "global" or "complex"
    top_k=15,
)
context = result["context"]  # structured markdown string
```

## Requirements

- Neo4j with Galaxy KG loaded
- Qdrant with indexed tool/workflow embeddings
- Env vars: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`