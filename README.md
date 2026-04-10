# Multi-Document GraphRAG Pipeline

A production-ready Knowledge Graph RAG system that processes heterogeneous documents (PDF, DOCX, TXT, Markdown) with structured cross-references, builds a Neo4j knowledge graph, and enables hybrid retrieval combining vector similarity with graph traversal.

> **Security Note**: `config.yaml` and `.env` are gitignored. Copy `config.yaml.example` to `config.yaml` and update the password.

## Features

- **Multi-format Document Support**: PDF, DOCX, TXT, Markdown
- **RFC/CFR Section Detection**: Extracts structured section IDs (e.g., `3.2.1`, `§ 1.1`)
- **Reference Extraction**: Detects cross-references between sections
- **Knowledge Graph**: Neo4j-backed graph for relationship traversal
- **Hybrid Retrieval**: Combines vector similarity (FAISS) with graph expansion
- **Query Routing**: Graph-first routing for navigation queries
- **Multi-hop Reasoning**: Traverses relationships for complex queries
- **Incremental Processing**: Only processes new sections on subsequent runs

## Requirements

- **Python**: 3.14+
- **Neo4j**: Community Edition (localhost:7474)
- **Ollama** (optional): For LLM-based answer generation

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Neo4j
neo4j start

# Set Neo4j password (open http://localhost:7474)
# Login with neo4j/neo4j and set a new password
```

## Quick Start

```bash
# Ingest documents (incremental)
bash scripts/ingest.sh

# Full rebuild
bash scripts/ingest.sh --rebuild

# Query the system
python scripts/run_query.py --query "What is HTTP?"

# Multi-hop retrieval
python scripts/run_query.py --query "..." --multi-hop

# Run tests
python3 -m pytest tests/ -v

# NEW: Unified CLI (recommended)
python scripts/rag_cli.py ingest [--rebuild]
python scripts/rag_cli.py query "What is HTTP?" [--multi-hop] [--no-llm]
python scripts/rag_cli.py serve [--port 8000]
python scripts/rag_cli.py eval tests/fixtures/eval.yaml
```

## Project Structure

```
├── parsing/           # Document parsing layer
│   └── __init__.py   # DocumentParser (RFC & CFR section extraction)
├── graph/             # Neo4j knowledge graph
│   └── neo4j_graph.py # Neo4jGraph operations
├── embeddings/        # Embedding & vector storage
│   ├── embedding_engine.py  # BGE embeddings + reference augmentation
│   └── vector_store.py      # FAISS index (doc_id/section_id keys)
├── retrieval/         # Hybrid retrieval
│   ├── hybrid_retriever.py # Query routing, reranking, graph expansion
│   └── query_router.py     # Intent classification
├── utils/            # Utilities
│   ├── config.py     # Config loader (yaml + env var override)
│   ├── llm.py        # LLMWrapper for Ollama
│   └── health.py     # Health checks
├── scripts/          # CLI entry points
│   ├── ingest.sh            # Auto-starts services + ingest
│   ├── ingest_documents.py # Document ingestion
│   ├── query.sh            # Auto-starts services + query
│   ├── run_query.py        # Query the system
│   ├── rag_cli.py          # Unified CLI (recommended)
│   ├── api.py              # FastAPI REST API
│   ├── evaluate.py         # Evaluation metrics
│   └── start_services.py  # Auto-start Neo4j/Ollama
├── tests/            # Test suite
│   └── fixtures/         # Test fixtures + eval.yaml
├── Data/             # Source documents (20 RFCs)
├── index/            # FAISS vector index
├── config.yaml       # Configuration
└── requirements.txt # Dependencies
```

## Configuration

Edit `config.yaml` to customize:
- Neo4j connection (URI, username, password)
- Embedding model
- Data and index directories

Or use environment variables (override config.yaml):
```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
export EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
export LLM_MODEL=qwen3:8b
```

## Data

Currently loaded: **20 RFC documents** (1458 sections)

| RFC | Sections |
|-----|----------|
| RFC 9110 | 291 |
| RFC 7231 | 140 |
| RFC 7540 | 97 |
| RFC 9113 | 95 |
| RFC 6455 | 88 |
| RFC 8446 | 86 |
| RFC 8447 | 22 |
| RFC 9112 | 59 |
| RFC 9111 | 65 |
| RFC 6265 | 59 |
| RFC 1035 | 71 |
| RFC 5322 | 54 |
| RFC 5246 | 57 |
| RFC 3986 | 68 |
| RFC 2045 | 32 |
| RFC 2046 | 43 |
| RFC 2119 | 9 |
| RFC 5890 | 33 |
| RFC 6066 | 23 |
| RFC 9204 | 66 |

## Tech Stack

- **LLM**: Qwen 3 via Ollama (local)
- **Embeddings**: BGE-base-en-v1.5 (local)
- **Vector DB**: FAISS
- **Graph DB**: Neo4j Community Edition
- **Orchestration**: LangChain

## Testing

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_parsing.py -v --tb=short

# Run evaluation
python scripts/rag_cli.py eval tests/fixtures/eval.yaml
```

## REST API

```bash
# Start server
python scripts/rag_cli.py serve
# or: python scripts/api.py

# Endpoints:
# GET  /health          - Health check
# GET  /documents       - List indexed documents
# POST /query           - Run retrieval query
# POST /ingest          - Trigger ingestion (not implemented)
```

## Troubleshooting

### Neo4j won't start
```bash
neo4j status
neo4j start
```

### Set Neo4j password
1. Open http://localhost:7474
2. Login with `neo4j/neo4j`
3. Set new password
4. Update `config.yaml` or script defaults

### Out of memory during ingestion
The script uses batched embedding generation. For extremely large document sets, reduce batch size in `scripts/ingest_documents.py`.

## License

MIT
