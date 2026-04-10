# Multi-Document GraphRAG Pipeline

## Essential Commands

```bash
# Activate venv (scripts auto-activate but be explicit for clarity)
source .venv/bin/activate

# Ingest documents (auto-starts Neo4j)
bash scripts/ingest.sh              # Incremental (skip existing)
bash scripts/ingest.sh --rebuild    # Full rebuild

# Query (auto-starts Neo4j + Ollama)
bash scripts/query.sh --query "What is HTTP?"
python scripts/run_query.py --query "..." --multi-hop  # Multi-hop mode

# NEW: Unified CLI (replaces scripts above)
python scripts/rag_cli.py ingest [--rebuild]
python scripts/rag_cli.py query "What is HTTP?" [--multi-hop] [--no-llm] [--k 10]
python scripts/rag_cli.py serve [--port 8000]
python scripts/rag_cli.py eval tests/fixtures/eval.yaml

# FastAPI server directly
python scripts/api.py  # starts on http://localhost:8000

# Test (no lint/typecheck configured)
python3 -m pytest tests/ -v
python3 -m pytest tests/test_parsing.py -v --tb=short  # Single file
```

## Prerequisites

- **Python**: 3.14+ (not 3.13 or earlier)
- **Neo4j**: Must be running. Start with `neo4j start` or `python scripts/start_services.py --neo4j`
- **Password**: Default `CHANGEME` in `config.yaml`. Update after first login at http://localhost:7474 (login `neo4j/neo4j`, set new password)
- **Security**: `config.yaml` and `.env` are gitignored. Copy `config.yaml.example` if it exists.

## Key Architecture

- **Data**: 20 RFC documents, 1458 sections in `Data/`
- **Index**: FAISS in `index/` with `doc_id/section_id` keys (e.g., `rfc9112/1`)
- **Graph**: Neo4j for relationship traversal
- **Section IDs**:
  - CFR: `§ 1.1`, `§ 405.2414` → regex `r"§\s*(\d+(?:\.\w+)*)"`
  - RFC: `3.2.1`, `1.1` → regex `r"^(\d+(?:\.\d+)*)\.\s+(\w|\b)"`

## Query Pipeline

1. Vector search (k×4 oversample) → 2. BM25 merge (50% weight) → 3. Cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`) → 4. Graph expansion (depth=2) → 5. Reference density boost

- Caches: Embedding (100 LRU), Query results (50, 5-min TTL)
- Graph-first queries skip vector search for navigation ("where is", "which section", "dependencies")

## Configuration

Edit `config.yaml` for Neo4j credentials, embedding model (`BAAI/bge-base-en-v1.5`), and paths.