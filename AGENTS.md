# Multi-Document GraphRAG Pipeline

## Essential Commands

```bash
source .venv/bin/activate

# Start API server (includes chat UI at http://localhost:8000)
python scripts/api.py

# Or use unified CLI
python scripts/rag_cli.py ingest [--rebuild]
python scripts/rag_cli.py query "What is HTTP?" [--multi-hop] [--no-llm] [--k 10]
python scripts/rag_cli.py serve [--port 8000]
python scripts/rag_cli.py eval tests/fixtures/eval.yaml
```

## Prerequisites

- **Python**: 3.14+ (not 3.13 or earlier)
- **Neo4j**: Must be running. Start with `neo4j start` or `python scripts/start_services.py --neo4j`
- **Ollama**: Required for LLM answer generation. Start with `ollama serve` or `python scripts/start_services.py --ollama`
- **Password**: Default `CHANGEME` in `config.yaml`. Update after first login at http://localhost:7474

## Key Architecture

- **Interface**: Chat-style web UI (FastAPI + HTML/JS). Open http://localhost:8000
- **Session-based**: Each browser session manages uploaded documents independently
- **Upload Flow**: Drag-drop files → parse → embed → store in Neo4j + FAISS → query against only user's docs

### Upload Features
- **Drag-and-drop**: Drop files onto sidebar to upload
- **Formats**: PDF, DOCX, TXT, Markdown
- **Isolation**: Each session only queries its own uploaded documents
- **Full pipeline**: Parses sections, extracts references, builds knowledge graph

## Query Pipeline

1. Vector search (k×4 oversample) → 2. BM25 merge (50% weight) → 3. Cross-encoder rerank → 4. Graph expansion → 5. Reference density boost → 6. LLM answer generation

- Graph-first queries skip vector search for navigation ("where is", "which section")
- LLM uses Ollama (qwen3:8b by default)

## Configuration

Edit `config.yaml` for:
- Neo4j credentials
- Embedding model (`BAAI/bge-base-en-v1.5`)
- Ollama model path
- Data and index directories