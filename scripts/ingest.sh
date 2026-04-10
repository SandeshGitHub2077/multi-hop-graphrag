#!/bin/bash
# Ingest wrapper - auto-starts services and ingests documents

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Source virtual environment if exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Start required services (only Neo4j needed for ingestion)
python scripts/start_services.py --neo4j --no-ollama

# Run ingestion (pass through all args)
python scripts/ingest_documents.py "$@"