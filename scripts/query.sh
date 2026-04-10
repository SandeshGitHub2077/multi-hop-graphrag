#!/bin/bash
# Quick query wrapper - auto-starts services and runs query

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Source virtual environment if exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Start required services
python scripts/start_services.py --neo4j --ollama

# Run the query
python scripts/run_query.py "$@"