#!/usr/bin/env python3
"""Unified RAG CLI - Single entry point for all RAG operations."""

import argparse
import sys
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def run_ingest(args: argparse.Namespace) -> int:
    """Run document ingestion."""
    rebuild = "--rebuild" if args.rebuild else ""
    cmd = f"python scripts/ingest_documents.py {rebuild}".strip()
    return subprocess.call(cmd, shell=True, cwd=PROJECT_DIR)


def run_query(args: argparse.Namespace) -> int:
    """Run a query."""
    cmd_parts = [
        "python scripts/run_query.py",
        f"--query {args.query}",
    ]
    if args.multi_hop:
        cmd_parts.append("--multi-hop")
    if args.no_llm:
        cmd_parts.append("--no-llm")
    if args.k:
        cmd_parts.append(f"--k {args.k}")
    
    cmd = " ".join(cmd_parts)
    return subprocess.call(cmd, shell=True, cwd=PROJECT_DIR)


def run_serve(args: argparse.Namespace) -> int:
    """Start the FastAPI server."""
    try:
        from scripts.api import app
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    except ImportError as e:
        print(f"Error: FastAPI not installed. Run: pip install fastapi uvicorn", file=sys.stderr)
        return 1


def run_eval(args: argparse.Namespace) -> int:
    """Run evaluation."""
    cmd = f"python scripts/evaluate.py {args.ground_truth}"
    return subprocess.call(cmd, shell=True, cwd=PROJECT_DIR)


def main():
    parser = argparse.ArgumentParser(
        prog="rag-cli",
        description="Unified CLI for Multi-Document GraphRAG Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument(
        "--rebuild", action="store_true",
        help="Full rebuild (delete existing index and graph data)"
    )

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Run a query")
    query_parser.add_argument("query", help="Query string")
    query_parser.add_argument("--multi-hop", action="store_true", help="Use multi-hop retrieval")
    query_parser.add_argument("--no-llm", action="store_true", help="Skip LLM answer generation")
    query_parser.add_argument("--k", type=int, default=10, help="Number of results")

    # Serve subcommand
    serve_parser = subparsers.add_parser("serve", help="Start FastAPI server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port")

    # Eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("ground_truth", help="Path to ground truth YAML file")

    args = parser.parse_args()

    if args.command == "ingest":
        return run_ingest(args)
    elif args.command == "query":
        return run_query(args)
    elif args.command == "serve":
        return run_serve(args)
    elif args.command == "eval":
        return run_eval(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())