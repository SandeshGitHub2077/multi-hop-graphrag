import warnings
import logging
import sys
import io
import os

original_stdout = sys.stdout
original_stderr = sys.stderr

# Suppress all output during imports
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph import Neo4jGraph
from embeddings import EmbeddingEngine, VectorStore
from retrieval import HybridRetriever, MultiHopRetriever
from utils.llm import LLMWrapper
from utils.health import HealthChecker

# Restore stdout/stderr
sys.stdout = original_stdout
sys.stderr = original_stderr


def format_context(results: list, max_chars: int = 2000) -> str:
    """Format retrieved sections into context for LLM."""
    context_parts = []
    total_chars = 0
    
    for result in results:
        section_text = f"Section {result.section_id}:\n{result.content}"
        if total_chars + len(section_text) > max_chars:
            break
        context_parts.append(section_text)
        total_chars += len(section_text)
    
    return "\n\n".join(context_parts)


def main():
    parser = argparse.ArgumentParser(description="Run queries against the RAG system")
    parser.add_argument("--query", required=True, help="Query string")
    parser.add_argument("--graph-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--graph-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--graph-password", default="password", help="Neo4j password")
    parser.add_argument("--index-dir", default="index", help="Directory containing FAISS index")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5", help="Embedding model")
    parser.add_argument("--k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--multi-hop", action="store_true", help="Use multi-hop retrieval")
    parser.add_argument("--iterative", action="store_true", help="Use iterative retrieval")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM answer generation")
    parser.add_argument("--llm-model", default="qwen3:8b", help="LLM model name")
    args = parser.parse_args()

    print("Loading vector store...")
    vector_store = VectorStore()
    vector_store.load(args.index_dir)
    
    print("Loading embedding model...")
    embedding_engine = EmbeddingEngine(args.embedding_model)
    embedding_engine.load_model()
    
    print("Connecting to graph...")
    graph = Neo4jGraph(args.graph_uri, args.graph_user, args.graph_password)
    graph.connect()

    health = HealthChecker(graph=graph, vector_store=vector_store)
    health_results = health.check_all()
    unhealthy = [(name, msg) for name, (ok, msg) in health_results.items() if not ok]
    if unhealthy:
        print("WARNING: Service issues detected:")
        for name, msg in unhealthy:
            print(f"  - {name}: {msg}")
    
    try:
        print(f"\nQuery: {args.query}")
        print("=" * 60)
        
        if args.iterative:
            retriever = HybridRetriever(vector_store, graph, embedding_engine)
            results, context = retriever.iterative_retrieve(args.query, max_iterations=3)
        elif args.multi_hop:
            base_retriever = HybridRetriever(vector_store, graph, embedding_engine)
            retriever = MultiHopRetriever(base_retriever)
            results = retriever.retrieve_with_hops(args.query, target_hops=2)
        else:
            retriever = HybridRetriever(vector_store, graph, embedding_engine)
            results = retriever.retrieve(args.query, k=args.k)
        
        print(f"\nRetrieved {len(results)} relevant sections")
        
        if not args.no_llm:
            print("\nGenerating answer with LLM...")
            llm = LLMWrapper(model=args.llm_model)
            llm.load()
            
            context = format_context(results)
            answer = llm.generate_with_context(args.query, context)
            
            print("\n" + "=" * 60)
            print("ANSWER:")
            print("=" * 60)
            print(answer)
            print()
        
        print("=" * 60)
        print("SOURCE SECTIONS:")
        print("=" * 60)
        
        for i, result in enumerate(results[:args.k], 1):
            print(f"{i}. [{result.section_id}] (score: {result.score:.4f})")
            content_preview = result.content[:150] + "..." if len(result.content) > 150 else result.content
            print(f"   {content_preview}")
            print()
        
    finally:
        graph.close()


if __name__ == "__main__":
    main()
