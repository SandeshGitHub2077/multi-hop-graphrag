#!/usr/bin/env python3
"""Evaluation script - computes precision@k, recall@k, MRR, nDCG."""

import warnings
import logging
import io
import sys
import os
import yaml

original_stdout = sys.stdout
original_stderr = sys.stderr

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
from retrieval import HybridRetriever
from utils.config import config

sys.stdout = original_stdout
sys.stderr = original_stderr


def precision_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Compute precision@k."""
    retrieved_k = retrieved[:k]
    relevant = sum(1 for r in retrieved_k if r in expected)
    return relevant / k if k > 0 else 0.0


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Compute recall@k."""
    retrieved_k = retrieved[:k]
    relevant = sum(1 for r in retrieved_k if r in expected)
    return relevant / len(expected) if expected else 0.0


def mrr(retrieved: list[str], expected: list[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, r in enumerate(retrieved, 1):
        if r in expected:
            return 1.0 / i
    return 0.0


def dcg(retrieved: list[str], expected: list[str], k: int) -> float:
    """Compute DCG@k."""
    dcg_sum = 0.0
    for i, r in enumerate(retrieved[:k], 1):
        if r in expected:
            dcg_sum += 1.0 / (i if i > 0 else 1)
    return dcg_sum


def ndcg(retrieved: list[str], expected: list[str], k: int) -> float:
    """Compute nDCG@k."""
    dcg_val = dcg(retrieved, expected, k)
    idcg_val = dcg(expected, expected, k)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


def evaluate_query(retriever: HybridRetriever, query: str, expected: list[str], k: int = 10) -> dict:
    """Evaluate a single query."""
    results = retriever.retrieve(query, k=k)
    retrieved = [r.section_id for r in results]
    
    return {
        "precision@5": precision_at_k(retrieved, expected, 5),
        "precision@10": precision_at_k(retrieved, expected, 10),
        "recall@5": recall_at_k(retrieved, expected, 5),
        "recall@10": recall_at_k(retrieved, expected, 10),
        "mrr": mrr(retrieved, expected),
        "ndcg@5": ndcg(retrieved, expected, 5),
        "ndcg@10": ndcg(retrieved, expected, 10),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument("ground_truth", help="Path to eval.yaml")
    parser.add_argument("--k", type=int, default=10, help="Max results to retrieve")
    args = parser.parse_args()

    with open(args.ground_truth) as f:
        eval_data = yaml.safe_load(f)
    
    queries = eval_data.get("queries", [])
    
    print("Loading vector store...")
    vector_store = VectorStore()
    vector_store.load(config.index_dir)
    
    print("Loading embedding model...")
    embedding_engine = EmbeddingEngine(config.embedding_model)
    embedding_engine.load_model()
    
    print("Connecting to graph...")
    graph = Neo4jGraph(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
    graph.connect()
    
    retriever = HybridRetriever(vector_store, graph, embedding_engine)
    
    all_results = []
    
    for item in queries:
        query = item["query"]
        expected = item["expected_sections"]
        
        print(f"\nQuery: {query}")
        print(f"Expected: {expected}")
        
        metrics = evaluate_query(retriever, query, expected, args.k)
        
        print(f"  Precision@5: {metrics['precision@5']:.3f}")
        print(f"  Precision@10: {metrics['precision@10']:.3f}")
        print(f"  Recall@5: {metrics['recall@5']:.3f}")
        print(f"  Recall@10: {metrics['recall@10']:.3f}")
        print(f"  MRR: {metrics['mrr']:.3f}")
        print(f"  nDCG@5: {metrics['ndcg@5']:.3f}")
        print(f"  nDCG@10: {metrics['ndcg@10']:.3f}")
        
        all_results.append(metrics)
    
    n = len(all_results)
    print("\n" + "=" * 60)
    print("AGGREGATE SCORES:")
    print("=" * 60)
    print(f"Queries evaluated: {n}")
    print(f"Precision@5:  {sum(r['precision@5'] for r in all_results) / n:.3f}")
    print(f"Precision@10: {sum(r['precision@10'] for r in all_results) / n:.3f}")
    print(f"Recall@5:     {sum(r['recall@5'] for r in all_results) / n:.3f}")
    print(f"Recall@10:    {sum(r['recall@10'] for r in all_results) / n:.3f}")
    print(f"MRR:          {sum(r['mrr'] for r in all_results) / n:.3f}")
    print(f"nDCG@5:       {sum(r['ndcg@5'] for r in all_results) / n:.3f}")
    print(f"nDCG@10:      {sum(r['ndcg@10'] for r in all_results) / n:.3f}")
    
    graph.close()


if __name__ == "__main__":
    main()