#!/usr/bin/env python3
"""FastAPI REST API for the RAG system."""

import warnings
import logging
import io
import sys
import os

original_stdout = sys.stdout
original_stderr = sys.stderr

sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph import Neo4jGraph
from embeddings import EmbeddingEngine, VectorStore
from retrieval import HybridRetriever
from utils.config import config
from utils.health import HealthChecker

sys.stdout = original_stdout
sys.stderr = original_stderr

app = FastAPI(title="GraphRAG API", version="1.0.0")

vector_store: Optional[VectorStore] = None
embedding_engine: Optional[EmbeddingEngine] = None
graph: Optional[Neo4jGraph] = None
retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global vector_store, embedding_engine, graph, retriever
    
    if retriever is None:
        vector_store = VectorStore()
        vector_store.load(config.index_dir)
        
        embedding_engine = EmbeddingEngine(config.embedding_model)
        embedding_engine.load_model()
        
        graph = Neo4jGraph(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
        graph.connect()
        
        retriever = HybridRetriever(vector_store, graph, embedding_engine)
    
    return retriever


class QueryRequest(BaseModel):
    query: str
    k: int = 10
    multi_hop: bool = False


class QueryResponse(BaseModel):
    answer: Optional[str] = None
    results: List[dict]


class HealthResponse(BaseModel):
    status: str
    details: dict


@app.get("/health")
async def health() -> HealthResponse:
    """Check service health."""
    try:
        r = get_retriever()
        health_checker = HealthChecker(graph=graph, vector_store=vector_store)
        results = health_checker.check_all()
        
        all_ok = all(ok for ok, _ in results.values())
        
        return HealthResponse(
            status="healthy" if all_ok else "degraded",
            details={name: ok for name, (ok, msg) in results.items()}
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/documents")
async def list_documents() -> dict:
    """List indexed documents."""
    try:
        r = get_retriever()
        docs = r._graph.get_all_documents()
        return {"documents": [{"doc_id": d.doc_id, "title": d.title} for d in docs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """Run retrieval query."""
    try:
        r = get_retriever()
        
        if req.multi_hop:
            from retrieval import MultiHopRetriever
            base_retriever = r
            mh_retriever = MultiHopRetriever(base_retriever)
            results = mh_retriever.retrieve_with_hops(req.query, target_hops=2)
        else:
            results = r.retrieve(req.query, k=req.k)
        
        return QueryResponse(
            results=[
                {
                    "section_id": res.section_id,
                    "doc_id": res.doc_id,
                    "content": res.content[:200] + "..." if len(res.content) > 200 else res.content,
                    "score": res.score
                }
                for res in results
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest() -> dict:
    """Trigger document ingestion."""
    return {"status": "not implemented - use bash scripts/ingest.sh"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)