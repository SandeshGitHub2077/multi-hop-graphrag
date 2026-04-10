from typing import Optional
from graph import Neo4jGraph
from embeddings import VectorStore


class HealthChecker:
    def __init__(self, graph: Optional[Neo4jGraph] = None, vector_store: Optional[VectorStore] = None):
        self.graph = graph
        self.vector_store = vector_store

    def check_neo4j(self) -> tuple[bool, str]:
        if self.graph is None:
            return False, "Neo4j not configured"
        
        try:
            self.graph.driver.verify_connectivity()
            return True, "Neo4j connected"
        except Exception as e:
            return False, f"Neo4j error: {e}"

    def check_vector_store(self) -> tuple[bool, str]:
        if self.vector_store is None:
            return False, "Vector store not configured"
        
        try:
            if self.vector_store.index is None:
                return False, "Vector index not loaded"
            if self.vector_store.index.ntotal == 0:
                return False, "Vector index is empty"
            return True, f"Vector store healthy ({self.vector_store.index.ntotal} vectors)"
        except Exception as e:
            return False, f"Vector store error: {e}"

    def check_all(self) -> dict[str, tuple[bool, str]]:
        results = {}
        
        if self.graph:
            results["neo4j"] = self.check_neo4j()
        
        if self.vector_store:
            results["vector_store"] = self.check_vector_store()
        
        return results

    def is_healthy(self) -> bool:
        results = self.check_all()
        return all(status for status, _ in results.values())