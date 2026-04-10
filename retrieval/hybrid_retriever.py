import re
from dataclasses import dataclass
from typing import Optional
from embeddings import EmbeddingEngine, VectorStore
from graph import Neo4jGraph
from rank_bm25 import BM25Okapi
import numpy as np
from collections import OrderedDict
import time
import hashlib


class QueryResultCache:
    def __init__(self, capacity: int = 50, ttl_seconds: int = 300):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.ttl = ttl_seconds

    def _make_key(self, query: str, k: int, max_hops: int) -> str:
        return hashlib.md5(f"{query}:{k}:{max_hops}".encode()).hexdigest()

    def get(self, query: str, k: int, max_hops: int) -> Optional[list[RetrievedSection]]:
        key = self._make_key(query, k, max_hops)
        if key in self.cache:
            timestamp, results = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.cache.move_to_end(key)
                return results
            else:
                del self.cache[key]
        return None

    def put(self, query: str, k: int, max_hops: int, results: list[RetrievedSection]):
        key = self._make_key(query, k, max_hops)
        self.cache[key] = (time.time(), results)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()


GRAPH_QUERY_PATTERNS = [
    r"\bwhere\s+is\b",
    r"\bwhich\s+section\b",
    r"\bwhat\s+refers\s+to\b",
    r"\bdependencies?\b",
    r"\bdepends\s+on\b",
    r"\blinks?\s+to\b",
    r"\breferences?\b",
]


@dataclass
class RetrievedSection:
    section_id: str
    content: str
    score: float
    source: str
    distance: int = 0
    doc_id: str = ""


class QueryRouter:
    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in GRAPH_QUERY_PATTERNS]

    def should_prioritize_graph(self, query: str) -> bool:
        for pattern in self.patterns:
            if pattern.search(query):
                return True
        return False


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        graph: Neo4jGraph,
        embedding_engine: EmbeddingEngine,
        use_cache: bool = True,
    ):
        self.vector_store = vector_store
        self.graph = graph
        self.embedding_engine = embedding_engine
        self.bm25: Optional[BM25Okapi] = None
        self._bm25_documents: list[tuple[str, str]] = []
        self._result_cache = QueryResultCache() if use_cache else None

    def _parse_key(self, key: str) -> tuple[str, str]:
        if "/" in key:
            doc_id, section_id = key.split("/", 1)
            return doc_id, section_id
        return "", key

    def build_bm25_index(self):
        all_keys = list(self.vector_store.contents.keys())
        self._bm25_documents = []
        for key in all_keys:
            content = self.vector_store.get_content(key)
            if content:
                self._bm25_documents.append((key, content))

        if not self._bm25_documents:
            self.bm25 = None
            return

        texts = [doc for _, doc in self._bm25_documents]
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)
        print(f"Built BM25 index with {len(self._bm25_documents)} documents")

    def _bm25_search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if self.bm25 is None:
            self.build_bm25_index()

        if self.bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                key, _ = self._bm25_documents[idx]
                results.append((key, float(scores[idx])))

        return results

    def retrieve(
        self,
        query: str,
        k: int = 10,
        max_hops: int = 2,
        use_graph_expansion: bool = True,
        use_cross_encoder: bool = True,
        use_bm25: bool = True,
    ) -> list[RetrievedSection]:
        if self._result_cache is not None:
            cached = self._result_cache.get(query, k, max_hops)
            if cached is not None:
                return cached[:k]

        query_embedding = self.embedding_engine.embed_text(query)

        vector_ids, vector_scores = self.vector_store.search(query_embedding, k=k * 4)

        results = []
        for key, score in zip(vector_ids, vector_scores):
            content = self.vector_store.get_content(key)
            doc_id, section_id = self._parse_key(key)
            if content:
                results.append(RetrievedSection(
                    section_id=key,
                    content=content,
                    score=score,
                    source="vector",
                    doc_id=doc_id
                ))

        if use_bm25:
            results = self._merge_bm25_results(query, results, k=k * 2)

        if use_cross_encoder and len(results) > 1:
            results = self._rerank_with_cross_encoder(query, results, k=k * 2)

        if use_graph_expansion:
            results = self._expand_with_graph(results, max_hops)

        results = self._rerank_results(results)

        final_results = results[:k]

        if self._result_cache is not None:
            self._result_cache.put(query, k, max_hops, final_results)

        return final_results

    def _merge_bm25_results(
        self, query: str, results: list[RetrievedSection], k: int = 20
    ) -> list[RetrievedSection]:
        bm25_results = self._bm25_search(query, k=k)
        if not bm25_results:
            return results

        section_id_to_result = {r.section_id: r for r in results}
        max_bm25_score = max(score for _, score in bm25_results) if bm25_results else 1.0

        for bm25_key, bm25_score in bm25_results:
            normalized_bm25 = bm25_score / max_bm25_score
            if bm25_key in section_id_to_result:
                section_id_to_result[bm25_key].score += normalized_bm25 * 0.5
                if section_id_to_result[bm25_key].source == "vector":
                    section_id_to_result[bm25_key].source = "vector+bm25"
            else:
                content = self.vector_store.get_content(bm25_key)
                if content:
                    doc_id, _ = self._parse_key(bm25_key)
                    results.append(RetrievedSection(
                        section_id=bm25_key,
                        content=content,
                        score=normalized_bm25 * 0.5,
                        source="bm25",
                        doc_id=doc_id
                    ))

        return results

    def _rerank_with_cross_encoder(
        self, query: str, results: list[RetrievedSection], k: int = 20
    ) -> list[RetrievedSection]:
        candidates = [(r.section_id, r.content) for r in results[:k]]

        reranked = self.embedding_engine.rerank(query, candidates, top_k=k)

        section_id_to_result = {r.section_id: r for r in results}
        new_results = []
        for section_id, score in reranked:
            if section_id in section_id_to_result:
                result = section_id_to_result[section_id]
                result.score = score
                result.source = "reranked"
                new_results.append(result)

        for r in results:
            if r.section_id not in [nr.section_id for nr in new_results]:
                r.source = "vector"
                new_results.append(r)

        return new_results

    def _expand_with_graph(
        self, initial_results: list[RetrievedSection], max_hops: int
    ) -> list[RetrievedSection]:
        expanded = {r.section_id: r for r in initial_results}
        
        for result in initial_results:
            doc_id = result.doc_id or self._parse_key(result.section_id)[0]
            for hop in range(1, max_hops + 1):
                neighbors = self.graph.get_neighbors(result.section_id, doc_id=doc_id, depth=hop)
                
                for neighbor in neighbors:
                    neighbor_id = neighbor["section_id"]
                    neighbor_doc_id = neighbor.get("doc_id", doc_id)
                    key = self.vector_store._make_key(neighbor_doc_id, neighbor_id)
                    if key not in expanded:
                        content = self.vector_store.get_content(key)
                        if content:
                            distance = neighbor["distance"]
                            hop_penalty = 1.0 / (distance + 1)
                            expanded[key] = RetrievedSection(
                                section_id=key,
                                content=content,
                                score=result.score * hop_penalty,
                                source=f"graph_hop_{distance}",
                                distance=distance,
                                doc_id=neighbor_doc_id
                            )
        
        return list(expanded.values())

    def _rerank_results(self, results: list[RetrievedSection]) -> list[RetrievedSection]:
        for result in results:
            ref_density = self.graph.get_reference_density(result.section_id)
            density_boost = 1.0 + (ref_density * 0.1)
            result.score *= density_boost
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def iterative_retrieve(
        self,
        query: str,
        max_iterations: int = 3,
        initial_k: int = 5,
    ) -> tuple[list[RetrievedSection], str]:
        context = ""
        all_results = []
        
        for iteration in range(max_iterations):
            current_query = f"{query} {context}".strip()
            
            results = self.retrieve(current_query, k=initial_k, use_graph_expansion=True)
            all_results.extend(results)
            
            top_context = "\n".join([
                f"[{r.section_id}]: {r.content[:300]}..."
                for r in results[:3]
            ])
            
            context = top_context
            
            if iteration >= max_iterations - 1:
                break
        
        seen = {}
        for result in all_results:
            if result.section_id not in seen:
                seen[result.section_id] = result
            else:
                seen[result.section_id].score = max(seen[result.section_id].score, result.score)
        
        final_results = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        
        return final_results, context


class MultiHopRetriever:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def retrieve_with_hops(
        self,
        query: str,
        target_hops: int = 2,
    ) -> list[RetrievedSection]:
        router = QueryRouter()
        
        if router.should_prioritize_graph(query):
            return self._graph_first_retrieve(query, target_hops)
        else:
            return self.retriever.retrieve(query, k=10, max_hops=target_hops)

    def _graph_first_retrieve(
        self, query: str, target_hops: int
    ) -> list[RetrievedSection]:
        query_embedding = self.retriever.embedding_engine.embed_text(query)
        vector_ids, vector_scores = self.retriever.vector_store.search(query_embedding, k=5)
        
        graph_results = []
        for section_id in vector_ids:
            neighbors = self.retriever.graph.get_neighbors(section_id, depth=target_hops)
            for neighbor in neighbors:
                content = self.retriever.vector_store.get_content(neighbor["section_id"])
                if content:
                    graph_results.append(RetrievedSection(
                        section_id=neighbor["section_id"],
                        content=content,
                        score=1.0 / (neighbor["distance"] + 1),
                        source="graph_traversal",
                        distance=neighbor["distance"]
                    ))
        
        seen = {}
        for result in graph_results:
            if result.section_id not in seen:
                seen[result.section_id] = result
            else:
                seen[result.section_id].score = max(seen[result.section_id].score, result.score)
        
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)
