import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import Optional


class VectorStore:
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.section_ids: list[str] = []
        self.contents: dict[str, dict] = {}

    def _make_key(self, doc_id: str, section_id: str) -> str:
        return f"{doc_id}/{section_id}"

    def build_index(self, embeddings: np.ndarray, section_ids: list[str], doc_ids: list[str], contents: dict[str, str]):
        if embeddings.shape[0] != len(section_ids):
            raise ValueError("Embeddings and section_ids length mismatch")
        
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        self.section_ids = [self._make_key(doc_id, sid) for doc_id, sid in zip(doc_ids, section_ids)]

        if isinstance(contents, list):
            self.contents = {self._make_key(doc_id, sid): content for doc_id, sid, content in zip(doc_ids, section_ids, contents)}
        else:
            self.contents = {self._make_key(doc_id, sid): contents.get(sid, "") or contents.get(self._make_key(doc_id, sid), "") for doc_id, sid in zip(doc_ids, section_ids)}

    def add_embeddings(self, embeddings: np.ndarray, section_ids: list[str], doc_ids: list[str], contents):
        if self.index is None:
            self.build_index(embeddings, section_ids, doc_ids, contents)
            return

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        new_keys = [self._make_key(doc_id, sid) for doc_id, sid in zip(doc_ids, section_ids)]
        self.section_ids.extend(new_keys)

        if isinstance(contents, list):
            new_contents = {self._make_key(doc_id, sid): content for doc_id, sid, content in zip(doc_ids, section_ids, contents)}
        else:
            new_contents = {self._make_key(doc_id, sid): contents.get(sid, "") or contents.get(self._make_key(doc_id, sid), "") for doc_id, sid in zip(doc_ids, section_ids)}
        self.contents.update(new_contents)
        
        new_contents = {self._make_key(doc_id, sid): content for doc_id, sid, content in zip(doc_ids, section_ids, contents)}
        self.contents.update(new_contents)

    def get_all_embeddings(self) -> np.ndarray:
        if self.index is None:
            raise ValueError("Index not built")
        return self.index.reconstruct_n(0, self.index.ntotal)

    def search(self, query_embedding: np.ndarray, k: int = 10) -> tuple[list[str], list[float]]:
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        distances, indices = self.index.search(query, k)
        
        retrieved_ids = [self.section_ids[i] for i in indices[0] if i < len(self.section_ids)]
        retrieved_scores = distances[0].tolist()
        
        return retrieved_ids, retrieved_scores

    def get_content(self, key: str) -> Optional[str]:
        return self.contents.get(key)

    def get_content_by_ids(self, doc_id: str, section_id: str) -> Optional[str]:
        return self.contents.get(self._make_key(doc_id, section_id))

    def save(self, path: str):
        path = Path(path)
        faiss.write_index(self.index, str(path / "faiss.index"))
        
        metadata = {
            "section_ids": self.section_ids,
            "contents": self.contents,
            "dimension": self.dimension,
        }
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def load(self, path: str):
        path = Path(path)
        self.index = faiss.read_index(str(path / "faiss.index"))
        
        with open(path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        self.section_ids = metadata["section_ids"]
        self.contents = metadata["contents"]
        self.dimension = metadata["dimension"]
