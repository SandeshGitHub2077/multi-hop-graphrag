import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Optional
import logging
import io
import sys
from functools import lru_cache
import hashlib
from collections import OrderedDict

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


class LRUCache:
    def __init__(self, capacity: int = 100):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: np.ndarray):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()


class EmbeddingEngine:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cache_capacity: int = 100,
    ):
        self.model_name = model_name
        self.cross_encoder_model = cross_encoder_model
        self.model: Optional[SentenceTransformer] = None
        self.cross_encoder: Optional[CrossEncoder] = None
        self._embedding_cache = LRUCache(cache_capacity)

    def load_model(self):
        import warnings
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model = SentenceTransformer(self.model_name)
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        print(f"Loaded embedding model: {self.model_name}")

    def _augment_with_references(
        self, text: str, references: list[str], section_map: dict[str, str]
    ) -> str:
        if not references:
            return text
        
        ref_descriptions = []
        for ref in references:
            if ref in section_map:
                ref_content = section_map[ref]
                ref_preview = ref_content[:200] + "..." if len(ref_content) > 200 else ref_content
                ref_descriptions.append(
                    f"This section references {ref} which discusses: {ref_preview}"
                )
            else:
                ref_descriptions.append(f"This section references {ref}")
        
        augmented = text + "\n\n" + " ".join(ref_descriptions)
        return augmented

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def embed_text(self, text: str) -> np.ndarray:
        cache_key = self._hash_text(text)
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            return cached

        if self.model is None:
            self.load_model()
        embedding = self.model.encode(text, normalize_embeddings=True)

        self._embedding_cache.put(cache_key, embedding)
        return embedding

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            self.load_model()
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings

    def embed_with_context(
        self, text: str, references: list[str], section_map: dict[str, str]
    ) -> np.ndarray:
        augmented_text = self._augment_with_references(text, references, section_map)
        return self.embed_text(augmented_text)

    def load_cross_encoder(self):
        import warnings

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.cross_encoder = CrossEncoder(self.cross_encoder_model)

        sys.stdout = old_stdout
        sys.stderr = old_stderr

        print(f"Loaded cross-encoder model: {self.cross_encoder_model}")

    def rerank(
        self, query: str, candidates: list[tuple[str, str]], top_k: int = 10
    ) -> list[tuple[str, float]]:
        if self.cross_encoder is None:
            self.load_cross_encoder()

        doc_texts = [doc for _, doc in candidates]
        pairs = [(query, doc) for doc in doc_texts]

        scores = self.cross_encoder.predict(pairs)

        results = []
        for (section_id, _), score in zip(candidates, scores):
            results.append((section_id, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
