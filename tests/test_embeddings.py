import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from embeddings import EmbeddingEngine, VectorStore


class TestEmbeddingEngine:
    @pytest.fixture
    def engine(self):
        return EmbeddingEngine("BAAI/bge-base-en-v1.5")

    def test_initialization(self, engine):
        assert engine.model_name == "BAAI/bge-base-en-v1.5"
        assert engine.model is None

    def test_augment_with_references_empty(self, engine):
        text = "Some content"
        result = engine._augment_with_references(text, [], {})
        
        assert result == text

    def test_augment_with_references_known(self, engine):
        text = "See § 1.1 for details"
        section_map = {"1.1": "This is the referenced content about food safety regulations"}
        
        result = engine._augment_with_references(text, ["1.1"], section_map)
        
        assert "1.1" in result
        assert "This section references" in result or "§ 1.1" in result

    def test_augment_with_references_unknown(self, engine):
        text = "See § 999.99 for details"
        
        result = engine._augment_with_references(text, ["999.99"], {})
        
        assert "999.99" in result


class TestVectorStore:
    @pytest.fixture
    def store(self):
        return VectorStore(dimension=768)

    @pytest.fixture
    def sample_embeddings(self):
        np.random.seed(42)
        return np.random.rand(5, 768).astype(np.float32)

    @pytest.fixture
    def sample_section_ids(self):
        return ["1.1", "1.2", "1.3", "2.1", "2.2"]

    @pytest.fixture
    def sample_contents(self):
        return {
            "1.1": "Content for section 1.1",
            "1.2": "Content for section 1.2",
            "1.3": "Content for section 1.3",
            "2.1": "Content for section 2.1",
            "2.2": "Content for section 2.2",
        }

    def test_initialization(self, store):
        assert store.dimension == 768
        assert store.index is None
        assert store.section_ids == []

    def test_build_index(self, store, sample_embeddings, sample_section_ids, sample_contents):
        doc_ids = ["doc1"] * 3 + ["doc2"] * 2
        store.build_index(sample_embeddings, sample_section_ids, doc_ids, sample_contents)

        assert store.index is not None
        assert store.index.ntotal == 5
        assert len(store.section_ids) == 5
        assert len(store.contents) == 5

    def test_build_index_mismatch(self, store, sample_embeddings, sample_contents):
        section_ids = ["1.1", "1.2"]
        doc_ids = ["doc1"] * 2

        with pytest.raises(ValueError, match="length mismatch"):
            store.build_index(sample_embeddings, section_ids, doc_ids, sample_contents)

    def test_search(self, store, sample_embeddings, sample_section_ids, sample_contents):
        doc_ids = ["doc1"] * 3 + ["doc2"] * 2
        store.build_index(sample_embeddings, sample_section_ids, doc_ids, sample_contents)

        query = np.random.rand(768).astype(np.float32)
        ids, scores = store.search(query, k=3)

        assert len(ids) <= 3
        assert len(scores) <= 3
        assert all(isinstance(sid, str) for sid in ids)
        assert all(isinstance(s, float) for s in scores)

    def test_search_before_build(self, store):
        query = np.random.rand(768).astype(np.float32)
        
        with pytest.raises(ValueError, match="Index not built"):
            store.search(query)

    def test_get_content(self, store, sample_embeddings, sample_section_ids, sample_contents):
        doc_ids = ["doc1"] * 3 + ["doc2"] * 2
        store.build_index(sample_embeddings, sample_section_ids, doc_ids, sample_contents)

        assert store.get_content("doc1/1.1") == "Content for section 1.1"
        assert store.get_content("nonexistent") is None

    def test_add_embeddings(self, store):
        np.random.seed(42)
        emb1 = np.random.rand(3, 768).astype(np.float32)
        emb2 = np.random.rand(2, 768).astype(np.float32)

        doc_ids1 = ["doc1"] * 3
        doc_ids2 = ["doc2"] * 2
        contents1 = {"doc1/1.1": "c1", "doc1/1.2": "c2", "doc1/1.3": "c3"}
        contents2 = {"doc2/2.1": "c4", "doc2/2.2": "c5"}

        store.add_embeddings(emb1, ["1.1", "1.2", "1.3"], doc_ids1, contents1)
        store.add_embeddings(emb2, ["2.1", "2.2"], doc_ids2, contents2)

        assert store.index.ntotal == 5
        assert len(store.section_ids) == 5

    def test_save_load(self, store, sample_embeddings, sample_section_ids, sample_contents, tmp_path):
        doc_ids = ["doc1"] * 3 + ["doc2"] * 2
        store.build_index(sample_embeddings, sample_section_ids, doc_ids, sample_contents)

        save_path = tmp_path / "test_index"
        save_path.mkdir()
        store.save(str(save_path))

        new_store = VectorStore()
        new_store.load(str(save_path))

        assert new_store.index is not None
        assert new_store.index.ntotal == 5

        expected = {"doc1/1.1": "Content for section 1.1", "doc1/1.2": "Content for section 1.2", "doc1/1.3": "Content for section 1.3", "doc2/2.1": "Content for section 2.1", "doc2/2.2": "Content for section 2.2"}
        assert new_store.contents == expected
        assert len(new_store.section_ids) == 5
