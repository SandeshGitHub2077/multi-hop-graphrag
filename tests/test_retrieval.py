import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from retrieval import QueryRouter, HybridRetriever, RetrievedSection


class TestQueryRouter:
    @pytest.fixture
    def router(self):
        return QueryRouter()

    def test_initialization(self, router):
        assert len(router.patterns) > 0

    def test_should_prioritize_graph_where_is(self, router):
        query = "Where is section 1.1?"
        assert router.should_prioritize_graph(query) == True

    def test_should_prioritize_graph_which_section(self, router):
        query = "Which section covers food safety?"
        assert router.should_prioritize_graph(query) == True

    def test_should_prioritize_graph_dependencies(self, router):
        router = QueryRouter()
        query = "What are the dependencies of section 1.1?"
        assert router.should_prioritize_graph(query) == True

    def test_should_prioritize_graph_references(self, router):
        query = "What references section 1.1?"
        assert router.should_prioritize_graph(query) == True

    def test_should_not_prioritize_graph_regular_query(self, router):
        query = "What are the food safety regulations?"
        assert router.should_prioritize_graph(query) == False

    def test_case_insensitive(self, router):
        query = "WHERE IS the section?"
        assert router.should_prioritize_graph(query) == True


class TestHybridRetriever:
    @pytest.fixture
    def mock_vector_store(self):
        store = MagicMock()
        store.search.return_value = (["1.1", "1.2"], [0.9, 0.8])
        store.get_content.side_effect = lambda sid: f"Content for {sid}"
        return store

    @pytest.fixture
    def mock_graph(self):
        graph = MagicMock()
        graph.get_neighbors.return_value = [
            {"section_id": "1.2", "content": "Content for 1.2", "distance": 1}
        ]
        graph.get_reference_density.return_value = 3
        return graph

    @pytest.fixture
    def mock_embedding_engine(self):
        engine = MagicMock()
        engine.embed_text.return_value = np.random.rand(768).astype(np.float32)
        return engine

    @pytest.fixture
    def retriever(self, mock_vector_store, mock_graph, mock_embedding_engine):
        return HybridRetriever(mock_vector_store, mock_graph, mock_embedding_engine)

    def test_initialization(self, retriever):
        assert retriever.vector_store is not None
        assert retriever.graph is not None
        assert retriever.embedding_engine is not None

    def test_retrieve_basic(self, retriever, mock_vector_store, mock_graph):
        results = retriever.retrieve("test query", k=5)
        
        assert len(results) > 0
        assert all(isinstance(r, RetrievedSection) for r in results)
        mock_vector_store.search.assert_called_once()

    def test_retrieve_with_graph_expansion(self, retriever, mock_vector_store, mock_graph):
        results = retriever.retrieve("test query", k=5, use_graph_expansion=True)
        
        assert len(results) > 0
        mock_graph.get_neighbors.assert_called()

    def test_rerank_uses_reference_density(self, retriever, mock_graph):
        results = [
            RetrievedSection("1.1", "content", 0.9, "vector"),
            RetrievedSection("1.2", "content", 0.8, "graph"),
        ]
        
        mock_graph.get_reference_density.side_effect = [5, 1]
        
        reranked = retriever._rerank_results(results)
        
        assert reranked[0].score >= reranked[1].score

    def test_expand_with_graph(self, retriever, mock_graph):
        initial = [RetrievedSection("1.1", "Content 1.1", 0.9, "vector")]
        mock_graph.get_neighbors.return_value = [
            {"section_id": "1.2", "content": "Content 1.2", "distance": 1}
        ]
        
        expanded = retriever._expand_with_graph(initial, max_hops=1)
        
        assert len(expanded) >= 1


class TestRetrievedSection:
    def test_dataclass_creation(self):
        section = RetrievedSection(
            section_id="1.1",
            content="Test content",
            score=0.95,
            source="vector",
            distance=0
        )
        
        assert section.section_id == "1.1"
        assert section.content == "Test content"
        assert section.score == 0.95
        assert section.source == "vector"
        assert section.distance == 0

    def test_default_distance(self):
        section = RetrievedSection(
            section_id="1.1",
            content="Test content",
            score=0.95,
            source="vector"
        )
        
        assert section.distance == 0


class TestMultiHopRetrieval:
    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            RetrievedSection("1.1", "Content", 0.9, "vector")
        ]
        return retriever

    @pytest.fixture
    def multi_hop(self, mock_retriever):
        from retrieval import MultiHopRetriever
        return MultiHopRetriever(mock_retriever)

    def test_retrieve_with_hops(self, multi_hop, mock_retriever):
        results = multi_hop.retrieve_with_hops("test query", target_hops=2)
        
        assert len(results) > 0
