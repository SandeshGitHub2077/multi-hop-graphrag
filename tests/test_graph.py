import pytest
from unittest.mock import Mock, patch, MagicMock
from graph import Neo4jGraph


class MockNeo4jDriver:
    def __init__(self):
        self.session_context = MagicMock()
        
    def session(self):
        return self.session_context
    
    def close(self):
        pass


class TestNeo4jGraph:
    @pytest.fixture
    def mock_driver(self, monkeypatch):
        mock_driver = MockNeo4jDriver()
        monkeypatch.setattr("graph.neo4j_graph.GraphDatabase.driver", lambda *args, **kwargs: mock_driver)
        return mock_driver

    def test_initialization(self):
        graph = Neo4jGraph("bolt://localhost:7687", "user", "pass")
        assert graph.uri == "bolt://localhost:7687"
        assert graph.username == "user"
        assert graph.password == "pass"

    def test_connect(self, mock_driver):
        graph = Neo4jGraph()
        graph.connect()
        assert graph.driver is not None
        graph.close()

    def test_ensure_constraints(self, mock_driver):
        mock_session = MagicMock()
        mock_driver.session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session_context.__exit__ = MagicMock(return_value=False)
        
        graph = Neo4jGraph()
        graph.connect()
        graph.ensure_constraints()
        
        assert mock_session.run.call_count == 2

    def test_upsert_section(self, mock_driver):
        mock_session = MagicMock()
        mock_driver.session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session_context.__exit__ = MagicMock(return_value=False)
        
        graph = Neo4jGraph()
        graph.connect()
        graph.upsert_section("1.1", "Test content", "doc1", {"refs": ["1.2"]})
        
        mock_session.run.assert_called_once()

    def test_upsert_document(self, mock_driver):
        mock_session = MagicMock()
        mock_driver.session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session_context.__exit__ = MagicMock(return_value=False)
        
        graph = Neo4jGraph()
        graph.connect()
        graph.upsert_document("doc1", "pdf")
        
        mock_session.run.assert_called_once()

    def test_create_reference_relationship(self, mock_driver):
        mock_session = MagicMock()
        mock_driver.session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session_context.__exit__ = MagicMock(return_value=False)
        
        graph = Neo4jGraph()
        graph.connect()
        graph.create_reference_relationship("1.1", "1.2")
        
        mock_session.run.assert_called_once()

    def test_get_section(self, mock_driver):
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {
            "section_id": "1.1",
            "content": "test content",
            "doc_id": "doc1"
        }
        mock_session.run.return_value = mock_result
        mock_driver.session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session_context.__exit__ = MagicMock(return_value=False)
        
        graph = Neo4jGraph()
        graph.connect()
        result = graph.get_section("1.1")
        
        assert result is not None
        assert result["section_id"] == "1.1"

    def test_get_reference_density(self, mock_driver):
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {"density": 5}
        mock_session.run.return_value = mock_result
        mock_driver.session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session_context.__exit__ = MagicMock(return_value=False)
        
        graph = Neo4jGraph()
        graph.connect()
        density = graph.get_reference_density("1.1")
        
        assert density == 5

    def test_get_all_sections(self, mock_driver):
        mock_session = MagicMock()
        mock_result = [
            {"section_id": "1.1", "content": "content1", "doc_id": "doc1"},
            {"section_id": "1.2", "content": "content2", "doc_id": "doc1"},
        ]
        mock_session.run.return_value = mock_result
        mock_driver.session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session_context.__exit__ = MagicMock(return_value=False)
        
        graph = Neo4jGraph()
        graph.connect()
        sections = graph.get_all_sections()
        
        assert len(sections) == 2

    def test_get_neighbors(self, mock_driver):
        mock_session = MagicMock()
        mock_result = [
            {"section_id": "1.2", "content": "content2", "distance": 1},
            {"section_id": "1.3", "content": "content3", "distance": 2},
        ]
        mock_session.run.return_value = mock_result
        mock_driver.session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session_context.__exit__ = MagicMock(return_value=False)
        
        graph = Neo4jGraph()
        graph.connect()
        neighbors = graph.get_neighbors("1.1", depth=2)
        
        assert len(neighbors) == 2
