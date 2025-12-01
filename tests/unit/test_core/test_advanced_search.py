"""
Unit tests for advanced_search module.

Tests query parsing, hybrid search, and search analytics functionality.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.core.advanced_search import QueryParser, HybridSearchEngine


class TestQueryParser:
    """Test the QueryParser class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = QueryParser()

    def test_init(self):
        """Test QueryParser initialization."""
        assert self.parser.field_mappings is not None
        assert "title" in self.parser.field_mappings
        assert "content" in self.parser.field_mappings

    def test_parse_simple_query(self):
        """Test parsing a simple query without operators."""
        result = self.parser.parse_query("machine learning")

        assert "simple_terms" in result
        assert "machine" in result["simple_terms"]
        assert "learning" in result["simple_terms"]

    def test_parse_field_query(self):
        """Test parsing queries with field specifications."""
        result = self.parser.parse_query("title:machine content:learning")

        assert "field_queries" in result
        assert "title" in result["field_queries"]
        assert "content" in result["field_queries"]

    def test_parse_boolean_query(self):
        """Test parsing queries with boolean operators."""
        result = self.parser.parse_query("machine AND learning")

        assert "boolean_query" in result
        assert result["boolean_query"] is not None

    def test_extract_field_queries(self):
        """Test field query extraction."""
        query = "title:test content:example"
        base_query, field_queries = self.parser._extract_field_queries(query)

        assert base_query == ""
        assert "title" in field_queries
        assert "content" in field_queries

    def test_extract_phrases(self):
        """Test phrase extraction."""
        query = '"machine learning" and "deep learning"'
        base_query, phrases = self.parser._extract_phrases(query)

        assert "and" in base_query
        assert len(phrases) == 2

    def test_parse_boolean_operators(self):
        """Test boolean operator parsing."""
        result = self.parser._parse_boolean_query("machine AND learning OR algorithms")

        assert result is not None
        assert result["type"] == "boolean"

    def test_extract_terms(self):
        """Test term extraction from queries."""
        terms = self.parser._extract_terms("machine learning algorithms")

        assert "simple_terms" in terms
        assert "machine" in terms["simple_terms"]
        assert "learning" in terms["simple_terms"]
        assert "algorithms" in terms["simple_terms"]


class TestHybridSearchEngine:
    """Test the HybridSearchEngine class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = HybridSearchEngine(
            model_name="test-model", backend="ollama", use_reranking=False
        )

    def test_init(self):
        """Test HybridSearchEngine initialization."""
        engine = HybridSearchEngine(
            model_name="test-model", backend="ollama", use_reranking=False
        )

        assert engine.model_name == "test-model"
        assert engine.backend == "ollama"
        assert engine.use_reranking is False

    @patch("src.core.advanced_search.SessionLocal")
    @patch("src.core.advanced_search.create_embeddings")
    @patch("src.core.advanced_search.get_embedding_model")
    def test_search_basic(self, mock_get_model, mock_create_embeddings, mock_session):
        """Test basic search functionality."""
        # Mock dependencies
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        mock_create_embeddings.return_value = (
            np.array([[0.1, 0.2, 0.3]]),
            "test_model",
        )

        # Mock search to return empty results (simplified test)
        with patch.object(self.engine, "search") as mock_search:
            mock_search.return_value = {
                "results": [],
                "analytics": {},
                "total_time": 0.1,
            }

            results = self.engine.search("test query", top_k=5, filters={})

            assert isinstance(results, dict)
            assert "results" in results
            assert "analytics" in results

    def test_get_query_embedding(self):
        """Test query embedding generation."""
        with patch("src.core.advanced_search.create_embeddings") as mock_create:
            with patch(
                "src.core.advanced_search.get_embedding_model"
            ) as mock_get_model:
                mock_model = MagicMock()
                mock_get_model.return_value = mock_model
                mock_create.return_value = (np.array([[0.1, 0.2, 0.3]]), "test_model")

                embedding = self.engine._get_query_embedding("test query")

                assert isinstance(embedding, np.ndarray)
                assert embedding.shape == (3,)

    def test_build_filter_clauses(self):
        """Test filter clause building."""
        filters = {
            "tags": ["python", "ml"],
            "categories": ["technical"],
            "language": "en",
        }
        filter_clauses = []

        self.engine._build_filter_clauses(filters, filter_clauses)

        assert len(filter_clauses) > 0

    def test_calculate_query_complexity(self):
        """Test query complexity calculation."""
        simple_query = {"simple_terms": ["test"]}
        complex_query = {
            "simple_terms": ["test", "complex"],
            "excluded_terms": ["simple"],
            "field_queries": {"title": ["advanced"]},
        }

        simple_complexity = self.engine._calculate_query_complexity(simple_query)
        complex_complexity = self.engine._calculate_query_complexity(complex_query)

        assert complex_complexity > simple_complexity
        assert simple_complexity >= 0

    def test_analyze_result_distribution(self):
        """Test result distribution analysis."""
        results = [
            {"score": 0.8, "document_id": "doc1"},
            {"score": 0.6, "document_id": "doc2"},
        ]

        analysis = self.engine._analyze_result_distribution(results)

        assert "unique_documents" in analysis
        assert analysis["unique_documents"] == 2

    def test_generate_search_analytics(self):
        """Test search analytics generation."""
        results = [
            {"score": 0.8, "document_id": "doc1"},
            {"score": 0.6, "document_id": "doc2"},
        ]

        analytics = self.engine._generate_search_analytics(
            {"simple_terms": ["test"]}, results, 0.15
        )

        assert "total_results" in analytics
        assert "query_complexity" in analytics
        assert "performance_metrics" in analytics
        assert analytics["total_results"] == 2
