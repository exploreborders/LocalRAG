"""
Unit tests for retrieval system components.

Tests DatabaseRetriever and RAGPipelineDB functionality.
"""

from unittest.mock import MagicMock, patch

from src.core.retrieval import (
    DatabaseRetriever,
    RAGPipelineDB,
    format_answer_db,
    format_results_db,
)


class TestDatabaseRetriever:
    """Test the DatabaseRetriever class."""

    @patch("src.core.retrieval.Elasticsearch")
    @patch("src.core.retrieval.SessionLocal")
    def test_init(self, mock_session, mock_es):
        """Test DatabaseRetriever initialization."""
        mock_es_instance = MagicMock()
        mock_es.return_value = mock_es_instance

        retriever = DatabaseRetriever()

        assert retriever is not None
        assert retriever.model_name == "embeddinggemma:latest"
        assert retriever.backend == "ollama"
        assert retriever.use_batch_processing is False
        assert retriever.hybrid_alpha == 0.7

    @patch("src.core.retrieval.Elasticsearch")
    @patch("src.core.retrieval.SessionLocal")
    def test_retrieve_with_knowledge_graph(self, mock_session, mock_es):
        """Test knowledge graph enhanced retrieval."""
        mock_es_instance = MagicMock()
        mock_es.return_value = mock_es_instance

        retriever = DatabaseRetriever()

        # Mock the method calls
        with patch.object(retriever, "_build_enhanced_context") as mock_build_context:
            mock_build_context.return_value = {
                "query": "test query",
                "filters": {},
                "kg_context": "knowledge graph context",
            }

            with patch.object(retriever, "hybrid_search") as mock_hybrid_search:
                mock_hybrid_search.return_value = {
                    "results": [{"content": "test result"}],
                    "total": 1,
                }

                result = retriever.retrieve_with_knowledge_graph("test query")

                assert result is not None
                mock_build_context.assert_called_once_with("test query", {})
                mock_hybrid_search.assert_called_once()

    @patch("src.core.retrieval.SessionLocal")
    def test_hybrid_search(self, mock_session):
        """Test hybrid search functionality."""
        retriever = DatabaseRetriever()

        # Mock embeddings creation
        with patch("src.core.embeddings.create_embeddings") as mock_create_embeddings:
            mock_create_embeddings.return_value = ([MagicMock()], "ollama")

            # Mock ES client and search
            mock_es_client = MagicMock()
            mock_es_client.search.return_value = {
                "hits": {
                    "hits": [
                        {
                            "_id": "chunk_123",
                            "_source": {
                                "content": "test content",
                                "metadata": {"title": "test"},
                                "document_id": 1,
                            },
                            "_score": 0.8,
                        }
                    ],
                    "total": {"value": 1},
                }
            }

            # Mock document query
            mock_document = MagicMock()
            mock_document.id = 1
            mock_document.filename = "test.pdf"
            mock_document.tags = []
            mock_document.categories = []
            mock_session.query.return_value.filter.return_value.first.return_value = mock_document

            with patch.object(retriever, "_get_es_client", return_value=mock_es_client):
                with patch.object(retriever, "_rerank_results") as mock_rerank:
                    mock_rerank.return_value = [{"content": "reranked content"}]

                    result = retriever.hybrid_search("test query")

                    assert isinstance(result, list)
                    assert len(result) == 1
                    assert result[0]["content"] == "reranked content"
                    mock_rerank.assert_called_once()


class TestRAGPipelineDB:
    """Test the RAGPipelineDB class."""

    @patch("src.core.retrieval.OllamaLLM")
    @patch("src.core.retrieval.SessionLocal")
    def test_init(self, mock_session, mock_llm):
        """Test RAGPipelineDB initialization."""
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance

        pipeline = RAGPipelineDB()

        assert pipeline is not None
        assert pipeline.llm == mock_llm_instance

    @patch("src.core.retrieval.OllamaLLM")
    @patch("src.core.retrieval.SessionLocal")
    def test_query(self, mock_session, mock_llm):
        """Test RAG pipeline query."""
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance

        pipeline = RAGPipelineDB()

        # Mock database query
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        mock_chunk = MagicMock()
        mock_chunk.content = "test content"
        mock_chunk.metadata = {"title": "test"}
        mock_session_instance.query.return_value.filter.return_value.limit.return_value.all.return_value = [
            mock_chunk
        ]

        with patch.object(pipeline, "_generate_answer") as mock_generate:
            mock_generate.return_value = "Generated answer"

            with patch.object(pipeline, "_format_sources") as mock_format:
                mock_format.return_value = [{"title": "test", "content": "test content"}]

                result = pipeline.query("test question")

                assert result is not None
                assert "answer" in result
                assert "sources" in result

    def test_detect_language(self):
        """Test language detection."""
        from src.core.retrieval import RAGPipelineDB

        pipeline = RAGPipelineDB()

        # Mock the detect function
        with patch("src.core.retrieval.detect") as mock_detect:
            mock_detect.return_value = "en"

            result = pipeline._detect_language("Hello world")
            assert result == "en"
            mock_detect.assert_called_once_with("Hello world")

    def test_generate_cache_key(self):
        """Test cache key generation."""
        from src.core.retrieval import RAGPipelineDB

        pipeline = RAGPipelineDB()

        key1 = pipeline._generate_cache_key("test query", 5, {}, "en")
        key2 = pipeline._generate_cache_key("test query", 5, {}, "en")

        assert key1 == key2  # Same inputs should generate same key
        assert isinstance(key1, str)
        assert len(key1) > 0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_format_results_db(self):
        """Test database results formatting."""
        results = [
            {
                "document": {"filename": "doc1.pdf", "page_content": "content1"},
                "score": 0.8,
            },
            {
                "document": {"filename": "doc2.pdf", "page_content": "content2"},
                "score": 0.7,
            },
        ]

        formatted = format_results_db(results)

        assert isinstance(formatted, str)
        assert "doc1.pdf" in formatted
        assert "content1" in formatted
        assert "doc2.pdf" in formatted
        assert "content2" in formatted

    def test_format_answer_db(self):
        """Test answer formatting."""
        answer = "This is a test answer."

        formatted = format_answer_db(answer)

        assert isinstance(formatted, str)
        assert formatted == answer
