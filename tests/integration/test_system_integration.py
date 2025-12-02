# Integration tests for system components

from unittest.mock import patch


class TestSystemIntegration:
    """Integration tests for system components."""

    def test_document_manager_creation(self):
        """Test that document manager can be created."""
        from src.core.document_manager import DocumentManager

        with patch("src.core.document_manager.SessionLocal") as mock_session:
            manager = DocumentManager(mock_session)
            assert manager is not None
            assert hasattr(manager, "process_document")
            assert hasattr(manager, "delete_document")

    def test_ai_enrichment_service_creation(self):
        """Test that AI enrichment service can be created."""
        from src.ai.enrichment import AIEnrichmentService

        with (
            patch("src.ai.enrichment.SessionLocal"),
            patch("src.ai.enrichment.TagManager"),
            patch("src.ai.enrichment.CategoryManager"),
        ):
            service = AIEnrichmentService()
            assert service is not None
            assert hasattr(service, "enrich_document")

    def test_hierarchical_chunker_creation(self):
        """Test that hierarchical chunker can be created."""
        from src.ai.pipeline.hierarchical_chunker import HierarchicalChunker

        chunker = HierarchicalChunker()
        assert chunker is not None
        assert hasattr(chunker, "chunk_document")

    def test_ai_tag_suggester_creation(self):
        """Test that AI tag suggester can be created."""
        from src.ai.tag_suggester import AITagSuggester

        suggester = AITagSuggester()
        assert suggester is not None
        assert hasattr(suggester, "suggest_tags")

    def test_advanced_search_creation(self):
        """Test that advanced search can be created."""
        from src.core.advanced_search import HybridSearchEngine

        with (
            patch("src.core.advanced_search.SessionLocal"),
            patch("src.core.advanced_search.KnowledgeGraph"),
            patch("src.core.embeddings.get_embedding_model"),
        ):
            search_engine = HybridSearchEngine()
            assert search_engine is not None
            assert hasattr(search_engine, "search")
            assert hasattr(search_engine, "get_search_facets")

    def test_retrieval_system_creation(self):
        """Test that retrieval system can be created."""
        from src.core.retrieval import DatabaseRetriever

        with (
            patch("src.core.retrieval.SessionLocal"),
            patch("src.core.retrieval.KnowledgeGraph"),
            patch("src.core.embeddings.get_embedding_model"),
        ):
            retriever = DatabaseRetriever()
            assert retriever is not None
            assert hasattr(retriever, "hybrid_search")

    def test_embeddings_creation(self):
        """Test that embeddings can be created."""
        from src.core.embeddings import create_embeddings

        with patch(
            "src.core.embeddings._encode_with_sentence_transformers"
        ) as mock_encode:
            mock_encode.return_value = [0.1, 0.2, 0.3]

            embeddings, model = create_embeddings(["test text"])
            assert embeddings is not None
            assert len(embeddings) == 1
            assert model is not None

    def test_knowledge_graph_creation(self):
        """Test that knowledge graph can be created."""
        from src.core.knowledge_graph import KnowledgeGraph

        with patch("src.core.knowledge_graph.Session") as mock_session:
            kg = KnowledgeGraph(mock_session)
            assert kg is not None
            assert hasattr(kg, "build_relationships_from_cooccurrence")
            assert hasattr(kg, "expand_query_context")
