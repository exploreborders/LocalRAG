"""
Unit tests for reprocess_documents module.

Tests document reprocessing functionality with mocked dependencies.
"""

from unittest.mock import MagicMock, patch

from src.core.reprocess_documents import reprocess_documents


class TestReprocessDocuments:
    """Test the reprocess_documents functionality."""

    @patch("src.core.reprocess_documents.SessionLocal")
    @patch("src.core.reprocess_documents.UploadProcessor")
    @patch("src.core.reprocess_documents.Document")
    def test_reprocess_documents_empty_database(
        self, mock_document_class, mock_upload_processor_class, mock_session_local
    ):
        """Test reprocessing when no documents exist."""
        # Mock empty database
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session
        mock_session.query.return_value.all.return_value = []

        # Mock upload processor
        mock_processor = MagicMock()
        mock_upload_processor_class.return_value = mock_processor

        # Call the function
        reprocess_documents()

        # Verify database was queried
        mock_session.query.assert_called_once()
        mock_session.close.assert_called_once()

    @patch("src.core.reprocess_documents.SessionLocal")
    @patch("src.core.reprocess_documents.UploadProcessor")
    @patch("src.core.reprocess_documents.Document")
    def test_reprocess_documents_with_documents_no_content(
        self, mock_document_class, mock_upload_processor_class, mock_session_local
    ):
        """Test reprocessing documents that have no stored content."""
        # Mock database with documents but no content
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        mock_doc = MagicMock()
        mock_doc.filename = "test.pdf"
        mock_doc.full_content = None  # No content
        mock_session.query.return_value.all.return_value = [mock_doc]

        # Mock upload processor
        mock_processor = MagicMock()
        mock_upload_processor_class.return_value = mock_processor

        # Call the function
        reprocess_documents()

        # Verify document was skipped
        mock_session.close.assert_called_once()

    @patch("src.core.reprocess_documents.SessionLocal")
    @patch("src.core.reprocess_documents.UploadProcessor")
    @patch("src.core.reprocess_documents.Document")
    @patch("src.core.reprocess_documents.DocumentProcessor")
    def test_reprocess_documents_successful_reprocessing(
        self,
        mock_doc_processor_class,
        mock_document_class,
        mock_upload_processor_class,
        mock_session_local,
    ):
        """Test successful document reprocessing."""
        # Mock database with document that has content
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        mock_doc = MagicMock()
        mock_doc.id = 1
        mock_doc.filename = "test.pdf"
        mock_doc.filepath = "/path/to/test.pdf"
        mock_doc.full_content = "This is test content with some chapters."
        mock_session.query.return_value.all.return_value = [mock_doc]

        # Mock upload processor
        mock_processor = MagicMock()
        mock_upload_processor_class.return_value = mock_processor
        mock_processor.reprocess_existing_document.return_value = {
            "success": True,
            "chunks_created": 5,
        }

        # Mock document processor
        mock_doc_processor = MagicMock()
        mock_doc_processor_class.return_value = mock_doc_processor
        mock_doc_processor._detect_all_chapters.return_value = [
            {"title": "Chapter 1", "content": "Content", "path": "1"}
        ]
        mock_doc_processor._process_document_standard.return_value = {
            "success": True,
            "chunks": [{"content": "Chunk 1"}],
        }

        # Mock database models
        with (
            patch("src.database.models.DocumentChapter"),
            patch("src.database.models.DocumentChunk"),
            patch("src.database.models.DocumentEmbedding"),
            patch("src.core.embeddings.create_embeddings") as mock_create_embeddings,
        ):
            # Mock embeddings creation
            mock_create_embeddings.return_value = ([MagicMock()], "ollama")

            # Call the function
            reprocess_documents()

            # Verify key operations were called
            mock_session.close.assert_called_once()
            mock_doc_processor._detect_all_chapters.assert_called_once()
            mock_doc_processor._process_document_standard.assert_called_once()
            mock_processor.reprocess_existing_document.assert_called_once()

    @patch("src.core.reprocess_documents.SessionLocal")
    def test_reprocess_documents_database_error(self, mock_session_local):
        """Test handling of database errors."""
        # Mock database error
        mock_session_local.side_effect = Exception("Database connection failed")

        # Call the function - should not crash
        reprocess_documents()

        # Function should complete without throwing exception

    @patch("src.core.reprocess_documents.SessionLocal")
    @patch("src.core.reprocess_documents.UploadProcessor")
    @patch("src.core.reprocess_documents.Document")
    @patch("src.core.reprocess_documents.DocumentProcessor")
    def test_reprocess_documents_processing_error(
        self,
        mock_doc_processor_class,
        mock_document_class,
        mock_upload_processor_class,
        mock_session_local,
    ):
        """Test handling of processing errors."""
        # Mock database with document
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        mock_doc = MagicMock()
        mock_doc.id = 1
        mock_doc.filename = "test.pdf"
        mock_doc.filepath = "/path/to/test.pdf"
        mock_doc.full_content = "Test content"
        mock_session.query.return_value.all.return_value = [mock_doc]

        # Mock processors to raise errors
        mock_processor = MagicMock()
        mock_upload_processor_class.return_value = mock_processor
        mock_processor.reprocess_existing_document.side_effect = Exception("Processing failed")

        mock_doc_processor = MagicMock()
        mock_doc_processor_class.return_value = mock_doc_processor
        mock_doc_processor._detect_all_chapters.return_value = []
        mock_doc_processor._create_chunks.return_value = []
        mock_doc_processor._process_document_standard.return_value = {
            "success": False,
            "error": "Processing failed",
        }

        # Call the function - should handle errors gracefully
        reprocess_documents()

        # Verify session was closed
        mock_session.close.assert_called_once()
