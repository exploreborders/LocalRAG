"""
Unit tests for UploadProcessor class.

Tests batch processing, reprocessing, file validation, and parallel execution.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from src.core.processing.upload_processor import UploadProcessor
from src.database.models import Document


class TestUploadProcessor:
    """Test the UploadProcessor class functionality."""

    @pytest.fixture
    def upload_processor(self, mock_db_session):
        """Create UploadProcessor instance with mocked database."""
        return UploadProcessor(
            db=mock_db_session, embedding_model="embeddinggemma:latest"
        )

    @pytest.fixture
    def mock_document(self):
        """Create a mock Document."""
        doc = MagicMock(spec=Document)
        doc.id = 1
        doc.filename = "test.pdf"
        doc.filepath = "/tmp/test.pdf"
        doc.status = "processed"
        return doc

    def test_init(self, mock_db_session):
        """Test UploadProcessor initialization."""
        processor = UploadProcessor(
            db=mock_db_session, embedding_model="embeddinggemma:latest"
        )
        assert processor.db == mock_db_session
        assert processor.max_workers >= 1
        assert processor.embedding_model == "embeddinggemma:latest"

    def test_init_default_workers(self):
        """Test UploadProcessor initialization with default workers."""
        with patch("multiprocessing.cpu_count", return_value=8):
            with patch(
                "src.core.processing.upload_processor.SessionLocal"
            ) as mock_session:
                mock_session.return_value = MagicMock()
                processor = UploadProcessor(embedding_model="embeddinggemma:latest")
                assert processor.max_workers == 4  # min(8, 4)

    def test_process_files_success(self, upload_processor):
        """Test successful batch file processing."""
        file_paths = ["/tmp/test1.pdf", "/tmp/test2.pdf"]

        with patch.object(upload_processor, "_process_single_file") as mock_process:
            mock_process.return_value = {"success": True, "filename": "test.pdf"}

            result = upload_processor.process_files(file_paths)

            assert result["total_files"] == 2
            assert result["successful"] == 2
            assert result["failed"] == 0
            assert len(result["results"]) == 2

    def test_process_files_with_failures(self, upload_processor):
        """Test batch processing with some failures."""
        file_paths = ["/tmp/test1.pdf", "/tmp/test2.pdf"]

        def mock_process_side_effect(file_path):
            if "test1" in file_path:
                return {"success": True, "filename": "test1.pdf"}
            else:
                raise Exception("Processing failed")

        with patch.object(upload_processor, "_process_single_file") as mock_process:
            mock_process.side_effect = mock_process_side_effect

            result = upload_processor.process_files(file_paths)

            assert result["total_files"] == 2
            assert result["successful"] == 1
            assert result["failed"] == 1

    def test_process_single_file_new_document(self, upload_processor, mock_db_session):
        """Test processing a new document."""
        # Mock file hash and document lookup
        with patch("src.core.processing.upload_processor.hashlib") as mock_hashlib:
            mock_hash = MagicMock()
            mock_hash.hexdigest.return_value = "test_hash"
            mock_hashlib.sha256.return_value = mock_hash

            # Mock document not existing
            mock_db_session.query.return_value.filter.return_value.filter.return_value.first.return_value = None

            # Mock document processor
            with patch(
                "src.core.processing.upload_processor.DocumentProcessor"
            ) as mock_doc_processor_class:
                mock_doc_processor = MagicMock()
                mock_doc_processor_class.return_value = mock_doc_processor
                mock_doc_processor.process_document.return_value = {
                    "success": True,
                    "document_id": 1,
                }

                result = upload_processor.process_single_file("/tmp/test.pdf")

                assert result["success"] is True
                mock_doc_processor.process_document.assert_called_once()

    def test_process_single_file_existing_no_force(
        self, upload_processor, mock_db_session, mock_document
    ):
        """Test processing existing document without force flag."""
        with patch("src.core.processing.upload_processor.hashlib") as mock_hashlib:
            mock_hash = MagicMock()
            mock_hash.hexdigest.return_value = "test_hash"
            mock_hashlib.sha256.return_value = mock_hash

            # Mock document existing
            mock_db_session.query.return_value.filter.return_value.filter.return_value.first.return_value = mock_document

            result = upload_processor.process_single_file("/tmp/test.pdf")

            assert result["success"] is True
            assert result["message"] == "Document already exists"
            assert result["document_id"] == 1

    def test_process_single_file_force_reprocess(
        self, upload_processor, mock_db_session, mock_document
    ):
        """Test force reprocessing of existing document."""
        with patch("src.core.processing.upload_processor.hashlib") as mock_hashlib:
            mock_hash = MagicMock()
            mock_hash.hexdigest.return_value = "test_hash"
            mock_hashlib.sha256.return_value = mock_hash

            # Mock document existing
            mock_db_session.query.return_value.filter.return_value.filter.return_value.first.return_value = mock_document

            # Mock advanced processor
            with patch(
                "src.data.loader.AdvancedDocumentProcessor"
            ) as mock_advanced_class:
                mock_advanced = MagicMock()
                mock_advanced_class.return_value = mock_advanced
                mock_advanced.process_document_comprehensive.return_value = {
                    "extracted_content": "test content",
                    "chapters_detected": 2,
                }

                # Mock reprocess method
                with patch.object(
                    upload_processor, "reprocess_existing_document"
                ) as mock_reprocess:
                    mock_reprocess.return_value = {"success": True, "document_id": 1}

                    result = upload_processor.process_single_file(
                        "/tmp/test.pdf", force_enrichment=True
                    )

                    mock_reprocess.assert_called_once()
                    assert result["success"] is True

    def test_process_single_file_advanced_processing_auto_detect(
        self, upload_processor, mock_db_session
    ):
        """Test auto-detection of advanced processing for PDFs."""
        with patch("src.core.processing.upload_processor.hashlib") as mock_hashlib:
            mock_hash = MagicMock()
            mock_hash.hexdigest.return_value = "test_hash"
            mock_hashlib.sha256.return_value = mock_hash

            # Mock document not existing
            mock_db_session.query.return_value.filter.return_value.filter.return_value.first.return_value = None

            # Mock scanned PDF detection
            with patch("src.data.loader.is_scanned_pdf", return_value=True):
                with patch(
                    "src.core.processing.upload_processor.DocumentProcessor"
                ) as mock_doc_processor_class:
                    mock_doc_processor = MagicMock()
                    mock_doc_processor_class.return_value = mock_doc_processor
                    mock_doc_processor.process_document.return_value = {
                        "success": True,
                        "document_id": 1,
                    }

                    result = upload_processor.process_single_file("test.pdf")

                    # Should call with use_advanced_processing=True due to scanned PDF
                    mock_doc_processor.process_document.assert_called_once()
                    args, kwargs = mock_doc_processor.process_document.call_args
                    assert kwargs.get("use_advanced_processing") is True

    def test_reprocess_existing_document_success(
        self, upload_processor, mock_db_session, mock_document
    ):
        """Test successful reprocessing of existing document."""
        processing_result = {
            "extracted_content": "Updated content for reprocessing",
            "chunks": [{"content": "chunk1"}, {"content": "chunk2"}],
            "chapters": [{"title": "Chapter 1"}],
        }

        # Mock content validation
        with patch(
            "src.utils.content_validator.ContentValidator.validate_content_quality"
        ) as mock_validate:
            mock_validate.return_value = {
                "is_valid": True,
                "quality_score": 0.9,
                "issues": [],
                "recommendations": [],
            }

            # Mock AI regeneration
            with patch.object(upload_processor, "tag_suggester") as mock_suggester:
                mock_suggester._call_llm.return_value = "Updated summary"

                result = upload_processor.reprocess_existing_document(
                    mock_document, processing_result, "/tmp/test.pdf"
                )

                assert result["success"] is True
                assert "chunks_created" in result
                assert "chapters_created" in result

    def test_reprocess_existing_document_validation_failure(
        self, upload_processor, mock_db_session, mock_document
    ):
        """Test reprocessing with content validation failure."""
        processing_result = {"extracted_content": "", "chunks": []}

        with patch(
            "src.utils.content_validator.ContentValidator.validate_content_quality"
        ) as mock_validate:
            mock_validate.return_value = {
                "is_valid": False,
                "quality_score": 0.3,
                "issues": ["Low quality content"],
                "recommendations": ["Improve content"],
            }

            result = upload_processor.reprocess_existing_document(
                mock_document, processing_result, "/tmp/test.pdf"
            )

            assert result["success"] is True  # Still succeeds but marks as needs_review
            assert mock_document.status == "needs_review"

    def test_reprocess_existing_document_processing_error(
        self, upload_processor, mock_db_session, mock_document
    ):
        """Test reprocessing with processing result being None."""
        result = upload_processor.reprocess_existing_document(
            mock_document, None, "/tmp/test.pdf"
        )

        assert result["success"] is False
        assert "Processing result is None" in result["error"]

    def test_upload_files_alias(self, upload_processor):
        """Test that upload_files is an alias for process_files."""
        with patch.object(upload_processor, "process_files") as mock_process:
            mock_process.return_value = {"total_files": 1, "successful": 1}

            result = upload_processor.upload_files(["/tmp/test.pdf"])

            mock_process.assert_called_once_with(["/tmp/test.pdf"], None, True)
            assert result == mock_process.return_value

    def test_del_method(self, upload_processor, mock_db_session):
        """Test cleanup in destructor."""
        upload_processor.__del__()

        # Should not raise exception
        assert True

    def test_process_single_file_error_handling(self, upload_processor):
        """Test error handling in single file processing."""
        with patch(
            "src.core.processing.upload_processor.hashlib",
            side_effect=Exception("Hash error"),
        ):
            result = upload_processor.process_single_file("/tmp/test.pdf")

            assert result["success"] is False
            assert "Hash error" in result["error"]
