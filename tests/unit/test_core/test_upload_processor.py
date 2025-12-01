"""
Unit tests for UploadProcessor class.

Tests batch processing, reprocessing, file validation, and parallel execution.
"""

from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.core.processing.upload_processor import UploadProcessor


class TestUploadProcessor:
    """Test the UploadProcessor class functionality."""

    @pytest.fixture
    def upload_processor(self, mock_db_session):
        """Create UploadProcessor instance with mocked database."""
        return UploadProcessor(db=mock_db_session, embedding_model="embeddinggemma:latest")

    @pytest.fixture
    def mock_document(self):
        """Create a mock Document."""
        doc = MagicMock()
        doc.id = 1
        doc.filename = "test.pdf"
        doc.filepath = "/tmp/test.pdf"
        doc.status = "processed"
        return doc

    def test_init(self, mock_db_session):
        """Test UploadProcessor initialization."""
        processor = UploadProcessor(db=mock_db_session, embedding_model="embeddinggemma:latest")
        assert processor.db == mock_db_session
        assert processor.max_workers >= 1
        assert processor.embedding_model == "embeddinggemma:latest"

    def test_init_default_workers(self):
        """Test UploadProcessor initialization with default workers."""
        with patch("multiprocessing.cpu_count", return_value=8):
            with patch("src.core.processing.upload_processor.SessionLocal") as mock_session:
                mock_session.return_value = MagicMock()
                processor = UploadProcessor(embedding_model="embeddinggemma:latest")
                assert processor.max_workers == 4  # min(8, 4)

    def test_process_files_success(self, upload_processor):
        """Test successful batch file processing."""
        file_paths = ["/tmp/test1.pdf", "/tmp/test2.pdf"]

        with patch.object(upload_processor, "_process_single_file") as mock_process:
            mock_process.return_value = {"success": True, "filename": "test.pdf"}

            # Use sequential processing to avoid pickle issues with mocks
            result = upload_processor.process_files(file_paths, use_parallel=False)

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

            # Use sequential processing to avoid pickle issues with mocks
            result = upload_processor.process_files(file_paths, use_parallel=False)

            assert result["total_files"] == 2
            assert result["successful"] == 1
            assert result["failed"] == 1
            assert len(result["results"]) == 2

    def test_process_single_file_new_document(self, upload_processor, mock_db_session):
        """Test processing a new document."""
        # Mock file operations and hash
        with (
            patch("builtins.open", mock_open(read_data=b"fake pdf content")),
            patch("src.core.processing.upload_processor.hashlib") as mock_hashlib,
            patch(
                "src.core.processing.upload_processor.SessionLocal",
                return_value=mock_db_session,
            ),
        ):
            mock_hash = MagicMock()
            mock_hash.hexdigest.return_value = "test_hash"
            mock_hashlib.sha256.return_value = mock_hash

            # Mock document not existing
            mock_db_session.query.return_value.filter.return_value.first.return_value = None

            # Mock document processor
            with patch(
                "src.core.processing.upload_processor.DocumentProcessor"
            ) as mock_doc_processor_class:
                mock_doc_processor = MagicMock()
                mock_doc_processor_class.return_value = mock_doc_processor
                mock_doc_processor.process_document.return_value = {
                    "success": True,
                    "document_id": 1,
                    "chunks_count": 5,
                }

                result = upload_processor._process_single_file("/tmp/test.pdf")

                assert result["success"] is True
                assert result["chunks_created"] == 5
                mock_doc_processor.process_document.assert_called_once()

    def test_process_single_file_existing_no_force(
        self, upload_processor, mock_db_session, mock_document
    ):
        """Test processing existing document without force flag."""
        with (
            patch("src.core.processing.upload_processor.hashlib") as mock_hashlib,
            patch("builtins.open", mock_open(read_data=b"fake content")),
            patch(
                "src.core.processing.upload_processor.SessionLocal",
                return_value=mock_db_session,
            ),
        ):
            mock_hash = MagicMock()
            mock_hash.hexdigest.return_value = "test_hash"
            mock_hashlib.sha256.return_value = mock_hash

            # Mock document existing
            mock_document.configure_mock(id=1)
            mock_db_session.query.return_value.filter.return_value.filter.return_value.first.return_value = (
                mock_document
            )

            result = upload_processor._process_single_file("/tmp/test.pdf")

            assert result["success"] is True
            assert result["chunks_created"] == 0  # No new chunks for existing doc
            assert result["message"] == "Document already exists"
            assert "document_id" in result

    def test_process_single_file_advanced_processing_auto_detect(
        self, upload_processor, mock_db_session
    ):
        """Test auto-detection of advanced processing for PDFs."""
        with (
            patch("builtins.open", mock_open(read_data=b"fake pdf content")),
            patch("src.core.processing.upload_processor.hashlib") as mock_hashlib,
            patch(
                "src.core.processing.upload_processor.SessionLocal",
                return_value=mock_db_session,
            ),
        ):
            mock_hash = MagicMock()
            mock_hash.hexdigest.return_value = "test_hash"
            mock_hashlib.sha256.return_value = mock_hash

            # Mock document not existing
            mock_db_session.query.return_value.filter.return_value.first.return_value = None

            # Mock scanned PDF detection
            mock_loader = MagicMock()
            mock_loader.is_scanned_pdf.return_value = True
            with patch.dict("sys.modules", {"src.data.loader": mock_loader}):
                with patch(
                    "src.core.processing.upload_processor.DocumentProcessor"
                ) as mock_doc_processor_class:
                    mock_doc_processor = MagicMock()
                    mock_doc_processor_class.return_value = mock_doc_processor
                    mock_doc_processor.process_document.return_value = {
                        "success": True,
                        "document_id": 1,
                        "chunks_count": 3,
                    }

                    result = upload_processor._process_single_file("test.pdf")

                    # Should call with use_advanced_processing=True due to scanned PDF
                    assert result["success"] is True
                    assert result["chunks_created"] == 3
                    mock_doc_processor.process_document.assert_called_once()

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

            # Mock DocumentProcessor for AI regeneration
            with patch("src.core.processing.upload_processor.DocumentProcessor") as mock_doc_class:
                mock_doc_instance = MagicMock()
                mock_doc_class.return_value = mock_doc_instance
                mock_doc_instance._generate_document_summary.return_value = "Updated summary"
                mock_doc_instance._suggest_categories_ai.return_value = ["test_category"]

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
        result = upload_processor.reprocess_existing_document(mock_document, None, "/tmp/test.pdf")

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

    def test_process_single_file_error_handling(self, upload_processor, mock_db_session):
        """Test error handling in single file processing."""
        with (
            patch("builtins.open", side_effect=Exception("File read error")),
            patch(
                "src.core.processing.upload_processor.SessionLocal",
                return_value=mock_db_session,
            ),
        ):
            result = upload_processor._process_single_file("/tmp/test.pdf")

            assert result["success"] is False
            assert "File read error" in result["error"]

    def test_process_files_with_progress_callback(self, upload_processor):
        """Test file processing with progress callback."""
        file_paths = ["/tmp/test1.pdf", "/tmp/test2.pdf"]
        progress_calls = []

        def progress_callback(progress, message):
            progress_calls.append((progress, message))

        with patch.object(upload_processor, "_process_single_file") as mock_process:
            mock_process.return_value = {"success": True, "filename": "test.pdf"}

            result = upload_processor.process_files(
                file_paths, use_parallel=False, progress_callback=progress_callback
            )

            assert result["total_files"] == 2
            assert len(progress_calls) == 2
            assert progress_calls[0][0] == 50.0  # 1/2 * 100
            assert progress_calls[1][0] == 100.0  # 2/2 * 100
