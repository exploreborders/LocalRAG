"""
Unit tests for BaseProcessor class.

Tests shared functionality for document processors.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.core.base_processor import BaseProcessor


class TestBaseProcessor:
    """Test the BaseProcessor class functionality."""

    @pytest.fixture
    def base_processor(self, mock_db_session):
        """Create BaseProcessor instance."""
        return BaseProcessor(mock_db_session)

    def test_init(self, mock_db_session):
        """Test BaseProcessor initialization."""
        processor = BaseProcessor(mock_db_session)
        assert processor.db == mock_db_session

    def test_process_existing_documents_no_pending(self, base_processor, mock_db_session):
        """Test processing when no pending documents exist."""
        # Mock empty query result
        mock_db_session.query.return_value.filter.return_value.all.return_value = []

        result = base_processor.process_existing_documents()

        assert result["success"] is True
        assert result["processed"] == 0
        assert "No pending documents" in result["message"]

    def test_process_existing_documents_with_pending(self, base_processor, mock_db_session):
        """Test processing pending documents."""
        # Create mock documents
        mock_doc1 = MagicMock()
        mock_doc1.filename = "doc1.pdf"
        mock_doc1.filepath = "/tmp/doc1.pdf"
        mock_doc1.status = "uploaded"

        mock_doc2 = MagicMock()
        mock_doc2.filename = "doc2.pdf"
        mock_doc2.filepath = "/tmp/doc2.pdf"
        mock_doc2.status = "pending"

        mock_db_session.query.return_value.filter.return_value.all.return_value = [
            mock_doc1,
            mock_doc2,
        ]

        # Mock file existence and processing
        with patch("os.path.exists", return_value=True):
            with patch.object(base_processor, "_process_single_document") as mock_process:
                mock_process.return_value = {"success": True}

                result = base_processor.process_existing_documents()

                assert result["success"] is True
                assert result["processed"] == 2
                assert result["failed"] == 0

    def test_process_existing_documents_file_not_found(self, base_processor, mock_db_session):
        """Test processing when document file doesn't exist."""
        mock_doc = MagicMock()
        mock_doc.filename = "missing.pdf"
        mock_doc.filepath = "/tmp/missing.pdf"
        mock_doc.status = "uploaded"

        mock_db_session.query.return_value.filter.return_value.all.return_value = [mock_doc]

        with patch("os.path.exists", return_value=False):
            result = base_processor.process_existing_documents()

            assert result["success"] is True
            assert result["processed"] == 0
            assert result["failed"] == 1
            assert mock_doc.status == "error"

    def test_process_existing_documents_processing_failure(self, base_processor, mock_db_session):
        """Test processing when document processing fails."""
        mock_doc = MagicMock()
        mock_doc.filename = "fail.pdf"
        mock_doc.filepath = "/tmp/fail.pdf"
        mock_doc.status = "uploaded"

        mock_db_session.query.return_value.filter.return_value.all.return_value = [mock_doc]

        with patch("os.path.exists", return_value=True):
            with patch.object(base_processor, "_process_single_document") as mock_process:
                mock_process.return_value = {
                    "success": False,
                    "error": "Processing failed",
                }

                result = base_processor.process_existing_documents()

                assert result["success"] is True
                assert result["processed"] == 0
                assert result["failed"] == 1

    def test_process_existing_documents_exception(self, base_processor, mock_db_session):
        """Test processing when an exception occurs."""
        mock_db_session.query.side_effect = Exception("Database error")

        result = base_processor.process_existing_documents()

        assert result["success"] is False
        assert "Database error" in result["error"]

    def test_process_single_document_not_implemented(self, base_processor):
        """Test that _process_single_document raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            base_processor._process_single_document("/tmp/test.pdf", "test.pdf")

    def test_validate_file_path_valid(self, base_processor):
        """Test file path validation with valid path."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"test content")
            tmp_file_path = tmp_file.name

        try:
            # Test with relative path
            relative_path = os.path.basename(tmp_file_path)
            with (
                patch("os.path.exists", return_value=True),
                patch("os.path.isfile", return_value=True),
                patch("os.path.isabs", return_value=False),
                patch("os.path.normpath", return_value=relative_path),
            ):
                result = base_processor._validate_file_path(relative_path)
                assert result is True
        finally:
            os.unlink(tmp_file_path)

    def test_validate_file_path_directory_traversal(self, base_processor):
        """Test file path validation with directory traversal."""
        malicious_path = "../../../etc/passwd"

        with patch("os.path.normpath", return_value="../../../etc/passwd"):
            result = base_processor._validate_file_path(malicious_path)
            assert result is False

    def test_validate_file_path_absolute_path(self, base_processor):
        """Test file path validation with absolute path."""
        absolute_path = "/etc/passwd"

        with (
            patch("os.path.normpath", return_value="/etc/passwd"),
            patch("os.path.isabs", return_value=True),
        ):
            result = base_processor._validate_file_path(absolute_path)
            assert result is False

    def test_validate_file_path_file_not_exists(self, base_processor):
        """Test file path validation when file doesn't exist."""
        nonexistent_path = "/tmp/nonexistent.txt"

        with (
            patch("os.path.normpath", return_value="/tmp/nonexistent.txt"),
            patch("os.path.isabs", return_value=False),
            patch("os.path.exists", return_value=False),
        ):
            result = base_processor._validate_file_path(nonexistent_path)
            assert result is False

    def test_validate_file_path_not_regular_file(self, base_processor):
        """Test file path validation when path is not a regular file."""
        dir_path = "/tmp"

        with (
            patch("os.path.normpath", return_value="/tmp"),
            patch("os.path.isabs", return_value=False),
            patch("os.path.exists", return_value=True),
            patch("os.path.isfile", return_value=False),
        ):
            result = base_processor._validate_file_path(dir_path)
            assert result is False

    def test_validate_file_path_exception(self, base_processor):
        """Test file path validation when an exception occurs."""
        with patch("os.path.normpath", side_effect=Exception("Path error")):
            result = base_processor._validate_file_path("invalid/path")
            assert result is False
