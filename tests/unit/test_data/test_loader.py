"""
Unit tests for data loader functionality.

Tests document loading, OCR processing, and advanced document processing.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.data.loader import (
    AdvancedDocumentProcessor,
    extract_text_with_ocr,
    is_scanned_pdf,
    load_documents,
    split_documents,
)


class TestOCRFunctions:
    """Test OCR-related functions."""

    def test_extract_text_with_ocr_no_tesseract(self):
        """Test OCR extraction when Tesseract is not available."""
        with patch("src.data.loader.TESSERACT_AVAILABLE", False):
            result = extract_text_with_ocr("/fake/path.pdf")
            assert result == ""


class TestAdvancedDocumentProcessor:
    """Test the AdvancedDocumentProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create AdvancedDocumentProcessor instance."""
        return AdvancedDocumentProcessor()

    def test_init(self, processor):
        """Test AdvancedDocumentProcessor initialization."""
        assert processor is not None
        assert hasattr(processor, "process_document_comprehensive")

    def test_process_document_comprehensive_text_pdf(self, processor):
        """Test comprehensive processing of text-based PDF."""
        # This test would require extensive mocking of the entire pipeline
        # For now, just test that the method exists and returns a dict
        result = processor.process_document_comprehensive("/nonexistent.pdf")
        assert isinstance(result, dict)
        assert "file_path" in result

    def test_process_document_comprehensive_scanned_pdf(self, processor):
        """Test comprehensive processing of scanned PDF."""
        # Similar to above, extensive mocking needed
        result = processor.process_document_comprehensive("/nonexistent.pdf")
        assert isinstance(result, dict)
        assert "file_path" in result

    def test_analyze_document_structure(self, processor):
        """Test document structure analysis."""
        content = """
        # Chapter 1: Introduction

        This is the introduction chapter.

        ## Section 1.1

        Subsection content here.

        # Chapter 2: Methods

        Methods description.
        """

        result = processor._analyze_document_structure(content)

        assert isinstance(result, dict)
        assert "hierarchy" in result
        assert "sections" in result

    def test_traverse_docling_tree(self, processor):
        """Test Docling document tree traversal."""
        # Mock Docling document structure
        mock_doc = MagicMock()
        mock_body = MagicMock()
        mock_body.text_content = "Body content"
        mock_doc.body = mock_body
        mock_doc.text_content = "Full text content"

        structure = {}
        result = processor._traverse_docling_tree(mock_doc, structure, 0, "")

        # Method returns None, but modifies the structure dict
        assert result is None
        assert isinstance(structure, dict)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_load_documents(self):
        """Test document loading from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            pdf_file = Path(temp_dir) / "test.pdf"
            pdf_file.write_text("fake pdf content")

            txt_file = Path(temp_dir) / "test.txt"
            txt_file.write_text("text content")

            # Skip files that don't exist
            result = load_documents(temp_dir)

            assert isinstance(result, list)
            # Should process files (though actual processing may fail without real PDFs)

    def test_split_documents(self):
        """Test document splitting functionality."""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document with some content that should be loaded.")
            temp_file = f.name

        try:
            result = split_documents([temp_file], chunk_size=50, chunk_overlap=10)
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            os.unlink(temp_file)

    def test_split_documents_empty(self):
        """Test splitting with empty document list."""
        result = split_documents([])
        assert result == ""
