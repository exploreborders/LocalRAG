"""
Unit tests for CaptionAwareProcessor class.

Tests caption extraction, context identification, and caption-centric chunking.
"""

from unittest.mock import MagicMock, patch

import pytest
from docling_core.types.doc.document import DoclingDocument

from src.data.caption_processor import CaptionAwareProcessor


class TestCaptionAwareProcessor:
    """Test the CaptionAwareProcessor class functionality."""

    def test_init(self):
        """Test CaptionAwareProcessor initialization."""
        processor = CaptionAwareProcessor(chunk_size=500, overlap=100)

        assert processor.chunk_size == 500
        assert processor.overlap == 100

    def test_init_defaults(self):
        """Test CaptionAwareProcessor initialization with defaults."""
        processor = CaptionAwareProcessor()

        assert processor.chunk_size == 1000
        assert processor.overlap == 200

    @patch("src.data.caption_processor.DoclingDocument")
    def test_extract_document_structure(self, mock_docling_doc):
        """Test document structure extraction."""
        # Mock the DoclingDocument
        mock_doc = MagicMock(spec=DoclingDocument)
        mock_doc.export_to_markdown.return_value = """
        # Document Title

        This is some content.

        Figure 1: A sample figure showing data.

        More content here.

        Table 2: Sample data table.

        Final content.
        """

        processor = CaptionAwareProcessor()
        result = processor.extract_document_structure(mock_doc)

        assert "full_text" in result
        assert "captions" in result
        assert "sections" in result
        assert len(result["captions"]) > 0

        # Check that export_to_markdown was called
        mock_doc.export_to_markdown.assert_called_once()

    def test_identify_captions_from_markdown(self):
        """Test caption identification from markdown text."""
        processor = CaptionAwareProcessor()

        markdown_text = """
        # Document

        Some content here.

        Figure 1: This is a figure caption.

        More content.

        Table 2: Data table description.

        Source: Research paper 2023.

        *Italic caption*

        _Underlined caption_

        ![Image alt](image.jpg)

        Final paragraph.
        """

        captions = processor._identify_captions_from_markdown(markdown_text)

        assert len(captions) >= 6  # Should find multiple caption types

        # Check specific captions
        figure_captions = [c for c in captions if "Figure 1" in c["text"]]
        assert len(figure_captions) == 1
        assert figure_captions[0]["type"] == "caption"

    def test_identify_captions_no_captions(self):
        """Test caption identification when no captions exist."""
        processor = CaptionAwareProcessor()

        markdown_text = """
        # Document

        This is just regular content without any captions.

        More paragraphs here.
        """

        captions = processor._identify_captions_from_markdown(markdown_text)

        assert len(captions) == 0

    def test_get_caption_context(self):
        """Test getting context around captions."""
        processor = CaptionAwareProcessor()

        lines = [
            "Previous line 1",
            "Previous line 2",
            "Figure 1: Caption text",
            "Next line 1",
            "Next line 2",
            "Next line 3",
        ]

        context = processor._get_caption_context(lines, 2, context_window=2)

        # Should include lines before and after, excluding the caption line
        assert "Previous line 1" in context
        assert "Previous line 2" in context
        assert "Next line 1" in context
        assert "Next line 2" in context
        assert "Figure 1: Caption text" not in context  # Caption line excluded

    def test_get_caption_context_edge_cases(self):
        """Test caption context with edge cases."""
        processor = CaptionAwareProcessor()

        # Caption at the beginning
        lines = ["Figure 1: Caption", "Line 2", "Line 3"]
        context = processor._get_caption_context(lines, 0, context_window=2)
        assert "Line 2" in context
        assert "Line 3" in context

        # Caption at the end
        lines = ["Line 1", "Line 2", "Figure 2: Caption"]
        context = processor._get_caption_context(lines, 2, context_window=2)
        assert "Line 1" in context
        assert "Line 2" in context

    @patch("src.data.caption_processor.DoclingDocument")
    def test_create_caption_centric_chunks_with_captions(self, mock_docling_doc):
        """Test creating chunks when captions are found."""
        # Mock DoclingDocument
        mock_doc = MagicMock(spec=DoclingDocument)
        mock_doc.export_to_markdown.return_value = """
        # Document

        Introduction text.

        Figure 1: Sample figure with description.

        More content after the figure.

        Table 1: Data table.

        Final content.
        """

        processor = CaptionAwareProcessor()
        chunks = processor.create_caption_centric_chunks(mock_doc)

        assert len(chunks) > 0

        # Check that chunks have caption metadata
        caption_chunks = [c for c in chunks if c.metadata.get("has_captions")]
        assert len(caption_chunks) > 0

        # Check metadata structure
        chunk = caption_chunks[0]
        assert "chunk_type" in chunk.metadata
        assert "has_captions" in chunk.metadata
        assert "caption_text" in chunk.metadata

    @patch("src.data.caption_processor.DoclingDocument")
    def test_create_caption_centric_chunks_no_captions(self, mock_docling_doc):
        """Test creating chunks when no captions are found."""
        # Mock DoclingDocument with no captions
        mock_doc = MagicMock(spec=DoclingDocument)
        mock_doc.export_to_markdown.return_value = """
        # Document

        This is regular content without any captions.

        More paragraphs here.
        """

        processor = CaptionAwareProcessor()
        chunks = processor.create_caption_centric_chunks(mock_doc)

        # Should fall back to regular chunking
        assert len(chunks) > 0

        # Check that chunks don't have caption metadata
        for chunk in chunks:
            assert chunk.metadata.get("has_captions") is False
            assert chunk.metadata.get("chunk_type") == "fallback"

    def test_create_chunks_with_captions(self):
        """Test the internal chunk creation method."""
        processor = CaptionAwareProcessor()

        # Create longer content to exceed the 100 character minimum
        long_content = (
            "Line 1\nFigure 1: Caption\nLine 3\nLine 4\nLine 5\n" + "Additional content line. " * 10
        )

        structure = {
            "full_text": long_content,
            "captions": [
                {
                    "text": "Figure 1: Caption",
                    "line_number": 1,
                    "type": "caption",
                    "context": "Line 3 Line 4",
                }
            ],
        }

        chunks = processor._create_chunks_with_captions(structure, {"source": "test"})

        assert len(chunks) > 0

        # Check chunk content includes context around caption
        chunk = chunks[0]
        assert "Figure 1: Caption" in chunk.page_content
        assert "Line 3" in chunk.page_content or "Line 4" in chunk.page_content

    def test_create_chunks_with_captions_empty(self):
        """Test chunk creation with empty captions."""
        processor = CaptionAwareProcessor()

        structure = {"full_text": "Just content", "captions": []}

        chunks = processor._create_chunks_with_captions(structure, {})

        assert len(chunks) == 0

    def test_fallback_chunking(self):
        """Test fallback chunking when no captions found."""
        processor = CaptionAwareProcessor(chunk_size=50, overlap=10)

        content = "This is a test document. " * 20  # Long content
        metadata = {"source": "test"}

        chunks = processor._fallback_chunking(content, metadata)

        assert len(chunks) > 0

        # Check that all chunks have correct metadata
        for chunk in chunks:
            assert chunk.metadata["source"] == "test"
            assert chunk.metadata["chunk_type"] == "fallback"
            assert chunk.metadata["has_captions"] is False

    def test_fallback_chunking_short_content(self):
        """Test fallback chunking with short content."""
        processor = CaptionAwareProcessor()

        content = "Short content"
        chunks = processor._fallback_chunking(content, {})

        assert len(chunks) == 1
        assert chunks[0].page_content == "Short content"
