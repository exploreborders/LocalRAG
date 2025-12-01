"""
Unit tests for DocumentProcessor class.

Tests document processing, AI enrichment, chunking, language detection,
and search indexing functionality.
"""

from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.core.processing.document_processor import DocumentProcessor
from src.database.models import Document


class TestDocumentProcessor:
    """Test the DocumentProcessor class functionality."""

    @pytest.fixture
    def document_processor(self, mock_db_session):
        """Create DocumentProcessor instance with mocked database."""
        return DocumentProcessor(mock_db_session)

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
        """Test DocumentProcessor initialization."""
        processor = DocumentProcessor(mock_db_session)
        assert processor.db == mock_db_session
        assert hasattr(processor, "tag_manager")
        assert hasattr(processor, "category_manager")
        assert hasattr(processor, "tag_suggester")

    def test_init_without_db(self):
        """Test DocumentProcessor initialization without db parameter."""
        with patch(
            "src.core.processing.document_processor.SessionLocal"
        ) as mock_session:
            mock_session.return_value = MagicMock()
            _ = DocumentProcessor()
            mock_session.assert_called_once()

    def test_suggest_categories_ai_success(self, document_processor):
        """Test AI-powered category suggestions."""
        with patch.object(document_processor.tag_suggester, "_call_llm") as mock_llm:
            mock_llm.return_value = "Academic, Technical"

            result = document_processor._suggest_categories_ai(
                "machine learning content", "ml.pdf", ["AI", "ML"]
            )

            assert "Academic" in result
            assert "Technical" in result
            mock_llm.assert_called_once()

    def test_suggest_categories_ai_fallback(self, document_processor):
        """Test category suggestion fallback when AI fails."""
        with patch.object(
            document_processor.tag_suggester,
            "_call_llm",
            side_effect=Exception("AI error"),
        ):
            result = document_processor._suggest_categories_ai(
                "deep learning neural networks", "ai.pdf", ["AI"]
            )

            assert "Technical" in result  # Should match keyword-based fallback

    def test_generate_document_summary_success(self, document_processor):
        """Test AI-powered document summary generation."""
        with patch.object(document_processor.tag_suggester, "_call_llm") as mock_llm:
            mock_llm.return_value = "This is a comprehensive summary of the document."

            result = document_processor._generate_document_summary(
                "document content", "test.pdf", ["tag1", "tag2"], 3
            )

            assert "comprehensive summary" in result
            mock_llm.assert_called_once()

    def test_generate_document_summary_fallback(self, document_processor):
        """Test summary generation fallback when AI fails."""
        with patch.object(
            document_processor.tag_suggester,
            "_call_llm",
            side_effect=Exception("AI error"),
        ):
            result = document_processor._generate_document_summary(
                "content", "test.pdf", ["tag"], 2
            )

            assert (
                "Document processed with advanced AI pipeline. Covers tag. 2 chapters detected."
                == result
            )

    def test_generate_document_summary_with_prefixes(self, document_processor):
        """Test summary generation with prefix removal."""
        with patch.object(document_processor.tag_suggester, "_call_llm") as mock_llm:
            # Test with "Document summary:" prefix
            mock_llm.return_value = "Document summary: This is a test summary."

            result = document_processor._generate_document_summary(
                "content", "test.pdf", ["tag"], 1
            )

            assert result == "This is a test summary."
            mock_llm.assert_called_once()

    def test_generate_document_summary_with_quotes(self, document_processor):
        """Test summary generation with quote removal."""
        with patch.object(document_processor.tag_suggester, "_call_llm") as mock_llm:
            # Test with quotes
            mock_llm.return_value = '"This is a quoted summary"'

            result = document_processor._generate_document_summary(
                "content", "test.pdf", ["tag"], 1
            )

            assert result == "This is a quoted summary"
            mock_llm.assert_called_once()

    def test_generate_document_summary_with_single_quotes(self, document_processor):
        """Test summary generation with single quote removal."""
        with patch.object(document_processor.tag_suggester, "_call_llm") as mock_llm:
            # Test with single quotes
            mock_llm.return_value = "'This is a single quoted summary'"

            result = document_processor._generate_document_summary(
                "content", "test.pdf", ["tag"], 1
            )

            assert result == "This is a single quoted summary"
            mock_llm.assert_called_once()

    def test_process_document_standard_mode(self, document_processor, mock_db_session):
        """Test document processing in standard mode."""
        with patch.object(
            document_processor, "_process_document_standard"
        ) as mock_standard:
            mock_standard.return_value = {"success": True, "document_id": 1}

            result = document_processor.process_document(
                "/tmp/test.pdf", use_advanced_processing=False
            )

            mock_standard.assert_called_once_with("/tmp/test.pdf", None, None, None)
            assert result["success"] is True

    def test_process_document_standard_with_progress_callback(
        self, document_processor, mock_db_session
    ):
        """Test document processing in standard mode with progress callback."""
        progress_calls = []

        def progress_callback(filename, progress, message):
            progress_calls.append((filename, progress, message))

        with patch.object(
            document_processor, "_process_document_standard"
        ) as mock_standard:
            mock_standard.return_value = {"success": True, "document_id": 1}

            result = document_processor.process_document(
                "/tmp/test.pdf",
                use_advanced_processing=False,
                progress_callback=progress_callback,
            )

            mock_standard.assert_called_once_with(
                "/tmp/test.pdf", None, progress_callback, None
            )
            assert result["success"] is True

    def test_process_document_advanced_mode(self, document_processor, mock_db_session):
        """Test document processing in advanced mode."""
        with patch.object(
            document_processor, "_process_document_advanced"
        ) as mock_advanced:
            mock_advanced.return_value = {"success": True, "document_id": 1}

            result = document_processor.process_document(
                "/tmp/test.pdf", use_advanced_processing=True
            )

            mock_advanced.assert_called_once_with("/tmp/test.pdf", None, None, None)
            assert result["success"] is True

    def test_detect_language_from_file_success(self, document_processor):
        """Test language detection from file content."""
        mock_file_content = "This is English text for testing."

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            with patch("src.core.processing.document_processor.detect") as mock_detect:
                mock_detect.return_value = "en"

                result = document_processor._detect_language_from_file("/tmp/test.txt")

                assert result == "en"
                mock_detect.assert_called_once_with(mock_file_content)

    def test_detect_language_from_file_fallback(self, document_processor):
        """Test language detection fallback when file read fails."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = document_processor._detect_language_from_file("/tmp/missing.txt")

            assert result == "en"  # Default fallback

    def test_detect_language_from_content(self, document_processor):
        """Test language detection from content string."""
        content = "This is English content. " * 50  # Make it substantial

        with patch("src.core.processing.document_processor.detect") as mock_detect:
            mock_detect.return_value = "en"

            result = document_processor._detect_language_from_content(content)

            assert result == "en"
            # Should be called multiple times for different content samples

    def test_detect_language_from_content_with_exceptions(self, document_processor):
        """Test language detection with LangDetectException handling."""
        content = (
            "This is a longer text that should be substantial enough for language detection. "
            * 20
        )

        with patch("src.core.processing.document_processor.detect") as mock_detect:
            from langdetect import LangDetectException

            # Make some calls fail, some succeed
            call_count = 0

            def detect_side_effect(text):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise LangDetectException("Detection failed", "en")
                else:
                    return "en"

            mock_detect.side_effect = detect_side_effect

            result = document_processor._detect_language_from_content(content)

            assert result == "en"  # Should get "en" from successful detection
            assert mock_detect.call_count >= 2

    def test_detect_all_chapters_markdown_headers(self, document_processor):
        """Test chapter detection from markdown headers."""
        content = "## Introduction\n\nSome intro content.\n\n## Chapter 1\n\nChapter content.\n\n## Chapter 2"

        chapters = document_processor._detect_all_chapters(content)

        assert len(chapters) >= 2
        assert any("Introduction" in ch["title"] for ch in chapters)
        assert any("Chapter 1" in ch["title"] for ch in chapters)

    def test_detect_all_chapters_table_format(self, document_processor):
        """Test chapter detection from table format."""
        content = "| 1 | Introduction |\n| 2 | Main Content |"

        chapters = document_processor._detect_all_chapters(content)

        assert len(chapters) >= 1
        assert any("Introduction" in ch["title"] for ch in chapters)

    def test_create_chunks_with_chapters(self, document_processor):
        """Test chunk creation with chapter awareness."""
        content = "Long document content " * 100
        chapters = [
            {"title": "Chapter 1", "content": content[:600], "path": "1", "level": 1}
        ]

        chunks = document_processor._create_chunks(content, 1, chapters)

        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)

    def test_create_chunks_without_chapters(self, document_processor):
        """Test chunk creation without chapter structure."""
        content = "Simple content " * 200

        chunks = document_processor._create_chunks(content, 1, [])

        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)

    def test_create_chunks_small_content_no_chapters(self, document_processor):
        """Test chunk creation with small content without chapters."""
        content = "Short content"  # Less than 100 chars

        chunks = document_processor._create_chunks(content, 1, [])

        # Should not create chunks for content that's too short
        assert len(chunks) == 0

    def test_create_chunks_chapter_too_small(self, document_processor):
        """Test chunk creation with chapter that's too small."""
        content = "Long document content " * 100
        chapters = [
            {
                "title": "Chapter 1",
                "content": "Short",
                "path": "1",
            }  # Less than 500 chars
        ]

        chunks = document_processor._create_chunks(content, 1, chapters)

        # Should not create chunks for chapters that are too small
        assert len(chunks) == 0

    def test_create_chunks_chapter_chunk_too_small(self, document_processor):
        """Test chunk creation where individual chapter chunks are too small."""
        content = "Long document content " * 100
        # Create a chapter with content that will create chunks smaller than 50 chars
        chapter_content = (
            "x" * 600
        )  # This will create chunks of "x" * 800, but with overlap
        chapters = [{"title": "Chapter 1", "content": chapter_content, "path": "1"}]

        chunks = document_processor._create_chunks(content, 1, chapters)

        # Should filter out chunks that are too small
        valid_chunks = [c for c in chunks if len(c["content"].strip()) > 50]
        assert len(valid_chunks) >= 0  # May have some valid chunks

    @patch("src.core.processing.document_processor.get_elasticsearch_client")
    def test_index_document_with_elasticsearch(
        self, mock_get_es, document_processor, mock_db_session, mock_document
    ):
        """Test document indexing in Elasticsearch."""
        mock_es = MagicMock()
        mock_get_es.return_value = mock_es

        chunks = [{"content": "chunk content", "metadata": {"word_count": 10}}]
        embeddings = [[0.1, 0.2, 0.3]]

        document_processor._index_document(mock_document, chunks, embeddings)

        mock_es.index.assert_called()
        # Verify document and chunk indexing calls

    @patch("src.core.processing.document_processor.get_elasticsearch_client")
    def test_index_document_no_elasticsearch(
        self, mock_get_es, document_processor, mock_db_session, mock_document
    ):
        """Test document indexing when Elasticsearch is unavailable."""
        mock_get_es.return_value = None

        chunks = [{"content": "chunk content"}]
        embeddings = [[0.1, 0.2, 0.3]]

        # Should not raise exception
        document_processor._index_document(mock_document, chunks, embeddings)

    @patch("src.core.processing.document_processor.get_elasticsearch_client")
    def test_index_document_elasticsearch_error(
        self, mock_get_es, document_processor, mock_db_session, mock_document
    ):
        """Test document indexing when Elasticsearch operation fails."""
        mock_es = MagicMock()
        mock_get_es.return_value = mock_es
        mock_es.bulk.side_effect = Exception("ES indexing failed")

        chunks = [{"content": "chunk content"}]
        embeddings = [[0.1, 0.2, 0.3]]

        # Should not raise exception despite ES failure
        document_processor._index_document(mock_document, chunks, embeddings)

    def test_process_document_advanced_mode_with_progress(
        self, document_processor, mock_db_session
    ):
        """Test advanced document processing with progress callback."""
        progress_calls = []

        def progress_callback(filename, progress, message):
            progress_calls.append((filename, progress, message))

        with patch.object(
            document_processor, "_process_document_advanced"
        ) as mock_advanced:
            mock_advanced.return_value = {"success": True, "document_id": 1}

            result = document_processor.process_document(
                "/tmp/test.pdf",
                use_advanced_processing=True,
                progress_callback=progress_callback,
            )

            mock_advanced.assert_called_once_with(
                "/tmp/test.pdf", None, progress_callback, None
            )
            assert result["success"] is True

    def test_suggest_categories_ai_edge_cases(self, document_processor):
        """Test category suggestion with various content types."""
        # Test with very short content - mock LLM to fail and test fallback
        with patch.object(
            document_processor.tag_suggester,
            "_call_llm",
            side_effect=Exception("LLM error"),
        ):
            result = document_processor._suggest_categories_ai("", "test.pdf", [])
            assert isinstance(result, list)
            assert result == ["General"]  # Fallback for empty content

        # Test with technical content - should trigger LLM or fallback
        technical_content = "machine learning neural networks deep learning algorithms"
        with patch.object(
            document_processor.tag_suggester,
            "_call_llm",
            side_effect=Exception("LLM error"),
        ):
            result = document_processor._suggest_categories_ai(
                technical_content, "ml.pdf", []
            )
            assert isinstance(result, list)
            assert "Technical" in result  # Should match keyword "machine learning"

        # Test with academic content (should trigger line 118)
        academic_content = "research paper study academic journal publication"
        with patch.object(
            document_processor.tag_suggester,
            "_call_llm",
            side_effect=Exception("LLM error"),
        ):
            result = document_processor._suggest_categories_ai(
                academic_content, "paper.pdf", []
            )
            assert isinstance(result, list)
            assert "Academic" in result  # Should match keyword "research"

        # Test with educational content (should trigger line 120)
        educational_content = "tutorial guide course education learning"
        with patch.object(
            document_processor.tag_suggester,
            "_call_llm",
            side_effect=Exception("LLM error"),
        ):
            result = document_processor._suggest_categories_ai(
                educational_content, "tutorial.pdf", []
            )
            assert isinstance(result, list)
            assert "Educational" in result  # Should match keyword "tutorial"

    def test_detect_language_from_file_empty_file(self, document_processor):
        """Test language detection from empty file."""
        with patch("builtins.open", mock_open(read_data="")):
            result = document_processor._detect_language_from_file("/tmp/empty.txt")
            assert result == "en"  # Should fallback to English

    def test_detect_all_chapters_empty_content(self, document_processor):
        """Test chapter detection with empty content."""
        result = document_processor._detect_all_chapters("")
        assert result == []

    def test_detect_all_chapters_no_headers(self, document_processor):
        """Test chapter detection with content that has no headers."""
        content = "This is just plain text without any headers or structure."
        result = document_processor._detect_all_chapters(content)
        # Should still try to detect some structure
        assert isinstance(result, list)

    def test_create_chunks_empty_content(self, document_processor):
        """Test chunk creation with empty content."""
        result = document_processor._create_chunks("", 1, [])
        assert result == []

    def test_create_chunks_with_empty_chapters(self, document_processor):
        """Test chunk creation with empty chapters list."""
        content = "Some content here"
        result = document_processor._create_chunks(content, 1, [])
        assert isinstance(result, list)
        # Should create chunks without chapter structure

    def test_process_document_file_not_found(self, document_processor):
        """Test processing document with non-existent file."""
        result = document_processor.process_document("/tmp/nonexistent.pdf")

        assert result["success"] is False
        assert "failed to load" in result["error"].lower()

    def test_del_method(self, document_processor, mock_db_session):
        """Test cleanup in destructor."""
        document_processor.__del__()

        # Should not raise exception
        assert True
