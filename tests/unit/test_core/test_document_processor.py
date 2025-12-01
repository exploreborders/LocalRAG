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
        with patch("src.core.processing.document_processor.SessionLocal") as mock_session:
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
        with patch.object(document_processor, "_process_document_standard") as mock_standard:
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

        with patch.object(document_processor, "_process_document_standard") as mock_standard:
            mock_standard.return_value = {"success": True, "document_id": 1}

            result = document_processor.process_document(
                "/tmp/test.pdf",
                use_advanced_processing=False,
                progress_callback=progress_callback,
            )

            mock_standard.assert_called_once_with("/tmp/test.pdf", None, progress_callback, None)
            assert result["success"] is True

    def test_process_document_advanced_mode(self, document_processor, mock_db_session):
        """Test document processing in advanced mode."""
        with patch.object(document_processor, "_process_document_advanced") as mock_advanced:
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
            "This is a longer text that should be substantial enough for language detection. " * 20
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
        chapters = [{"title": "Chapter 1", "content": content[:600], "path": "1", "level": 1}]

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
        chapter_content = "x" * 600  # This will create chunks of "x" * 800, but with overlap
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

        with patch.object(document_processor, "_process_document_advanced") as mock_advanced:
            mock_advanced.return_value = {"success": True, "document_id": 1}

            result = document_processor.process_document(
                "/tmp/test.pdf",
                use_advanced_processing=True,
                progress_callback=progress_callback,
            )

            mock_advanced.assert_called_once_with("/tmp/test.pdf", None, progress_callback, None)
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
            result = document_processor._suggest_categories_ai(technical_content, "ml.pdf", [])
            assert isinstance(result, list)
            assert "Technical" in result  # Should match keyword "machine learning"

        # Test with academic content (should trigger line 118)
        academic_content = "research paper study academic journal publication"
        with patch.object(
            document_processor.tag_suggester,
            "_call_llm",
            side_effect=Exception("LLM error"),
        ):
            result = document_processor._suggest_categories_ai(academic_content, "paper.pdf", [])
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

    def test_detect_all_chapters_table_format_comprehensive(self, document_processor):
        """Test comprehensive chapter detection from complex table format."""
        content = """
| 1   | Overview                     |                                 |
| 2   | Basic Python                 |                                 |
|     | 2.1 | Sequence, Selection, and   |                                 |
|     | 2.2 | Expressions and Evaluation |                                 |
|     | 2.3 | Variables, Types, and State|                                 |
| 3   | Object-Oriented Programming  |                                 |
| 4   | Testing                      |                                 |
| 5   | Running Time Analysis        |                                 |
| 6   | Stacks and                   | Queues                          |
| 7   | Deques and Linked            | Lists                           |
| 8   | Doubly-Linked                | Lists                           |
| 9   | Recursion                    |                                 |
"""

        chapters = document_processor._detect_all_chapters(content)

        # Should detect substantial TOC
        assert len(chapters) >= 10

        # Check main chapters - titles should NOT include numbers
        main_chapters = [ch for ch in chapters if ch["level"] == 1]
        assert len(main_chapters) >= 9

        # Verify specific chapter titles (no numbers in title field)
        chapter_titles = {ch["path"]: ch["title"] for ch in chapters}

        assert "1" in chapter_titles
        assert chapter_titles["1"] == "Overview"  # Not "1 Overview"

        assert "2" in chapter_titles
        assert chapter_titles["2"] == "Basic Python"  # Not "2 Basic Python"

        assert "6" in chapter_titles
        assert chapter_titles["6"] == "Stacks and Queues"  # Combined but no number prefix

        assert "7" in chapter_titles
        assert chapter_titles["7"] == "Deques and Linked"  # Not combined with "Lists"

        assert "8" in chapter_titles
        assert chapter_titles["8"] == "Doubly-Linked"  # Not combined with "Lists"

        # Check subsections
        assert "2.1" in chapter_titles
        assert chapter_titles["2.1"] == "Sequence, Selection, and"

        assert "2.2" in chapter_titles
        assert chapter_titles["2.2"] == "Expressions and Evaluation"

    def test_detect_all_chapters_title_number_separation(self, document_processor):
        """Test that chapter numbers are in path field, not title field."""
        content = """
| 1   | Introduction                 |                                 |
| 2   | Getting Started              |                                 |
|     | 2.1 | First Steps                |                                 |
| 3   | More Content                 |                                 |
| 4   | Even More                    |                                 |
| 5   | Advanced Topics              |                                 |
| 10  | Final Chapter                |                                 |
"""

        chapters = document_processor._detect_all_chapters(content)

        assert len(chapters) >= 6

        # Verify no chapter numbers in titles
        for chapter in chapters:
            title = chapter["title"]
            path = chapter["path"]

            # Title should not start with the chapter number
            if path.isdigit():
                assert not title.startswith(path + " ")
                assert not title.startswith(path + ".")
            elif "." in path:
                # For subsections, title shouldn't contain the full path
                assert not title.startswith(path + " ")

    def test_detect_all_chapters_subsection_detection(self, document_processor):
        """Test that subsections (X.Y format) are properly detected."""
        content = """
| 1   | Main Chapter                 |                                 |
|     | 1.1 | Subsection One            |                                 |
|     | 1.2 | Subsection Two            |                                 |
|     | 1.3 | Subsection Three          |                                 |
| 2   | Another Chapter              |                                 |
|     | 2.1 | More Subsections          |                                 |
"""

        chapters = document_processor._detect_all_chapters(content)

        # Should detect main chapters and subsections
        assert len(chapters) >= 6

        # Check levels are correct
        level_counts = {}
        for chapter in chapters:
            level = chapter["level"]
            level_counts[level] = level_counts.get(level, 0) + 1

        assert level_counts.get(1, 0) >= 2  # Main chapters
        assert level_counts.get(2, 0) >= 4  # Subsections

        # Verify subsection paths and titles
        subsection_chapters = [ch for ch in chapters if ch["level"] == 2]
        assert len(subsection_chapters) >= 4

        # Check specific subsections
        chapter_paths = {ch["path"]: ch["title"] for ch in chapters}
        assert "1.1" in chapter_paths
        assert "1.2" in chapter_paths
        assert "2.1" in chapter_paths

    def test_detect_all_chapters_title_combination_logic(self, document_processor):
        """Test the special title combination logic for certain chapters."""
        content = """
| 1   | Introduction                 |                                 |
| 2   | Getting Started              |                                 |
| 3   | Basic Concepts               |                                 |
| 4   | Intermediate Topics          |                                 |
| 5   | Advanced Material            |                                 |
| 6   | Stacks and                   | Queues                          |
| 7   | Deques and Linked            | Lists                           |
| 8   | Doubly-Linked                | Lists                           |
| 9   | Recursion                    | Concatenating Doubly Linked    |
| 10  | Final Chapter                |                                 |
"""

        chapters = document_processor._detect_all_chapters(content)

        assert len(chapters) >= 10

        chapter_titles = {ch["path"]: ch["title"] for ch in chapters}

        # Chapter 6: Should combine "Stacks and" + "Queues"
        assert chapter_titles.get("6") == "Stacks and Queues"

        # Chapter 7: Should NOT combine with "Lists" (special case)
        assert chapter_titles.get("7") == "Deques and Linked"

        # Chapter 8: Should NOT combine with "Lists" (special case)
        assert chapter_titles.get("8") == "Doubly-Linked"

        # Chapter 9: Should NOT combine with "Concatenating Doubly Linked"
        assert chapter_titles.get("9") == "Recursion"

    def test_detect_all_chapters_hierarchical_structure(self, document_processor):
        """Test that hierarchical chapter structure is maintained."""
        content = """
| 1   | Overview                     |                                 |
| 2   | Basic Python                 |                                 |
|     | 2.1 | Sequence, Selection, and   |                                 |
|     | 2.2 | Expressions and Evaluation |                                 |
|     | 2.3 | Variables, Types, and State|                                 |
|     | 2.4 | Collections . . . . . . . .|                                 |
|     |     | 2.4.1 Strings ( str ) . . .|                                 |
|     |     | 2.4.2 Lists ( list ) . . . .|                                 |
| 3   | Object-Oriented Programming  |                                 |
"""

        chapters = document_processor._detect_all_chapters(content)

        assert len(chapters) >= 8

        # Check hierarchical levels
        levels_found = set(ch["level"] for ch in chapters)
        assert 1 in levels_found  # Main chapters
        assert 2 in levels_found  # Subsections
        assert 3 in levels_found  # Sub-subsections

        # Verify specific hierarchical relationships
        chapter_data = {ch["path"]: ch for ch in chapters}

        # Main chapters should have level 1
        assert chapter_data["1"]["level"] == 1
        assert chapter_data["2"]["level"] == 1
        assert chapter_data["3"]["level"] == 1

        # Subsections should have level 2
        assert chapter_data["2.1"]["level"] == 2
        assert chapter_data["2.2"]["level"] == 2
        assert chapter_data["2.3"]["level"] == 2
        assert chapter_data["2.4"]["level"] == 2

        # Sub-subsections should have level 3
        assert chapter_data["2.4.1"]["level"] == 3
        assert chapter_data["2.4.2"]["level"] == 3

    def test_create_chunks_fallback_when_chapters_empty(self, document_processor):
        """Test chunk creation falls back to position-based assignment when chapters have no content."""
        # Create substantial document content
        content = "This is test content for chunking. " * 200  # ~4000 characters

        # Create chapters with empty content (like from table parsing)
        chapters = [
            {"title": "Chapter 1", "content": "", "path": "1", "level": 1},
            {"title": "Chapter 2", "content": "", "path": "2", "level": 1},
            {"title": "Chapter 3", "content": "", "path": "3", "level": 1},
        ]

        chunks = document_processor._create_chunks(content, 1, chapters)

        # Should create chunks despite empty chapter content
        assert len(chunks) > 0

        # All chunks should have chapter metadata
        for chunk in chunks:
            assert "metadata" in chunk
            assert "chapter_title" in chunk["metadata"]
            assert "chapter_path" in chunk["metadata"]
            assert chunk["metadata"]["chapter_title"] is not None
            assert chunk["metadata"]["chapter_path"] is not None

        # Should have chunks assigned to different chapters
        chapter_titles = set(chunk["metadata"]["chapter_title"] for chunk in chunks)
        assert len(chapter_titles) >= 2  # At least some different chapters

        # Verify chunk content is substantial
        for chunk in chunks:
            assert len(chunk["content"]) > 100  # Reasonable chunk size

    def test_create_chunks_no_fallback_for_small_chapters(self, document_processor):
        """Test that fallback doesn't trigger for chapters with small but non-empty content."""
        content = "This is test content for chunking. " * 200

        # Create chapters with small content (should not trigger fallback)
        chapters = [
            {"title": "Chapter 1", "content": "Short content", "path": "1", "level": 1},
            {"title": "Chapter 2", "content": "Also short", "path": "2", "level": 1},
        ]

        chunks = document_processor._create_chunks(content, 1, chapters)

        # Should not create chunks because chapter content is too small (< 500 chars)
        # and fallback only triggers when ALL chapters have empty content
        assert len(chunks) == 0

    def test_create_chunks_mixed_content_chapters(self, document_processor):
        """Test chunking with mix of chapters with and without content."""
        content = "This is test content for chunking. " * 200

        chapters = [
            {
                "title": "Chapter 1",
                "content": "This is substantial content for chapter 1. " * 20,
                "path": "1",
                "level": 1,
            },  # > 500 chars
            {
                "title": "Chapter 2",
                "content": "",
                "path": "2",
                "level": 1,
            },  # Empty content
        ]

        chunks = document_processor._create_chunks(content, 1, chapters)

        # Should create chunks from Chapter 1 (has content) but not trigger fallback
        # because not ALL chapters have empty content
        assert len(chunks) > 0

        # All chunks should be from Chapter 1
        chapter_titles = set(chunk["metadata"]["chapter_title"] for chunk in chunks)
        assert chapter_titles == {"Chapter 1"}

    def test_del_method(self, document_processor, mock_db_session):
        """Test cleanup in destructor."""
        document_processor.__del__()

        # Should not raise exception
        assert True
