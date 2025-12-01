"""
Unit tests for ContentValidator class.

Tests content quality validation, OCR artifact detection, structure analysis,
and chunk quality assessment.
"""

from src.utils.content_validator import ContentValidator


class TestContentValidator:
    """Test the ContentValidator class functionality."""

    def test_validate_content_quality_empty_content(self):
        """Test validation of empty content."""
        result = ContentValidator.validate_content_quality("")

        assert result["is_valid"] is False
        assert result["quality_score"] == 0.0
        assert "Empty content" in result["issues"]

    def test_validate_content_quality_short_content(self):
        """Test validation of very short content."""
        short_content = "This is short."
        result = ContentValidator.validate_content_quality(short_content)

        assert result["is_valid"] is False
        assert "Content too short" in result["issues"]
        assert result["quality_score"] < 1.0

    def test_validate_content_quality_good_content(self):
        """Test validation of good quality content."""
        good_content = """
        This is a comprehensive document about artificial intelligence.
        Artificial intelligence is a method of data analysis that automates analytical model building.
        It is a branch of computer science based on the idea that systems can learn from data,
        identify patterns and make decisions with minimal human intervention.

        The process involves training algorithms on large datasets to recognize patterns.
        These patterns can then be used to make predictions or decisions on new data.
        """
        result = ContentValidator.validate_content_quality(good_content)

        assert result["is_valid"] is True
        assert result["quality_score"] >= 0.5
        # Allow some minor issues as long as overall quality is good
        assert (
            len(
                [
                    issue
                    for issue in result["issues"]
                    if "critical" in issue.lower() or "poor" in issue.lower()
                ]
            )
            == 0
        )

    def test_validate_content_quality_with_ocr_artifacts(self):
        """Test validation of content with OCR artifacts."""
        content_with_artifacts = """
        === page 1 ===
        This document contains tesseract OCR artifacts.
        The text has some andrew corruption from OCR processing.
        paddleocr has also left some marks.
        """

        result = ContentValidator.validate_content_quality(content_with_artifacts)

        assert "High OCR artifact content" in result["issues"]
        assert result["metrics"]["ocr_artifact_score"] < 1.0

    def test_validate_content_quality_poor_structure(self):
        """Test validation of content with poor structure."""
        poor_content = (
            "word word word word word word word word word word"  # No sentences, just words
        )

        result = ContentValidator.validate_content_quality(poor_content)

        assert "Poor text structure" in result["issues"]
        assert result["metrics"]["structure_score"] < 0.5

    def test_validate_content_quality_with_chunks(self):
        """Test validation with chunk quality assessment."""
        content = "This is a test document with multiple sentences. It has proper structure and meaningful content."
        chunks = [
            "This is a test document",
            "with multiple sentences.",
            "It has proper structure",
            "and meaningful content.",
        ]

        result = ContentValidator.validate_content_quality(content, chunks)

        # chunk_score is only calculated if chunks are provided
        if chunks:
            assert "chunk_score" in result["metrics"]
            assert isinstance(result["metrics"]["chunk_score"], float)
        else:
            assert "chunk_score" not in result["metrics"]

    def test_check_ocr_artifacts_clean_content(self):
        """Test OCR artifact detection on clean content."""
        clean_content = "This is clean, readable text without any OCR artifacts."

        score = ContentValidator._check_ocr_artifacts(clean_content)
        assert score == 1.0  # Perfect score for clean content

    def test_check_ocr_artifacts_with_patterns(self):
        """Test OCR artifact detection with various patterns."""
        content_with_patterns = """
        === page 1 ===
        This has tesseract artifacts and andrew corruption.
        Also contains paddleocr marks and easyocr traces.
        """

        score = ContentValidator._check_ocr_artifacts(content_with_patterns)
        assert score < 1.0  # Should be reduced due to artifacts

    def test_check_ocr_artifacts_special_characters(self):
        """Test OCR artifact detection with excessive special characters."""
        content_with_specials = "Normal text with @#$%^&*() excessive special characters!"

        score = ContentValidator._check_ocr_artifacts(content_with_specials)
        assert score < 1.0  # Should be reduced due to special chars

    def test_check_text_structure_good_structure(self):
        """Test text structure analysis on well-structured content."""
        good_content = """
        This is the first paragraph. It contains multiple sentences.
        Each sentence ends with proper punctuation!

        This is the second paragraph. It also has good structure.
        The content flows logically from one idea to another.
        """

        score = ContentValidator._check_text_structure(good_content)
        assert score > 0.5  # Should have good structure score

    def test_check_text_structure_poor_structure(self):
        """Test text structure analysis on poorly structured content."""
        poor_content = (
            "word word word word word word word word word word word word word word word word"
        )

        score = ContentValidator._check_text_structure(poor_content)
        assert score < 0.5  # Should have poor structure score

    def test_check_text_structure_with_patterns(self):
        """Test text structure with meaningful patterns."""
        content_with_patterns = """
        This document discusses machine learning algorithms.
        The algorithm section explains the technical approach.
        Chapter 3 covers implementation details.
        """

        score = ContentValidator._check_text_structure(content_with_patterns)
        assert score > 0.3  # Should detect meaningful patterns

    def test_validate_chunks_quality_good_chunks(self):
        """Test chunk quality validation with good chunks."""
        good_chunks = [
            "This is a meaningful chunk with proper content.",
            "It contains multiple sentences and good structure.",
            "The chunk has enough words to be substantial.",
        ]

        score = ContentValidator._validate_chunks_quality(good_chunks)
        assert score > 0.5  # Should have good chunk quality

    def test_validate_chunks_quality_poor_chunks(self):
        """Test chunk quality validation with poor chunks."""
        poor_chunks = [
            "Short",
            "Very short chunk",
            "=== page 1 ===",  # OCR artifact
            "tesseract",  # OCR artifact
        ]

        score = ContentValidator._validate_chunks_quality(poor_chunks)
        assert score < 0.5  # Should have poor chunk quality

    def test_validate_chunks_quality_empty_list(self):
        """Test chunk quality validation with empty chunk list."""
        score = ContentValidator._validate_chunks_quality([])
        assert score == 0.0

    def test_is_chunk_meaningful_good_chunk(self):
        """Test chunk meaningfulness detection for good chunks."""
        good_chunk = "This is a meaningful chunk with proper sentences and structure."

        assert ContentValidator._is_chunk_meaningful(good_chunk) is True

    def test_is_chunk_meaningful_short_chunk(self):
        """Test chunk meaningfulness detection for short chunks."""
        short_chunk = "Short"

        assert ContentValidator._is_chunk_meaningful(short_chunk) is False

    def test_is_chunk_meaningful_with_ocr_artifacts(self):
        """Test chunk meaningfulness detection with OCR artifacts."""
        chunk_with_artifacts = "This chunk contains tesseract OCR artifacts."

        assert ContentValidator._is_chunk_meaningful(chunk_with_artifacts) is False

    def test_is_chunk_meaningful_no_sentences(self):
        """Test chunk meaningfulness detection for chunks without sentences."""
        # Test with a truly meaningless chunk
        meaningless = "a a a a a"
        assert ContentValidator._is_chunk_meaningful(meaningless) is False

    def test_suggest_reprocessing_method_ocr_artifacts(self):
        """Test reprocessing method suggestion for OCR artifacts."""
        validation_result = {"issues": ["High OCR artifact content"]}

        method = ContentValidator.suggest_reprocessing_method(validation_result)
        assert method == "ocr_fallback"

    def test_suggest_reprocessing_method_poor_structure(self):
        """Test reprocessing method suggestion for poor structure."""
        validation_result = {"issues": ["Poor text structure"]}

        method = ContentValidator.suggest_reprocessing_method(validation_result)
        assert method == "direct_extraction"

    def test_suggest_reprocessing_method_poor_chunks(self):
        """Test reprocessing method suggestion for poor chunks."""
        validation_result = {"issues": ["Poor chunk quality"]}

        method = ContentValidator.suggest_reprocessing_method(validation_result)
        assert method == "chunking_adjustment"

    def test_suggest_reprocessing_method_short_content(self):
        """Test reprocessing method suggestion for short content."""
        validation_result = {"issues": ["Content too short"]}

        method = ContentValidator.suggest_reprocessing_method(validation_result)
        assert method == "hybrid_extraction"

    def test_suggest_reprocessing_method_default(self):
        """Test default reprocessing method suggestion."""
        validation_result = {"issues": ["Some other issue"]}

        method = ContentValidator.suggest_reprocessing_method(validation_result)
        assert method == "standard_reprocessing"

    def test_validate_content_quality_dict_input(self):
        """Test validation with dict input (should handle gracefully)."""
        # This tests the type checking/conversion logic
        result = ContentValidator.validate_content_quality({"invalid": "input"})

        assert result["is_valid"] is False
        # Dict gets converted to string, so it should have content but may have other issues
        assert len(result["issues"]) > 0

    def test_validate_content_quality_none_chunks(self):
        """Test validation with None chunks parameter."""
        content = (
            "This is comprehensive content with multiple sentences. " * 20
        )  # Make it long enough
        result = ContentValidator.validate_content_quality(content, None)

        assert result["is_valid"] is True
        assert result["metrics"]["chunk_score"] is None  # Should be None when no chunks

    def test_validate_content_quality_string_chunks(self):
        """Test validation with string chunks (should convert to list)."""
        content = "This is test content."
        chunks = "single chunk"  # String instead of list

        result = ContentValidator.validate_content_quality(content, chunks)

        # Should handle the string gracefully (convert to list)
        assert "chunk_score" in result["metrics"]

    def test_check_ocr_artifacts_no_artifacts(self):
        """Test OCR artifact detection with clean content."""
        clean_content = "This is perfectly clean content with no OCR artifacts."

        score = ContentValidator._check_ocr_artifacts(clean_content)
        assert score == 1.0

    def test_check_ocr_artifacts_mixed_content(self):
        """Test OCR artifact detection with mixed clean and artifact content."""
        mixed_content = "This is clean text with tesseract artifacts and some andrew corruption."

        score = ContentValidator._check_ocr_artifacts(mixed_content)
        assert 0 < score < 1.0  # Should be reduced but not zero

    def test_check_text_structure_minimal_structure(self):
        """Test text structure with minimal but acceptable structure."""
        minimal_content = "This is a sentence. This is another sentence."

        score = ContentValidator._check_text_structure(minimal_content)
        assert score > 0  # Should have some structure score

    def test_check_text_structure_no_structure(self):
        """Test text structure with no discernible structure."""
        no_structure = "word word word word word word word word word word word word"

        score = ContentValidator._check_text_structure(no_structure)
        assert score < 0.5  # Should have low structure score

    def test_validate_chunks_quality_mixed_quality(self):
        """Test chunk quality with mixed good and bad chunks."""
        mixed_chunks = [
            "This is a good chunk with proper content and structure.",
            "Short",  # Bad chunk
            "This is another good chunk with meaningful content.",
            "tesseract",  # OCR artifact - bad
        ]

        score = ContentValidator._validate_chunks_quality(mixed_chunks)
        assert 0 < score < 1.0  # Should be between 0 and 1

    def test_is_chunk_meaningful_edge_cases(self):
        """Test chunk meaningfulness with various edge cases."""
        # Test chunk with only punctuation
        assert ContentValidator._is_chunk_meaningful("!!!???") is False

        # Test chunk with numbers only
        assert ContentValidator._is_chunk_meaningful("123 456 789") is False

        # Test chunk with single long word
        long_word = "a" * 50  # Very long word
        assert ContentValidator._is_chunk_meaningful(long_word) is False

    def test_suggest_reprocessing_method_various_issues(self):
        """Test reprocessing suggestions for different issue combinations."""
        # Test with multiple issues
        validation_result = {"issues": ["Poor text structure", "Content too short"]}

        method = ContentValidator.suggest_reprocessing_method(validation_result)
        assert method in [
            "direct_extraction",
            "hybrid_extraction",
            "standard_reprocessing",
        ]

    def test_validate_content_quality_edge_cases(self):
        """Test validation with various edge cases."""
        # Test with None content
        result = ContentValidator.validate_content_quality(None)
        assert result["is_valid"] is False

        # Test with very long content
        long_content = "This is a sentence. " * 1000  # Very long content
        result = ContentValidator.validate_content_quality(long_content)
        assert "is_valid" in result
        assert "quality_score" in result

    def test_validate_content_quality_ocr_only_content(self):
        """Test validation of content that consists only of OCR artifacts."""
        ocr_only = "tesseract andrew paddleocr easyocr ocr mac"

        result = ContentValidator.validate_content_quality(ocr_only)
        assert result["is_valid"] is False
        assert "High OCR artifact content" in result["issues"]
        assert result["metrics"]["ocr_artifact_score"] <= 0.5
