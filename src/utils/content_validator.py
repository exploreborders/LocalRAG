"""
Content quality validation utilities for LocalRAG system.

This module provides validation functions to ensure extracted document content
is meaningful and not corrupted by OCR artifacts or processing errors.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ContentValidator:
    """
    Validates the quality of extracted document content.
    """

    # Common OCR artifacts and corrupted text patterns
    OCR_ARTIFACTS = [
        r"=== page \d+ ===",
        r"=== page \d+ \(.*?\) ===",
        r"tesseract",
        r"easyocr",
        r"paddleocr",
        r"andrew",  # Common OCR corruption
        r"ocr mac",  # OCR engine artifacts
        r"ocrmac",
        r"\b[a-z]{20,}\b",  # Very long words (likely OCR errors)
        r"\b[A-Z]{10,}\b",  # Very long uppercase words
    ]

    # Patterns that indicate meaningful content
    MEANINGFUL_PATTERNS = [
        r"\b(the|and|or|but|in|on|at|to|for|of|with|by)\b",  # Common words
        r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # Proper names
        r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",  # Dates
        r"chapter|section|introduction|conclusion",  # Document structure
        r"algorithm|method|approach|technique",  # Technical terms
    ]

    @staticmethod
    def validate_content_quality(
        content: str, chunks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive content quality validation.

        Args:
            content: Full document content
            chunks: List of document chunks

        Returns:
            Dict with validation results and quality metrics
        """
        # Ensure content is a string
        if isinstance(content, dict):
            logger.error(f"Content is a dict instead of string: {content}")
            content = str(content)
        elif not isinstance(content, str):
            content = str(content)

        if not content:
            return {
                "is_valid": False,
                "quality_score": 0.0,
                "issues": ["Empty content"],
                "recommendations": ["Reprocess document with different extraction method"],
            }

        issues = []
        quality_score = 1.0

        # Check content length
        if len(content) < 500:
            issues.append("Content too short")
            quality_score *= 0.3

        # Check for OCR artifacts
        ocr_score = ContentValidator._check_ocr_artifacts(content)
        if ocr_score < 0.7:
            issues.append("High OCR artifact content")
            quality_score *= ocr_score

        # Check text structure
        structure_score = ContentValidator._check_text_structure(content)
        if structure_score < 0.5:
            issues.append("Poor text structure")
            quality_score *= structure_score

        # Check chunk quality if provided
        chunk_score = None
        if chunks:
            # Ensure chunks is a list of strings
            if isinstance(chunks, list):
                chunks = [str(chunk) if not isinstance(chunk, str) else chunk for chunk in chunks]
            else:
                chunks = [str(chunks)] if chunks else []

            chunk_score = ContentValidator._validate_chunks_quality(chunks)
            if chunk_score < 0.6:
                issues.append("Poor chunk quality")
                quality_score *= chunk_score

        # Determine overall validity
        is_valid = quality_score >= 0.5 and len(issues) <= 2

        recommendations = []
        if not is_valid:
            if "High OCR artifact content" in issues:
                recommendations.append("Try different OCR engine or preprocessing")
            if "Poor text structure" in issues:
                recommendations.append("Use alternative PDF text extraction method")
            if "Poor chunk quality" in issues:
                recommendations.append("Adjust chunking parameters or reprocess")
            if "Content too short" in issues:
                recommendations.append("Check if document contains text content")

        return {
            "is_valid": is_valid,
            "quality_score": quality_score,
            "issues": issues,
            "recommendations": recommendations,
            "metrics": {
                "content_length": len(content),
                "ocr_artifact_score": ocr_score,
                "structure_score": structure_score,
                "chunk_score": chunk_score,
            },
        }

    @staticmethod
    def _check_ocr_artifacts(content: str) -> float:
        """
        Check for OCR artifacts and corrupted text patterns.

        Returns:
            Score between 0-1, where 1 is clean content
        """
        content_lower = content.lower()
        artifact_score = 0

        for pattern in ContentValidator.OCR_ARTIFACTS:
            matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
            if matches > 0:
                # Penalize based on frequency of artifacts
                artifact_score += min(matches * 0.1, 0.5)

        # Check for excessive special characters (OCR corruption indicator)
        special_chars = len(re.findall(r"[^a-zA-Z0-9\s.,!?-]", content))
        special_ratio = special_chars / len(content) if content else 1

        if special_ratio > 0.1:  # More than 10% special characters
            artifact_score += min(special_ratio * 2, 0.5)

        return max(0, 1 - artifact_score)

    @staticmethod
    def _check_text_structure(content: str) -> float:
        """
        Check if content has proper text structure.

        Returns:
            Score between 0-1, where 1 indicates good structure
        """
        structure_score = 0

        # Check for meaningful patterns
        for pattern in ContentValidator.MEANINGFUL_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                structure_score += 0.2

        # Check sentence structure (periods followed by capital letters)
        sentences = re.findall(r"\. [A-Z]", content)
        if len(sentences) > 0:
            structure_score += min(len(sentences) / 50, 0.3)  # Up to 0.3 for sentence structure

        # Check paragraph structure (multiple lines)
        lines = content.split("\n")
        long_lines = [line for line in lines if len(line.strip()) > 50]
        if len(long_lines) > 0:
            structure_score += min(len(long_lines) / 20, 0.3)  # Up to 0.3 for paragraphs

        # Check word distribution (not too many very short/long words)
        words = re.findall(r"\b\w+\b", content)
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if 3 <= avg_word_length <= 8:  # Reasonable word length
                structure_score += 0.2

        return min(structure_score, 1.0)

    @staticmethod
    def _validate_chunks_quality(chunks: List[str]) -> float:
        """
        Validate the quality of document chunks.

        Returns:
            Score between 0-1, where 1 indicates good chunk quality
        """
        if not chunks:
            return 0.0

        meaningful_chunks = 0
        total_chunks = min(len(chunks), 20)  # Check first 20 chunks

        for chunk in chunks[:20]:
            if ContentValidator._is_chunk_meaningful(chunk):
                meaningful_chunks += 1

        return meaningful_chunks / total_chunks if total_chunks > 0 else 0.0

    @staticmethod
    def _is_chunk_meaningful(chunk: str) -> bool:
        """
        Check if a chunk contains meaningful content.
        """
        if len(chunk.strip()) < 20:
            return False

        # Check for OCR artifacts
        chunk_lower = chunk.lower()
        for pattern in ContentValidator.OCR_ARTIFACTS:
            if re.search(pattern, chunk_lower):
                return False

        # Check for meaningful content indicators
        has_meaningful_pattern = any(
            re.search(pattern, chunk, re.IGNORECASE)
            for pattern in ContentValidator.MEANINGFUL_PATTERNS
        )

        # Check word count and structure
        words = re.findall(r"\b\w+\b", chunk)
        has_good_word_count = len(words) >= 5
        has_sentences = "." in chunk or "!" in chunk or "?" in chunk

        return has_meaningful_pattern or (has_good_word_count and has_sentences)

    @staticmethod
    def suggest_reprocessing_method(validation_result: Dict[str, Any]) -> str:
        """
        Suggest the best reprocessing method based on validation results.
        """
        issues = validation_result.get("issues", [])

        if "High OCR artifact content" in issues:
            return "ocr_fallback"
        elif "Poor text structure" in issues:
            return "direct_extraction"
        elif "Content too short" in issues:
            return "hybrid_extraction"
        elif "Poor chunk quality" in issues:
            return "chunking_adjustment"
        else:
            return "standard_reprocessing"
