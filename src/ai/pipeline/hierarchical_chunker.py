"""
Hierarchical document chunking with chapter-aware processing.

This module provides intelligent text chunking that respects document
structure and maintains hierarchical relationships for better retrieval.
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class HierarchicalChunker:
    """
    Chapter-aware document chunking with hierarchical structure preservation.

    Creates chunks that maintain document hierarchy and provide better
    context for retrieval and generation.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        min_chunk_size: int = 50,  # Reduced for hierarchical chunks
        max_chunk_size: int = 2000,
    ):
        """
        Initialize the hierarchical chunker.

        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk_document(
        self, text: str, structure: Dict[str, Any], filename: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Create hierarchical chunks from document text and structure.

        Args:
            text: Full document text
            structure: Document structure analysis
            filename: Document filename

        Returns:
            List of chunk dictionaries with metadata
        """
        logger.info(f"Creating hierarchical chunks for: {filename}")

        # Extract hierarchical elements
        hierarchy = structure.get("hierarchy", [])
        logger.info(f"Found {len(hierarchy)} hierarchy items")

        if not hierarchy:
            # Fallback to standard chunking if no hierarchy
            logger.info("Using standard chunking (no hierarchy available)")
            return self._standard_chunking(text, filename)

        # Create chapter-based chunks
        chunks = []

        for chapter in hierarchy:
            chapter_chunks = self._chunk_chapter(text, chapter, hierarchy, filename)
            chunks.extend(chapter_chunks)

        # If no chunks were created, fallback to standard chunking
        if not chunks:
            return self._standard_chunking(text, filename)

        # Post-process chunks
        chunks = self._post_process_chunks(chunks)

        logger.info(f"Created {len(chunks)} hierarchical chunks")
        return chunks

    def _chunk_chapter(
        self,
        full_text: str,
        chapter: Dict[str, Any],
        hierarchy: List[Dict[str, Any]],
        filename: str,
    ) -> List[Dict[str, Any]]:
        """
        Create chunks for a specific chapter with hierarchical context.
        """
        chunks = []
        chapter_path = chapter.get("path", "")
        chapter_title = chapter.get("title", "")

        logger.debug(f"Processing chapter: {chapter_title} (path: {chapter_path})")

        # Find chapter content in full text
        chapter_content = self._extract_chapter_content(full_text, chapter, hierarchy)

        logger.debug(f"Chapter content length: {len(chapter_content) if chapter_content else 0}")

        if not chapter_content:
            logger.debug(f"No content found for chapter: {chapter_title}")
            return chunks

        # Split chapter into semantic units
        sections = self._split_into_sections(chapter_content)
        logger.debug(f"Split into {len(sections)} sections")

        current_pos = 0
        chunk_index = 0

        for section in sections:
            logger.debug(f"Processing section with {len(section)} chars")
            section_chunks = self._chunk_section(
                section, chapter_path, chapter_title, current_pos, chunk_index, filename
            )
            logger.debug(f"Created {len(section_chunks)} chunks from section")
            chunks.extend(section_chunks)
            current_pos += len(section)
            chunk_index += len(section_chunks)

        logger.debug(f"Total chunks from chapter: {len(chunks)}")
        return chunks

    def _extract_chapter_content(
        self, full_text: str, chapter: Dict[str, Any], hierarchy: List[Dict[str, Any]]
    ) -> str:
        """
        Extract content belonging to a specific chapter.
        """
        # Find the start position of this chapter
        chapter_title = chapter.get("title", "")
        chapter_path = chapter.get("path", "")

        # Look for chapter markers in text
        patterns = [
            rf"(?i)(chapter|section)\s*{chapter_path}\b.*?$(.*?)(?=(chapter|section)\s*\d|\Z)",
            rf"(?i){re.escape(chapter_title)}.*?$(.*?)(?=(chapter|section)\s*\d|#+\s|\Z)",
            rf"#+\s*{re.escape(chapter_title)}.*?$(.*?)(?=#+\s|\Z)",  # Markdown headers
        ]

        for pattern in patterns:
            match = re.search(pattern, full_text, re.DOTALL | re.MULTILINE)
            if match and len(match.groups()) > 1 and match.group(2):
                return match.group(2).strip()

        # Fallback: estimate based on hierarchy
        return self._estimate_chapter_content(full_text, chapter, hierarchy)

    def _estimate_chapter_content(
        self, full_text: str, chapter: Dict[str, Any], hierarchy: List[Dict[str, Any]]
    ) -> str:
        """
        Estimate chapter content when direct extraction fails.
        """
        # Simple estimation: divide text by number of chapters
        total_chapters = len([h for h in hierarchy if h.get("level") == 1])
        if total_chapters == 0:
            return full_text

        chapter_index = int(chapter.get("path", "1").split(".")[0]) - 1
        text_parts = self._split_text_by_chapters(full_text, total_chapters)

        if chapter_index < len(text_parts):
            return text_parts[chapter_index]

        return full_text

    def _split_text_by_chapters(self, text: str, num_chapters: int) -> List[str]:
        """Split text into roughly equal parts based on chapter count."""
        if num_chapters <= 1:
            return [text]

        total_length = len(text)
        part_length = total_length // num_chapters

        parts = []
        start = 0

        for i in range(num_chapters):
            end = start + part_length
            if i == num_chapters - 1:  # Last part
                end = total_length

            # Try to end at a sentence boundary
            while end < total_length and text[end] not in ".!?\n":
                end += 1
            end = min(end + 1, total_length)

            parts.append(text[start:end])
            start = end

        return parts

    def _split_into_sections(self, chapter_content: str) -> List[str]:
        """
        Split chapter content into logical sections.
        """
        # Look for section markers
        section_patterns = [
            r"(?m)^(?:\d+\.\d+|\([a-z]\)|\•|\-)\s+",  # Numbered sections, bullets
            r"(?m)^[A-Z][^.!?\n]{10,80}?\n",  # Short capitalized lines (headings)
        ]

        sections = []
        last_pos = 0

        for pattern in section_patterns:
            for match in re.finditer(pattern, chapter_content):
                start = match.start()
                if start > last_pos:
                    sections.append(chapter_content[last_pos:start])
                    last_pos = start

        # Add remaining content
        if last_pos < len(chapter_content):
            sections.append(chapter_content[last_pos:])

        # If no sections found, treat whole chapter as one section
        if not sections:
            sections = [chapter_content]

        return sections

    def _chunk_section(
        self,
        section_text: str,
        chapter_path: str,
        chapter_title: str,
        global_pos: int,
        start_index: int,
        filename: str,
    ) -> List[Dict[str, Any]]:
        """
        Create chunks from a section with appropriate metadata.
        """
        chunks = []
        words = section_text.split()
        current_chunk = []
        current_length = 0
        chunk_start = 0

        i = 0
        while i < len(words):
            word = words[i]
            word_length = len(word) + 1  # +1 for space

            # Check if adding this word would exceed chunk size
            if current_length + word_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        chunk_text,
                        chapter_path,
                        chapter_title,
                        start_index + len(chunks),
                        global_pos + chunk_start,
                        filename,
                    )
                    chunks.append(chunk)

                # Start new chunk with overlap
                overlap_words = self._get_overlap_words(current_chunk, self.overlap)
                current_chunk = overlap_words + [word]
                current_length = sum(len(w) + 1 for w in current_chunk)
                chunk_start += len(" ".join(current_chunk[: -len(overlap_words)]))
            else:
                current_chunk.append(word)
                current_length += word_length

            i += 1

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunk = self._create_chunk(
                    chunk_text,
                    chapter_path,
                    chapter_title,
                    start_index + len(chunks),
                    global_pos + chunk_start,
                    filename,
                )
                chunks.append(chunk)

        return chunks

    def _get_overlap_words(self, words: List[str], max_overlap: int) -> List[str]:
        """Get overlap words from the end of a chunk."""
        overlap_words = []
        current_length = 0

        for word in reversed(words):
            word_length = len(word) + 1
            if current_length + word_length > max_overlap:
                break
            overlap_words.insert(0, word)
            current_length += word_length

        return overlap_words

    def _create_chunk(
        self,
        text: str,
        chapter_path: str,
        chapter_title: str,
        chunk_index: int,
        global_pos: int,
        filename: str,
    ) -> Dict[str, Any]:
        """
        Create a chunk dictionary with metadata.
        """
        # Determine section type based on path
        path_parts = chapter_path.split(".")
        level = len(path_parts)

        if level == 1:
            section_type = "chapter"
        elif level == 2:
            section_type = "section"
        elif level >= 3:
            section_type = "subsection"
        else:
            section_type = "paragraph"

        # Calculate content relevance score
        relevance_score = self._calculate_relevance(text, chapter_title)

        return {
            "content": text,
            "chunk_index": chunk_index,
            "chapter_title": chapter_title,
            "chapter_path": chapter_path,
            "section_type": section_type,
            "content_relevance": relevance_score,
            "word_count": len(text.split()),
            "filename": filename,
            "global_position": global_pos,
        }

    def _calculate_relevance(self, text: str, chapter_title: str) -> float:
        """
        Calculate content relevance score based on text characteristics.
        """
        if not text or not chapter_title:
            return 0.5

        # Simple relevance calculation
        text_lower = text.lower()
        title_words = set(chapter_title.lower().split())

        # Count title words in text
        title_word_count = sum(1 for word in title_words if word in text_lower)

        # Calculate density
        density = title_word_count / len(text.split()) if text.split() else 0

        # Technical content indicators
        technical_indicators = ["algorithm", "method", "analysis", "system", "model"]
        technical_score = (
            sum(1 for indicator in technical_indicators if indicator in text_lower) * 0.1
        )

        # Length factor (longer chunks tend to be more substantial)
        length_factor = min(1.0, len(text) / 500)

        relevance = min(1.0, density * 2 + technical_score + length_factor * 0.2)
        return round(relevance, 3)

    def _standard_chunking(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """
        Fallback to standard chunking when hierarchical structure is unavailable.
        """
        logger.info("Using standard chunking (no hierarchy available)")

        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        chunk_index = 0

        i = 0
        while i < len(words):
            word = words[i]
            word_length = len(word) + 1

            if current_length + word_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = {
                        "content": chunk_text,
                        "chunk_index": chunk_index,
                        "chapter_title": "General Content",
                        "chapter_path": "1",
                        "section_type": "general",
                        "content_relevance": 0.5,
                        "word_count": len(chunk_text.split()),
                        "filename": filename,
                        "global_position": i - len(current_chunk),
                    }
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk with overlap
                overlap_words = self._get_overlap_words(current_chunk, self.overlap)
                current_chunk = overlap_words + [word]
                current_length = sum(len(w) + 1 for w in current_chunk)
            else:
                current_chunk.append(word)
                current_length += word_length

            i += 1

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunk = {
                    "content": chunk_text,
                    "chunk_index": chunk_index,
                    "chapter_title": "General Content",
                    "chapter_path": "1",
                    "section_type": "general",
                    "content_relevance": 0.5,
                    "word_count": len(chunk_text.split()),
                    "filename": filename,
                    "global_position": i - len(current_chunk),
                }
                chunks.append(chunk)

        return chunks

    def _post_process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Post-process chunks to ensure quality and consistency.
        """
        processed_chunks = []

        for chunk in chunks:
            original_word_count = chunk["word_count"]

            # More aggressive filtering for vision OCR content
            content = chunk["content"]

            # Skip chunks that are mostly OCR artifacts
            if self._is_ocr_artifact_chunk(content):
                logger.debug(f"Filtering out OCR artifact chunk: {content[:100]}...")
                continue

            # Skip very small chunks, with higher minimum for vision content
            min_words = max(10, self.min_chunk_size // 10)  # 10 words minimum for vision content
            if chunk["word_count"] < min_words:
                logger.debug(
                    f"Filtering out small chunk: {chunk['word_count']} words < {min_words} (content: {content[:100]}...)"
                )
                continue

            # Trim excessive whitespace
            chunk["content"] = " ".join(chunk["content"].split())
            new_word_count = len(chunk["content"].split())

            # Update word count if it changed
            if new_word_count != original_word_count:
                chunk["word_count"] = new_word_count
                logger.debug(f"Updated word count from {original_word_count} to {new_word_count}")
                continue

            # Trim excessive whitespace
            chunk["content"] = " ".join(chunk["content"].split())
            new_word_count = len(chunk["content"].split())

            # Update word count if it changed
            if new_word_count != original_word_count:
                chunk["word_count"] = new_word_count
                logger.debug(f"Updated word count from {original_word_count} to {new_word_count}")
                continue

            # Trim excessive whitespace
            chunk["content"] = " ".join(chunk["content"].split())
            new_word_count = len(chunk["content"].split())

            # Update word count if it changed
            if new_word_count != original_word_count:
                chunk["word_count"] = new_word_count
                logger.debug(f"Updated word count from {original_word_count} to {new_word_count}")

            # Ensure chunk size is within bounds
            if len(chunk["content"]) > self.max_chunk_size:
                chunk["content"] = chunk["content"][: self.max_chunk_size]
                chunk["word_count"] = len(chunk["content"].split())
                logger.debug(f"Truncated chunk to max size: {self.max_chunk_size} chars")

            processed_chunks.append(chunk)

        logger.info(f"Post-processed {len(chunks)} chunks -> {len(processed_chunks)} kept")
        return processed_chunks

    def _is_ocr_artifact_chunk(self, content: str) -> bool:
        """
        Check if a chunk contains mostly OCR artifacts and should be filtered out.
        """
        if not content or len(content.strip()) < 20:
            return True

        content_lower = content.lower()

        # Check for page markers and OCR artifacts
        ocr_indicators = [
            "=== page",
            "tesseract",
            "no text found",
            "ocr failed",
            "easyocr",
            "paddleocr",
        ]

        # Direct check for page markers - these should always be filtered
        for indicator in ocr_indicators:
            if indicator in content_lower:
                # If the chunk contains OCR indicators, be more aggressive
                words = content.split()
                if not words:
                    return True

                ocr_words = sum(
                    1
                    for word in words
                    if any(indicator in word.lower() for indicator in ocr_indicators)
                )
                if ocr_words / len(words) > 0.2:  # More than 20% OCR artifacts
                    return True

                # Also filter if the content contains page markers at the start
                if content_lower.strip().startswith("=== page"):
                    return True

                ocr_words = sum(
                    1
                    for word in words
                    if any(indicator in word.lower() for indicator in ocr_indicators)
                )
                if ocr_words / len(words) > 0.3:  # More than 30% OCR artifacts
                    return True

                # Also filter if the content is very short and contains OCR indicators
                if len(content.strip()) < 100 and ocr_words > 0:
                    return True

        # Check for chunks that are just page numbers or minimal content
        if len(content.strip()) < 50 and not any(char.isalpha() for char in content):
            return True

        # Check for chunks with excessive special characters (OCR corruption)
        special_chars = sum(1 for char in content if not char.isalnum() and not char.isspace())
        if (
            len(content) > 0 and special_chars / len(content) > 0.3
        ):  # More than 30% special characters
            return True

        return False
