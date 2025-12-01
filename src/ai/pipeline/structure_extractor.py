"""
Structure extractor for hierarchical document analysis using llama3.2.

This module analyzes document content to extract hierarchical structure
(chapters, sections, subsections) and identify key topics.
"""

import json
import logging
import re
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)


class StructureExtractor:
    """
    AI-powered document structure extraction using llama3.2.

    Extracts hierarchical document structure and identifies topics for
    enhanced retrieval and organization.
    """

    def __init__(
        self,
        model_name: str = "llama3.2:latest",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the structure extractor.

        Args:
            model_name: Ollama model name for structure analysis
            base_url: Ollama API base URL
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/generate"

    def is_available(self) -> bool:
        """Check if the structure analysis model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model["name"] == self.model_name for model in models)
            return False
        except Exception as e:
            logger.warning(f"Structure model availability check failed: {e}")
            return False

    def extract_structure(self, text: str, filename: str = "") -> Dict[str, Any]:
        """
        Extract hierarchical structure and topics from document text.

        Args:
            text: Full document text
            filename: Document filename for context

        Returns:
            Dict containing structure analysis results
        """
        logger.info(f"Extracting structure from document: {filename}")

        # Prepare analysis prompt
        prompt = self._build_structure_prompt(text, filename)

        try:
            # Make API request to phi3.5
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,  # Balanced creativity for structure analysis
                    "num_predict": 2048,  # Allow detailed responses
                },
            }

            response = requests.post(self.api_url, json=payload, timeout=120)

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")

                # Parse the structured response
                structure_data = self._parse_structure_response(response_text)

                # Enhance with additional analysis
                structure_data.update(
                    {
                        "document_title": self._extract_title(text, filename),
                        "key_topics": self._extract_topics(text),
                        "reading_time_minutes": self._estimate_reading_time(text),
                        "complexity_score": self._calculate_complexity(text),
                    }
                )

                return structure_data
            else:
                logger.error(f"Structure extraction API failed: {response.status_code}")
                return self._fallback_structure(text, filename)

        except Exception as e:
            logger.error(f"Structure extraction failed: {e}")
            return self._fallback_structure(text, filename)

    def _build_structure_prompt(self, text: str, filename: str) -> str:
        """Build the prompt for structure analysis."""
        # Truncate text if too long (keep first and last parts)
        if len(text) > 8000:
            text_sample = text[:4000] + "\n...\n" + text[-4000:]
        else:
            text_sample = text

        prompt = f"""
        Analyze this German technical document and identify its chapter structure.

        Document: {filename}
        Content:
        {text_sample}

        TASK: Identify all chapters and sections in this document. Look for:
        - Chapter headers like "Kapitel 1", "1. Einleitung", "Chapter 1"
        - Section headers with numbers like "1.1", "2.3", etc.
        - German technical terms that indicate chapters: Grundlagen, Theorie, Methoden, Algorithmen, Anwendung, etc.

        Return ONLY valid JSON with the document structure:
        {{
            "hierarchy": [
                {{"level": 1, "path": "1", "title": "Chapter Title Here", "content_preview": "", "word_count": 0, "type": "chapter"}},
                {{"level": 2, "path": "1.1", "title": "Section Title Here", "content_preview": "", "word_count": 0, "type": "section"}}
            ],
            "document_type": "technical_manual",
            "primary_topic": "artificial_intelligence",
            "secondary_topics": ["machine_learning", "deep_learning"],
            "technical_level": "intermediate",
            "content_quality": 0.8,
            "structure_confidence": 0.9
        }}

        If you cannot find clear chapters, create reasonable chapter divisions based on content topics.
        """

        return prompt

    def _parse_structure_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the JSON response from the structure analysis."""
        try:
            # Extract JSON from response
            json_start = response_text.find("{")

            if json_start >= 0:
                # Find the first complete JSON object by counting braces
                json_str = response_text[json_start:]
                brace_count = 0
                end_pos = 0

                for i, char in enumerate(json_str):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break

                if end_pos > 0:
                    json_str = json_str[:end_pos]

                    # Clean JSON by removing JavaScript-style comments
                    import re

                    # Remove // comments
                    json_str = re.sub(r"//.*$", "", json_str, flags=re.MULTILINE)
                    # Remove /* */ comments
                    json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)

                    parsed = json.loads(json_str)

                    # Validate required fields
                    if "hierarchy" not in parsed:
                        parsed["hierarchy"] = []

                    if "document_type" not in parsed:
                        parsed["document_type"] = "unknown"

                    if "primary_topic" not in parsed:
                        parsed["primary_topic"] = "general"

                    # Filter out non-chapter entries from hierarchy
                    filtered_hierarchy = []
                    for item in parsed["hierarchy"]:
                        title = item.get("title", "").strip()
                        if title and not self._is_non_chapter_text(title):
                            filtered_hierarchy.append(item)

                    parsed["hierarchy"] = filtered_hierarchy

                    return dict(parsed)  # Ensure it's a dict

            # If we get here, no valid JSON was found
            logger.warning("No complete JSON object found in structure response")
            return self._create_default_structure()

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structure JSON: {e}")
            return self._create_default_structure()

    def _extract_table_of_contents(self, text: str) -> List[Dict[str, Any]]:
        """Extract table of contents from document text."""
        lines = text.split("\n")
        hierarchy = []

        # Look for table of contents section
        toc_start = -1
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if (
                "inhaltsverzeichnis" in line_lower  # German
                or "table of contents" in line_lower
                or "contents" in line_lower
            ):
                toc_start = i
                break

        if toc_start == -1:
            return []

        # Parse markdown table format
        # Look for table rows starting from toc_start
        for i in range(toc_start, min(toc_start + 50, len(lines))):
            line = lines[i].strip()

            # Skip table separators (|---| or |-----|)
            if re.match(r"^\s*\|[\s\-\|]+\|\s*$", line):
                continue

            # Look for table rows with | number | title |
            match = re.match(r"^\s*\|\s*([\d\.]+)\s*\|\s*(.+?)\s*\|.*\|?\s*$", line)
            if match:
                path = match.group(1).strip()
                title = match.group(2).strip()

                # Skip if it's not a proper chapter/section number
                if not re.match(r"^\d+(\.\d+)*$", path):
                    continue

                # Filter out non-chapter text
                if self._is_non_chapter_text(title):
                    continue

                # Determine level based on dots in path
                level = path.count(".") + 1

                hierarchy.append(
                    {
                        "level": level,
                        "path": path,
                        "title": title,
                        "content_preview": "",
                        "word_count": 0,
                        "type": "chapter" if level == 1 else "section",
                    }
                )

        return hierarchy

    def _is_non_chapter_text(self, line: str) -> bool:
        """Check if a line contains text that shouldn't be considered a chapter heading."""
        # Common non-chapter patterns (case insensitive, with optional content after colon)
        non_chapter_patterns = [
            r"^wobei:?.*",  # "Wobei:" - German "whereas" or "where"
            r"^note:?.*",  # "Note:"
            r"^see also:?.*",  # "See also:"
            r"^example:?.*",  # "Example:"
            r"^figure:?.*",  # "Figure:"
            r"^table:?.*",  # "Table:"
            r"^source:?.*",  # "Source:"
            r"^references?:?.*",  # "Reference(s):"
            r"^abstract:?.*",  # "Abstract:"
            r"^summary:?.*",  # "Summary:"
            r"^introduction:?.*",  # "Introduction:" (too generic)
            r"^conclusion:?.*",  # "Conclusion:"
            r"^acknowledgments?:?.*",  # "Acknowledgment(s):"
            r"^appendix:?.*",  # "Appendix:"
            r"^glossary:?.*",  # "Glossary:"
            r"^index:?.*",  # "Index:"
            r"^contents?:?.*",  # "Content(s):"
            r"^table of contents?:?.*",  # "Table of contents:"
            r"^in this (chapter|section|part):?.*",  # "In this chapter/section/part:"
            r"^the following:?.*",  # "The following:"
            r"^overview:?.*",  # "Overview:"
            r"^background:?.*",  # "Background:"
            r"^related work:?.*",  # "Related work:"
            r"^methodology:?.*",  # "Methodology:"
            r"^results:?.*",  # "Results:"
            r"^discussion:?.*",  # "Discussion:"
            r"^future work:?.*",  # "Future work:"
        ]

        line_lower = line.lower().strip()
        for pattern in non_chapter_patterns:
            if re.match(pattern, line_lower):
                return True

        # Also exclude very short lines (likely not chapter headings)
        if len(line.strip()) < 5:
            return True

        # Exclude lines that are just numbers or symbols
        if re.match(r"^[\d\s\.\-\(\)]+$", line.strip()):
            return True

        return False

    def _create_default_structure(self) -> Dict[str, Any]:
        """Create a default structure when parsing fails."""
        return {
            "hierarchy": [],
            "document_type": "unknown",
            "primary_topic": "general",
            "secondary_topics": [],
            "technical_level": "intermediate",
            "content_quality": 0.5,
            "structure_confidence": 0.3,
        }

    def _fallback_structure(self, text: str, filename: str) -> Dict[str, Any]:
        """Fallback structure extraction using heuristics."""
        logger.info("Using fallback structure extraction")

        # First priority: extract table of contents if it exists
        toc_hierarchy = self._extract_table_of_contents(text)
        if toc_hierarchy:
            logger.info(f"Successfully extracted {len(toc_hierarchy)} items from table of contents")
            return {
                "hierarchy": toc_hierarchy,
                "document_type": self._guess_document_type(text, filename),
                "primary_topic": "general",
                "secondary_topics": [],
                "technical_level": "intermediate",
                "content_quality": 0.8,
                "structure_confidence": 0.9,
            }

        # Second priority: try LLM analysis if available
        if self.is_available():
            logger.info("No table of contents found, trying LLM analysis")
            try:
                # Use LLM for analysis but with a much simpler prompt
                prompt = f"""
                Analyze this document and identify its main sections.

                Document: {filename}
                Content: {text[:2000]}

                Return a simple JSON list of section titles:
                {{"sections": ["Section 1", "Section 2", "Subsection 1.1"]}}
                """

                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 500},
                }

                import requests

                response = requests.post(self.api_url, json=payload, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "")

                    # Simple JSON extraction
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1

                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        import json

                        parsed = json.loads(json_str)

                        if "sections" in parsed:
                            hierarchy = []
                            for i, title in enumerate(parsed["sections"], 1):
                                hierarchy.append(
                                    {
                                        "level": 1,
                                        "path": str(i),
                                        "title": title,
                                        "content_preview": "",
                                        "word_count": 0,
                                        "type": "chapter",
                                    }
                                )

                            logger.info(f"LLM extracted {len(hierarchy)} sections")
                            return {
                                "hierarchy": hierarchy,
                                "document_type": self._guess_document_type(text, filename),
                                "primary_topic": "general",
                                "secondary_topics": [],
                                "technical_level": "intermediate",
                                "content_quality": 0.5,
                                "structure_confidence": 0.6,
                            }

            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")

        # Final fallback: simple heuristic-based structure detection
        logger.info("Using heuristic structure extraction")
        lines = text.split("\n")
        hierarchy = []

        current_chapter = 0
        current_section = 0

        for line in lines[:100]:  # Analyze first 100 lines
            line = line.strip()
            if not line:
                continue

            # Look for chapter-like patterns (with filtering)
            if (
                re.match(r"^(chapter|Chapter|CHAPTER)\s+\d+", line)
                or re.match(r"^\d+\.?\s+[A-Z]", line)
            ) and not self._is_non_chapter_text(line):
                current_chapter += 1
                hierarchy.append(
                    {
                        "level": 1,
                        "path": str(current_chapter),
                        "title": line[:100],
                        "content_preview": "",
                        "word_count": 0,
                        "type": "chapter",
                    }
                )
                current_section = 0

            # Look for section-like patterns (more selective)
            elif re.match(r"^\d+\.\d+", line) or (
                re.match(r"^[A-Z][^.!?]*$", line)
                and len(line) < 80
                and not self._is_non_chapter_text(line)
            ):
                current_section += 1
                hierarchy.append(
                    {
                        "level": 2,
                        "path": f"{current_chapter}.{current_section}",
                        "title": line[:100],
                        "content_preview": "",
                        "word_count": 0,
                        "type": "section",
                    }
                )

        return {
            "hierarchy": hierarchy,
            "document_type": self._guess_document_type(text, filename),
            "primary_topic": "general",
            "secondary_topics": [],
            "technical_level": "intermediate",
            "content_quality": 0.4,
            "structure_confidence": 0.2,
        }

    def _extract_title(self, text: str, filename: str) -> str:
        """Extract document title from content or filename."""
        # Look for title-like patterns at the beginning
        lines = text.split("\n")[:10]

        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200 and not line.isupper():
                # Check if it looks like a title
                if not re.match(r"^\d+", line) and not line.startswith("http"):
                    return line

        # Fallback to filename
        return filename.replace("_", " ").replace("-", " ").title()

    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from document content."""
        # Simple keyword-based topic extraction
        # In a real implementation, this would use more sophisticated NLP
        text_lower = text.lower()

        topics = []
        topic_keywords = {
            "machine learning": [
                "machine learning",
                "ml",
                "neural network",
                "deep learning",
            ],
            "data science": [
                "data science",
                "statistics",
                "analytics",
                "data analysis",
            ],
            "programming": ["python", "javascript", "java", "programming", "code"],
            "research": ["research", "study", "analysis", "methodology"],
            "business": ["business", "management", "strategy", "marketing"],
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)

        return topics[:3]  # Return top 3 topics

    def _estimate_reading_time(self, text: str) -> int:
        """Estimate reading time in minutes."""
        words = len(text.split())
        # Average reading speed: 200 words per minute
        return max(1, words // 200)

    def _calculate_complexity(self, text: str) -> float:
        """Calculate content complexity score (0-1)."""
        # Simple complexity metrics
        avg_word_length = sum(len(word) for word in text.split()) / len(text.split())
        long_words = sum(1 for word in text.split() if len(word) > 6)
        technical_terms = len(re.findall(r"\b[A-Z]{2,}\b", text))  # Acronyms

        complexity = min(1.0, (avg_word_length * 0.1 + long_words * 0.001 + technical_terms * 0.01))
        return round(complexity, 2)

    def _guess_document_type(self, text: str, filename: str) -> str:
        """Guess document type based on content and filename."""
        text_lower = text.lower()
        filename_lower = filename.lower()

        # Check filename patterns
        if "paper" in filename_lower or "research" in filename_lower:
            return "research_paper"
        elif "manual" in filename_lower or "guide" in filename_lower:
            return "manual"
        elif "report" in filename_lower:
            return "report"
        elif "book" in filename_lower:
            return "book"

        # Check content patterns
        if "abstract" in text_lower or "introduction" in text_lower:
            return "research_paper"
        elif "chapter" in text_lower:
            return "book"
        elif "section" in text_lower and "subsection" in text_lower:
            return "manual"

        return "article"
