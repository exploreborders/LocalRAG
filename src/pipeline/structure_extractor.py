"""
Structure extractor for hierarchical document analysis using phi3.5.

This module analyzes document content to extract hierarchical structure
(chapters, sections, subsections) and identify key topics.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import re
import requests

logger = logging.getLogger(__name__)

class StructureExtractor:
    """
    AI-powered document structure extraction using phi3.5.

    Extracts hierarchical document structure and identifies topics for
    enhanced retrieval and organization.
    """

    def __init__(self, model_name: str = "phi3.5:3.8b", base_url: str = "http://localhost:11434"):
        """
        Initialize the structure extractor.

        Args:
            model_name: Ollama model name for structure analysis
            base_url: Ollama API base URL
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/generate"

    def is_available(self) -> bool:
        """Check if the structure analysis model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(model['name'] == self.model_name for model in models)
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
                    "num_predict": 2048  # Allow detailed responses
                }
            }

            response = requests.post(self.api_url, json=payload, timeout=120)

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')

                # Parse the structured response
                structure_data = self._parse_structure_response(response_text)

                # Enhance with additional analysis
                structure_data.update({
                    'document_title': self._extract_title(text, filename),
                    'key_topics': self._extract_topics(text),
                    'reading_time_minutes': self._estimate_reading_time(text),
                    'complexity_score': self._calculate_complexity(text)
                })

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
        Analyze the following document and extract its hierarchical structure.

        Document: {filename}
        Content Sample:
        {text_sample}

        Instructions:
        1. Identify the document's hierarchical structure (chapters, sections, subsections)
        2. Assign proper hierarchical levels and numbering (1, 1.1, 1.1.1, etc.)
        3. Extract chapter titles and section headings
        4. Identify the main topics and themes
        5. Determine the document type and purpose
        6. Assess content complexity and technical level

        Return your analysis as a JSON object with this exact structure:
        {{
            "hierarchy": [
                {{
                    "level": 1,
                    "path": "1",
                    "title": "Chapter Title",
                    "content_preview": "First few sentences...",
                    "word_count": 150,
                    "type": "chapter"
                }},
                {{
                    "level": 2,
                    "path": "1.1",
                    "title": "Section Title",
                    "content_preview": "Section content preview...",
                    "word_count": 75,
                    "type": "section"
                }}
            ],
            "document_type": "research_paper|manual|report|article|book",
            "primary_topic": "Main subject area",
            "secondary_topics": ["topic1", "topic2"],
            "technical_level": "beginner|intermediate|advanced|expert",
            "content_quality": 0.85,
            "structure_confidence": 0.9
        }}

        Be precise and base your analysis only on the provided content.
        """

        return prompt

    def _parse_structure_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the JSON response from the structure analysis."""
        try:
            import json

            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)

                # Validate required fields
                if 'hierarchy' not in parsed:
                    parsed['hierarchy'] = []

                if 'document_type' not in parsed:
                    parsed['document_type'] = 'unknown'

                if 'primary_topic' not in parsed:
                    parsed['primary_topic'] = 'general'

                return parsed
            else:
                logger.warning("No JSON found in structure response")
                return self._create_default_structure()

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structure JSON: {e}")
            return self._create_default_structure()

    def _create_default_structure(self) -> Dict[str, Any]:
        """Create a default structure when parsing fails."""
        return {
            'hierarchy': [],
            'document_type': 'unknown',
            'primary_topic': 'general',
            'secondary_topics': [],
            'technical_level': 'intermediate',
            'content_quality': 0.5,
            'structure_confidence': 0.3
        }

    def _fallback_structure(self, text: str, filename: str) -> Dict[str, Any]:
        """Fallback structure extraction using heuristics."""
        logger.info("Using fallback structure extraction")

        # Simple heuristic-based structure detection
        lines = text.split('\n')
        hierarchy = []

        current_chapter = 0
        current_section = 0

        for line in lines[:50]:  # Analyze first 50 lines
            line = line.strip()
            if not line:
                continue

            # Look for chapter-like patterns
            if re.match(r'^(chapter|Chapter|CHAPTER)\s+\d+', line) or \
               re.match(r'^\d+\.?\s+[A-Z]', line):
                current_chapter += 1
                hierarchy.append({
                    'level': 1,
                    'path': str(current_chapter),
                    'title': line[:100],
                    'content_preview': '',
                    'word_count': 0,
                    'type': 'chapter'
                })
                current_section = 0

            # Look for section-like patterns
            elif re.match(r'^\d+\.\d+', line) or \
                 re.match(r'^[A-Z][^.!?]*$', line) and len(line) < 80:
                current_section += 1
                hierarchy.append({
                    'level': 2,
                    'path': f"{current_chapter}.{current_section}",
                    'title': line[:100],
                    'content_preview': '',
                    'word_count': 0,
                    'type': 'section'
                })

        return {
            'hierarchy': hierarchy,
            'document_type': self._guess_document_type(text, filename),
            'primary_topic': 'general',
            'secondary_topics': [],
            'technical_level': 'intermediate',
            'content_quality': 0.4,
            'structure_confidence': 0.2
        }

    def _extract_title(self, text: str, filename: str) -> str:
        """Extract document title from content or filename."""
        # Look for title-like patterns at the beginning
        lines = text.split('\n')[:10]

        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200 and not line.isupper():
                # Check if it looks like a title
                if not re.match(r'^\d+', line) and not line.startswith('http'):
                    return line

        # Fallback to filename
        return filename.replace('_', ' ').replace('-', ' ').title()

    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from document content."""
        # Simple keyword-based topic extraction
        # In a real implementation, this would use more sophisticated NLP
        text_lower = text.lower()

        topics = []
        topic_keywords = {
            'machine learning': ['machine learning', 'ml', 'neural network', 'deep learning'],
            'data science': ['data science', 'statistics', 'analytics', 'data analysis'],
            'programming': ['python', 'javascript', 'java', 'programming', 'code'],
            'research': ['research', 'study', 'analysis', 'methodology'],
            'business': ['business', 'management', 'strategy', 'marketing']
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
        technical_terms = len(re.findall(r'\b[A-Z]{2,}\b', text))  # Acronyms

        complexity = min(1.0, (avg_word_length * 0.1 + long_words * 0.001 + technical_terms * 0.01))
        return round(complexity, 2)

    def _guess_document_type(self, text: str, filename: str) -> str:
        """Guess document type based on content and filename."""
        text_lower = text.lower()
        filename_lower = filename.lower()

        # Check filename patterns
        if 'paper' in filename_lower or 'research' in filename_lower:
            return 'research_paper'
        elif 'manual' in filename_lower or 'guide' in filename_lower:
            return 'manual'
        elif 'report' in filename_lower:
            return 'report'
        elif 'book' in filename_lower:
            return 'book'

        # Check content patterns
        if 'abstract' in text_lower or 'introduction' in text_lower:
            return 'research_paper'
        elif 'chapter' in text_lower:
            return 'book'
        elif 'section' in text_lower and 'subsection' in text_lower:
            return 'manual'

        return 'article'