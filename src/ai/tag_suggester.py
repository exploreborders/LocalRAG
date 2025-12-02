"""
AI-powered tag suggestion system for intelligent document tagging.

This module provides intelligent tag recommendations based on document content analysis,
using LLM to extract relevant keywords, topics, and entities for automated tagging.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class AITagSuggester:
    """
    AI-powered tag suggestion system using Ollama LLM.

    Provides intelligent tag recommendations based on document content analysis,
    with confidence scoring and relevance ranking.
    """

    def __init__(
        self,
        model_name: str = "llama3.2:latest",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the AI tag suggester.

        Args:
            model_name: Ollama model name for tag generation
            base_url: Ollama API base URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.max_content_length = 2000  # Limit content for analysis

    def _call_llm(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Call Ollama LLM with the given prompt.

        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response text
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3,  # Lower temperature for more consistent results
                },
            }

            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                return str(response_text).strip()
            else:
                logger.error(f"LLM call failed: {response.status_code} - {response.text}")
                return ""

        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return ""

    def suggest_tags(
        self,
        document_content: str,
        document_title: str = "",
        existing_tags: Optional[List[str]] = None,
        max_suggestions: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Generate intelligent tag suggestions based on document content.

        Args:
            document_content: Full document content
            document_title: Document title/filename
            existing_tags: List of existing tags to avoid duplicates
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of tag suggestions with confidence scores
        """
        if not document_content.strip():
            return []

        # Prepare content for analysis
        analysis_content = self._prepare_content(document_content)

        # Generate tag suggestions
        raw_suggestions = self._generate_tag_suggestions(analysis_content, document_title)

        # Parse and clean suggestions
        parsed_tags = self._parse_tag_response(raw_suggestions)

        # Filter out existing tags and rank by relevance
        filtered_tags = self._filter_existing_tags(parsed_tags, existing_tags or [])

        # Calculate confidence scores and rank
        ranked_suggestions = self._rank_suggestions(filtered_tags, analysis_content)

        return ranked_suggestions[:max_suggestions]

    def _prepare_content(self, content: str) -> str:
        """
        Prepare document content for analysis by truncating and cleaning.

        Args:
            content: Raw document content

        Returns:
            Prepared content for analysis
        """
        # Truncate content if too long
        if len(content) > self.max_content_length:
            content = content[: self.max_content_length] + "..."

        # Basic cleaning
        content = re.sub(r"\s+", " ", content)  # Normalize whitespace
        content = content.strip()

        return content

    def _generate_tag_suggestions(self, content: str, title: str) -> str:
        """
        Generate tag suggestions using LLM.

        Args:
            content: Prepared document content
            title: Document title

        Returns:
            Raw LLM response with tag suggestions
        """
        prompt = f"""
        Analyze the following document and suggest 3-5 high-level, reusable tags that would help categorize and find this document.
        Focus on BROAD categories and main topics that could connect multiple related documents.

        Prioritize tags like:
        - Technical domains: "Machine Learning", "Database", "Web Development", "AI", "Computer Vision"
        - Academic fields: "Computer Science", "Mathematics", "Engineering"
        - Business areas: "Project Management", "Business Analysis", "Marketing"
        - Content types: "Tutorial", "Reference", "Research", "Documentation"

        Avoid overly specific tags that would only apply to this one document.
        Return only a comma-separated list of tags, no explanations.

        Document Title: {title}
        Document Content: {content}

        Tags:
        """

        return self._call_llm(prompt, max_tokens=100)

    def _parse_tag_response(self, response: str) -> List[str]:
        """
        Parse LLM response into clean tag list.

        Args:
            response: Raw LLM response

        Returns:
            List of cleaned tag strings
        """
        if not response.strip():
            return []

        tags = []

        # Try different separators
        for separator in [",", "\n", ";", "|"]:
            if separator in response:
                candidates = response.split(separator)
                break
        else:
            candidates = [response]

        for tag in candidates:
            tag = tag.strip()

            # Remove numbering and bullet points
            tag = re.sub(r"^\d+\.?\s*", "", tag)
            tag = re.sub(r"^[-•*]\s*", "", tag)
            tag = re.sub(r'^["\'](.*)["\']$', r"\1", tag)  # Remove quotes

            # Clean and validate
            tag = tag.strip()
            if tag and 2 <= len(tag) <= 50:  # Reasonable length limits
                # Capitalize first letter of each word
                tag = " ".join(word.capitalize() for word in tag.split())
                tags.append(tag)

        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower not in seen:
                seen.add(tag_lower)
                unique_tags.append(tag)

        return unique_tags

    def _filter_existing_tags(self, suggestions: List[str], existing_tags: List[str]) -> List[str]:
        """
        Filter out tags that already exist.

        Args:
            suggestions: List of suggested tags
            existing_tags: List of existing tags to avoid

        Returns:
            Filtered list of new tag suggestions
        """
        existing_lower = {tag.lower() for tag in existing_tags}
        filtered = []

        for tag in suggestions:
            # Check for exact match and partial matches
            tag_lower = tag.lower()

            # Skip if exact match
            if tag_lower in existing_lower:
                continue

            # Skip if very similar (simple fuzzy check)
            similar = False
            for existing in existing_lower:
                # Check if one is substring of the other (for short tags)
                if len(tag) <= 10 or len(existing) <= 10:
                    if tag_lower in existing or existing in tag_lower:
                        similar = True
                        break

            if not similar:
                filtered.append(tag)

        return filtered

    def _rank_suggestions(self, tags: List[str], content: str) -> List[Dict[str, Any]]:
        """
        Rank tag suggestions by relevance and assign confidence scores.

        Args:
            tags: List of tag suggestions
            content: Document content for relevance scoring

        Returns:
            List of tags with confidence scores and metadata
        """
        if not tags:
            return []

        content_lower = content.lower()
        scored_tags = []

        for tag in tags:
            confidence = self._calculate_confidence(tag, content_lower)
            scored_tags.append(
                {
                    "tag": tag,
                    "confidence": confidence,
                    "relevance_score": confidence,
                    "source": "ai_generated",
                }
            )

        # Sort by confidence (highest first)
        scored_tags.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        return scored_tags

    def _calculate_confidence(self, tag: str, content_lower: str) -> float:
        """
        Calculate confidence score for a tag based on content analysis.

        Args:
            tag: Tag to score
            content_lower: Lowercase document content

        Returns:
            Confidence score between 0.0 and 1.0
        """
        tag_lower = tag.lower()
        tag_words = tag_lower.split()

        # Base confidence
        confidence = 0.5

        # Boost for exact matches
        if tag_lower in content_lower:
            confidence += 0.3

        # Boost for partial matches of multi-word tags
        if len(tag_words) > 1:
            matched_words = sum(1 for word in tag_words if word in content_lower)
            if matched_words > 0:
                confidence += 0.2 * (matched_words / len(tag_words))

        # Boost for technical/academic terms (contains numbers, special chars)
        if re.search(r"[0-9\-_]", tag):
            confidence += 0.1

        # Penalize very generic terms
        generic_terms = {"document", "file", "text", "content", "information", "data"}
        if tag_lower in generic_terms:
            confidence -= 0.2

        # Ensure confidence is within bounds
        confidence = max(0.1, min(0.95, confidence))

        return confidence

        """
        Analyze document content to understand tagging patterns and themes.

        Args:
            document_content: Document content to analyze

        Returns:
            Analysis results with themes, entities, and suggested categories
        """
        analysis_content = self._prepare_content(document_content)

        prompt = f"""
        Analyze this document and provide insights for tagging:

        1. Main themes/topics (comma-separated)
        2. Key entities mentioned
        3. Suggested category
        4. Content type (academic, technical, business, etc.)

        Document content: {analysis_content}

        Return as JSON format:
        {{
            "themes": ["theme1", "theme2"],
            "entities": ["entity1", "entity2"],
            "category": "suggested_category",
            "content_type": "content_type"
        }}
        """

        response = self._call_llm(prompt, max_tokens=200)

        try:
            # Try to parse JSON response
            result = json.loads(response)
            return dict(result)  # Ensure it's a dict
        except json.JSONDecodeError:
            # Fallback to basic analysis
            return {
                "themes": [],
                "entities": [],
                "category": "general",
                "content_type": "document",
            }
