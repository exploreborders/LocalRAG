"""
Content relevance scoring for document chunks.

This module provides semantic and topic-aware relevance scoring
to improve retrieval quality and ranking.
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class RelevanceScorer:
    """
    Multi-dimensional content relevance scoring for document chunks.

    Combines semantic analysis, topic relevance, and structural importance
    to provide comprehensive relevance scores.
    """

    def __init__(self):
        """Initialize the relevance scorer."""
        # Common stop words for text analysis
        self.stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
        }

    def score_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_topics: List[str],
        document_structure: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Score chunks for relevance using multiple criteria.

        Args:
            chunks: List of chunk dictionaries
            document_topics: Document's main topics
            document_structure: Document structure analysis

        Returns:
            Chunks with enhanced relevance scores
        """
        logger.info(f"Scoring {len(chunks)} chunks for relevance")

        scored_chunks = []

        for chunk in chunks:
            # Calculate different relevance components
            semantic_score = self._calculate_semantic_relevance(chunk, document_topics)
            structural_score = self._calculate_structural_relevance(
                chunk, document_structure
            )
            topical_score = self._calculate_topical_relevance(chunk, document_topics)
            positional_score = self._calculate_positional_relevance(chunk, len(chunks))

            # Combine scores with weights
            final_score = self._combine_scores(
                {
                    "semantic": semantic_score,
                    "structural": structural_score,
                    "topical": topical_score,
                    "positional": positional_score,
                }
            )

            # Update chunk with scores
            enhanced_chunk = chunk.copy()
            enhanced_chunk.update(
                {
                    "relevance_score": final_score,
                    "relevance_components": {
                        "semantic": semantic_score,
                        "structural": structural_score,
                        "topical": topical_score,
                        "positional": positional_score,
                    },
                    "relevance_factors": self._identify_relevance_factors(
                        chunk, document_topics
                    ),
                }
            )

            scored_chunks.append(enhanced_chunk)

        # Sort by relevance score
        scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)

        logger.info(
            f"Completed relevance scoring, top score: {scored_chunks[0]['relevance_score']:.3f}"
        )
        return scored_chunks

    def _calculate_semantic_relevance(
        self, chunk: Dict[str, Any], document_topics: List[str]
    ) -> float:
        """
        Calculate semantic relevance based on content density and coherence.
        """
        content = chunk.get("content", "").lower()
        if not content:
            return 0.0

        # Content density (ratio of meaningful words to total words)
        words = [w for w in re.findall(r"\b\w+\b", content) if w not in self.stop_words]
        total_words = len(re.findall(r"\b\w+\b", content))

        if total_words == 0:
            return 0.0

        density = len(words) / total_words

        # Lexical diversity (unique words / total words)
        unique_words = len(set(words))
        diversity = unique_words / len(words) if words else 0

        # Technical term density
        technical_indicators = [
            "algorithm",
            "method",
            "analysis",
            "system",
            "model",
            "process",
            "function",
            "variable",
            "parameter",
            "result",
            "conclusion",
            "theory",
            "principle",
            "concept",
            "framework",
            "approach",
        ]

        technical_count = sum(1 for word in words if word in technical_indicators)
        technical_density = technical_count / len(words) if words else 0

        # Combine factors
        semantic_score = density * 0.4 + diversity * 0.3 + technical_density * 0.3

        return min(1.0, semantic_score)

    def _calculate_structural_relevance(
        self, chunk: Dict[str, Any], document_structure: Dict[str, Any]
    ) -> float:
        """
        Calculate relevance based on structural position and type.
        """
        section_type = chunk.get("section_type", "general")
        chapter_path = chunk.get("chapter_path", "")

        # Base scores by section type
        type_scores = {
            "chapter": 0.9,  # Chapter introductions are highly relevant
            "section": 0.8,  # Major sections are very relevant
            "subsection": 0.7,  # Subsections are relevant
            "paragraph": 0.5,  # Regular paragraphs are moderately relevant
            "general": 0.4,  # General content is less specific
        }

        base_score = type_scores.get(section_type, 0.4)

        # Bonus for early chapters (often contain key concepts)
        try:
            chapter_num = int(chapter_path.split(".")[0])
            if chapter_num <= 3:  # First 3 chapters get bonus
                base_score += 0.1
        except (ValueError, IndexError):
            pass

        # Check if chunk contains structural elements
        content = chunk.get("content", "")
        structural_indicators = [
            "summary",
            "conclusion",
            "introduction",
            "overview",
            "key points",
            "main idea",
            "important",
            "note that",
        ]

        has_structural_content = any(
            indicator in content.lower() for indicator in structural_indicators
        )

        if has_structural_content:
            base_score += 0.2

        return min(1.0, base_score)

    def _calculate_topical_relevance(
        self, chunk: Dict[str, Any], document_topics: List[str]
    ) -> float:
        """
        Calculate relevance based on topic alignment.
        """
        content = chunk.get("content", "").lower()
        if not content or not document_topics:
            return 0.5

        # Count topic mentions
        topic_matches = 0
        total_topics = len(document_topics)

        for topic in document_topics:
            topic_words = set(topic.lower().split())
            content_words = set(re.findall(r"\b\w+\b", content))

            # Calculate overlap
            overlap = len(topic_words.intersection(content_words))
            if overlap > 0:
                topic_matches += 1

        if total_topics == 0:
            return 0.5

        topical_score = topic_matches / total_topics

        # Boost score for chunks that mention multiple topics
        if topic_matches > 1:
            topical_score *= 1.2

        return min(1.0, topical_score)

    def _calculate_positional_relevance(
        self, chunk: Dict[str, Any], total_chunks: int
    ) -> float:
        """
        Calculate relevance based on position in document.
        """
        chunk_index = chunk.get("chunk_index", 0)

        if total_chunks <= 1:
            return 0.5

        # Early chunks often contain important introductory content
        position_ratio = chunk_index / total_chunks

        if position_ratio < 0.2:  # First 20% of chunks
            return 0.8
        elif position_ratio < 0.5:  # Next 30% of chunks
            return 0.6
        else:  # Later chunks
            return 0.4

    def _combine_scores(self, scores: Dict[str, float]) -> float:
        """
        Combine multiple relevance scores into a final score.
        """
        weights = {
            "semantic": 0.4,  # Most important - content quality
            "structural": 0.3,  # Important - document position
            "topical": 0.2,  # Moderately important - topic alignment
            "positional": 0.1,  # Least important - document position
        }

        final_score = sum(
            scores[component] * weight for component, weight in weights.items()
        )

        return round(final_score, 3)

    def _identify_relevance_factors(
        self, chunk: Dict[str, Any], document_topics: List[str]
    ) -> List[str]:
        """
        Identify specific factors contributing to chunk relevance.
        """
        factors = []
        content = chunk.get("content", "").lower()

        # Check for technical content
        technical_terms = ["algorithm", "method", "analysis", "system", "model"]
        if any(term in content for term in technical_terms):
            factors.append("technical_content")

        # Check for structural importance
        structural_terms = ["summary", "conclusion", "introduction", "overview"]
        if any(term in content for term in structural_terms):
            factors.append("structural_importance")

        # Check for topic alignment
        if document_topics:
            topic_words = set()
            for topic in document_topics:
                topic_words.update(topic.lower().split())

            content_words = set(re.findall(r"\b\w+\b", content))
            if topic_words.intersection(content_words):
                factors.append("topic_alignment")

        # Check for early position
        chunk_index = chunk.get("chunk_index", 0)
        if chunk_index < 5:  # Early chunks
            factors.append("early_position")

        # Check for high word count (substantial content)
        word_count = chunk.get("word_count", 0)
        if word_count > 100:
            factors.append("substantial_content")

        return factors

    def rank_chunks_for_query(
        self, chunks: List[Dict[str, Any]], query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rank chunks specifically for a given query.

        Args:
            chunks: List of chunks with relevance scores
            query: Search query
            top_k: Number of top chunks to return

        Returns:
            Top-ranked chunks for the query
        """
        if not query or not chunks:
            return chunks[:top_k]

        query_lower = query.lower()
        query_words = set(re.findall(r"\b\w+\b", query_lower))

        ranked_chunks = []

        for chunk in chunks:
            content = chunk.get("content", "").lower()
            content_words = set(re.findall(r"\b\w+\b", content))

            # Calculate query relevance
            word_overlap = len(query_words.intersection(content_words))
            query_density = word_overlap / len(query_words) if query_words else 0

            # Boost for exact phrase matches
            exact_matches = len(re.findall(re.escape(query_lower), content))
            phrase_boost = min(0.5, exact_matches * 0.1)

            # Combine with existing relevance score
            base_relevance = chunk.get("relevance_score", 0.5)
            query_relevance = base_relevance + query_density + phrase_boost

            # Create ranked chunk
            ranked_chunk = chunk.copy()
            ranked_chunk["query_relevance"] = round(query_relevance, 3)
            ranked_chunk["query_matches"] = word_overlap

            ranked_chunks.append(ranked_chunk)

        # Sort by query relevance
        ranked_chunks.sort(key=lambda x: x["query_relevance"], reverse=True)

        return ranked_chunks[:top_k]

    def get_relevance_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate relevance statistics for a set of chunks.
        """
        if not chunks:
            return {}

        scores = [chunk.get("relevance_score", 0) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_relevance": round(sum(scores) / len(scores), 3),
            "max_relevance": max(scores),
            "min_relevance": min(scores),
            "high_relevance_count": sum(1 for s in scores if s >= 0.8),
            "medium_relevance_count": sum(1 for s in scores if 0.5 <= s < 0.8),
            "low_relevance_count": sum(1 for s in scores if s < 0.5),
        }
