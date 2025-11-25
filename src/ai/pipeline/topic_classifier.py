"""
Advanced topic classification system for cross-document relationships.

This module provides intelligent topic extraction, classification, and
relationship mapping across multiple documents for enhanced knowledge synthesis.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
import re
import requests

logger = logging.getLogger(__name__)


class TopicClassifier:
    """
    Multi-strategy topic classification and relationship mapping.

    Uses AI analysis combined with statistical methods to identify topics
    and create meaningful relationships between documents.
    """

    def __init__(
        self,
        model_name: str = "llama3.2:latest",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the topic classifier.

        Args:
            model_name: Ollama model for topic analysis
            base_url: Ollama API base URL
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/generate"

        # Predefined topic categories for classification
        self.topic_categories = {
            "academic": [
                "research",
                "study",
                "analysis",
                "methodology",
                "theory",
                "literature",
                "publication",
                "academic",
                "scholarly",
            ],
            "technical": [
                "algorithm",
                "implementation",
                "system",
                "architecture",
                "framework",
                "protocol",
                "standard",
                "specification",
            ],
            "business": [
                "management",
                "strategy",
                "market",
                "finance",
                "economics",
                "organization",
                "enterprise",
                "corporate",
                "industry",
            ],
            "scientific": [
                "experiment",
                "data",
                "measurement",
                "observation",
                "hypothesis",
                "validation",
                "empirical",
                "quantitative",
            ],
            "educational": [
                "learning",
                "teaching",
                "education",
                "training",
                "course",
                "tutorial",
                "guide",
                "instruction",
                "curriculum",
            ],
        }

    def is_available(self) -> bool:
        """Check if the topic classification model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model["name"] == self.model_name for model in models)
            return False
        except Exception as e:
            logger.warning(f"Topic model availability check failed: {e}")
            return False

    def classify_document_topics(
        self, text: str, title: str = "", existing_topics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Classify topics for a single document.

        Args:
            text: Document text
            title: Document title
            existing_topics: Previously identified topics

        Returns:
            Topic classification results
        """
        logger.info(f"Classifying topics for document: {title}")

        # Prepare analysis prompt
        prompt = self._build_topic_prompt(text, title, existing_topics)

        try:
            # Make API request
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Balanced creativity for topic analysis
                    "num_predict": 1024,
                },
            }

            response = requests.post(self.api_url, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")

                # Parse structured response
                topics_data = self._parse_topic_response(response_text)

                # Enhance with statistical analysis
                topics_data = self._enhance_with_statistics(topics_data, text)

                return topics_data
            else:
                logger.error(f"Topic classification API failed: {response.status_code}")
                return self._fallback_classification(text, title)

        except Exception as e:
            logger.error(f"Topic classification failed: {e}")
            return self._fallback_classification(text, title)

    def analyze_cross_document_relationships(
        self, documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze relationships between multiple documents based on topics.

        Args:
            documents: List of document dictionaries with topic information

        Returns:
            Cross-document relationship analysis
        """
        logger.info(f"Analyzing relationships across {len(documents)} documents")

        # Extract topic information from all documents
        doc_topics = {}
        all_topics = set()

        for doc in documents:
            doc_id = doc.get("id") or doc.get("filename", "")
            topics = doc.get("topics", []) or doc.get("secondary_topics", [])
            primary_topic = doc.get("primary_topic", "")

            if primary_topic:
                topics = [primary_topic] + topics

            doc_topics[doc_id] = set(topics)
            all_topics.update(topics)

        # Find topic clusters
        topic_clusters = self._find_topic_clusters(doc_topics)

        # Identify related documents
        relationships = self._identify_relationships(doc_topics, topic_clusters)

        # Calculate topic distribution
        topic_distribution = self._calculate_topic_distribution(doc_topics)

        return {
            "total_documents": len(documents),
            "unique_topics": len(all_topics),
            "topic_clusters": topic_clusters,
            "document_relationships": relationships,
            "topic_distribution": topic_distribution,
            "most_common_topics": self._get_most_common_topics(doc_topics),
        }

    def _build_topic_prompt(
        self, text: str, title: str, existing_topics: Optional[List[str]] = None
    ) -> str:
        """Build the topic classification prompt."""
        # Truncate text if too long
        if len(text) > 3000:
            text = text[:1500] + "\n...\n" + text[-1500:]

        existing_str = ""
        if existing_topics:
            existing_str = (
                f"Previously identified topics: {', '.join(existing_topics)}\n"
            )

        prompt = f"""
        Analyze the following document and identify its key topics and themes.

        Document Title: {title}
        {existing_str}
        Content Sample:
        {text}

        Instructions:
        1. Identify the primary topic or main subject of the document
        2. List 3-5 secondary topics or sub-themes
        3. Determine the most appropriate category (academic, technical, business, scientific, educational)
        4. Assess the technical/complexity level (beginner, intermediate, advanced, expert)
        5. Identify any specialized domains or fields

        Return your analysis as a JSON object with this exact structure:
        {{
            "primary_topic": "Main subject area",
            "secondary_topics": ["topic1", "topic2", "topic3"],
            "category": "academic|technical|business|scientific|educational",
            "technical_level": "beginner|intermediate|advanced|expert",
            "specialized_domains": ["domain1", "domain2"],
            "confidence_score": 0.85,
            "key_concepts": ["concept1", "concept2", "concept3"]
        }}

        Be specific and base your analysis only on the provided content.
        """

        return prompt

    def _parse_topic_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the JSON response from topic classification."""
        try:
            import json

            # Extract JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)

                # Validate required fields
                if "primary_topic" not in parsed:
                    parsed["primary_topic"] = "general"

                if "secondary_topics" not in parsed:
                    parsed["secondary_topics"] = []

                if "category" not in parsed:
                    parsed["category"] = "general"

                return parsed
            else:
                logger.warning("No JSON found in topic response")
                return self._create_default_topics()

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse topic JSON: {e}")
            return self._create_default_topics()

    def _create_default_topics(self) -> Dict[str, Any]:
        """Create default topic classification when parsing fails."""
        return {
            "primary_topic": "general",
            "secondary_topics": [],
            "category": "general",
            "technical_level": "intermediate",
            "specialized_domains": [],
            "confidence_score": 0.3,
            "key_concepts": [],
        }

    def _enhance_with_statistics(
        self, topics_data: Dict[str, Any], text: str
    ) -> Dict[str, Any]:
        """
        Enhance topic classification with statistical analysis.
        """
        # Extract keywords using frequency analysis
        words = re.findall(r"\b\w+\b", text.lower())
        word_freq = Counter(words)

        # Remove stop words (basic list)
        stop_words = {
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
        }
        filtered_words = {
            word: freq
            for word, freq in word_freq.items()
            if word not in stop_words and len(word) > 3
        }

        # Get top keywords
        top_keywords = [word for word, _ in word_freq.most_common(10)]

        # Add statistical insights
        topics_data["statistical_keywords"] = top_keywords
        topics_data["vocabulary_richness"] = (
            len(filtered_words) / len(words) if words else 0
        )

        return topics_data

    def _fallback_classification(self, text: str, title: str) -> Dict[str, Any]:
        """Fallback topic classification using statistical methods."""
        logger.info("Using fallback topic classification")

        # Simple keyword-based classification
        text_lower = text.lower()

        # Check against predefined categories
        category_scores = {}
        for category, keywords in self.topic_categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score

        # Determine primary category
        if category_scores:
            primary_category = max(category_scores, key=category_scores.get)
        else:
            primary_category = "general"

        # Extract potential topics from title and frequent words
        potential_topics = []

        if title:
            # Use title words as potential topics
            title_words = [w for w in title.split() if len(w) > 3]
            potential_topics.extend(title_words[:3])

        # Add frequent meaningful words
        words = re.findall(r"\b\w+\b", text_lower)
        word_freq = Counter(words)
        frequent_words = [
            word
            for word, freq in word_freq.most_common(5)
            if len(word) > 4 and word not in {"which", "their", "there", "would"}
        ]

        potential_topics.extend(frequent_words[:2])

        return {
            "primary_topic": potential_topics[0] if potential_topics else "general",
            "secondary_topics": potential_topics[1:4],
            "category": primary_category,
            "technical_level": "intermediate",
            "specialized_domains": [primary_category],
            "confidence_score": 0.4,
            "key_concepts": potential_topics[:3],
            "statistical_keywords": frequent_words,
            "vocabulary_richness": 0.0,
        }

    def _find_topic_clusters(
        self, doc_topics: Dict[str, Set[str]]
    ) -> List[Dict[str, Any]]:
        """
        Find clusters of documents that share similar topics.
        """
        clusters = []
        processed_docs = set()

        for doc_id, topics in doc_topics.items():
            if doc_id in processed_docs:
                continue

            # Find documents with significant topic overlap
            cluster_docs = {doc_id}
            cluster_topics = topics.copy()

            for other_doc, other_topics in doc_topics.items():
                if other_doc != doc_id and other_doc not in processed_docs:
                    overlap = len(topics.intersection(other_topics))
                    if overlap >= 2:  # At least 2 topics in common
                        cluster_docs.add(other_doc)
                        cluster_topics.update(other_topics)

            if len(cluster_docs) > 1:  # Only create cluster if multiple documents
                clusters.append(
                    {
                        "documents": list(cluster_docs),
                        "shared_topics": list(cluster_topics),
                        "topic_overlap": len(cluster_topics),
                        "document_count": len(cluster_docs),
                    }
                )

            processed_docs.update(cluster_docs)

        return clusters

    def _identify_relationships(
        self, doc_topics: Dict[str, Set[str]], clusters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify specific relationships between documents.
        """
        relationships = []

        # Process each cluster
        for cluster in clusters:
            cluster_docs = cluster["documents"]

            # Find pairwise relationships within cluster
            for i, doc1 in enumerate(cluster_docs):
                for doc2 in cluster_docs[i + 1 :]:
                    shared_topics = doc_topics[doc1].intersection(doc_topics[doc2])

                    if shared_topics:
                        relationships.append(
                            {
                                "document1": doc1,
                                "document2": doc2,
                                "relationship_type": "shared_topics",
                                "shared_topics": list(shared_topics),
                                "strength": len(shared_topics),
                                "cluster_id": len(relationships),  # Simple ID
                            }
                        )

        return relationships

    def _calculate_topic_distribution(
        self, doc_topics: Dict[str, Set[str]]
    ) -> Dict[str, int]:
        """Calculate the distribution of topics across documents."""
        all_topics = []
        for topics in doc_topics.values():
            all_topics.extend(topics)

        return dict(Counter(all_topics))

    def _get_most_common_topics(
        self, doc_topics: Dict[str, Set[str]]
    ) -> List[Tuple[str, int]]:
        """Get the most common topics across all documents."""
        distribution = self._calculate_topic_distribution(doc_topics)
        return sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:10]

    def suggest_related_documents(
        self,
        target_doc_topics: Set[str],
        all_documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Suggest related documents based on topic similarity.

        Args:
            target_doc_topics: Topics of the target document
            all_documents: All available documents with topic info
            top_k: Number of suggestions to return

        Returns:
            List of related document suggestions with similarity scores
        """
        suggestions = []

        for doc in all_documents:
            doc_topics = set(
                doc.get("topics", [])
                + doc.get("secondary_topics", [])
                + [doc.get("primary_topic", "")]
            )

            # Calculate topic overlap
            overlap = len(target_doc_topics.intersection(doc_topics))
            if overlap > 0:
                similarity_score = overlap / len(target_doc_topics.union(doc_topics))

                suggestions.append(
                    {
                        "document": doc,
                        "similarity_score": similarity_score,
                        "shared_topics": list(
                            target_doc_topics.intersection(doc_topics)
                        ),
                        "overlap_count": overlap,
                    }
                )

        # Sort by similarity and return top k
        suggestions.sort(key=lambda x: x["similarity_score"], reverse=True)
        return suggestions[:top_k]

    def create_topic_taxonomy(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a taxonomy of topics from a collection of documents.

        Args:
            documents: List of documents with topic information

        Returns:
            Topic taxonomy with hierarchy and relationships
        """
        # Collect all topics
        all_topics = set()
        topic_documents = defaultdict(list)

        for doc in documents:
            topics = set(
                doc.get("topics", [])
                + doc.get("secondary_topics", [])
                + [doc.get("primary_topic", "")]
            )
            all_topics.update(topics)

            for topic in topics:
                topic_documents[topic].append(doc.get("id") or doc.get("filename", ""))

        # Create basic taxonomy
        taxonomy = {
            "topics": list(all_topics),
            "topic_hierarchy": {},  # Could be enhanced with hierarchical relationships
            "topic_documents": dict(topic_documents),
            "topic_categories": self._categorize_topics(list(all_topics)),
            "total_topics": len(all_topics),
            "total_documents": len(documents),
        }

        return taxonomy

    def _categorize_topics(self, topics: List[str]) -> Dict[str, List[str]]:
        """Categorize topics into predefined categories."""
        categories = defaultdict(list)

        for topic in topics:
            topic_lower = topic.lower()

            # Find best category match
            best_category = "general"
            best_score = 0

            for category, keywords in self.topic_categories.items():
                score = sum(1 for keyword in keywords if keyword in topic_lower)
                if score > best_score:
                    best_score = score
                    best_category = category

            categories[best_category].append(topic)

        return dict(categories)
