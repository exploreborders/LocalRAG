"""
AI-powered document enrichment service.

Provides automatic tagging, summarization, and topic extraction
using LLM capabilities for enhanced document management.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from src.database.models import Document, DocumentChunk, SessionLocal
from src.core.document_manager import TagManager, CategoryManager


class AIEnrichmentService:
    """
    Service for AI-powered document enrichment.

    Uses LLM to automatically analyze documents and add metadata.
    """

    def __init__(self, llm_client=None):
        """
        Initialize the AI enrichment service.

        Args:
            llm_client: LLM client for AI operations (optional, will try to import)
        """
        self.llm_client = llm_client
        if self.llm_client is None:
            try:
                from langchain_ollama import OllamaLLM

                self.llm_client = OllamaLLM(
                    model="llama3.2:latest"
                )  # Use available model
            except ImportError:
                print("⚠️ Ollama LLM not available for AI enrichment")
                self.llm_client = None

        self.db = SessionLocal()
        self.tag_manager = TagManager(self.db)
        self.category_manager = CategoryManager(self.db)

    def __del__(self):
        """Clean up database connections."""
        self.db.close()

    def enrich_document(self, document_id: int, force: bool = False) -> Dict[str, Any]:
        """
        Enrich a document with AI-generated metadata.

        Args:
            document_id: Document ID to enrich
            force: Whether to re-enrich already enriched documents

        Returns:
            Dict with enrichment results
        """
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return {"success": False, "error": "Document not found"}

        # Check if already enriched (unless force is True)
        if not force and (document.document_summary or document.key_topics):
            return {"success": False, "error": "Document already enriched"}

        if not self.llm_client:
            return {"success": False, "error": "LLM client not available"}

        try:
            # Get document content (first few chunks for analysis)
            chunks = (
                self.db.query(DocumentChunk)
                .filter(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.chunk_index)
                .limit(5)
                .all()
            )

            content = " ".join([chunk.content for chunk in chunks])

            # Generate enrichment data
            enrichment_data = self._generate_enrichment_data(content, document.filename)

            # Apply enrichment
            self._apply_enrichment(document, enrichment_data)

            return {
                "success": True,
                "document_id": document_id,
                "enrichment": enrichment_data,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_enrichment_data(self, content: str, filename: str) -> Dict[str, Any]:
        """
        Generate AI enrichment data for document content with enhanced categorization.

        Args:
            content: Document content to analyze
            filename: Document filename

        Returns:
            Dict with tags, summary, topics, categories, etc.
        """
        # Truncate content if too long (keep first 2000 chars for analysis)
        analysis_content = content[:2000] if len(content) > 2000 else content

        # Generate summary
        summary_prompt = f"""
        Please provide a concise summary of the following document in 2-3 sentences:

        Document: {filename}
        Content: {analysis_content}

        Summary:
        """

        summary = self._call_llm(summary_prompt, max_tokens=150)

        # Clean up the summary response - remove introductory text
        summary_lower = summary.lower()
        if (
            summary_lower.startswith("here is a summary")
            or summary_lower.startswith("here is a concise summary")
            or summary_lower.startswith("summary:")
        ):
            # Find the first colon and take everything after it
            colon_index = summary.find(":")
            if colon_index != -1:
                summary = summary[colon_index + 1 :].strip()

        # Generate tags
        tags_prompt = f"""
        Analyze the following document and suggest 3-5 relevant tags (single words or short phrases) that would help categorize and find this document.
        Return only a comma-separated list of tags, no explanations or numbering.

        Document: {filename}
        Content: {analysis_content}

        Tags:
        """

        tags_response = self._call_llm(tags_prompt, max_tokens=50)
        # Clean up the response - handle various formats (comma-separated, bullet points, numbered)
        tags = []
        # Split on common separators
        for separator in [",", "\n", "*", "-", "•"]:
            if separator in tags_response:
                candidates = tags_response.split(separator)
                break
        else:
            candidates = [tags_response]

        for tag in candidates:
            tag = tag.strip()
            # Remove numbering like "1. Tag" and bullet points
            tag = re.sub(r"^\d+\.\s*", "", tag)
            tag = re.sub(r"^[-•*]\s*", "", tag)
            # Skip empty tags or very long ones
            if tag and len(tag) <= 50 and len(tag) >= 2:  # At least 2 chars
                tags.append(tag)

        tags = tags[:5]  # Limit to 5 tags

        # Generate topics
        topics_prompt = f"""
        Extract the main topics/themes from the following document.
        Return 2-4 key topics as a comma-separated list, no explanations.

        Document: {filename}
        Content: {analysis_content}

        Topics:
        """

        topics_response = self._call_llm(topics_prompt, max_tokens=50)
        # Clean up the response
        topics = []
        for topic in topics_response.split(","):
            topic = topic.strip()
            # Remove numbering and clean up
            topic = re.sub(r"^\d+\.\s*", "", topic)
            topic = re.sub(r"^[-•*]\s*", "", topic)
            if topic and len(topic) <= 100:  # Limit topic length
                topics.append(topic)
        topics = topics[:4]  # Limit to 4 topics

        # AI-powered category classification
        category_data = self._classify_document_category(
            analysis_content, filename, tags, topics
        )

        # Estimate reading time (rough calculation: 200 words per minute)
        word_count = len(re.findall(r"\w+", content))
        reading_time = max(1, round(word_count / 200))

        return {
            "summary": summary.strip(),
            "tags": tags,
            "topics": topics,
            "primary_category": category_data.get("primary_category"),
            "subcategories": category_data.get("subcategories", []),
            "category_confidence": category_data.get("confidence", 0.0),
            "alternative_categories": category_data.get("alternatives", []),
            "reading_time": reading_time,
            "word_count": word_count,
            "generated_at": datetime.now().isoformat(),
        }

    def _classify_document_category(
        self, content: str, filename: str, tags: List[str], topics: List[str]
    ) -> Dict[str, Any]:
        """
        Use AI to classify document into categories and subcategories.

        Args:
            content: Document content
            filename: Document filename
            tags: Generated tags
            topics: Generated topics

        Returns:
            Dict with category classification data
        """
        if not self.llm_client:
            return {
                "primary_category": "General",
                "subcategories": [],
                "confidence": 0.0,
                "alternatives": [],
            }

        # Primary category classification
        category_prompt = f"""
        Analyze this document and assign it to the most appropriate primary category.
        Choose from: Academic, Technical, Business, Scientific, Educational, Legal, Medical, Creative, Reference, General

        Document: {filename}
        Content preview: {content[:500]}
        Tags: {", ".join(tags)}
        Topics: {", ".join(topics)}

        Return only the category name (one word):
        """

        primary_category = self._call_llm(category_prompt, max_tokens=20).strip()

        # Validate category
        valid_categories = [
            "Academic",
            "Technical",
            "Business",
            "Scientific",
            "Educational",
            "Legal",
            "Medical",
            "Creative",
            "Reference",
            "General",
        ]
        if primary_category not in valid_categories:
            primary_category = "General"

        # Generate subcategories
        subcategory_prompt = f"""
        For the category "{primary_category}", suggest 1-2 relevant subcategories for this document.
        Consider the tags and topics provided.

        Document: {filename}
        Tags: {", ".join(tags)}
        Topics: {", ".join(topics)}

        IMPORTANT: Return ONLY clean subcategory names separated by commas.
        Do NOT include explanations, prefixes, or quotes.
        Examples: "Machine Learning, Deep Learning" or "Computer Vision" or "Natural Language Processing, Neural Networks"
        """

        subcategory_response = self._call_llm(subcategory_prompt, max_tokens=50).strip()
        subcategories = [
            s.strip() for s in subcategory_response.split(",") if s.strip()
        ]

        # Clean subcategory names - extract actual category names
        cleaned_subcategories = []
        for subcat in subcategories[:2]:  # Limit to 2
            # Remove common AI prefixes and clean up
            cleaned = self._clean_category_name(subcat)
            if cleaned and len(cleaned) > 1:  # Avoid single characters
                cleaned_subcategories.append(cleaned)

        subcategories = cleaned_subcategories

        # Alternative categories with confidence
        alternatives_prompt = f"""
        Suggest 2 alternative categories for this document (besides {primary_category}).
        Include confidence scores (0.0-1.0) in format: Category:Score

        Document: {filename}
        Tags: {", ".join(tags)}
        Topics: {", ".join(topics)}

        Return format: Category1:0.8, Category2:0.6
        """

        alternatives_response = self._call_llm(
            alternatives_prompt, max_tokens=40
        ).strip()
        alternatives = []
        for alt in alternatives_response.split(","):
            if ":" in alt:
                cat, score = alt.split(":", 1)
                try:
                    score_val = float(score.strip())
                    cleaned_cat = self._clean_category_name(cat.strip())
                    if cleaned_cat:
                        alternatives.append(
                            {"category": cleaned_cat, "confidence": score_val}
                        )
                except ValueError:
                    continue

        return {
            "primary_category": primary_category,
            "subcategories": subcategories,
            "confidence": 0.8,  # Default high confidence for AI classification
            "alternatives": alternatives[:2],  # Limit to 2 alternatives
        }

    def _clean_category_name(self, raw_name: str) -> str:
        """
        Clean AI-generated category names to extract actual category names.

        Args:
            raw_name: Raw category name from AI

        Returns:
            Cleaned category name
        """
        import re

        cleaned = raw_name.strip()

        # Remove common AI prefixes and verbose phrases
        prefixes_to_remove = [
            r"I suggest the following relevant subcategories for the .*? category:\s*",
            r"I would suggest the following relevant subcategories for the .*? category:\s*",
            r"For the .*? category, I suggest:\s*",
            r"Based on the .*?(?:provided|document|tags|topics).*?:\s*",
            r"Two relevant subcategories for .*? could be:\s*",
            r"Relevant subcategories?:\s*",
            r"Suggested subcategories?:\s*",
            r"Subcategories?:\s*",
            r"Suggested:\s*",
            r"^\s*[-•*]\s*",  # Bullet points
            r"^\s*\d+\.\s*",  # Numbered lists
        ]

        # Remove prefixes
        for prefix in prefixes_to_remove:
            cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)

        # Split on common separators and take the first meaningful part
        parts = re.split(r"[,;]|\sand\s|\sor\s|\scould be\s", cleaned)
        if parts:
            # Take the first non-empty part
            cleaned = parts[0].strip()

        # Remove quotes and extra punctuation
        cleaned = re.sub(r'^["\']|["\']$', "", cleaned)  # Remove surrounding quotes
        cleaned = re.sub(
            r"[^\w\s\-&]", "", cleaned
        )  # Keep letters, numbers, spaces, hyphens, ampersands
        cleaned = re.sub(r"\s+", " ", cleaned)  # Normalize whitespace
        cleaned = cleaned.strip()

        # Skip if it's too short or contains unwanted words
        skip_words = [
            "the",
            "and",
            "or",
            "for",
            "with",
            "from",
            "this",
            "that",
            "these",
            "those",
        ]
        if len(cleaned.split()) <= 1 or any(
            word in cleaned.lower() for word in skip_words
        ):
            return ""

        # Capitalize properly (title case for category names)
        if cleaned:
            cleaned = cleaned.title()

        return cleaned

    def _apply_enrichment(self, document: Document, enrichment_data: Dict[str, Any]):
        """
        Apply enrichment data to document with enhanced category support.

        Args:
            document: Document object to update
            enrichment_data: Enrichment data to apply
        """
        # Update dedicated AI enrichment columns
        document.document_summary = enrichment_data.get("summary")
        document.key_topics = enrichment_data.get("topics", [])
        document.reading_time_minutes = enrichment_data.get("reading_time")

        # Update custom metadata for backward compatibility and additional metadata
        custom_metadata = document.custom_metadata or {}
        custom_metadata.update(
            {
                "ai_enriched": True,
                "word_count": enrichment_data.get("word_count"),
                "ai_generated_at": enrichment_data.get("generated_at"),
                "ai_category_confidence": enrichment_data.get(
                    "category_confidence", 0.0
                ),
                "ai_alternative_categories": enrichment_data.get(
                    "alternative_categories", []
                ),
            }
        )
        document.custom_metadata = custom_metadata

        # Create and assign tags
        for tag_name in enrichment_data.get("tags", []):
            try:
                tag = self.tag_manager.get_tag_by_name(tag_name)
                if not tag:
                    # Create new tag with AI-generated unique color
                    tag = self.tag_manager.create_tag_with_ai_color(tag_name)

                # Add tag to document if not already assigned
                if tag:
                    self.tag_manager.add_tag_to_document(document.id, tag.id)
            except Exception as e:
                print(f"Warning: Failed to add tag '{tag_name}': {e}")
                continue

        # Create and assign AI-suggested categories
        primary_category = enrichment_data.get("primary_category")
        if primary_category:
            try:
                # Check if primary category exists
                category = self.category_manager.get_category_by_name(primary_category)
                if not category:
                    # Create new category
                    category = self.category_manager.create_category(
                        name=primary_category,
                        description=f"AI-classified {primary_category.lower()} category",
                    )

                # Add category to document if not already assigned
                if category:
                    self.category_manager.add_category_to_document(
                        document.id, category.id
                    )

                    # Create subcategories if provided
                    parent_id = category.id
                    for subcategory_name in enrichment_data.get("subcategories", []):
                        subcat = self.category_manager.get_category_by_name(
                            subcategory_name, parent_id
                        )
                        if not subcat:
                            subcat = self.category_manager.create_category(
                                name=subcategory_name,
                                description=f"AI-generated subcategory of {primary_category}",
                                parent_id=parent_id,
                            )
                        if subcat:
                            self.category_manager.add_category_to_document(
                                document.id, subcat.id
                            )

            except Exception as e:
                print(f"Warning: Failed to add AI categories: {e}")

        self.db.commit()

    def _call_llm(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Call LLM with prompt.

        Args:
            prompt: Prompt to send to LLM
            max_tokens: Maximum tokens to generate

        Returns:
            LLM response text
        """
        if not self.llm_client:
            return "LLM not available"

        try:
            # Use OllamaLLM invoke method
            response = self.llm_client.invoke(prompt)
            return response.strip() if response else ""
        except Exception as e:
            print(f"LLM call failed: {e}")
            return "Error generating response"

    def get_document_summary(self, document_id: int) -> Optional[str]:
        """
        Get AI-generated summary for a document.

        Args:
            document_id: Document ID

        Returns:
            Summary text or None if not available
        """
        document = self.db.query(Document).filter(Document.id == document_id).first()
        return document.document_summary if document else None

    def get_document_topics(self, document_id: int) -> List[str]:
        """
        Get AI-extracted topics for a document.

        Args:
            document_id: Document ID

        Returns:
            List of topics
        """
        document = self.db.query(Document).filter(Document.id == document_id).first()
        return document.key_topics if document and document.key_topics else []

    def batch_enrich_documents(
        self, document_ids: List[int], force: bool = False
    ) -> Dict[str, Any]:
        """
        Enrich multiple documents in batch.

        Args:
            document_ids: List of document IDs to enrich
            force: Whether to re-enrich already enriched documents

        Returns:
            Dict with batch results
        """
        results = []
        successful = 0
        failed = 0

        for doc_id in document_ids:
            result = self.enrich_document(doc_id, force)
            results.append(result)
            if result["success"]:
                successful += 1
            else:
                failed += 1

        return {
            "total": len(document_ids),
            "successful": successful,
            "failed": failed,
            "results": results,
        }

    def find_similar_documents(
        self, document_id: int, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find documents similar to the given document based on AI-generated topics and tags.

        Args:
            document_id: Source document ID
            limit: Maximum number of similar documents to return

        Returns:
            List of similar documents with similarity scores
        """
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return []

        # Get source document's tags and topics
        source_tags = [tag.name for tag in document.tags]
        source_topics = []
        if document.custom_metadata:
            source_topics = document.custom_metadata.get("topics", [])

        # Find documents with overlapping tags/topics
        similar_docs = []
        all_docs = self.db.query(Document).filter(Document.id != document_id).all()

        for doc in all_docs:
            score = 0
            doc_tags = [tag.name for tag in doc.tags]
            doc_topics = []
            if doc.custom_metadata:
                doc_topics = doc.custom_metadata.get("topics", [])

            # Calculate similarity score
            tag_overlap = len(set(source_tags) & set(doc_tags))
            topic_overlap = len(set(source_topics) & set(doc_topics))

            score = tag_overlap * 2 + topic_overlap * 3  # Weight topics higher

            if score > 0:
                similar_docs.append(
                    {
                        "document": doc,
                        "score": score,
                        "shared_tags": list(set(source_tags) & set(doc_tags)),
                        "shared_topics": list(set(source_topics) & set(doc_topics)),
                    }
                )

        # Sort by score and return top results
        similar_docs.sort(key=lambda x: x["score"], reverse=True)
        return similar_docs[:limit]
