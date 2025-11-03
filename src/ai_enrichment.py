"""
AI-powered document enrichment service.

Provides automatic tagging, summarization, and topic extraction
using LLM capabilities for enhanced document management.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from .database.models import Document, DocumentChunk, SessionLocal
from .document_managers import TagManager, CategoryManager


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
                self.llm_client = OllamaLLM(model="llama3.2:latest")  # Use available model
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
            return {'success': False, 'error': 'Document not found'}

        # Check if already enriched (unless force is True)
        if not force and (document.document_summary or document.key_topics):
            return {'success': False, 'error': 'Document already enriched'}

        if not self.llm_client:
            return {'success': False, 'error': 'LLM client not available'}

        try:
            # Get document content (first few chunks for analysis)
            chunks = self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).order_by(DocumentChunk.chunk_index).limit(5).all()

            content = ' '.join([chunk.content for chunk in chunks])

            # Generate enrichment data
            enrichment_data = self._generate_enrichment_data(content, document.filename)

            # Apply enrichment
            self._apply_enrichment(document, enrichment_data)

            return {
                'success': True,
                'document_id': document_id,
                'enrichment': enrichment_data
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _generate_enrichment_data(self, content: str, filename: str) -> Dict[str, Any]:
        """
        Generate AI enrichment data for document content.

        Args:
            content: Document content to analyze
            filename: Document filename

        Returns:
            Dict with tags, summary, topics, etc.
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
        for separator in [',', '\n', '*', '-', '•']:
            if separator in tags_response:
                candidates = tags_response.split(separator)
                break
        else:
            candidates = [tags_response]

        for tag in candidates:
            tag = tag.strip()
            # Remove numbering like "1. Tag" and bullet points
            tag = re.sub(r'^\d+\.\s*', '', tag)
            tag = re.sub(r'^[-•*]\s*', '', tag)
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
        for topic in topics_response.split(','):
            topic = topic.strip()
            # Remove numbering and clean up
            topic = re.sub(r'^\d+\.\s*', '', topic)
            topic = re.sub(r'^[-•*]\s*', '', topic)
            if topic and len(topic) <= 100:  # Limit topic length
                topics.append(topic)
        topics = topics[:4]  # Limit to 4 topics

        # Estimate reading time (rough calculation: 200 words per minute)
        word_count = len(re.findall(r'\w+', content))
        reading_time = max(1, round(word_count / 200))

        return {
            'summary': summary.strip(),
            'tags': tags,
            'topics': topics,
            'reading_time': reading_time,
            'word_count': word_count,
            'generated_at': datetime.now().isoformat()
        }

    def _apply_enrichment(self, document: Document, enrichment_data: Dict[str, Any]):
        """
        Apply enrichment data to document.

        Args:
            document: Document object to update
            enrichment_data: Enrichment data to apply
        """
        # Update dedicated AI enrichment columns
        document.document_summary = enrichment_data.get('summary')
        document.key_topics = enrichment_data.get('topics', [])
        document.reading_time_minutes = enrichment_data.get('reading_time')

        # Update custom metadata for backward compatibility and additional metadata
        custom_metadata = document.custom_metadata or {}
        custom_metadata.update({
            'ai_enriched': True,
            'word_count': enrichment_data.get('word_count'),
            'ai_generated_at': enrichment_data.get('generated_at')
        })
        document.custom_metadata = custom_metadata

        # Create and assign tags
        for tag_name in enrichment_data.get('tags', []):
            try:
                tag = self.tag_manager.get_tag_by_name(tag_name)
                if not tag:
                    # Create new tag with default color
                    tag = self.tag_manager.create_tag(tag_name, color="#6c757d")  # Gray color

                # Add tag to document if not already assigned
                if tag:
                    self.tag_manager.add_tag_to_document(document.id, tag.id)
            except Exception as e:
                print(f"Warning: Failed to add tag '{tag_name}': {e}")
                continue

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

    def batch_enrich_documents(self, document_ids: List[int], force: bool = False) -> Dict[str, Any]:
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
            if result['success']:
                successful += 1
            else:
                failed += 1

        return {
            'total': len(document_ids),
            'successful': successful,
            'failed': failed,
            'results': results
        }

    def find_similar_documents(self, document_id: int, limit: int = 5) -> List[Dict[str, Any]]:
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
            source_topics = document.custom_metadata.get('topics', [])

        # Find documents with overlapping tags/topics
        similar_docs = []
        all_docs = self.db.query(Document).filter(Document.id != document_id).all()

        for doc in all_docs:
            score = 0
            doc_tags = [tag.name for tag in doc.tags]
            doc_topics = []
            if doc.custom_metadata:
                doc_topics = doc.custom_metadata.get('topics', [])

            # Calculate similarity score
            tag_overlap = len(set(source_tags) & set(doc_tags))
            topic_overlap = len(set(source_topics) & set(doc_topics))

            score = tag_overlap * 2 + topic_overlap * 3  # Weight topics higher

            if score > 0:
                similar_docs.append({
                    'document': doc,
                    'score': score,
                    'shared_tags': list(set(source_tags) & set(doc_tags)),
                    'shared_topics': list(set(source_topics) & set(doc_topics))
                })

        # Sort by score and return top results
        similar_docs.sort(key=lambda x: x['score'], reverse=True)
        return similar_docs[:limit]