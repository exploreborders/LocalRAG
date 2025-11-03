"""
Enhanced retrieval system with hybrid search, topic filtering, and hierarchical navigation.

This module provides advanced document retrieval capabilities including:
- Hybrid BM25 + Vector search
- Topic-based filtering and ranking
- Hierarchical document navigation
- Cross-document relationship discovery
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from elasticsearch import Elasticsearch
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import numpy as np

from .database.models import (
    SessionLocal, Document, DocumentChunk, DocumentChapter,
    Topic, DocumentTopic, DocumentTag, DocumentCategory
)
from .retrieval_db import DatabaseRetriever
from .pipeline.relevance_scorer import RelevanceScorer

logger = logging.getLogger(__name__)

class EnhancedRetriever:
    """
    Enhanced document retrieval with hybrid search and advanced filtering.

    Combines multiple retrieval strategies for optimal results.
    """

    def __init__(self,
                 vector_retriever: Optional[DatabaseRetriever] = None,
                 hybrid_weight: float = 0.7):
        """
        Initialize the enhanced retriever.

        Args:
            vector_retriever: Existing vector retriever instance
            hybrid_weight: Weight for vector search vs text search (0-1)
        """
        self.vector_retriever = vector_retriever or DatabaseRetriever()
        self.hybrid_weight = hybrid_weight
        self.relevance_scorer = RelevanceScorer()

        # Initialize Elasticsearch for BM25 search
        self.es = Elasticsearch(
            hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}],
            verify_certs=False
        )

        self.db: Session = SessionLocal()

    def __del__(self):
        """Clean up database connections."""
        self.db.close()

    def hybrid_search(self,
                     query: str,
                     top_k: int = 5,
                     filters: Optional[Dict[str, Any]] = None,
                     search_mode: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and vector similarity.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters (topics, tags, categories, etc.)
            search_mode: "hybrid", "vector", "text", or "topic"

        Returns:
            Ranked search results with metadata
        """
        logger.info(f"Performing {search_mode} search for: {query}")

        if search_mode == "vector":
            return self._vector_search(query, top_k, filters)
        elif search_mode == "text":
            return self._text_search(query, top_k, filters)
        elif search_mode == "topic":
            return self._topic_search(query, top_k, filters)
        else:  # hybrid
            return self._hybrid_search(query, top_k, filters)

    def _hybrid_search(self,
                      query: str,
                      top_k: int,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Combine BM25 text search with vector similarity search.
        """
        # Get results from both search methods
        vector_results = self._vector_search(query, top_k * 2, filters)
        text_results = self._text_search(query, top_k * 2, filters)

        # Combine and rerank results
        combined_results = self._combine_search_results(
            vector_results, text_results, query, top_k
        )

        return combined_results

    def _vector_search(self,
                      query: str,
                      top_k: int,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Pure vector similarity search.
        """
        # Use existing vector retriever
        results = self.vector_retriever.retrieve(query, top_k, filters)

        # Add search method metadata
        for result in results:
            result['search_method'] = 'vector'
            result['hybrid_score'] = result.get('score', 0) * self.hybrid_weight

        return results

    def _text_search(self,
                    query: str,
                    top_k: int,
                    filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        BM25 text search using Elasticsearch.
        """
        # Build Elasticsearch query
        es_query = {
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["content", "chapter_title", "filename"],
                            "type": "best_fields"
                        }
                    }
                }
            },
            "size": top_k * 2,  # Get more for filtering
            "_source": ["document_id", "chunk_id", "content", "chapter_title", "chapter_path"]
        }

        # Add filters if provided
        if filters:
            es_query["query"]["bool"]["filter"] = self._build_es_filters(filters)

        try:
            response = self.es.search(index="rag_chunks", body=es_query)
            hits = response['hits']['hits']

            results = []
            for hit in hits:
                source = hit['_source']
                result = {
                    'document_id': source['document_id'],
                    'chunk_id': source.get('chunk_id'),
                    'content': source['content'],
                    'score': hit['_score'],
                    'search_method': 'text',
                    'chapter_title': source.get('chapter_title'),
                    'chapter_path': source.get('chapter_path'),
                    'hybrid_score': hit['_score'] * (1 - self.hybrid_weight)
                }
                results.append(result)

            # Enrich with document metadata
            results = self._enrich_results_with_metadata(results)

            return results

        except Exception as e:
            logger.error(f"Elasticsearch text search failed: {e}")
            return []

    def _topic_search(self,
                     query: str,
                     top_k: int,
                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Topic-aware search that prioritizes documents with relevant topics.
        """
        # First, find documents with relevant topics
        topic_docs = self._find_documents_by_topics(query)

        if not topic_docs:
            # Fallback to regular search
            return self._vector_search(query, top_k, filters)

        # Search within topic-relevant documents
        topic_filters = filters.copy() if filters else {}
        topic_filters['document_ids'] = [doc['id'] for doc in topic_docs]

        results = self._vector_search(query, top_k, topic_filters)

        # Boost scores for topic-relevant documents
        for result in results:
            doc_id = result.get('document_id')
            topic_boost = next((doc['relevance'] for doc in topic_docs
                              if doc['id'] == doc_id), 1.0)
            result['score'] *= topic_boost
            result['topic_boost'] = topic_boost

        return results

    def _combine_search_results(self,
                               vector_results: List[Dict[str, Any]],
                               text_results: List[Dict[str, Any]],
                               query: str,
                               top_k: int) -> List[Dict[str, Any]]:
        """
        Combine and rerank results from multiple search methods.
        """
        # Create a combined result set
        result_map = {}

        # Add vector results
        for result in vector_results:
            key = f"{result['document_id']}_{result.get('chunk_id', 'no_chunk')}"
            result_map[key] = result

        # Add or merge text results
        for result in text_results:
            key = f"{result['document_id']}_{result.get('chunk_id', 'no_chunk')}"

            if key in result_map:
                # Merge results - combine scores
                existing = result_map[key]
                existing['hybrid_score'] = (
                    existing.get('hybrid_score', 0) +
                    result.get('hybrid_score', 0)
                )
                existing['search_methods'] = list(set(
                    existing.get('search_methods', [existing.get('search_method')]) +
                    [result.get('search_method')]
                ))
            else:
                result_map[key] = result

        # Convert to list and sort by combined score
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)

        # Apply query-specific reranking
        combined_results = self.relevance_scorer.rank_chunks_for_query(
            combined_results, query, top_k
        )

        return combined_results[:top_k]

    def _find_documents_by_topics(self, query: str) -> List[Dict[str, Any]]:
        """
        Find documents that match query topics.
        """
        # Extract potential topics from query
        query_words = set(query.lower().split())

        # Find topics that match query words
        matching_topics = self.db.query(Topic).filter(
            or_(*[Topic.name.ilike(f"%{word}%") for word in query_words])
        ).all()

        if not matching_topics:
            return []

        # Get documents related to these topics
        topic_ids = [t.id for t in matching_topics]

        doc_topics = self.db.query(DocumentTopic).filter(
            DocumentTopic.topic_id.in_(topic_ids)
        ).all()

        # Group by document and calculate relevance
        doc_relevance = {}
        for dt in doc_topics:
            doc_id = dt.document_id
            if doc_id not in doc_relevance:
                doc_relevance[doc_id] = {'relevance': 0, 'topics': []}
            doc_relevance[doc_id]['relevance'] += dt.relevance_score
            doc_relevance[doc_id]['topics'].append(dt.topic.name)

        # Convert to list format
        results = []
        for doc_id, data in doc_relevance.items():
            results.append({
                'id': doc_id,
                'relevance': data['relevance'],
                'topics': data['topics']
            })

        return sorted(results, key=lambda x: x['relevance'], reverse=True)

    def _build_es_filters(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build Elasticsearch filters from filter dictionary.
        """
        es_filters = []

        # Document ID filter
        if 'document_ids' in filters:
            es_filters.append({
                "terms": {"document_id": filters['document_ids']}
            })

        # Language filter
        if 'detected_language' in filters:
            es_filters.append({
                "term": {"detected_language": filters['detected_language']}
            })

        # Topic filter
        if 'topics' in filters:
            # This would require joining with document_topics
            # For now, we'll filter at the application level
            pass

        return es_filters

    def _enrich_results_with_metadata(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich search results with document metadata.
        """
        if not results:
            return results

        # Get unique document IDs
        doc_ids = list(set(r['document_id'] for r in results))

        # Batch query for document metadata
        documents = self.db.query(Document).filter(Document.id.in_(doc_ids)).all()
        doc_map = {doc.id: doc for doc in documents}

        # Enrich results
        for result in results:
            doc_id = result['document_id']
            if doc_id in doc_map:
                doc = doc_map[doc_id]
                result.update({
                    'filename': doc.filename,
                    'filepath': doc.filepath,
                    'detected_language': doc.detected_language,
                    'content_type': doc.content_type,
                    'document_summary': doc.document_summary,
                    'primary_topic': getattr(doc, 'primary_topic', None)
                })

        return results

    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply advanced filters to search results.
        """
        filtered_results = results

        # Topic filter
        if 'topics' in filters:
            topic_names = set(filters['topics'])
            filtered_results = [
                r for r in filtered_results
                if r.get('primary_topic') in topic_names or
                any(t in topic_names for t in r.get('secondary_topics', []))
            ]

        # Tag filter
        if 'tags' in filters:
            tag_names = set(filters['tags'])
            # This would require additional queries - simplified for now
            pass

        # Category filter
        if 'categories' in filters:
            category_names = set(filters['categories'])
            # This would require additional queries - simplified for now
            pass

        # Language filter
        if 'detected_language' in filters:
            lang = filters['detected_language']
            filtered_results = [
                r for r in filtered_results
                if r.get('detected_language') == lang
            ]

        # Date filters
        if 'date_from' in filters:
            from_date = filters['date_from']
            filtered_results = [
                r for r in filtered_results
                if r.get('upload_date', '') >= from_date
            ]

        if 'date_to' in filters:
            to_date = filters['date_to']
            filtered_results = [
                r for r in filtered_results
                if r.get('upload_date', '') <= to_date
            ]

        return filtered_results

    def hierarchical_search(self,
                           query: str,
                           hierarchy_level: str = "chapter",
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search at specific hierarchical levels (chapter, section, etc.).

        Args:
            query: Search query
            hierarchy_level: Level to search at ("chapter", "section", "subsection")
            top_k: Number of results

        Returns:
            Hierarchical search results
        """
        # Map hierarchy levels to database fields
        level_map = {
            "chapter": 1,
            "section": 2,
            "subsection": 3
        }

        target_level = level_map.get(hierarchy_level, 1)

        # Search for chapters at the target level
        chapters = self.db.query(DocumentChapter).filter(
            DocumentChapter.level == target_level
        ).all()

        # Score chapters based on query relevance
        scored_chapters = []
        query_embedding = self.vector_retriever.embed_query(query)

        for chapter in chapters:
            # Simple text similarity (could be enhanced with embeddings)
            content_lower = chapter.content.lower()
            query_lower = query.lower()

            # Calculate simple relevance score
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())

            overlap = len(query_words.intersection(content_words))
            relevance = overlap / len(query_words) if query_words else 0

            if relevance > 0:
                scored_chapters.append({
                    'chapter_id': chapter.id,
                    'document_id': chapter.document_id,
                    'chapter_title': chapter.chapter_title,
                    'chapter_path': chapter.chapter_path,
                    'content': chapter.content[:500] + '...' if len(chapter.content) > 500 else chapter.content,
                    'relevance_score': relevance,
                    'level': chapter.level
                })

        # Sort and return top results
        scored_chapters.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Enrich with document metadata
        return self._enrich_results_with_metadata(scored_chapters[:top_k])

    def get_related_documents(self,
                            document_id: int,
                            top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find documents related to the given document based on topics.

        Args:
            document_id: ID of the source document
            top_k: Number of related documents to return

        Returns:
            List of related documents with similarity scores
        """
        # Get topics for the source document
        source_topics = self.db.query(DocumentTopic).filter(
            DocumentTopic.document_id == document_id
        ).all()

        if not source_topics:
            return []

        topic_ids = [dt.topic_id for dt in source_topics]

        # Find other documents with these topics
        related_docs = self.db.query(DocumentTopic).filter(
            and_(
                DocumentTopic.topic_id.in_(topic_ids),
                DocumentTopic.document_id != document_id
            )
        ).all()

        # Calculate document similarity based on shared topics
        doc_scores = {}
        for dt in related_docs:
            doc_id = dt.document_id
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0
            doc_scores[doc_id] += dt.relevance_score

        # Get document details
        related_documents = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            doc = self.db.query(Document).filter(Document.id == doc_id).first()
            if doc:
                related_documents.append({
                    'id': doc.id,
                    'filename': doc.filename,
                    'filepath': doc.filepath,
                    'similarity_score': score,
                    'shared_topics': []  # Could be populated with actual shared topics
                })

        return related_documents

    def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """
        Get search suggestions based on existing topics and content.

        Args:
            partial_query: Partial query string
            limit: Maximum number of suggestions

        Returns:
            List of suggested search terms
        """
        suggestions = []

        # Get matching topics
        topics = self.db.query(Topic).filter(
            Topic.name.ilike(f"%{partial_query}%")
        ).limit(limit).all()

        suggestions.extend([topic.name for topic in topics])

        # Get matching chapter titles
        chapters = self.db.query(DocumentChapter).filter(
            DocumentChapter.chapter_title.ilike(f"%{partial_query}%")
        ).limit(limit).all()

        suggestions.extend([chapter.chapter_title for chapter in chapters])

        return list(set(suggestions))[:limit]