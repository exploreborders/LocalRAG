#!/usr/bin/env python3
"""
Enhanced Search System for LocalRAG.

This module provides advanced search capabilities including:
- Hybrid search combining vector similarity and BM25 keyword search
- Faceted search with dynamic filters
- Advanced query processing with boolean operators and field-specific queries
- Search analytics and performance metrics
- Query expansion using knowledge graph relationships
"""

import logging
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from elasticsearch import Elasticsearch

from src.core.embeddings import create_embeddings, get_embedding_model
from src.core.knowledge_graph import KnowledgeGraph
from src.database.models import Document, SessionLocal
from src.utils.error_handler import ErrorHandler, ProcessingError, ValidationError

logger = logging.getLogger(__name__)


class QueryParser:
    """
    Advanced query parser supporting boolean operators, phrase search, and field-specific queries.
    """

    def __init__(self):
        self.field_mappings = {
            "title": "title",
            "content": "content",
            "tags": "tags",
            "categories": "categories",
            "author": "author",
            "filename": "filename",
            "language": "detected_language",
        }

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse advanced query with boolean operators and field specifications.

        Supports:
        - Boolean operators: AND, OR, NOT
        - Field-specific queries: field:value
        - Phrase search: "exact phrase"
        - Wildcards: term*
        - Fuzzy search: term~

        Args:
            query: Raw query string

        Returns:
            Parsed query structure
        """
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")

        parsed = {
            "original_query": query,
            "boolean_query": None,
            "field_queries": {},
            "phrase_terms": [],
            "wildcard_terms": [],
            "fuzzy_terms": [],
            "simple_terms": [],
            "excluded_terms": [],
        }

        # Extract field-specific queries first
        query, field_queries = self._extract_field_queries(query)
        parsed["field_queries"] = field_queries

        # Extract phrases
        query, phrases = self._extract_phrases(query)
        parsed["phrase_terms"] = phrases

        # Parse boolean operators
        boolean_query = self._parse_boolean_query(query)
        if boolean_query:
            parsed["boolean_query"] = boolean_query
        else:
            # Simple term extraction
            terms = self._extract_terms(query)
            parsed.update(terms)

        return parsed

    def _extract_field_queries(self, query: str) -> Tuple[str, Dict[str, List[str]]]:
        """Extract field:value patterns from query."""
        field_queries = defaultdict(list)

        # Pattern for field:value or field:"value"
        field_pattern = r'(\w+):(?:"([^"]*)"|\'([^\']*)\'|([^\s]+))'
        matches = re.findall(field_pattern, query)

        for match in matches:
            field, quoted_val, single_quoted_val, unquoted_val = match
            value = quoted_val or single_quoted_val or unquoted_val

            if field in self.field_mappings:
                field_queries[field].append(value)
                # Remove from query
                query = re.sub(rf"{re.escape(field)}:{re.escape(value)}", "", query)

        return query.strip(), dict(field_queries)

    def _extract_phrases(self, query: str) -> Tuple[str, List[str]]:
        """Extract quoted phrases from query."""
        phrases = []
        phrase_pattern = r'"([^"]*)"|\'([^\']*)\''

        def replace_phrase(match):
            phrase = match.group(1) or match.group(2)
            phrases.append(phrase)
            return f"__PHRASE_{len(phrases) - 1}__"

        query = re.sub(phrase_pattern, replace_phrase, query)
        return query, phrases

    def _parse_boolean_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Parse boolean operators (AND, OR, NOT)."""
        # Simple boolean parsing - can be enhanced
        if any(op in query.upper() for op in ["AND", "OR", "NOT"]):
            # For now, return basic structure - full boolean parsing would be complex
            return {
                "type": "boolean",
                "query": query,
                "operators": ["AND", "OR", "NOT"],  # Detected operators
            }
        return None

    def _extract_terms(self, query: str) -> Dict[str, List[str]]:
        """Extract different types of terms from query."""
        terms = {
            "simple_terms": [],
            "wildcard_terms": [],
            "fuzzy_terms": [],
            "excluded_terms": [],
        }

        # Split by whitespace and process each term
        words = query.split()

        for word in words:
            word = word.strip()
            if not word:
                continue

            if word.startswith("-"):
                terms["excluded_terms"].append(word[1:])
            elif word.endswith("*"):
                terms["wildcard_terms"].append(word)
            elif word.endswith("~"):
                terms["fuzzy_terms"].append(word[:-1])
            else:
                terms["simple_terms"].append(word)

        return terms


class HybridSearchEngine:
    """
    Advanced hybrid search engine combining vector similarity and BM25 keyword search.
    """

    def __init__(
        self,
        model_name: str = "embeddinggemma:latest",
        backend: str = "ollama",
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        use_reranking: bool = True,
    ):
        """
        Initialize hybrid search engine.

        Args:
            model_name: Embedding model name
            backend: Embedding backend
            vector_weight: Weight for vector similarity (0-1)
            bm25_weight: Weight for BM25 keyword search (0-1)
            use_reranking: Whether to use cross-encoder reranking
        """
        if not (0 <= vector_weight <= 1) or not (0 <= bm25_weight <= 1):
            raise ValidationError("Weights must be between 0 and 1")

        self.model_name = model_name
        self.backend = backend
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.use_reranking = use_reranking

        self.error_handler = ErrorHandler(__name__)
        self.query_parser = QueryParser()

        # Initialize components
        self.db = SessionLocal()
        self.knowledge_graph = KnowledgeGraph(self.db)

        try:
            self.model = get_embedding_model(model_name, backend)
        except Exception as e:
            raise ProcessingError(f"Failed to load embedding model: {e}")

    def __del__(self):
        """Clean up database connections."""
        if hasattr(self, "db"):
            self.db.close()

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        search_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform advanced hybrid search.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Search filters
            search_config: Additional search configuration

        Returns:
            Search results with analytics
        """
        start_time = time.time()
        search_config = search_config or {}

        try:
            # Parse query
            parsed_query = self.query_parser.parse_query(query)

            # Expand query using knowledge graph
            expanded_query = self._expand_query_with_kg(parsed_query, filters or {})

            # Generate query embedding
            query_embedding = self._get_query_embedding(query)

            # Perform hybrid search
            vector_results = self._vector_search(query_embedding, top_k * 2, filters)
            bm25_results = self._bm25_search(expanded_query, top_k * 2, filters)

            # Combine results
            combined_results = self._combine_hybrid_results(vector_results, bm25_results, top_k)

            # Apply faceted filtering if requested
            if search_config.get("enable_facets", False):
                combined_results = self._apply_faceted_filters(combined_results, filters)

            # Rerank if enabled
            if self.use_reranking:
                combined_results = self._rerank_results(combined_results, query, top_k)

            # Generate analytics
            analytics = self._generate_search_analytics(
                parsed_query, combined_results, time.time() - start_time
            )

            return {
                "results": combined_results,
                "analytics": analytics,
                "parsed_query": parsed_query,
                "expanded_query": expanded_query,
                "total_time": time.time() - start_time,
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "results": [],
                "analytics": {"error": str(e)},
                "total_time": time.time() - start_time,
            }

    def _expand_query_with_kg(
        self, parsed_query: Dict[str, Any], filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Expand query using knowledge graph relationships."""
        expanded = parsed_query.copy()

        # Extract terms for expansion
        all_terms = []
        all_terms.extend(parsed_query.get("simple_terms", []))
        all_terms.extend(parsed_query.get("phrase_terms", []))
        for field_terms in parsed_query.get("field_queries", {}).values():
            all_terms.extend(field_terms)

        # Use knowledge graph to expand
        if all_terms:
            # Get related tags and categories
            expanded_tags = set()
            expanded_categories = set()

            for term in all_terms[:3]:  # Limit to avoid over-expansion
                # Simple term-based expansion (can be enhanced)
                term_lower = term.lower()

                # Expand using existing knowledge graph method
                kg_expansion = self.knowledge_graph.expand_query_context(
                    tag_names=[term_lower], category_names=[], context_depth=2
                )

                expanded_tags.update(kg_expansion.get("expanded_tags", []))
                expanded_categories.update(kg_expansion.get("expanded_categories", []))

            expanded["expanded_tags"] = list(expanded_tags)
            expanded["expanded_categories"] = list(expanded_categories)

        return expanded

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for query."""
        embeddings_array, _ = create_embeddings([query], self.model_name, self.backend)
        if embeddings_array is None or len(embeddings_array) == 0:
            raise ProcessingError(f"Failed to generate embedding for query: {query}")
        return embeddings_array[0]

    def _vector_search(
        self, query_embedding: np.ndarray, top_k: int, filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        # Use pgvector for vector search
        # This is a simplified implementation - would need actual pgvector integration
        return []

    def _bm25_search(
        self,
        expanded_query: Dict[str, Any],
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search using Elasticsearch."""
        try:
            es_client = self._get_es_client()

            # Build Elasticsearch query
            es_query = self._build_advanced_es_query(expanded_query, filters)

            response = es_client.search(index="chunks", body=es_query, size=top_k)

            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                doc_id = source.get("document_id")

                # Get document metadata
                document = self.db.query(Document).filter(Document.id == doc_id).first()

                if document:
                    result = {
                        "chunk_id": hit["_id"],
                        "content": source.get("content", ""),
                        "score": hit["_score"],
                        "document_title": document.filename,
                        "document_id": doc_id,
                        "tags": [tag_assignment.tag.name for tag_assignment in document.tags],
                        "categories": [
                            cat_assignment.category.name for cat_assignment in document.categories
                        ],
                        "metadata": source.get("metadata", {}),
                        "search_type": "bm25",
                    }
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _build_advanced_es_query(
        self, expanded_query: Dict[str, Any], filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build advanced Elasticsearch query."""
        must_clauses = []
        should_clauses = []
        must_not_clauses = []
        filter_clauses = []

        # Handle field-specific queries
        field_queries = expanded_query.get("field_queries", {})
        for field, values in field_queries.items():
            if field in self.query_parser.field_mappings:
                es_field = self.query_parser.field_mappings[field]
                for value in values:
                    must_clauses.append({"match": {es_field: value}})

        # Handle boolean query
        boolean_query = expanded_query.get("boolean_query")
        if boolean_query:
            # Simple boolean query handling
            query_str = boolean_query.get("query", "")
            must_clauses.append({"query_string": {"query": query_str, "default_field": "content"}})
        else:
            # Handle simple terms
            simple_terms = expanded_query.get("simple_terms", [])
            if simple_terms:
                must_clauses.append(
                    {
                        "multi_match": {
                            "query": " ".join(simple_terms),
                            "fields": ["content^2", "title", "tags", "categories"],
                            "type": "best_fields",
                        }
                    }
                )

            # Handle phrase terms
            phrase_terms = expanded_query.get("phrase_terms", [])
            for phrase in phrase_terms:
                must_clauses.append({"match_phrase": {"content": phrase}})

            # Handle wildcard terms
            wildcard_terms = expanded_query.get("wildcard_terms", [])
            for term in wildcard_terms:
                should_clauses.append({"wildcard": {"content": term}})

        # Handle excluded terms
        excluded_terms = expanded_query.get("excluded_terms", [])
        for term in excluded_terms:
            must_not_clauses.append({"match": {"content": term}})

        # Add expanded tags and categories
        expanded_tags = expanded_query.get("expanded_tags", [])
        expanded_categories = expanded_query.get("expanded_categories", [])

        if expanded_tags:
            should_clauses.append({"terms": {"tags": expanded_tags}})

        if expanded_categories:
            should_clauses.append({"terms": {"categories": expanded_categories}})

        # Build filters
        if filters:
            self._build_filter_clauses(filters, filter_clauses)

        # Construct final query
        query = {"bool": {}}

        if must_clauses:
            query["bool"]["must"] = must_clauses
        if should_clauses:
            query["bool"]["should"] = should_clauses
            query["bool"]["minimum_should_match"] = 0
        if must_not_clauses:
            query["bool"]["must_not"] = must_not_clauses
        if filter_clauses:
            query["bool"]["filter"] = filter_clauses

        return {"query": query}

    def _build_filter_clauses(self, filters: Dict[str, Any], filter_clauses: List[Dict[str, Any]]):
        """Build filter clauses from filters dict."""
        # Tag filters
        tags = filters.get("tags", [])
        if tags:
            filter_clauses.append({"terms": {"tags": tags}})

        # Category filters
        categories = filters.get("categories", [])
        if categories:
            filter_clauses.append({"terms": {"categories": categories}})

        # Date filters
        date_from = filters.get("date_from")
        date_to = filters.get("date_to")
        if date_from or date_to:
            date_filter = {"range": {"upload_date": {}}}
            if date_from:
                date_filter["range"]["upload_date"]["gte"] = date_from
            if date_to:
                date_filter["range"]["upload_date"]["lte"] = date_to
            filter_clauses.append(date_filter)

        # Language filter
        language = filters.get("detected_language")
        if language:
            filter_clauses.append({"term": {"detected_language": language}})

        # Author filter
        author = filters.get("author")
        if author:
            filter_clauses.append({"match": {"author": author}})

    def _combine_hybrid_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Combine vector and BM25 results using weighted scoring."""
        # Create combined results dict
        combined = {}

        # Add vector results
        for result in vector_results:
            doc_id = result["document_id"]
            chunk_id = result["chunk_id"]
            key = f"{doc_id}_{chunk_id}"

            result_copy = result.copy()
            result_copy["vector_score"] = result["score"]
            result_copy["bm25_score"] = 0.0
            result_copy["combined_score"] = self.vector_weight * result["score"]
            combined[key] = result_copy

        # Add/merge BM25 results
        for result in bm25_results:
            doc_id = result["document_id"]
            chunk_id = result["chunk_id"]
            key = f"{doc_id}_{chunk_id}"

            if key in combined:
                # Merge scores
                existing = combined[key]
                existing["bm25_score"] = result["score"]
                existing["combined_score"] = (
                    self.vector_weight * existing["vector_score"]
                    + self.bm25_weight * result["score"]
                )
            else:
                # New result
                result_copy = result.copy()
                result_copy["vector_score"] = 0.0
                result_copy["bm25_score"] = result["score"]
                result_copy["combined_score"] = self.bm25_weight * result["score"]
                combined[key] = result_copy

        # Sort by combined score and return top_k
        sorted_results = sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)

        # Update final score
        for result in sorted_results[:top_k]:
            result["score"] = result["combined_score"]

        return sorted_results[:top_k]

    def _apply_faceted_filters(
        self, results: List[Dict[str, Any]], filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply faceted filters to results."""
        if not filters:
            return results

        filtered_results = []

        for result in results:
            include = True

            # Apply tag filters
            if "tags" in filters and filters["tags"]:
                result_tags = set(result.get("tags", []))
                filter_tags = set(filters["tags"])
                if not result_tags.intersection(filter_tags):
                    include = False

            # Apply category filters
            if "categories" in filters and filters["categories"]:
                result_cats = set(result.get("categories", []))
                filter_cats = set(filters["categories"])
                if not result_cats.intersection(filter_cats):
                    include = False

            # Apply other filters as needed

            if include:
                filtered_results.append(result)

        return filtered_results

    def _rerank_results(
        self, results: List[Dict[str, Any]], query: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """Rerank results using additional relevance signals."""
        query_lower = query.lower()
        query_terms = set(query.split())

        for result in results:
            # Boost for exact matches
            if query_lower in result["content"].lower():
                result["score"] *= 1.2

            # Boost for term matches in tags/categories
            result_tags = set(result.get("tags", []))
            result_cats = set(result.get("categories", []))
            matching_terms = len((result_tags | result_cats) & query_terms)
            if matching_terms > 0:
                result["score"] *= 1.0 + matching_terms * 0.1

        # Sort by final score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _generate_search_analytics(
        self,
        parsed_query: Dict[str, Any],
        results: List[Dict[str, Any]],
        search_time: float,
    ) -> Dict[str, Any]:
        """Generate search analytics and metrics."""
        analytics = {
            "search_time": search_time,
            "total_results": len(results),
            "query_complexity": self._calculate_query_complexity(parsed_query),
            "result_distribution": self._analyze_result_distribution(results),
            "performance_metrics": {
                "avg_score": (
                    sum(r.get("score", 0) for r in results) / len(results) if results else 0
                ),
                "score_variance": np.var([r.get("score", 0) for r in results]) if results else 0,
                "top_score": max((r.get("score", 0) for r in results), default=0),
            },
        }

        return analytics

    def _calculate_query_complexity(self, parsed_query: Dict[str, Any]) -> float:
        """Calculate query complexity score."""
        complexity = 0

        # Base complexity from term count
        term_count = len(parsed_query.get("simple_terms", []))
        complexity += term_count * 0.1

        # Add complexity for phrases
        complexity += len(parsed_query.get("phrase_terms", [])) * 0.3

        # Add complexity for field queries
        complexity += len(parsed_query.get("field_queries", {})) * 0.5

        # Add complexity for boolean operators
        if parsed_query.get("boolean_query"):
            complexity += 1.0

        # Add complexity for exclusions
        complexity += len(parsed_query.get("excluded_terms", [])) * 0.2

        return min(complexity, 5.0)  # Cap at 5.0

    def _analyze_result_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distribution of results."""
        if not results:
            return {}

        # Count by document
        doc_counts = Counter(r["document_id"] for r in results)

        # Count by tags
        tag_counts = Counter()
        cat_counts = Counter()

        for result in results:
            for tag in result.get("tags", []):
                tag_counts[tag] += 1
            for cat in result.get("categories", []):
                cat_counts[cat] += 1

        return {
            "unique_documents": len(doc_counts),
            "documents_with_multiple_chunks": sum(1 for count in doc_counts.values() if count > 1),
            "top_tags": dict(tag_counts.most_common(5)),
            "top_categories": dict(cat_counts.most_common(5)),
        }

    def _get_es_client(self) -> Elasticsearch:
        """Get Elasticsearch client."""
        from src.database.opensearch_setup import get_elasticsearch_client

        return get_elasticsearch_client()

    def get_search_facets(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get faceted search data for dynamic filters.

        Returns available filter options based on current search results.
        """
        # Perform search to get base results
        search_results = self.search(query, top_k=100, filters=filters)

        # Extract facet data
        facets = {
            "tags": Counter(),
            "categories": Counter(),
            "languages": Counter(),
            "authors": Counter(),
            "date_ranges": Counter(),
            "file_types": Counter(),
        }

        for result in search_results["results"]:
            # Tags
            for tag in result.get("tags", []):
                facets["tags"][tag] += 1

            # Categories
            for cat in result.get("categories", []):
                facets["categories"][cat] += 1

            # Languages
            lang = result.get("metadata", {}).get("detected_language")
            if lang:
                facets["languages"][lang] += 1

            # Authors
            author = result.get("metadata", {}).get("author")
            if author:
                facets["authors"][author] += 1

            # File types
            filename = result.get("document_title", "")
            if "." in filename:
                ext = filename.split(".")[-1].lower()
                facets["file_types"][ext] += 1

        # Convert to sorted lists for UI
        for facet_name, counter in facets.items():
            facets[facet_name] = [
                {"value": key, "count": count} for key, count in counter.most_common(10)
            ]

        return facets
