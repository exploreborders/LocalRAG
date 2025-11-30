"""
Knowledge Graph System for Tag Relationships and Category Integration.

This module provides a comprehensive knowledge graph that connects tags, categories,
and documents through various relationship types, enabling rich contextual retrieval
and enhanced LLM understanding.
"""

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from src.database.models import (
    Document,
    DocumentCategory,
    DocumentCategoryAssignment,
    DocumentTag,
    DocumentTagAssignment,
    SessionLocal,
)

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Knowledge graph managing relationships between tags, categories, and documents.

    Provides relationship inference, graph traversal, and contextual expansion
    for enhanced retrieval and RAG operations.
    """

    def __init__(self, db: Session):
        self.db = db
        self._relationship_cache = {}
        self._graph_cache = {}

    def build_relationships_from_cooccurrence(
        self, min_occurrences: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build tag relationships based on co-occurrence analysis.

        Args:
            min_occurrences: Minimum times tags must co-occur to create relationship

        Returns:
            Dict mapping tag names to their relationships
        """
        logger.info("Building tag relationships from co-occurrence analysis")

        # Get all tag assignments with document counts
        tag_assignments = (
            self.db.query(DocumentTagAssignment.document_id, DocumentTag.name.label("tag_name"))
            .join(DocumentTag)
            .all()
        )

        # Group by document
        doc_tags = defaultdict(set)
        for doc_id, tag_name in tag_assignments:
            doc_tags[doc_id].add(tag_name)

        # Count co-occurrences (both inter-document and intra-document)
        co_occurrences = defaultdict(lambda: defaultdict(float))
        for tags in doc_tags.values():
            tag_list = list(tags)
            # Intra-document relationships (tags that appear together in same document)
            for i, tag1 in enumerate(tag_list):
                for tag2 in tag_list[i + 1 :]:
                    # Give higher weight to intra-document relationships
                    co_occurrences[tag1][tag2] += 2  # Weight: 2 for same document
                    co_occurrences[tag2][tag1] += 2

        # Also add inter-document relationships (tags that appear in multiple documents)
        # This is less common but still valuable
        tag_document_count = defaultdict(int)
        for tags in doc_tags.values():
            for tag in tags:
                tag_document_count[tag] += 1

        # For tags that appear in multiple documents, create weaker relationships
        for tag1 in tag_document_count:
            for tag2 in tag_document_count:
                if tag1 != tag2 and tag_document_count[tag1] > 1 and tag_document_count[tag2] > 1:
                    # Very weak relationship for tags that just happen to be in multiple docs
                    co_occurrences[tag1][tag2] += 0.1
                    co_occurrences[tag2][tag1] += 0.1

        # Create relationships
        relationships = defaultdict(list)
        for tag1, related in co_occurrences.items():
            for tag2, count in related.items():
                # For intra-document relationships (weight >= 2), require lower threshold
                # For inter-document relationships (weight < 2), require higher threshold
                min_required = 1.5 if count >= 2 else min_occurrences
                if count >= min_required:
                    strength = min(1.0, count / 10.0)  # Normalize strength
                    relationships[tag1].append(
                        {
                            "related_tag": tag2,
                            "type": "co_occurs_with",
                            "strength": strength,
                            "evidence_count": count,
                        }
                    )

        # Merge with semantic relationships
        semantic_relationships = self.build_semantic_relationships()
        for tag, rels in semantic_relationships.items():
            if tag not in relationships:
                relationships[tag] = []
            relationships[tag].extend(rels)

        self._relationship_cache = dict(relationships)
        return dict(relationships)

    def infer_tag_category_relationships(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Infer relationships between tags and categories based on document assignments.

        Returns:
            Dict mapping tags to their category relationships
        """
        logger.info("Inferring tag-category relationships")

        # Get documents with both tags and categories
        # Use table aliases to avoid SQLAlchemy 2.0 join ambiguity
        from sqlalchemy.orm import aliased

        doc_alias = aliased(Document)
        tag_assign_alias = aliased(DocumentTagAssignment)
        tag_alias = aliased(DocumentTag)
        cat_assign_alias = aliased(DocumentCategoryAssignment)
        cat_alias = aliased(DocumentCategory)

        doc_data = (
            self.db.query(
                doc_alias.id,
                tag_alias.name.label("tag_name"),
                cat_alias.name.label("category_name"),
            )
            .select_from(doc_alias)
            .join(tag_assign_alias, doc_alias.id == tag_assign_alias.document_id)
            .join(tag_alias, tag_assign_alias.tag_id == tag_alias.id)
            .join(cat_assign_alias, doc_alias.id == cat_assign_alias.document_id)
            .join(cat_alias, cat_assign_alias.category_id == cat_alias.id)
            .all()
        )

        # Group by tag
        tag_categories = defaultdict(lambda: defaultdict(int))
        for doc_id, tag_name, cat_name in doc_data:
            tag_categories[tag_name][cat_name] += 1

        # Create relationships
        relationships = defaultdict(list)
        for tag_name, categories in tag_categories.items():
            total_docs = sum(categories.values())
            for cat_name, count in categories.items():
                strength = count / total_docs
                relationships[tag_name].append(
                    {
                        "category": cat_name,
                        "type": "belongs_to",
                        "strength": strength,
                        "evidence_count": count,
                    }
                )

        return dict(relationships)

    def build_semantic_relationships(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build semantic relationships between tags based on shared categories
        and name similarity.

        Returns:
            Dict mapping tags to their semantic relationships
        """
        logger.info("Building semantic relationships between tags")

        relationships = defaultdict(list)

        # Get all tags with their categories
        from sqlalchemy.orm import aliased

        tag_alias = aliased(DocumentTag)
        cat_assign_alias = aliased(DocumentCategoryAssignment)
        cat_alias = aliased(DocumentCategory)
        doc_assign_alias = aliased(DocumentTagAssignment)

        tag_data = (
            self.db.query(tag_alias.name.label("tag_name"), cat_alias.name.label("category_name"))
            .select_from(tag_alias)
            .join(doc_assign_alias, tag_alias.id == doc_assign_alias.tag_id)
            .join(
                cat_assign_alias,
                doc_assign_alias.document_id == cat_assign_alias.document_id,
            )
            .join(cat_alias, cat_assign_alias.category_id == cat_alias.id)
            .distinct()
            .all()
        )

        # Group categories by tag
        tag_categories = defaultdict(set)
        for tag_name, cat_name in tag_data:
            tag_categories[tag_name].add(cat_name)

        # Get all tag names
        all_tags = list(tag_categories.keys())

        # Build relationships based on shared categories (only specific categories)
        specific_categories = {
            "Machine Learning",
            "Deep Learning",
            "Database",
            "Project Management",
            "AI",
            "Computer Vision",
        }
        for i, tag1 in enumerate(all_tags):
            for tag2 in all_tags[i + 1 :]:
                shared_categories = tag_categories[tag1] & tag_categories[tag2]
                # Only create relationships for specific, meaningful categories
                meaningful_shared = shared_categories & specific_categories
                if meaningful_shared and len(meaningful_shared) >= 1:
                    # Calculate relationship strength based on shared meaningful categories
                    strength = min(
                        1.0, len(meaningful_shared) * 0.5
                    )  # 0.5 per meaningful shared category
                    relationships[tag1].append(
                        {
                            "related_tag": tag2,
                            "type": "shares_meaningful_categories",
                            "strength": strength,
                            "shared_categories": list(meaningful_shared),
                            "evidence_count": len(meaningful_shared),
                        }
                    )
                    relationships[tag2].append(
                        {
                            "related_tag": tag1,
                            "type": "shares_meaningful_categories",
                            "strength": strength,
                            "shared_categories": list(meaningful_shared),
                            "evidence_count": len(meaningful_shared),
                        }
                    )

        # Add name-based similarity relationships
        for i, tag1 in enumerate(all_tags):
            tag1_lower = tag1.lower()
            for tag2 in all_tags[i + 1 :]:
                tag2_lower = tag2.lower()

                # Check for substring relationships
                if tag1_lower in tag2_lower or tag2_lower in tag1_lower:
                    if len(tag1) > 3 and len(tag2) > 3:  # Avoid very short tags
                        strength = 0.2  # Lower strength for name similarity
                        relationships[tag1].append(
                            {
                                "related_tag": tag2,
                                "type": "name_similarity",
                                "strength": strength,
                                "evidence_count": 1,
                            }
                        )
                        relationships[tag2].append(
                            {
                                "related_tag": tag1,
                                "type": "name_similarity",
                                "strength": strength,
                                "evidence_count": 1,
                            }
                        )

        return dict(relationships)

    def get_related_tags(
        self, tag_names: List[str], depth: int = 2, min_strength: float = 0.3
    ) -> Set[str]:
        """
        Get related tags through graph traversal.

        Args:
            tag_names: Starting tag names
            depth: How many relationship levels to traverse
            min_strength: Minimum relationship strength to include

        Returns:
            Set of related tag names
        """
        if not self._relationship_cache:
            self.build_relationships_from_cooccurrence()

        visited = set(tag_names)
        queue = deque([(tag, 0) for tag in tag_names])

        while queue:
            current_tag, current_depth = queue.popleft()

            if current_depth >= depth:
                continue

            # Get relationships for current tag
            relationships = self._relationship_cache.get(current_tag, [])

            for rel in relationships:
                if rel["strength"] >= min_strength:
                    related_tag = rel["related_tag"]
                    if related_tag not in visited:
                        visited.add(related_tag)
                        queue.append((related_tag, current_depth + 1))

        # Remove original tags from results
        return visited - set(tag_names)

    def get_related_categories(self, category_names: List[str], depth: int = 1) -> Set[str]:
        """
        Get related categories through hierarchical relationships.

        Args:
            category_names: Starting category names
            depth: How many hierarchy levels to traverse

        Returns:
            Set of related category names
        """
        visited = set(category_names)
        queue = deque([(cat, 0) for cat in category_names])

        while queue:
            current_cat, current_depth = queue.popleft()

            if current_depth >= depth:
                continue

            # Get parent categories
            parent = (
                self.db.query(DocumentCategory).filter(DocumentCategory.name == current_cat).first()
            )

            if parent and parent.parent_category_id:
                parent_cat = (
                    self.db.query(DocumentCategory)
                    .filter(DocumentCategory.id == parent.parent_category_id)
                    .first()
                )

                if parent_cat and parent_cat.name not in visited:
                    visited.add(parent_cat.name)
                    queue.append((parent_cat.name, current_depth + 1))

            # Get child categories
            if parent:
                children = (
                    self.db.query(DocumentCategory)
                    .filter(DocumentCategory.parent_category_id == parent.id)
                    .all()
                )
            else:
                children = []

            for child in children:
                if child.name not in visited:
                    visited.add(child.name)
                    queue.append((child.name, current_depth + 1))

        return visited - set(category_names)

    def expand_query_context(
        self, tag_names: List[str], category_names: List[str], context_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Expand query context using knowledge graph relationships.

        Args:
            tag_names: Tags from the query
            category_names: Categories from the query
            context_depth: How far to expand relationships

        Returns:
            Dict with expanded context information
        """
        expanded_tags = self.get_related_tags(tag_names, depth=context_depth)
        expanded_categories = self.get_related_categories(category_names, depth=context_depth)

        # Get tag-category relationships for additional context
        tag_category_rels = self.infer_tag_category_relationships()

        # Find categories related to query tags
        tag_related_categories = set()
        for tag in tag_names:
            if tag in tag_category_rels:
                for rel in tag_category_rels[tag]:
                    if rel["strength"] > 0.5:  # High confidence relationships
                        tag_related_categories.add(rel["category"])

        return {
            "expanded_tags": list(expanded_tags),
            "expanded_categories": list(expanded_categories),
            "tag_related_categories": list(tag_related_categories),
            "context_depth": context_depth,
            "total_expansions": len(expanded_tags)
            + len(expanded_categories)
            + len(tag_related_categories),
        }

    def find_documents_by_relationships(
        self,
        tag_names: List[str],
        category_names: List[str],
        include_related: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Find documents using tag and category relationships.

        Args:
            tag_names: Primary tags to search for
            category_names: Primary categories to search for
            include_related: Whether to include related tags/categories

        Returns:
            List of documents with relationship context
        """
        # Build query with explicit joins
        if tag_names and category_names:
            # Both tags and categories - need explicit joins
            from sqlalchemy import select

            tag_ids = select(DocumentTag.id).where(DocumentTag.name.in_(tag_names))
            cat_ids = select(DocumentCategory.id).where(DocumentCategory.name.in_(category_names))

            base_query = (
                self.db.query(Document)
                .select_from(Document)
                .join(
                    DocumentTagAssignment,
                    Document.id == DocumentTagAssignment.document_id,
                )
                .join(
                    DocumentCategoryAssignment,
                    Document.id == DocumentCategoryAssignment.document_id,
                )
                .filter(DocumentTagAssignment.tag_id.in_(tag_ids))
                .filter(DocumentCategoryAssignment.category_id.in_(cat_ids))
                .distinct()
            )

        elif tag_names:
            # Tags only
            from sqlalchemy import select

            tag_ids = select(DocumentTag.id).where(DocumentTag.name.in_(tag_names))
            base_query = (
                self.db.query(Document)
                .select_from(Document)
                .join(
                    DocumentTagAssignment,
                    Document.id == DocumentTagAssignment.document_id,
                )
                .filter(DocumentTagAssignment.tag_id.in_(tag_ids))
                .distinct()
            )

        elif category_names:
            # Categories only
            from sqlalchemy import select

            cat_ids = select(DocumentCategory.id).where(DocumentCategory.name.in_(category_names))
            base_query = (
                self.db.query(Document)
                .select_from(Document)
                .join(
                    DocumentCategoryAssignment,
                    Document.id == DocumentCategoryAssignment.document_id,
                )
                .filter(DocumentCategoryAssignment.category_id.in_(cat_ids))
                .distinct()
            )

        else:
            # No filtering
            base_query = self.db.query(Document).distinct()

        direct_docs = base_query.all()

        results = []
        for doc in direct_docs:
            # Get actual tag and category names through relationships
            doc_tags = [assignment.tag.name for assignment in doc.tags if assignment.tag]
            doc_categories = [
                assignment.category.name for assignment in doc.categories if assignment.category
            ]

            results.append(
                {
                    "document": doc,
                    "tags": doc_tags,
                    "categories": doc_categories,
                    "match_type": "direct",
                    "relevance_score": 1.0,
                }
            )

        # Add related documents if requested
        if include_related:
            related_tags = self.get_related_tags(tag_names) if tag_names else set()
            related_categories = (
                self.get_related_categories(category_names) if category_names else set()
            )

            if related_tags or related_categories:
                related_query = self.db.query(Document).distinct()

                if related_tags:
                    from sqlalchemy import select

                    related_tag_ids = (
                        select(DocumentTag.id)
                        .where(DocumentTag.name.in_(related_tags))
                        .scalar_subquery()
                    )
                    related_query = related_query.join(DocumentTagAssignment).filter(
                        DocumentTagAssignment.tag_id.in_(related_tag_ids)
                    )

                if related_categories:
                    from sqlalchemy import select

                    related_cat_ids = (
                        select(DocumentCategory.id)
                        .where(DocumentCategory.name.in_(related_categories))
                        .scalar_subquery()
                    )
                    related_query = related_query.join(DocumentCategoryAssignment).filter(
                        DocumentCategoryAssignment.category_id.in_(related_cat_ids)
                    )

                # Exclude already found documents
                direct_doc_ids = {doc.id for doc in direct_docs}
                related_docs = related_query.filter(Document.id.notin_(direct_doc_ids)).all()

                for doc in related_docs:
                    doc_tags = [tag.name for tag in doc.tags]
                    doc_categories = [cat.name for cat in doc.categories]

                    # Calculate relevance score based on relationship strength
                    relevance = 0.7  # Base score for related documents

                    results.append(
                        {
                            "document": doc,
                            "tags": doc_tags,
                            "categories": doc_categories,
                            "match_type": "related",
                            "relevance_score": relevance,
                        }
                    )

        return results

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dict with graph statistics
        """
        # Tag statistics
        tag_count = self.db.query(func.count(DocumentTag.id)).scalar()
        tag_relationships = len(self._relationship_cache) if self._relationship_cache else 0

        # Category statistics
        category_count = self.db.query(func.count(DocumentCategory.id)).scalar()

        # Document relationships
        doc_tag_assignments = self.db.query(func.count(DocumentTagAssignment.id)).scalar()
        doc_cat_assignments = self.db.query(func.count(DocumentCategoryAssignment.id)).scalar()

        return {
            "tags": {"total": tag_count, "with_relationships": tag_relationships},
            "categories": {"total": category_count},
            "assignments": {
                "tag_assignments": doc_tag_assignments,
                "category_assignments": doc_cat_assignments,
            },
            "relationships": {"tag_relationships_cached": len(self._relationship_cache)},
        }

    def clear_cache(self):
        """Clear relationship and graph caches."""
        self._relationship_cache.clear()
        self._graph_cache.clear()
        logger.info("Knowledge graph cache cleared")
