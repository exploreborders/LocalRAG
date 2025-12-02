#!/usr/bin/env python3
"""
Lightweight document management facade for LocalRAG.

Provides a unified interface to the extracted document management components.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.core.categorization.category_manager import CategoryManager
from src.core.processing.document_processor import DocumentProcessor
from src.core.tagging.tag_manager import TagManager
from src.database.models import (
    Document,
    DocumentCategory,
    DocumentCategoryAssignment,
    DocumentChapter,
    DocumentChunk,
    DocumentEmbedding,
    DocumentTag,
    DocumentTagAssignment,
    DocumentTopic,
    ProcessingJob,
    SessionLocal,
)

logger = logging.getLogger(__name__)


class DocumentManager:
    """
    Lightweight facade for document management operations.

    Provides a unified interface to the extracted document management components.
    """

    def __init__(self, db: Optional[Session] = None):
        self.db = db or SessionLocal()
        self.tag_manager = TagManager(self.db)
        self.category_manager = CategoryManager(self.db)
        self.document_processor = DocumentProcessor(self.db)

    # Delegate tag operations
    def get_tag_by_name(self, name: str) -> Optional[DocumentTag]:
        """Get a tag by name."""
        return self.tag_manager.get_tag_by_name(name)

    def create_tag(self, name: str, color: Optional[str] = None) -> DocumentTag:
        """Create a new tag with optional color."""
        return self.tag_manager.create_tag(name, color)

    def add_tag_to_document(self, document_id: int, tag_id: int) -> bool:
        """Add a tag to a document."""
        return self.tag_manager.add_tag_to_document(document_id, tag_id)

    def remove_tag_from_document(self, document_id: int, tag_id: int) -> bool:
        """Remove a tag from a document."""
        return self.tag_manager.remove_tag_from_document(document_id, tag_id)

    def get_document_tags(self, document_id: int) -> List[DocumentTag]:
        """Get all tags for a document."""
        return self.tag_manager.get_document_tags(document_id)

    def suggest_tags_for_document(self, document_id: int, max_suggestions: int = 5) -> List[str]:
        """Suggest tags for a document using AI."""
        return self.tag_manager.suggest_tags_for_document(document_id, max_suggestions)

    def delete_tag(self, tag_name: str) -> bool:
        """Delete a tag and all its assignments."""
        return self.tag_manager.delete_tag(tag_name)

    def get_all_tags(self) -> List[DocumentTag]:
        """Get all tags."""
        return self.tag_manager.get_all_tags()

    def get_popular_tags(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most popular tags by usage count."""
        return self.tag_manager.get_popular_tags(limit)

    # Delegate category operations
    def get_category_by_name(
        self, name: str, parent_id: Optional[int] = None
    ) -> Optional[DocumentCategory]:
        """Get a category by name and optional parent."""
        return self.category_manager.get_category_by_name(name, parent_id)

    def create_category(
        self,
        name: str,
        description: Optional[str] = None,
        parent_id: Optional[int] = None,
    ) -> DocumentCategory:
        """Create a new category."""
        return self.category_manager.create_category(name, description, parent_id)

    def add_category_to_document(self, document_id: int, category_id: int) -> bool:
        """Add a category to a document."""
        return self.category_manager.add_category_to_document(document_id, category_id)

    def remove_category_from_document(self, document_id: int, category_id: int) -> bool:
        """Remove a category from a document."""
        return self.category_manager.remove_category_from_document(document_id, category_id)

    def get_document_categories(self, document_id: int) -> List[DocumentCategory]:
        """Get all categories for a document."""
        return self.category_manager.get_document_categories(document_id)

    def get_category_hierarchy(self, category_id: int) -> List[DocumentCategory]:
        """Get the full hierarchy path for a category."""
        return self.category_manager.get_category_hierarchy(category_id)

    def get_root_categories(self) -> List[DocumentCategory]:
        """Get all root categories (categories with no parent)."""
        return self.category_manager.get_root_categories()

    def get_category_tree(self) -> List[Dict[str, Any]]:
        """Get the complete category tree with hierarchy."""
        return self.category_manager.get_category_tree()

    def delete_category(self, category_id: int) -> bool:
        """Delete a category and all its subcategories."""
        return self.category_manager.delete_category(category_id)

    def delete_document(self, document_id: int) -> bool:
        """
        Delete a document and all associated data including Elasticsearch indices.

        Args:
            document_id: ID of the document to delete

        Returns:
            bool: True if deletion was successful
        """
        try:
            # Get document info before deletion
            document = self.db.query(Document).filter(Document.id == document_id).first()
            if not document:
                return False

            file_path = document.filepath

            # Delete from Elasticsearch first
            try:
                from elasticsearch import Elasticsearch

                es = Elasticsearch(
                    hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}],
                    verify_certs=False,
                )
                if es.ping():
                    # Delete document metadata
                    es.delete(index="documents", id=str(document_id), ignore=[404])

                    # Delete all chunks for this document
                    es.delete_by_query(
                        index="chunks",
                        body={"query": {"term": {"document_id": document_id}}},
                        ignore=[404],
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to clean Elasticsearch indices for document {document_id}: {e}"
                )

            # Delete from database in correct order (reverse dependencies)
            # Delete embeddings (references chunks)
            self.db.query(DocumentEmbedding).filter(
                DocumentEmbedding.chunk_id.in_(
                    self.db.query(DocumentChunk.id).filter(DocumentChunk.document_id == document_id)
                )
            ).delete()

            # Delete chunks
            self.db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()

            # Delete chapters
            self.db.query(DocumentChapter).filter(
                DocumentChapter.document_id == document_id
            ).delete()

            # Delete topic relationships
            self.db.query(DocumentTopic).filter(DocumentTopic.document_id == document_id).delete()

            # Delete tag relationships
            self.db.query(DocumentTagAssignment).filter(
                DocumentTagAssignment.document_id == document_id
            ).delete()

            # Delete category relationships
            self.db.query(DocumentCategoryAssignment).filter(
                DocumentCategoryAssignment.document_id == document_id
            ).delete()

            # Delete processing jobs
            self.db.query(ProcessingJob).filter(ProcessingJob.document_id == document_id).delete()

            # Finally delete the document
            self.db.query(Document).filter(Document.id == document_id).delete()

            self.db.commit()

            # Remove physical file
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Could not delete physical file {file_path}: {e}")

            logger.info(f"Successfully deleted document {document_id}")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    def synchronize_indices(self) -> Dict[str, Any]:
        """
        Synchronize Elasticsearch indices with database state.

        Returns:
            Dict with synchronization results
        """
        results = {
            "documents_indexed": 0,
            "chunks_indexed": 0,
            "orphaned_documents_removed": 0,
            "orphaned_chunks_removed": 0,
            "errors": [],
        }

        try:
            from elasticsearch import Elasticsearch

            es = Elasticsearch(
                hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}],
                verify_certs=False,
            )

            if not es.ping():
                results["errors"].append("Elasticsearch not available")
                return results

            # Get all document IDs from database
            db_document_ids = {doc.id for doc in self.db.query(Document).all()}

            # Check documents index
            try:
                # Get all document IDs in documents index
                es_docs = es.search(
                    index="documents",
                    body={"query": {"match_all": {}}, "size": 10000, "_source": ["id"]},
                )
                es_document_ids = {int(hit["_source"]["id"]) for hit in es_docs["hits"]["hits"]}

                # Remove orphaned documents from index
                orphaned_docs = es_document_ids - db_document_ids
                for doc_id in orphaned_docs:
                    try:
                        es.delete(index="documents", id=str(doc_id), ignore=[404])
                        results["orphaned_documents_removed"] += 1
                    except Exception as e:
                        results["errors"].append(
                            f"Failed to remove orphaned document {doc_id}: {e}"
                        )

                # Re-index missing documents
                missing_docs = db_document_ids - es_document_ids
                for doc_id in missing_docs:
                    try:
                        doc = self.db.query(Document).filter(Document.id == doc_id).first()
                        if doc:
                            self._reindex_document(doc)
                            results["documents_indexed"] += 1
                    except Exception as e:
                        results["errors"].append(f"Failed to re-index document {doc_id}: {e}")

            except Exception as e:
                results["errors"].append(f"Error synchronizing documents index: {e}")

            # Check chunks index
            try:
                # Get all document IDs that have chunks in ES
                es_chunks = es.search(
                    index="chunks",
                    body={
                        "query": {"match_all": {}},
                        "size": 10000,
                        "_source": ["document_id"],
                        "collapse": {"field": "document_id"},
                    },
                )
                es_chunk_doc_ids = {
                    int(hit["_source"]["document_id"]) for hit in es_chunks["hits"]["hits"]
                }

                # Remove chunks for non-existent documents
                orphaned_chunk_docs = es_chunk_doc_ids - db_document_ids
                for doc_id in orphaned_chunk_docs:
                    try:
                        es.delete_by_query(
                            index="chunks",
                            body={"query": {"term": {"document_id": doc_id}}},
                            ignore=[404],
                        )
                        results["orphaned_chunks_removed"] += 1
                    except Exception as e:
                        results["errors"].append(
                            f"Failed to remove orphaned chunks for document {doc_id}: {e}"
                        )

                # Re-index chunks for documents that exist but may have missing chunks
                for doc_id in db_document_ids:
                    try:
                        # Check if document has chunks in database
                        chunk_count = (
                            self.db.query(DocumentChunk)
                            .filter(DocumentChunk.document_id == doc_id)
                            .count()
                        )
                        if chunk_count > 0:
                            # Check if chunks exist in ES
                            es_chunk_count = es.count(
                                index="chunks",
                                body={"query": {"term": {"document_id": doc_id}}},
                            )["count"]

                            if es_chunk_count == 0:
                                # Re-index chunks for this document
                                doc = self.db.query(Document).filter(Document.id == doc_id).first()
                                if doc:
                                    self._reindex_document_chunks(doc)
                                    results["chunks_indexed"] += chunk_count
                    except Exception as e:
                        results["errors"].append(
                            f"Error checking chunks for document {doc_id}: {e}"
                        )

            except Exception as e:
                results["errors"].append(f"Error synchronizing chunks index: {e}")

        except Exception as e:
            results["errors"].append(f"General synchronization error: {e}")

        return results

    def _reindex_document(self, document: Document) -> None:
        """Re-index a single document in Elasticsearch."""
        try:
            from elasticsearch import Elasticsearch

            es = Elasticsearch(
                hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}],
                verify_certs=False,
            )

            doc_data = {
                "id": document.id,
                "filename": document.filename,
                "filepath": document.filepath,
                "file_hash": document.file_hash,
                "detected_language": document.detected_language,
                "document_summary": document.document_summary,
                "key_topics": document.key_topics,
                "reading_time_minutes": document.reading_time_minutes,
                "status": document.status,
                "created_at": (document.upload_date.isoformat() if document.upload_date else None),
            }

            es.index(index="documents", id=str(document.id), body=doc_data)

        except Exception as e:
            logger.error(f"Failed to re-index document {document.id}: {e}")

    def _reindex_document_chunks(self, document: Document) -> None:
        """Re-index all chunks for a document in Elasticsearch."""
        try:
            from elasticsearch import Elasticsearch

            es = Elasticsearch(
                hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}],
                verify_certs=False,
            )

            # Get all chunks for this document
            chunks = (
                self.db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).all()
            )

            for chunk in chunks:
                # Get embedding if it exists
                embedding = (
                    self.db.query(DocumentEmbedding)
                    .filter(DocumentEmbedding.chunk_id == chunk.id)
                    .first()
                )

                chunk_data = {
                    "document_id": document.id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "chapter_title": chunk.chapter_title,
                    "chapter_path": chunk.chapter_path,
                    "embedding": embedding.embedding.tolist() if embedding else [],
                }

                chunk_id = f"{document.id}_{chunk.chunk_index}"
                es.index(index="chunks", id=chunk_id, body=chunk_data)

        except Exception as e:
            logger.error(f"Failed to re-index chunks for document {document.id}: {e}")

    # Delegate document processing operations
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document and return processing results."""
        return self.document_processor.process_document(file_path)
