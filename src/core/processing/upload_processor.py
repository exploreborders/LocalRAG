"""
Document upload processor for LocalRAG.

Handles batch processing, reprocessing, and file upload operations.
"""

import hashlib
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from src.core.base_processor import BaseProcessor
from src.core.embeddings import create_embeddings
from src.core.processing.document_processor import DocumentProcessor
from src.database.models import (
    Document,
    DocumentCategoryAssignment,
    DocumentChapter,
    DocumentChunk,
    DocumentEmbedding,
    DocumentTagAssignment,
    SessionLocal,
)
from src.utils.config_manager import config
from src.utils.content_validator import ContentValidator

logger = logging.getLogger(__name__)


class UploadProcessor(BaseProcessor):
    """
    Enhanced document upload processor with batch processing capabilities.

    Handles parallel file processing, reprocessing of existing documents,
    and comprehensive upload workflows.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        db: Optional[Session] = None,
        embedding_model: str = "embeddinggemma:latest",
    ):
        super().__init__(db or SessionLocal())
        self.max_workers = max_workers or min(mp.cpu_count(), 4)
        self.embedding_model = embedding_model

    def __del__(self):
        """Clean up database connections."""
        if hasattr(self, "db"):
            self.db.close()

    def process_files(
        self,
        file_paths: List[str],
        progress_callback: Optional[Callable] = None,
        use_parallel: bool = True,
    ) -> Dict[str, Any]:
        """
        Process multiple files, optionally in parallel.

        Args:
            file_paths: List of file paths to process
            progress_callback: Optional callback for overall progress
            use_parallel: Whether to use parallel processing

        Returns:
            Dict with batch processing results
        """
        results = []
        total_files = len(file_paths)

        if use_parallel and len(file_paths) > 1:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self._process_single_file, file_path): file_path
                    for file_path in file_paths
                }

                # Collect results as they complete
                for i, future in enumerate(as_completed(future_to_file)):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)

                        if progress_callback:
                            progress = (i + 1) / total_files * 100
                            progress_callback(progress, f"Processed {i + 1}/{total_files} files")

                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")
                        results.append({"file_path": file_path, "success": False, "error": str(e)})
        else:
            # Sequential processing
            for i, file_path in enumerate(file_paths):
                try:
                    result = self._process_single_file(file_path)
                    results.append(result)

                    if progress_callback:
                        progress = (i + 1) / total_files * 100
                        progress_callback(progress, f"Processed {i + 1}/{total_files} files")

                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    results.append({"file_path": file_path, "success": False, "error": str(e)})

        # Summarize results
        successful = sum(1 for r in results if r.get("success", False))
        failed = total_files - successful

        # Calculate totals for compatibility with web interface
        total_chunks = sum(r.get("chunks_created", 0) for r in results if r.get("success", False))
        total_chapters = sum(
            r.get("chapters_created", 0) for r in results if r.get("success", False)
        )

        # Extract errors for compatibility
        errors = [r.get("error", "Unknown error") for r in results if not r.get("success", False)]

        return {
            "total_files": total_files,
            "successful": successful,
            "successful_uploads": successful,  # For backward compatibility
            "failed": failed,
            "failed_uploads": failed,  # For backward compatibility
            "total_chunks": total_chunks,
            "total_chapters": total_chapters,
            "results": results,
            "file_results": results,  # For backward compatibility
            "errors": errors,  # For backward compatibility
        }

    def process_single_file(
        self,
        file_path: str,
        filename: Optional[str] = None,
        file_hash: Optional[str] = None,
        force_enrichment: bool = False,
        use_advanced_processing: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Process a single file with optional reprocessing capabilities.

        Args:
            file_path: Path to the file to process
            filename: Optional filename (used for database lookup)
            file_hash: Optional file hash (used for database lookup)
            force_enrichment: Whether to force AI enrichment even if document exists
            use_advanced_processing: Whether to use advanced AI processing

        Returns:
            Dict with processing results
        """
        try:
            # Calculate file hash if not provided
            if not file_hash:
                with open(file_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()

            # Use filename from path if not provided
            if not filename:
                filename = Path(file_path).name

            # Check if document already exists in database
            existing_doc = None
            if filename and file_hash:
                existing_doc = (
                    self.db.query(Document)
                    .filter(Document.filename == filename, Document.file_hash == file_hash)
                    .first()
                )

            if existing_doc and not force_enrichment:
                # Document exists and we're not forcing reprocessing
                return {
                    "success": True,
                    "filename": filename,
                    "message": "Document already exists",
                    "document_id": existing_doc.id,
                    "chunks_created": 0,
                    "chapters_created": 0,
                }

            # Handle reprocessing of existing documents differently
            if existing_doc and force_enrichment:
                # For reprocessing, run advanced processing directly
                from src.data.loader import AdvancedDocumentProcessor

                advanced_processor = AdvancedDocumentProcessor()
                processing_result = advanced_processor.process_document_comprehensive(file_path)

                # Check if processing was successful
                if processing_result is None:
                    logger.error(
                        f"Document processing failed for {file_path} - received None result"
                    )
                    return {
                        "success": False,
                        "filename": filename,
                        "error": "Document processing returned no results",
                        "document_id": existing_doc.id,
                    }

                # Update the existing document with fresh processing results
                result = self.reprocess_existing_document(
                    existing_doc, processing_result, file_path
                )
                return result

            # For new documents, proceed with normal processing
            # Determine if we should use advanced processing
            if use_advanced_processing is None:
                # Auto-detect scanned PDFs and enable advanced processing for them
                if file_path.lower().endswith(".pdf"):
                    from src.data.loader import is_scanned_pdf

                    is_scanned = is_scanned_pdf(file_path)
                    use_advanced_processing = is_scanned
                    logger.info(
                        f"Auto-detected scanned PDF: {is_scanned}, using advanced processing: {use_advanced_processing}"
                    )
                else:
                    use_advanced_processing = False

            # Process the document
            processor = DocumentProcessor()
            result = processor.process_document(
                file_path, filename, use_advanced_processing=use_advanced_processing
            )

            if result.get("success"):
                # DocumentProcessor already stored the document, just format the result
                result["chunks_created"] = result.get("chunks_count", 0)
                result["chapters_created"] = 0  # DocumentProcessor doesn't create chapters yet
                result["filename"] = filename

            return result

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return {
                "success": False,
                "filename": filename,
                "error": str(e),
                "chunks_created": 0,
                "chapters_created": 0,
            }

    def reprocess_existing_document(
        self, existing_doc: Document, processing_result: Dict[str, Any], file_path: str
    ) -> Dict[str, Any]:
        """
        Reprocess an existing document with new processing results.

        Args:
            existing_doc: The existing document record
            processing_result: Results from document processing
            file_path: Path to the document file

        Returns:
            Dict with reprocessing results
        """
        # Validate processing_result
        if processing_result is None:
            logger.error(f"Cannot reprocess document {existing_doc.id} - processing_result is None")
            return {
                "success": False,
                "filename": existing_doc.filename,
                "error": "Processing result is None",
                "document_id": existing_doc.id,
                "chunks_created": 0,
                "chapters_created": 0,
            }

        if not isinstance(processing_result, dict):
            logger.error(
                f"Cannot reprocess document {existing_doc.id} - processing_result is not a dict: {type(processing_result)}"
            )
            return {
                "success": False,
                "filename": existing_doc.filename,
                "error": f"Invalid processing result type: {type(processing_result)}",
                "document_id": existing_doc.id,
                "chunks_created": 0,
                "chapters_created": 0,
            }

        try:
            # Validate content quality before processing
            extracted_content = processing_result.get("extracted_content", "")
            chunks = processing_result.get("chunks", [])

            validation_result = ContentValidator.validate_content_quality(extracted_content, chunks)

            # Log validation results and handle quality issues
            if not validation_result["is_valid"]:
                logger.warning(
                    f"Content quality issues detected for document {existing_doc.id}: "
                    f"{validation_result['issues']}"
                )
                logger.info(f"Quality score: {validation_result['quality_score']:.2f}")

                # Add quality issues to processing result for user feedback
                processing_result["quality_issues"] = validation_result["issues"]
                processing_result["quality_score"] = validation_result["quality_score"]
                processing_result["quality_recommendations"] = validation_result["recommendations"]

            # Update document metadata
            existing_doc.last_modified = datetime.now()

            # Set status based on content quality
            if validation_result["is_valid"]:
                existing_doc.status = "processed"
            else:
                existing_doc.status = "needs_review"  # New status for quality issues

            # Regenerate AI content for reprocessing
            extracted_content = processing_result.get("extracted_content", "")
            if extracted_content and isinstance(extracted_content, str):
                # Use DocumentProcessor for AI content regeneration
                doc_processor = DocumentProcessor()

                # Generate fresh AI content
                content_for_analysis = extracted_content[:2000]

                # Generate new tags
                suggested_tags_data = doc_processor.tag_suggester.suggest_tags(
                    content_for_analysis, existing_doc.filename
                )
                suggested_tags = [
                    tag.get("tag", "") for tag in suggested_tags_data if tag.get("tag")
                ]

                # Generate new categories
                suggested_categories = doc_processor._suggest_categories_ai(
                    content_for_analysis, existing_doc.filename, suggested_tags
                )

                # Generate new summary
                document_summary = doc_processor._generate_document_summary(
                    content_for_analysis,
                    existing_doc.filename,
                    suggested_tags,
                    processing_result.get("chapters_detected", 0),
                )

                # Update AI-enriched fields
                existing_doc.document_summary = document_summary
                existing_doc.key_topics = suggested_tags[:5]  # Store as key topics

                # Clear existing tags and categories
                self.db.query(DocumentTagAssignment).filter(
                    DocumentTagAssignment.document_id == existing_doc.id
                ).delete()
                self.db.query(DocumentCategoryAssignment).filter(
                    DocumentCategoryAssignment.document_id == existing_doc.id
                ).delete()

                # Add new tags
                for tag_name in suggested_tags:
                    if tag_name:
                        tag = doc_processor.tag_manager.get_tag_by_name(tag_name)
                        if not tag:
                            tag = doc_processor.tag_manager.create_tag(tag_name)
                        doc_processor.tag_manager.add_tag_to_document(existing_doc.id, tag.id)

                # Add new categories
                for category_name in suggested_categories:
                    if category_name:
                        category = doc_processor.category_manager.get_category_by_name(
                            category_name
                        )
                        if not category:
                            category = doc_processor.category_manager.create_category(category_name)
                        doc_processor.category_manager.add_category_to_document(
                            existing_doc.id, category.id
                        )

            # Update other fields if available
            if "reading_time_minutes" in processing_result:
                existing_doc.reading_time_minutes = processing_result["reading_time_minutes"]
            if "author" in processing_result:
                existing_doc.author = processing_result["author"]
            if "publication_date" in processing_result:
                existing_doc.publication_date = processing_result["publication_date"]
            if "detected_language" in processing_result:
                existing_doc.detected_language = processing_result["detected_language"]

            # Clear existing chunks, chapters, and embeddings for this document
            # First, get existing chunk IDs to clear embeddings
            existing_chunk_ids = [
                chunk.id
                for chunk in self.db.query(DocumentChunk)
                .filter(DocumentChunk.document_id == existing_doc.id)
                .all()
            ]

            # Clear embeddings first (before deleting chunks due to foreign key constraints)
            if existing_chunk_ids:
                self.db.query(DocumentEmbedding).filter(
                    DocumentEmbedding.chunk_id.in_(existing_chunk_ids)
                ).delete()

            # Clear existing chunks and chapters
            self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == existing_doc.id
            ).delete()
            self.db.query(DocumentChapter).filter(
                DocumentChapter.document_id == existing_doc.id
            ).delete()

            # Add new chunks and chapters
            chunks_created = 0
            chapters_created = 0

            if "chunks" in processing_result and processing_result["chunks"] is not None:
                chunks_list = processing_result["chunks"]
                if isinstance(chunks_list, list):
                    for i, chunk_data in enumerate(chunks_list):
                        if isinstance(chunk_data, dict) and "content" in chunk_data:
                            # Ensure content is a string
                            content = chunk_data["content"]
                            if isinstance(content, dict):
                                logger.warning(
                                    f"Chunk content is a dict, converting to string: {content}"
                                )
                                content = str(content)
                            elif not isinstance(content, str):
                                content = str(content)

                            chunk = DocumentChunk(
                                document_id=existing_doc.id,
                                content=content,
                                chunk_index=i,  # Use sequential index starting from 0
                                embedding_model="nomic-ai/nomic-embed-text-v1.5",
                                chapter_path=chunk_data.get("chapter_path"),
                                chapter_title=chunk_data.get("chapter_title"),
                            )
                            self.db.add(chunk)
                            chunks_created += 1

            if "chapters" in processing_result and processing_result["chapters"] is not None:
                chapters_list = processing_result["chapters"]
                if isinstance(chapters_list, list):
                    for i, chapter_data in enumerate(chapters_list):
                        if isinstance(chapter_data, dict) and "title" in chapter_data:
                            chapter = DocumentChapter(
                                document_id=existing_doc.id,
                                chapter_title=chapter_data["title"],
                                chapter_path=chapter_data.get("path", f"chapter_{i + 1}"),
                                level=chapter_data.get("level", 1),
                                word_count=chapter_data.get(
                                    "word_count", len(chapter_data["title"].split())
                                ),
                                content=chapter_data.get(
                                    "content",
                                    chapter_data.get("content_preview", chapter_data["title"]),
                                ),
                            )
                            self.db.add(chapter)
                            chapters_created += 1

            # Commit chunks and chapters before creating embeddings
            self.db.commit()

            # Create embeddings for new chunks in batches to avoid memory issues
            if chunks_created > 0:
                # Get the newly created chunks
                chunks = (
                    self.db.query(DocumentChunk)
                    .filter(DocumentChunk.document_id == existing_doc.id)
                    .order_by(DocumentChunk.chunk_index)
                    .all()
                )

                # Create embeddings in batches to avoid memory issues
                embedding_batch_size = 100  # Process 100 chunks at a time
                total_chunks = len(chunks)

                logger.info(
                    f"Creating embeddings for {total_chunks} chunks in batches of {embedding_batch_size}"
                )

                for batch_start in range(0, total_chunks, embedding_batch_size):
                    batch_end = min(batch_start + embedding_batch_size, total_chunks)
                    batch_chunks = chunks[batch_start:batch_end]
                    batch_texts = [chunk.content for chunk in batch_chunks]

                    logger.debug(
                        f"Processing embedding batch {batch_start // embedding_batch_size + 1}/{(total_chunks + embedding_batch_size - 1) // embedding_batch_size}"
                    )

                    try:
                        # Create embeddings for this batch
                        embeddings_array, _ = create_embeddings(
                            batch_texts,
                            model_name=config.models.embedding_model,
                            backend=config.models.embedding_backend,
                        )

                        # Store embeddings if creation was successful
                        if embeddings_array is not None and len(embeddings_array) > 0:
                            for chunk, embedding in zip(batch_chunks, embeddings_array):
                                # Check if embedding is valid (not None and not all zeros)
                                if (
                                    embedding is not None
                                    and hasattr(embedding, "shape")
                                    and embedding.shape[0] > 0
                                ):
                                    try:
                                        doc_embedding = DocumentEmbedding(
                                            chunk_id=chunk.id,
                                            embedding=embedding.tolist(),
                                            embedding_model="nomic-ai/nomic-embed-text-v1.5",
                                        )
                                        self.db.add(doc_embedding)
                                    except Exception as embed_error:
                                        logger.warning(
                                            f"Failed to store embedding for chunk {chunk.id}: {embed_error}"
                                        )
                                        continue
                        else:
                            logger.warning(
                                f"Failed to create embeddings for batch {batch_start // embedding_batch_size + 1}"
                            )

                    except Exception as e:
                        logger.error(
                            f"Embedding creation failed for batch {batch_start // embedding_batch_size + 1}: {e}"
                        )
                        # Continue with next batch

                logger.info(f"Completed embedding creation for document {existing_doc.id}")

                # Commit embeddings to database before indexing
                self.db.commit()

                # Update search index with enhanced metadata
                # Get all embeddings for this document
                all_embeddings = []
                for chunk in chunks:
                    embedding_record = (
                        self.db.query(DocumentEmbedding)
                        .filter(DocumentEmbedding.chunk_id == chunk.id)
                        .first()
                    )
                    if embedding_record:
                        all_embeddings.append(embedding_record.embedding)

                if all_embeddings and len(all_embeddings) == len(chunks):
                    logger.info(f"Indexing {len(all_embeddings)} chunks in Elasticsearch")
                    try:
                        processor = DocumentProcessor()
                        processor._index_document(
                            existing_doc,
                            [
                                {
                                    "content": chunk.content,
                                    "metadata": {"word_count": len(chunk.content.split())},
                                }
                                for chunk in chunks
                            ],
                            all_embeddings,
                        )
                        logger.info(f"Successfully indexed document {existing_doc.id} in search")
                    except Exception as index_error:
                        logger.error(
                            f"Failed to index document {existing_doc.id} in search: {index_error}"
                        )
                else:
                    logger.warning(
                        f"Embeddings mismatch for document {existing_doc.id}: {len(all_embeddings)} embeddings vs {len(chunks)} chunks - skipping search indexing"
                    )

            self.db.commit()

            return {
                "success": True,
                "filename": existing_doc.filename,
                "message": f"Reprocessed document with {chunks_created} chunks and {chapters_created} chapters",
                "document_id": existing_doc.id,
                "chunks_created": chunks_created,
                "chapters_created": chapters_created,
            }

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to reprocess document {existing_doc.id}: {e}")
            return {
                "success": False,
                "filename": existing_doc.filename,
                "error": str(e),
                "chunks_created": 0,
                "chapters_created": 0,
            }

    def upload_files(
        self,
        files,
        progress_callback: Optional[Callable] = None,
        data_dir: Optional[str] = None,
        use_parallel: bool = True,
        max_workers: Optional[int] = None,
        enable_streaming: bool = False,
        memory_limit_mb: int = 512,
    ) -> Dict[str, Any]:
        """
        Upload and process multiple files.

        Args:
            files: List of file paths (str) or Streamlit UploadedFile objects
            progress_callback: Optional callback for progress updates
            data_dir: Optional data directory for saving uploaded files
            use_parallel: Whether to use parallel processing
            max_workers: Maximum number of worker threads
            enable_streaming: Whether to use streaming for large files
            memory_limit_mb: Memory limit in MB

        Returns:
            Dict with upload results
        """
        import os
        import tempfile

        # Store processing options (could be used in future enhancements)
        self.enable_streaming = enable_streaming
        self.memory_limit_mb = memory_limit_mb

        if max_workers:
            self.max_workers = max_workers

        # Handle both file paths and UploadedFile objects
        processed_files = []
        temp_files = []

        for file_obj in files:
            if hasattr(file_obj, "getvalue"):  # Streamlit UploadedFile
                # Save UploadedFile to temporary location
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f"_{file_obj.name}"
                ) as tmp_file:
                    tmp_file.write(file_obj.getvalue())
                    temp_file_path = tmp_file.name

                temp_files.append(temp_file_path)
                processed_files.append(temp_file_path)
            else:  # Assume it's a file path (string)
                processed_files.append(str(file_obj))

        # Process files
        result = self.process_files(processed_files, progress_callback, use_parallel)

        # Clean up temporary files after processing
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass  # Ignore cleanup errors

        return result

    def _process_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file (used by ProcessPoolExecutor).

        Args:
            file_path: Path to the file to process

        Returns:
            Dict with processing results
        """
        try:
            # Create a new database session for this process
            db = SessionLocal()

            # Calculate file hash
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            filename = Path(file_path).name

            # Check if document already exists
            existing_doc = (
                db.query(Document)
                .filter(Document.filename == filename, Document.file_hash == file_hash)
                .first()
            )

            if existing_doc:
                return {
                    "success": True,
                    "filename": filename,
                    "message": "Document already exists",
                    "document_id": existing_doc.id,
                    "file_path": file_path,
                    "chunks_created": 0,  # Not creating new chunks
                    "chapters_created": 0,  # Not creating new chapters
                }

            # Process the document
            processor = DocumentProcessor(db, embedding_model=self.embedding_model)
            result = processor.process_document(file_path, filename)

            db.close()

            if result.get("success"):
                # Format result for compatibility with web interface expectations
                return {
                    "success": True,
                    "filename": filename,
                    "file_path": file_path,
                    "document_id": result.get("document_id"),
                    "chunks_created": result.get("chunks_count", 0),
                    "chapters_created": 0,  # DocumentProcessor doesn't create chapters directly
                    "message": f"Successfully processed document with {result.get('chunks_count', 0)} chunks",
                }

            return result

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return {
                "success": False,
                "filename": Path(file_path).name if file_path else "unknown",
                "error": str(e),
                "file_path": file_path,
                "chunks_created": 0,
                "chapters_created": 0,
            }
