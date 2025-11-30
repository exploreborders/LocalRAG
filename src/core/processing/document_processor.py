"""
Document processing system for LocalRAG.

Handles document analysis, structure extraction, chunking, and metadata generation.
"""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langdetect import LangDetectException, detect
from sqlalchemy.orm import Session

from src.ai.tag_suggester import AITagSuggester
from src.core.base_processor import BaseProcessor
from src.core.categorization.category_manager import CategoryManager
from src.core.embeddings import create_embeddings
from src.core.tagging.tag_manager import TagManager
from src.utils.config_manager import config
from src.database.models import (
    Document,
    DocumentChapter,
    DocumentChunk,
    DocumentEmbedding,
    SessionLocal,
)
from src.database.opensearch_setup import get_elasticsearch_client

logger = logging.getLogger(__name__)


class DocumentProcessor(BaseProcessor):
    """
    Enhanced document processor with integrated structure extraction and chapter-aware processing.

    This module provides the DocumentProcessor class which handles:
    - Integrated Docling document structure extraction during upload
    - Chapter-aware chunking with hierarchical metadata
    - Parallel processing for multiple files
    - Progress tracking and error handling
    - Automatic language detection and preprocessing
    """

    def __init__(self, db: Optional[Session] = None):
        super().__init__(db or SessionLocal())
        self.tag_manager = TagManager(self.db)
        self.category_manager = CategoryManager(self.db)
        self.tag_suggester = AITagSuggester()

    def _suggest_categories_ai(
        self, content: str, filename: str, tags: List[str]
    ) -> List[str]:
        """
        Suggest categories for a document using AI-based classification.

        Args:
            content: Document content
            filename: Document filename
            tags: Generated tags

        Returns:
            List of suggested category names
        """
        try:
            # Use the tag suggester's LLM to classify categories
            category_prompt = f"""
            Analyze this document and suggest 1-3 relevant categories.
            Choose from: Academic, Technical, Business, Scientific, Educational, Legal, Medical, Creative, Reference, General

            Document: {filename}
            Content preview: {content[:500]}
            Tags: {", ".join(tags)}

            Return only category names separated by commas (no explanations):
            """

            response = self.tag_suggester._call_llm(
                category_prompt, max_tokens=50
            ).strip()

            # Parse the response
            categories = [cat.strip() for cat in response.split(",") if cat.strip()]

            # Validate categories
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

            # Filter to valid categories and limit to 3
            validated_categories = [
                cat for cat in categories if cat in valid_categories
            ][:3]

            return validated_categories

        except Exception as e:
            # Fallback to simple keyword-based categorization
            content_lower = content.lower()
            categories = []

            if any(
                word in content_lower
                for word in ["deep learning", "neural", "machine learning", "ai"]
            ):
                categories.append("Technical")
            if any(
                word in content_lower
                for word in ["academic", "research", "paper", "study"]
            ):
                categories.append("Academic")
            if any(
                word in content_lower
                for word in ["tutorial", "guide", "course", "education"]
            ):
                categories.append("Educational")

            return categories[:3] if categories else ["General"]

    def _generate_document_summary(
        self, content: str, filename: str, tags: List[str], chapters_count: int
    ) -> str:
        """
        Generate an AI-powered document summary.

        Args:
            content: Document content sample
            filename: Document filename
            tags: Generated tags
            chapters_count: Number of chapters detected

        Returns:
            AI-generated document summary
        """
        try:
            # Create a comprehensive summary prompt
            summary_prompt = f"""
            Create a concise but informative summary of this document in 2-3 sentences.

            Content preview: {content[:800]}
            Tags: {", ".join(tags[:5])}  # Limit tags for prompt

            Focus on:
            - Main topics and subject matter
            - Key concepts covered
            - Document type and purpose
            - Target audience (if apparent)

            Summary should be professional and informative. Do not mention the filename or chapter/section counts.
            """

            summary = self.tag_suggester._call_llm(
                summary_prompt, max_tokens=200
            ).strip()

            # Clean up the summary - remove unwanted prefixes
            if summary:
                # Remove common AI prefixes
                prefixes_to_remove = [
                    "Here is a concise summary of the document in 2-3 sentences:",
                    "Here is a concise summary of the document:",
                    "Here is a summary of the document:",
                    "Summary:",
                    "Document summary:",
                ]

                for prefix in prefixes_to_remove:
                    if summary.lower().startswith(prefix.lower()):
                        summary = summary[len(prefix) :].strip()
                        break

                # Remove any leading/trailing artifacts
                summary = summary.strip()
                if summary.startswith('"') and summary.endswith('"'):
                    summary = summary[1:-1]
                if summary.startswith("'") and summary.endswith("'"):
                    summary = summary[1:-1]

                # Return clean summary without structure info
                return summary
            else:
                # Fallback summary
                return f"Document about {', '.join(tags[:3]) if tags else 'various topics'}. {chapters_count} chapters detected."

        except Exception as e:
            # Fallback to basic summary
            return f"Document processed with advanced AI pipeline. Covers {', '.join(tags[:3]) if tags else 'various topics'}. {chapters_count} chapters detected."

    def __del__(self):
        """Clean up database connections."""
        if hasattr(self, "db"):
            self.db.close()

    def process_document(
        self,
        file_path: str,
        filename: Optional[str] = None,
        use_advanced_processing: bool = False,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process a document with optional advanced AI-powered pipeline.

        Args:
            file_path: Path to the document file
            filename: Optional filename override
            use_advanced_processing: Whether to use comprehensive AI pipeline
            progress_callback: Optional progress callback

        Returns:
            Processing results
        """
        if use_advanced_processing:
            logger.info(
                f"Using ADVANCED processing for {filename or os.path.basename(file_path)}"
            )
            return self._process_document_advanced(
                file_path, filename, progress_callback
            )
        else:
            logger.info(
                f"Using STANDARD processing for {filename or os.path.basename(file_path)}"
            )
            return self._process_document_standard(
                file_path, filename, progress_callback
            )

    def _process_document_advanced(
        self,
        file_path: str,
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process document using comprehensive AI-powered pipeline.
        """
        try:
            # Lazy import to avoid dependency issues during testing
            from src.data.loader import AdvancedDocumentProcessor

            # Initialize advanced processor
            advanced_processor = AdvancedDocumentProcessor()

            if progress_callback:
                progress_callback(
                    filename or os.path.basename(file_path),
                    5,
                    "Starting advanced AI processing...",
                )

            # Run comprehensive processing
            results = advanced_processor.process_document_comprehensive(file_path)

            if progress_callback:
                progress_callback(
                    filename or os.path.basename(file_path),
                    50,
                    "AI processing complete, storing results...",
                )

            # Store results in database (simplified for now)
            # This would need to be expanded to store all the advanced metadata
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            # Generate tags using AI
            content_for_analysis = results.get("extracted_content", "")[
                :2000
            ]  # Sample for analysis
            suggested_tags_data = self.tag_suggester.suggest_tags(
                content_for_analysis, filename or os.path.basename(file_path)
            )
            # Extract tag names from the suggestion data
            suggested_tags = [
                tag.get("tag", "") for tag in suggested_tags_data if tag.get("tag")
            ]

            # Generate categories using AI-based classification
            suggested_categories = self._suggest_categories_ai(
                content_for_analysis,
                filename or os.path.basename(file_path),
                suggested_tags,
            )

            # Generate AI-powered document summary
            document_summary = self._generate_document_summary(
                content_for_analysis,
                filename or os.path.basename(file_path),
                suggested_tags,
                results.get("chapters_detected", 0),
            )

            processing_result = {
                "file_hash": file_hash,
                "detected_language": results.get(
                    "language", "de"
                ),  # Default to German for technical docs
                "document_summary": document_summary,
                "key_topics": results.get("topics", []),
                "reading_time_minutes": max(
                    1, len(results.get("extracted_content", "")) // 2000
                ),  # Rough estimate
                "suggested_tags": suggested_tags,
                "suggested_categories": suggested_categories,
            }

            document = Document(
                filename=filename or os.path.basename(file_path),
                filepath=file_path,
                file_hash=processing_result["file_hash"],
                status="processed",
                detected_language=processing_result["detected_language"],
                document_summary=processing_result["document_summary"],
                key_topics=processing_result["key_topics"],
                reading_time_minutes=processing_result["reading_time_minutes"],
            )
            self.db.add(document)
            self.db.flush()

            # Store suggested tags
            for tag_name in processing_result.get("suggested_tags", []):
                if tag_name:
                    tag = self.tag_manager.get_tag_by_name(tag_name)
                    if not tag:
                        tag = self.tag_manager.create_tag(tag_name)
                    self.tag_manager.add_tag_to_document(document.id, tag.id)

            # Store suggested categories
            for category_name in processing_result.get("suggested_categories", []):
                if category_name:
                    category = self.category_manager.get_category_by_name(category_name)
                    if not category:
                        category = self.category_manager.create_category(category_name)
                    self.category_manager.add_category_to_document(
                        document.id, category.id
                    )

            # Store chapters from advanced processing
            structure_info = results.get("structure_analysis", {})
            hierarchy = structure_info.get("hierarchy", [])
            for chapter_data in hierarchy:
                chapter_record = DocumentChapter(
                    document_id=document.id,
                    chapter_title=chapter_data.get("title", "")[
                        :255
                    ],  # Limit to 255 chars
                    chapter_path=chapter_data.get("path", ""),
                    level=chapter_data.get("level", 1),
                    word_count=chapter_data.get("word_count", 0),
                    content=chapter_data.get(
                        "content_preview", chapter_data.get("title", "")
                    ),
                    section_type="chapter"
                    if chapter_data.get("level", 1) == 1
                    else "section",
                )
                self.db.add(chapter_record)

            # Store all chunks from advanced processing
            for i, chunk_data in enumerate(results.get("chunks", [])):
                chunk = DocumentChunk(
                    document_id=document.id,
                    content=chunk_data["content"],
                    chunk_index=i,
                    chapter_title=chunk_data.get("chapter", "auto"),
                    embedding_model="sentence-transformers",  # Default embedding model
                )
                self.db.add(chunk)

            self.db.commit()

            if progress_callback:
                progress_callback(
                    filename or os.path.basename(file_path),
                    100,
                    "Advanced processing complete!",
                )

            return {
                "success": True,
                "document_id": document.id,
                "advanced_processing": True,
                "processing_stages": results.get("processing_stages", []),
                "chapters_detected": len(results.get("chapters", [])),
                "chunks_created": len(results.get("chunks", [])),
                "topics_identified": len(results.get("topics", [])),
            }

        except Exception as e:
            logger.error(f"Advanced document processing failed for {file_path}: {e}")
            # Fallback to standard processing
            return self._process_document_standard(
                file_path, filename, progress_callback
            )

    def _process_document_standard(
        self,
        file_path: str,
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single document with enhanced structure extraction.

        Args:
            file_path: Path to the document file
            filename: Optional custom filename
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with processing results
        """
        if filename is None:
            filename = Path(file_path).name

        if progress_callback:
            progress_callback(filename, 0, f"Starting processing of {filename}")

        try:
            # Detect language
            language = self._detect_language_from_file(file_path)
            if progress_callback:
                progress_callback(filename, 10, f"Detected language: {language}")

            # Lazy import to avoid dependency issues during testing
            from src.data.loader import split_documents

            # Load document content
            doc_content = split_documents([file_path])
            if not doc_content:
                return {"success": False, "error": "Failed to load document"}

            # Detect language from extracted content
            language = self._detect_language_from_content(doc_content)

            # Create document record
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            document = Document(
                filename=filename,
                filepath=file_path,
                file_hash=file_hash,
                detected_language=language,
                status="processing",
                full_content=doc_content,
            )
            self.db.add(document)
            self.db.commit()

            if progress_callback:
                progress_callback(filename, 20, "Document record created")

            # Detect chapters from full content first
            all_chapters = self._detect_all_chapters(doc_content)

            # Process content into chunks
            chunks = self._create_chunks(doc_content, document.id, all_chapters)

            if progress_callback:
                progress_callback(
                    filename,
                    60,
                    f"Created {len(chunks)} chunks, detected {len(all_chapters)} chapters",
                )

            # Store chapters in database
            for chapter in all_chapters:
                chapter_record = DocumentChapter(
                    document_id=document.id,
                    chapter_title=chapter["title"],  # Short title (max 255 chars)
                    chapter_path=chapter["path"],
                    level=chapter.get("level", 1),
                    word_count=len(chapter["title"].split()),
                    content=chapter.get("content", chapter["title"]),  # Full content
                )
                self.db.add(chapter_record)
            self.db.commit()

            # Generate embeddings
            embeddings_array, model = create_embeddings(
                [chunk["content"] for chunk in chunks],
                model_name=config.models.embedding_model,
                backend=config.models.embedding_backend,
            )

            # Store chunks with embeddings and enhanced metadata
            for i, chunk_data in enumerate(chunks):
                metadata = chunk_data.get("metadata", {})

                chunk = DocumentChunk(
                    document_id=document.id,
                    chunk_index=i,
                    content=chunk_data["content"],
                    embedding_model="nomic-ai/nomic-embed-text-v1.5",
                    chapter_title=metadata.get("chapter_title"),
                    chapter_path=metadata.get("chapter_path"),
                )
                self.db.add(chunk)
                self.db.flush()  # Get chunk ID

                # Create embedding record if embeddings were generated
                if embeddings_array is not None and i < len(embeddings_array):
                    embedding_record = DocumentEmbedding(
                        chunk_id=chunk.id,
                        embedding=embeddings_array[i].tolist(),
                        embedding_model="nomic-ai/nomic-embed-text-v1.5",
                    )
                    self.db.add(embedding_record)

            self.db.commit()

            if progress_callback:
                progress_callback(filename, 90, "Embeddings generated and stored")

            # Index in Elasticsearch
            if embeddings_array is not None:
                self._index_document(document, chunks, embeddings_array.tolist())
            else:
                self._index_document(document, chunks, [])

            # AI enrichment
            if progress_callback:
                progress_callback(filename, 95, "Running AI enrichment...")

            try:
                from src.ai.enrichment import AIEnrichmentService

                enrichment_service = AIEnrichmentService()
                enrichment_result = enrichment_service.enrich_document(document.id)

                # Update document with enrichment data
                if "summary" in enrichment_result:
                    document.document_summary = enrichment_result["summary"]
                if "topics" in enrichment_result:
                    document.key_topics = enrichment_result["topics"]

                self.db.commit()

                if progress_callback:
                    progress_callback(filename, 98, "AI enrichment completed")

            except Exception as e:
                print(f"⚠️ AI enrichment failed: {e}")
                # Continue without enrichment

            # Update document status to processed
            document.status = "processed"
            self.db.commit()

            if progress_callback:
                progress_callback(filename, 100, "Document processing complete")

            return {
                "success": True,
                "document_id": document.id,
                "chunks_count": len(chunks),
                "language": language,
            }

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            self.db.rollback()
            return {"success": False, "error": str(e)}

    def _detect_language_from_file(self, file_path: str) -> str:
        """Detect language from file content."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                sample = f.read(1000)
                return detect(sample)
        except (LangDetectException, FileNotFoundError):
            return "en"

    def _detect_language_from_content(self, content: str) -> str:
        """Detect language from extracted content."""
        try:
            # Use multiple samples for better detection
            samples = [
                content[:2000],  # Beginning
                (
                    content[len(content) // 3 : len(content) // 3 + 2000]
                    if len(content) > 6000
                    else content[len(content) // 2 : len(content) // 2 + 1000]
                ),  # Middle
                content[-2000:]
                if len(content) > 2000
                else content[len(content) // 2 :],  # End
            ]

            # Try to detect language from each sample
            detections = []
            for sample in samples:
                if len(sample.strip()) > 100:  # Only use substantial samples
                    try:
                        lang = detect(sample)
                        detections.append(lang)
                    except LangDetectException:
                        continue

            # Return the most common detection, or 'en' as fallback
            if detections:
                return max(set(detections), key=detections.count)
            else:
                return "en"
        except (LangDetectException, ValueError):
            return "en"

    def _detect_all_chapters(self, content: str) -> List[Dict[str, Any]]:
        """Detect all chapters from the full document content."""
        chapters = []

        # Look for ## headers (markdown)
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("##"):
                header_text = line.strip()[2:].strip()  # Remove ##
                if header_text and len(header_text) > 3:
                    # Extract just the title (first line, truncated to 255 chars)
                    title_lines = header_text.split("\n")
                    short_title = title_lines[0].strip()[:255]  # Limit to 255 chars

                    chapters.append(
                        {
                            "title": short_title,
                            "content": header_text,  # Store full content separately
                            "path": f"section_{len(chapters) + 1}",
                            "start_line": i,
                            "level": 2,
                        }
                    )

        # Look for table format | number | title | (markdown tables)
        table_matches = re.findall(
            r"\|\s*(\d+(?:\.\d+)*)\s*\|\s*([^\|]+?)\s*\|", content
        )
        for number, title in table_matches:
            title = title.strip()
            if title and len(title) > 3:
                # Truncate title to 255 characters for database
                short_title = title[:255]

                chapters.append(
                    {
                        "title": short_title,
                        "content": title,  # Store full content separately
                        "path": number,
                        "start_line": -1,  # Table entries don't have line numbers
                        "level": 1 if "." not in number else len(number.split(".")) + 1,
                    }
                )

        # Additional patterns for scanned/OCR text (plain text patterns)
        if not chapters:  # Only try these if no structured chapters found
            # Look for numbered chapters (1. Chapter Title, 2. Chapter Title, etc.)
            chapter_patterns = [
                r"^\s*(\d+)\.?\s+(.+)$",  # 1. Chapter Title or 1 Chapter Title
                r"^\s*Kapitel\s*(\d+):?\s*(.+)$",  # Kapitel 1: Title (German)
                r"^\s*Chapter\s*(\d+):?\s*(.+)$",  # Chapter 1: Title (English)
                r"^\s*(\d+)\s+(.+)$",  # 1 Title (simple numbered)
            ]

            for pattern in chapter_patterns:
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue

                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        number = match.group(1)
                        title = match.group(2).strip()

                        # Skip very short titles or common false positives
                        if len(title) < 3 or title.lower() in [
                            "page",
                            "seiten",
                            "chapter",
                            "kapitel",
                        ]:
                            continue

                        # Avoid duplicates
                        if not any(ch["title"] == title[:255] for ch in chapters):
                            chapters.append(
                                {
                                    "title": title[:255],
                                    "content": title,
                                    "path": number,
                                    "start_line": i,
                                    "level": 1,
                                }
                            )

            # Look for lines that look like chapter titles (capitalized, standalone)
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                # Skip lines that are too short or too long
                if len(line) < 5 or len(line) > 100:
                    continue

                # Look for patterns that suggest chapter titles
                if (
                    line[0].isupper()  # Starts with capital
                    and not line.endswith(".")  # Not a sentence
                    and not any(
                        char.isdigit() for char in line[:10]
                    )  # No early numbers
                    and sum(1 for c in line if c.isupper()) / len(line) < 0.5
                ):  # Not ALL CAPS
                    # Additional heuristics for chapter-like content
                    chapter_keywords = [
                        "introduction",
                        "grundlagen",
                        "einführung",
                        "overview",
                        "grundlagen",
                        "theorie",
                        "praxis",
                        "anwendung",
                        "methoden",
                        "algorithmen",
                    ]
                    if any(keyword in line.lower() for keyword in chapter_keywords):
                        if not any(ch["title"] == line[:255] for ch in chapters):
                            chapters.append(
                                {
                                    "title": line[:255],
                                    "content": line,
                                    "path": f"section_{len(chapters) + 1}",
                                    "start_line": i,
                                    "level": 1,
                                }
                            )

        # For scanned PDFs with no clear chapter structure, create synthetic chapters
        # based on document length and common technical content
        if (
            not chapters and len(content) > 1000
        ):  # Any reasonable content but no chapters found
            logger.info(
                "No chapters found in scanned PDF, creating synthetic chapters for technical content"
            )

            # Try to use advanced structure analysis if available
            try:
                from src.data.loader import AdvancedDocumentProcessor

                processor = AdvancedDocumentProcessor()
                structure_analysis = processor._analyze_document_structure(content)

                if structure_analysis.get("sections"):
                    # Use AI-analyzed structure
                    for section in structure_analysis["sections"]:
                        chapters.append(
                            {
                                "title": section.get(
                                    "title", f"Chapter {len(chapters) + 1}"
                                )[:255],
                                "content": section.get("title", ""),
                                "path": str(len(chapters) + 1),
                                "start_line": section.get("start_line", 0),
                                "level": section.get("level", 1),
                            }
                        )
                    logger.info(
                        f"Created {len(chapters)} AI-analyzed chapters for scanned PDF"
                    )
                else:
                    # Fallback to synthetic chapters
                    synthetic_chapters = [
                        "Einführung und Grundlagen",
                        "Mathematische Grundlagen",
                        "Kernkonzepte und Architekturen",
                        "Training und Optimierung",
                        "Erweiterte Techniken",
                        "Praktische Anwendungen",
                        "Fallstudien und Beispiele",
                        "Best Practices und Tipps",
                        "Troubleshooting und Debugging",
                        "Performance und Optimierung",
                        "Deployment und Produktion",
                        "Zukunftsaussichten",
                    ]

                    # Create chapters distributed throughout the document
                    content_length = len(content)
                    num_synthetic = min(
                        len(synthetic_chapters), max(3, content_length // 5000)
                    )
                    for i in range(num_synthetic):
                        start_pos = (content_length * i) // num_synthetic
                        end_pos = (content_length * (i + 1)) // num_synthetic
                        chapter_content = content[start_pos:end_pos]

                        chapters.append(
                            {
                                "title": synthetic_chapters[i][:255],
                                "content": chapter_content[:1000],  # Store preview
                                "path": str(i + 1),
                                "start_line": start_pos // 100,  # Rough line estimate
                                "level": 1,
                            }
                        )
                    logger.info(
                        f"Created {len(chapters)} synthetic chapters for technical content"
                    )
            except Exception as e:
                logger.warning(f"Advanced chapter detection failed: {e}")
                # Continue with empty chapters list

        return chapters

    def _create_chunks(
        self, content: str, document_id: int, chapters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create chunks from document content with chapter awareness."""
        chunks = []

        if not chapters:
            # Simple chunking without chapters
            chunk_size = 1000
            overlap = 200

            for i in range(0, len(content), chunk_size - overlap):
                chunk_content = content[i : i + chunk_size]
                if len(chunk_content.strip()) > 100:  # Only substantial chunks
                    chunks.append(
                        {
                            "content": chunk_content,
                            "metadata": {
                                "chapter_title": None,
                                "chapter_path": None,
                            },
                        }
                    )
        else:
            # Chapter-aware chunking
            for chapter in chapters:
                chapter_content = chapter.get("content", "")
                if len(chapter_content) > 500:  # Substantial chapter content
                    # Create chunks within this chapter
                    chunk_size = 800
                    overlap = 150

                    for i in range(0, len(chapter_content), chunk_size - overlap):
                        chunk_content = chapter_content[i : i + chunk_size]
                        if len(chunk_content.strip()) > 50:
                            chunks.append(
                                {
                                    "content": chunk_content,
                                    "metadata": {
                                        "chapter_title": chapter["title"],
                                        "chapter_path": chapter["path"],
                                    },
                                }
                            )

        return chunks

    def _index_document(
        self,
        document: Document,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """Index document in Elasticsearch for search."""
        try:
            es_client = get_elasticsearch_client()
            if es_client:
                # Index document metadata
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
                    "created_at": document.created_at.isoformat()
                    if document.created_at
                    else None,
                }

                es_client.index(index="documents", id=str(document.id), body=doc_data)

                # Index chunks with embeddings
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_data = {
                        "document_id": document.id,
                        "chunk_index": i,
                        "content": chunk["content"],
                        "chapter_title": chunk.get("metadata", {}).get("chapter_title"),
                        "chapter_path": chunk.get("metadata", {}).get("chapter_path"),
                        "embedding": embedding,
                    }

                    chunk_id = f"{document.id}_{i}"
                    es_client.index(index="chunks", id=chunk_id, body=chunk_data)

                logger.info(f"Indexed document {document.id} with {len(chunks)} chunks")
            else:
                logger.warning("Elasticsearch client not available, skipping indexing")

        except Exception as e:
            logger.error(f"Failed to index document {document.id}: {e}")
            # Don't fail the entire process if indexing fails
