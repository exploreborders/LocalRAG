from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
import os
import logging
import fitz  # PyMuPDF for direct PDF processing
from PIL import Image
import io
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning(
        "Tesseract OCR not available. Install with: pip install pytesseract tesseract"
    )

# Vision model setup for advanced image analysis
try:
    import ollama

    VISION_MODEL_AVAILABLE = True
    VISION_BACKEND = "ollama"
    logger.info("Ollama vision model available")
except ImportError:
    try:
        from transformers import (
            Qwen2VLForConditionalGeneration,
            AutoTokenizer,
            AutoProcessor,
        )
        import torch

        VISION_MODEL_AVAILABLE = True
        VISION_BACKEND = "transformers"
        logger.info("Transformers vision model dependencies available")
    except ImportError:
        VISION_MODEL_AVAILABLE = False
        VISION_BACKEND = None
        logger.warning(
            "No vision model dependencies available. Install ollama or transformers"
        )


def extract_text_with_ocr(pdf_path: str) -> str:
    """
    Extract text from scanned PDF using comprehensive OCR processing.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text content
    """
    if not TESSERACT_AVAILABLE:
        logger.warning("OCR requested but Tesseract not available")
        return ""

    try:
        import fitz
        from PIL import Image
        import io

        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        text_content = []

        logger.info(f"Starting comprehensive OCR for {total_pages} pages")

        # Process ALL pages for scanned PDFs - don't skip any
        for page_num in range(total_pages):
            try:
                page = doc.load_page(page_num)

                # Get existing text (might be minimal for scanned docs)
                existing_text = page.get_text().strip()

                # Always attempt OCR for better extraction
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2.5, 2.5)
                )  # Higher scaling for better OCR
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                # OCR with multiple configurations for better results
                ocr_configs = [
                    ("deu+eng", "--psm 3"),  # Auto page segmentation
                    ("deu+eng", "--psm 6"),  # Uniform block of text
                    ("eng+deu", "--psm 3"),  # Try English first
                ]

                best_ocr_text = ""
                max_length = 0

                for lang, config in ocr_configs:
                    try:
                        page_text = pytesseract.image_to_string(
                            img, lang=lang, config=config
                        )
                        if len(page_text.strip()) > max_length:
                            max_length = len(page_text.strip())
                            best_ocr_text = page_text
                    except Exception as config_error:
                        logger.debug(
                            f"OCR config {lang} {config} failed: {config_error}"
                        )
                        continue

                # Combine existing text and OCR text
                combined_text = existing_text
                if best_ocr_text.strip():
                    if combined_text and not combined_text.endswith("\n"):
                        combined_text += "\n"
                    combined_text += best_ocr_text

                # Only add pages with substantial content
                if len(combined_text.strip()) > 20:  # Minimum content threshold
                    text_content.append(
                        f"=== PAGE {page_num + 1} ===\n{combined_text}\n"
                    )
                    logger.debug(
                        f"Processed page {page_num + 1}: {len(combined_text)} chars"
                    )

            except Exception as page_error:
                logger.warning(f"Failed to process page {page_num + 1}: {page_error}")
                continue

        doc.close()

        full_text = "\n".join(text_content)
        logger.info(
            f"OCR completed: {len(full_text)} characters from {len(text_content)} pages with content"
        )

        # Post-process the text to clean up common OCR errors
        full_text = _post_process_ocr_text(full_text)

        return full_text

    except Exception as e:
        logger.error(f"OCR processing failed for {pdf_path}: {e}")
        return ""


def _post_process_ocr_text(text: str) -> str:
    """
    Clean up common OCR errors and improve text quality.

    Args:
        text: Raw OCR text

    Returns:
        Cleaned text
    """
    import re

    # Remove excessive whitespace
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Fix common OCR character errors
    replacements = {
        " ,": ",",  # Space before comma
        " .": ".",  # Space before period
        "  +": " ",  # Multiple spaces
        r"(\w)-\s*\n\s*(\w)": r"\1-\2",  # Hyphenated words split across lines
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    # Clean up page markers
    text = re.sub(r"=== PAGE \d+ ===\s*\n", "", text)

    return text.strip()

    try:
        import fitz
        from PIL import Image
        import io

        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        text_content = []

        # For scanned PDFs, process efficiently
        # Focus on first 20 pages (TOC and main content usually here)
        priority_pages = list(range(min(20, len(doc))))

        # Add a few sampled pages to find chapter headers
        if len(doc) > 20:
            step = max(15, len(doc) // 10)  # Sample every 10% of document
            for page_num in range(20, min(len(doc), 80), step):  # Max 80 pages total
                priority_pages.append(page_num)

        # Add last page
        if len(doc) > 1:
            priority_pages.append(len(doc) - 1)

        # Remove duplicates and sort
        priority_pages = sorted(list(set(priority_pages)))

        logger.info(
            f"OCR processing {len(priority_pages)} pages out of {len(doc)} total for scanned PDF"
        )

        logger.info(
            f"OCR processing {len(priority_pages)} priority pages out of {len(doc)} total"
        )

        for page_num in priority_pages:
            try:
                page = doc.load_page(page_num)

                # Skip pages with too much text (likely already OCR'd by Docling)
                existing_text = page.get_text()
                if (
                    len(existing_text.strip()) > 500
                ):  # Page already has substantial text
                    text_content.append(
                        f"Page {page_num + 1} (existing text):\n{existing_text}\n"
                    )
                    continue

                # Get page as image for OCR
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2)
                )  # 2x scaling for better OCR
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                # Extract text with OCR
                page_text = pytesseract.image_to_string(
                    img, lang="deu+eng"
                )  # German + English

                if page_text.strip():  # Only add if OCR found text
                    text_content.append(f"Page {page_num + 1} (OCR):\n{page_text}\n")
                    logger.debug(
                        f"OCR extracted {len(page_text)} chars from page {page_num + 1}"
                    )

            except Exception as page_error:
                logger.warning(f"OCR failed on page {page_num + 1}: {page_error}")
                continue

        doc.close()

        full_text = "\n".join(text_content)
        logger.info(
            f"OCR completed: {len(full_text)} characters from {len(priority_pages)} pages"
        )
        return full_text

    except Exception as e:
        logger.error(f"OCR processing failed for {pdf_path}: {e}")
        return ""

    try:
        import fitz
        from PIL import Image
        import io

        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        text_content = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Get page as image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better OCR
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            # Extract text with OCR
            page_text = pytesseract.image_to_string(
                img, lang="deu+eng"
            )  # German + English
            text_content.append(f"Page {page_num + 1}:\n{page_text}\n")

        doc.close()
        return "\n".join(text_content)

    except Exception as e:
        logger.error(f"OCR processing failed for {pdf_path}: {e}")
        return ""


def is_scanned_pdf(pdf_path: str) -> bool:
    """
    Check if a PDF is likely scanned/image-based by analyzing text content.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        True if PDF appears to be scanned
    """
    try:
        import fitz

        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        # Check first few pages
        total_text = ""
        pages_to_check = []

        # Always check first page
        pages_to_check.append(0)

        # Check a few more pages distributed throughout
        if total_pages > 1:
            pages_to_check.append(
                min(5, total_pages - 1)
            )  # Page 6 or last page if shorter
        if total_pages > 10:
            pages_to_check.append(
                min(10, total_pages - 1)
            )  # Page 11 or last page if shorter

        pages_checked = len(pages_to_check)

        for page_num in pages_to_check:
            try:
                page = doc.load_page(page_num)
                page_text = page.get_text()
                total_text += page_text

                # Also check for images (strong indicator of scanned content)
                images = page.get_images(full=True)
                if len(images) > 0:
                    logger.info(
                        f"Page {page_num + 1} contains {len(images)} images - likely scanned"
                    )
                    doc.close()
                    return True

            except Exception as e:
                logger.warning(f"Failed to check page {page_num + 1}: {e}")

        doc.close()

        # If very little text is found, it's likely scanned
        text_ratio = len(total_text.strip()) / max(
            1, pages_checked * 500
        )  # More aggressive heuristic - expect at least 500 chars per page
        is_scanned = text_ratio < 0.05  # Less than 5% of expected text content

        logger.info(
            f"PDF {pdf_path}: {len(total_text.strip())} chars across {pages_checked} pages, ratio={text_ratio:.4f}, scanned={is_scanned}"
        )
        return is_scanned

    except Exception as e:
        logger.error(f"Failed to analyze PDF {pdf_path}: {e}")
        return False


class AdvancedDocumentProcessor:
    """
    Advanced document processor with vision models and multi-stage AI analysis.

    Pipeline:
    1. Document Input
    2. Docling Parser (baseline extraction)
    3. Quality Check → Vision Fallback (qwen2.5vl:7b) [if needed]
    4. Structure Analysis (phi3.5:3.8b for hierarchy + topics)
    5. Topic Classification (multi-strategy approach)
    6. Hierarchical Chunking (chapter-aware, token-based)
    7. Relevance Scoring (semantic + topic-aware)
    8. Embedding Generation (nomic-embed-text-v1.5)
    9. Storage: PostgreSQL + JSONB structures + topic relationships
    10. Search: BM25 (Elasticsearch) + Vector (pgvector) hybrid
    """

    def __init__(self):
        self.vision_available = VISION_MODEL_AVAILABLE and VISION_BACKEND == "ollama"
        logger.info(
            f"AdvancedDocumentProcessor initialized - Vision available: {self.vision_available}"
        )

    def _initialize_models(self):
        """Initialize AI models for the processing pipeline."""
        # Models are initialized on-demand to avoid loading issues
        pass

    def process_document_comprehensive(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process document through the complete AI-powered pipeline.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Comprehensive processing results
        """
        logger.info(f"Starting comprehensive processing for {pdf_path}")

        results = {
            "file_path": pdf_path,
            "processing_stages": [],
            "extracted_content": "",
            "structure_analysis": {},
            "topics": [],
            "chapters": [],
            "chunks": [],
            "embeddings": [],
            "quality_metrics": {},
        }

        # Stage 1: Baseline extraction with Docling
        results["processing_stages"].append("docling_baseline")
        docling_content = self._extract_with_docling(pdf_path)
        results["extracted_content"] = docling_content

        # Stage 2: Quality check and vision fallback
        quality_score = self._assess_content_quality(docling_content, pdf_path)
        results["quality_metrics"]["baseline_quality"] = quality_score

        if quality_score < 0.7:  # Poor quality, use vision fallback
            results["processing_stages"].append("vision_fallback")
            vision_content = self._extract_with_vision_model(pdf_path)
            if len(vision_content) > len(docling_content) * 1.2:
                results["extracted_content"] = vision_content
                results["processing_stages"].append("vision_improved")

        # Stage 3: Structure analysis
        results["processing_stages"].append("structure_analysis")
        structure_info = self._analyze_document_structure(results["extracted_content"])
        results["structure_analysis"] = structure_info

        # Stage 4: Topic classification
        results["processing_stages"].append("topic_classification")
        topics = self._classify_topics(results["extracted_content"], structure_info)
        results["topics"] = topics

        # Stage 5: Hierarchical chunking
        results["processing_stages"].append("hierarchical_chunking")
        chunks = self._create_hierarchical_chunks(
            results["extracted_content"], structure_info
        )
        results["chunks"] = chunks

        # Stage 6: Relevance scoring
        results["processing_stages"].append("relevance_scoring")
        scored_chunks = self._score_chunk_relevance(chunks, topics)
        results["chunks"] = scored_chunks

        # Stage 7: Embedding generation
        results["processing_stages"].append("embedding_generation")
        embeddings = self._generate_embeddings(scored_chunks)
        results["embeddings"] = embeddings

        logger.info(f"Comprehensive processing completed for {pdf_path}")
        return results

    def _extract_with_docling(self, pdf_path: str) -> str:
        """Extract content using Docling parser."""
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            pipeline_options = PdfPipelineOptions(do_ocr=True, do_table_structure=True)
            doc_converter = DocumentConverter(format_options={"pdf": pipeline_options})

            result = doc_converter.convert(pdf_path)
            return result.document.export_to_markdown()
        except Exception as e:
            logger.error(f"Docling extraction failed: {e}")
            return ""

    def _assess_content_quality(self, content: str, pdf_path: str) -> float:
        """Assess the quality of extracted content."""
        if not content:
            return 0.0

        # Quality metrics
        content_length = len(content)
        avg_words_per_page = content_length / max(1, self._get_pdf_page_count(pdf_path))

        # Check for structured elements
        has_headers = "##" in content or "#" in content
        has_lists = (
            "- " in content
            or "* " in content
            or content.count("\n") > content_length * 0.1
        )

        # Calculate quality score
        quality_score = min(
            1.0,
            (
                min(1.0, content_length / 10000)
                * 0.4  # Content length (up to 10k chars = 0.4)
                + min(1.0, avg_words_per_page / 1000) * 0.3  # Words per page
                + (0.3 if has_headers else 0.0)  # Headers present
                + (0.2 if has_lists else 0.0)  # Lists/tables present
            ),
        )

        return quality_score

    def _extract_with_vision_model(self, pdf_path: str) -> str:
        """Extract content using vision model for images."""
        if not VISION_MODEL_AVAILABLE:
            logger.warning("Vision model not available, falling back to enhanced OCR")
            return self._extract_with_enhanced_ocr(pdf_path)

        try:
            import fitz
            from PIL import Image
            import io
            import base64

            doc = fitz.open(pdf_path)
            vision_content = []

            logger.info(
                f"AI vision processing for {len(doc)} pages using {VISION_BACKEND}"
            )

            # Process more pages for comprehensive content extraction and chapter detection
            # Prioritize: TOC pages (first 10), content pages (sampled throughout), last page
            important_pages = list(
                range(min(10, len(doc)))
            )  # First 10 pages (TOC, intro, early chapters)

            # Add more content pages for better chapter detection and content coverage
            if len(doc) > 10:
                # Sample pages throughout the document to capture all chapters
                content_pages = [
                    15,
                    20,
                    30,
                    40,
                    50,
                    60,
                    80,
                    100,
                    120,
                    150,
                    200,
                    250,
                    300,
                    350,
                ]  # More comprehensive sampling
                important_pages.extend([p for p in content_pages if p < len(doc)])

            # Add last page
            if len(doc) > 1:
                important_pages.append(len(doc) - 1)

            # Remove duplicates and limit to reasonable number for comprehensive processing
            important_pages = sorted(list(set(important_pages)))[
                :25
            ]  # Max 25 pages for better coverage

            logger.info(
                f"Vision processing {len(important_pages)} key pages out of {len(doc)} total"
            )

            # Process pages individually to avoid timeouts
            for page_num in important_pages:
                try:
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB
                    )
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))

                    # Convert to base64 for Ollama
                    img_base64 = base64.b64encode(img_data).decode("utf-8")

                    # Process single page with vision model
                    page_content = self._process_single_image_with_vision(
                        {
                            "page_num": page_num + 1,
                            "image": img_base64,
                            "image_obj": img,
                        }
                    )

                    if page_content:
                        vision_content.append(page_content)

                except Exception as page_error:
                    logger.warning(
                        f"Vision processing failed for page {page_num + 1}: {page_error}"
                    )
                    continue

            doc.close()

            full_content = "\n\n".join(vision_content)
            logger.info(
                f"Vision model extracted {len(full_content)} characters from {len(vision_content)} pages"
            )
            return full_content

        except Exception as e:
            logger.error(f"Vision model processing failed: {e}")
            return self._extract_with_enhanced_ocr(pdf_path)

    def _process_single_image_with_vision(self, img_data: Dict[str, Any]) -> str:
        """Process a single image with the vision model."""
        try:
            page_num = img_data["page_num"]

            if VISION_BACKEND == "ollama":
                # Use Ollama vision model
                response = self._query_ollama_vision(img_data["image"], page_num)
                if response:
                    return f"=== PAGE {page_num} ===\n{response}"

            elif VISION_BACKEND == "transformers":
                # Use transformers vision model (fallback)
                response = self._query_transformers_vision(
                    img_data["image_obj"], page_num
                )
                if response:
                    return f"=== PAGE {page_num} ===\n{response}"

        except Exception as page_error:
            logger.warning(
                f"Vision processing failed for page {img_data['page_num']}: {page_error}"
            )

        # Fallback to OCR for this page
        try:
            page_text = pytesseract.image_to_string(
                img_data["image_obj"], lang="deu+eng", config="--psm 6"
            )
            if page_text.strip():
                return f"=== PAGE {page_num} (OCR Fallback) ===\n{page_text}"
        except Exception as ocr_error:
            logger.warning(
                f"OCR fallback also failed for page {img_data['page_num']}: {ocr_error}"
            )

        return ""

    def _query_ollama_vision(self, image_base64: str, page_num: int) -> str:
        """Query Ollama vision model for image analysis."""
        try:
            import ollama

            prompt = f"""Extract all readable text from this technical document page ({page_num}).

Focus on:
- Chapter/section titles and headers
- Technical content, explanations, algorithms
- Important terms, definitions, formulas
- Table of contents or index entries

Extract the raw text content concisely, preserving technical terminology."""

            response = ollama.chat(
                model="qwen2.5vl:latest",
                messages=[
                    {"role": "user", "content": prompt, "images": [image_base64]}
                ],
                options={
                    "temperature": 0.1,  # Low temperature for accurate extraction
                    "num_predict": 1024,  # Reasonable length limit
                    "timeout": 90,  # 1.5 minute timeout per page for more comprehensive processing
                },
            )

            content = response["message"]["content"]
            logger.debug(
                f"Ollama vision extracted {len(content)} chars from page {page_num}"
            )
            return content

        except Exception as e:
            logger.error(f"Ollama vision query failed for page {page_num}: {e}")
            return ""

    def _query_transformers_vision(self, image: Image.Image, page_num: int) -> str:
        """Query transformers vision model (fallback)."""
        try:
            # Placeholder for transformers implementation
            logger.info(
                f"Transformers vision processing for page {page_num} (placeholder)"
            )
            # This would implement the transformers-based vision model
            return f"Transformers vision processing for page {page_num} (not implemented yet)"

        except Exception as e:
            logger.error(f"Transformers vision query failed for page {page_num}: {e}")
            return ""

    def _extract_with_enhanced_ocr(self, pdf_path: str) -> str:
        """Enhanced OCR processing as fallback."""
        return extract_text_with_ocr(pdf_path)

    def _analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure from content."""
        lines = content.split("\n")
        structure = {
            "hierarchy": [],
            "sections": [],
            "estimated_chapters": 0,
            "headers_found": [],
            "content_sections": [],
        }

        # Look for headers and structure patterns
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Detect potential headers
            if len(line) > 5 and len(line) < 150:  # Reasonable header length
                # Check for header-like patterns
                if (
                    line[0].isupper()  # Starts with capital
                    or line.isupper()  # ALL CAPS
                    or any(
                        keyword in line.lower()
                        for keyword in [
                            "kapitel",
                            "chapter",
                            "section",
                            "abschnitt",
                            "teil",
                            "introduction",
                            "einführung",
                            "grundlagen",
                            "theory",
                            "praxis",
                            "methoden",
                            "algorithmen",
                            "anwendung",
                            "examples",
                            "beispiele",
                        ]
                    )
                ):
                    # Determine header level
                    level = 1
                    if line.isupper():
                        level = 1  # Main headers
                    elif any(
                        line.lower().startswith(word) for word in ["kapitel", "chapter"]
                    ):
                        level = 1  # Chapter headers
                    elif any(
                        word in line.lower()
                        for word in ["grundlagen", "theory", "introduction"]
                    ):
                        level = 2  # Section headers

                    structure["headers_found"].append(
                        {
                            "text": line,
                            "line_number": i,
                            "level": level,
                            "position": i / len(lines),  # Relative position in document
                        }
                    )

        # Group headers into chapters
        if structure["headers_found"]:
            # Simple chapter detection: group related headers
            chapters = []
            current_chapter = None

            for header in structure["headers_found"]:
                if header["level"] == 1 or not current_chapter:
                    # Start new chapter
                    if current_chapter:
                        chapters.append(current_chapter)

                    current_chapter = {
                        "title": header["text"],
                        "start_line": header["line_number"],
                        "headers": [header],
                        "level": header["level"],
                    }
                else:
                    # Add to current chapter
                    if current_chapter:
                        current_chapter["headers"].append(header)

            if current_chapter:
                chapters.append(current_chapter)

            structure["sections"] = chapters
            structure["estimated_chapters"] = len(chapters)

        # Fallback if no structure found
        if not structure["sections"]:
            # Create synthetic chapters based on content length
            content_length = len(content)
            num_chapters = max(
                3, min(12, content_length // 6000)
            )  # 1 chapter per ~6k chars

            for i in range(num_chapters):
                start_pos = i * content_length // num_chapters
                end_pos = (i + 1) * content_length // num_chapters

                # Extract a title from the content around this position
                section_content = content[start_pos:end_pos]
                section_lines = section_content.split("\n")[:10]  # First 10 lines

                # Find a good title line
                title = f"Chapter {i + 1}"
                for line in section_lines:
                    line = line.strip()
                    if 10 < len(line) < 80 and line[0].isupper():
                        title = line
                        break

                structure["sections"].append(
                    {
                        "title": title,
                        "start_line": start_pos,
                        "content_preview": section_content[:200] + "...",
                        "level": 1,
                    }
                )

            structure["estimated_chapters"] = num_chapters

        return structure

    def _classify_topics(self, content: str, structure: Dict[str, Any]) -> List[str]:
        """Classify document topics using multi-strategy approach."""
        # Placeholder for topic classification
        # This would use multiple strategies to identify topics
        return ["technical", "educational", "machine_learning"]

    def _create_hierarchical_chunks(
        self, content: str, structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create hierarchical chunks based on document structure."""
        # Placeholder for hierarchical chunking
        # This would create chapter-aware chunks with proper boundaries
        chunk_size = 1000
        chunks = []

        for i in range(0, len(content), chunk_size):
            chunk_text = content[i : i + chunk_size]
            chunks.append(
                {
                    "content": chunk_text,
                    "start_pos": i,
                    "end_pos": min(i + chunk_size, len(content)),
                    "chapter": "auto",
                    "hierarchy_level": 1,
                }
            )

        return chunks

    def _score_chunk_relevance(
        self, chunks: List[Dict[str, Any]], topics: List[str]
    ) -> List[Dict[str, Any]]:
        """Score chunk relevance using semantic and topic-aware methods."""
        # Placeholder for relevance scoring
        for chunk in chunks:
            chunk["relevance_score"] = 0.8  # Placeholder score
        return chunks

    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """Generate embeddings for chunks."""
        # Placeholder for embedding generation
        return [[0.1] * 768 for _ in chunks]  # Placeholder embeddings

    def _get_pdf_page_count(self, pdf_path: str) -> int:
        """Get the number of pages in a PDF."""
        try:
            import fitz

            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            return page_count
        except:
            return 1


def load_documents(data_dir="data"):
    """
    Load documents from the specified directory using Docling with enhanced OCR support.

    Supports multiple file formats: .txt, .pdf, .docx, .pptx, .xlsx, .html, .md, etc.
    Enhanced for scanned/image-based PDFs with OCR capabilities.

    Args:
        data_dir (str): Path to the directory containing documents

    Returns:
        list: List of Document objects with page_content and metadata
    """
    documents = []

    # Use Docling with default settings (it handles OCR automatically)
    doc_converter = DocumentConverter()

    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if not os.path.isfile(file_path):
            continue

        try:
            if file.endswith(".txt"):
                # Use LangChain TextLoader for simple text files
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            else:
                # Enhanced processing for PDFs with vision model priority
                if file.endswith(".pdf"):
                    text_content = ""
                    ocr_used = False
                    vision_used = False

                    # Check if this is a scanned PDF
                    is_scanned = is_scanned_pdf(file_path)
                    logger.info(f"PDF {file} detected as scanned: {is_scanned}")

                    # Vision model will be used if Docling gives poor results
                    vision_used = (
                        False  # Will be set to True if vision model is actually used
                    )

                    # Try Docling first for all PDFs
                    docling_text = ""
                    try:
                        result = doc_converter.convert(file_path)
                        docling_text = result.document.export_to_markdown()
                        text_content = docling_text
                        logger.info(
                            f"Docling extracted {len(docling_text)} characters from {file}"
                        )
                    except Exception as e:
                        logger.warning(f"Docling failed for {file}: {e}")

                    # For scanned PDFs, try vision model if Docling gave poor results
                    if is_scanned and VISION_MODEL_AVAILABLE:
                        # Check if Docling result is mostly image tags (poor extraction)
                        image_tag_ratio = (
                            docling_text.count("<!-- image -->") / len(docling_text)
                            if docling_text
                            else 1.0
                        )
                        substantial_text_ratio = (
                            len(
                                [
                                    line
                                    for line in docling_text.split("\n")
                                    if line.strip()
                                    and not line.strip().startswith("<!--")
                                ]
                            )
                            / len(docling_text.split("\n"))
                            if docling_text
                            else 0
                        )

                        logger.info(
                            f"Docling quality check for {file}: image_ratio={image_tag_ratio:.3f}, text_ratio={substantial_text_ratio:.3f}"
                        )

                        if (
                            image_tag_ratio > 0.05
                            or substantial_text_ratio < 0.2
                            or is_scanned
                        ):  # Poor Docling result or scanned PDF
                            logger.info(
                                f"Triggering vision model for {file}: scanned={is_scanned}, image_ratio={image_tag_ratio:.3f}, text_ratio={substantial_text_ratio:.3f}"
                            )
                            try:
                                advanced_processor = AdvancedDocumentProcessor()
                                vision_content = (
                                    advanced_processor._extract_with_vision_model(
                                        file_path
                                    )
                                )
                                logger.info(
                                    f"Vision model extracted {len(vision_content) if vision_content else 0} characters"
                                )

                                if (
                                    vision_content
                                    and len(vision_content) > len(docling_text) * 1.2
                                ):
                                    text_content = vision_content
                                    vision_used = True
                                    logger.info(
                                        f"Vision model improved extraction for {file}: {len(vision_content)} vs {len(docling_text)} chars"
                                    )
                                else:
                                    logger.info(
                                        f"Vision model did not improve extraction for {file}: {len(vision_content) if vision_content else 0} vs {len(docling_text)} chars"
                                    )
                            except Exception as e:
                                logger.warning(f"Vision model failed for {file}: {e}")
                                vision_used = False
                        else:
                            logger.info(
                                f"Docling result acceptable for {file}, skipping vision model"
                            )

                    # Final fallback to basic OCR if still no content
                    if not text_content and is_scanned:
                        try:
                            logger.info(f"Final fallback to basic OCR for {file}")
                            text_content = extract_text_with_ocr(file_path)
                            ocr_used = True
                            logger.info(
                                f"Basic OCR extracted {len(text_content)} characters from {file}"
                            )
                        except Exception as e:
                            logger.error(
                                f"All extraction methods failed for {file}: {e}"
                            )
                            text_content = (
                                "No content could be extracted from this document."
                            )

                    documents.append(
                        Document(
                            page_content=text_content,
                            metadata={
                                "source": file_path,
                                "file_type": "pdf",
                                "ocr_used": ocr_used,
                                "is_scanned": is_scanned_pdf(file_path)
                                if not ocr_used
                                else True,
                                "docling_metadata": result.document.origin.model_dump()
                                if "result" in locals()
                                and hasattr(result.document, "origin")
                                and result.document.origin
                                else {},
                            },
                        )
                    )
                else:
                    # Use Docling for all other formats
                    result = doc_converter.convert(file_path)
                    # Extract text content from Docling document
                    text_content = result.document.export_to_markdown()
                    documents.append(
                        Document(
                            page_content=text_content,
                            metadata={
                                "source": file_path,
                                "file_type": file.split(".")[-1],
                                "docling_metadata": result.document.origin.model_dump()
                                if hasattr(result.document, "origin")
                                and result.document.origin
                                else {},
                            },
                        )
                    )
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Fallback to basic text loading if Docling fails
            try:
                # Only try text fallback for text-based files, not binaries like PDFs
                if file_path.lower().endswith(
                    (".txt", ".md", ".py", ".json", ".xml", ".html")
                ):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={"source": file_path, "fallback": True},
                        )
                    )
                else:
                    print(f"Skipping {file_path} - not a text file and Docling failed")
            except Exception as e2:
                print(f"Failed to load {file_path} with fallback method: {e2}")

    return documents


# Individual load functions removed - now using Docling for unified document processing


def split_documents(file_paths, chunk_size=1000, chunk_overlap=200):
    """
    Load and split documents into smaller chunks for better retrieval performance.

    Args:
        file_paths (list): List of file paths to load and split
        chunk_size (int): Maximum size of each chunk in characters
        chunk_overlap (int): Number of characters to overlap between chunks

    Returns:
        str: The content of the first document (for backward compatibility)
    """
    if not file_paths:
        return ""

    # Load the first document
    file_path = file_paths[0]

    try:
        # Handle .txt files directly
        if file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            # Use Docling for other document formats
            doc_converter = DocumentConverter()
            result = doc_converter.convert(file_path)
            content = result.document.export_to_markdown()
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        # Fallback to basic text loading
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e2:
            print(f"Failed to load {file_path} with fallback: {e2}")
            return ""

    return content


def load_and_chunk_with_captions(data_dir="data", chunk_size=1000, chunk_overlap=200):
    """
    Load documents and split them using caption-aware chunking.

    This function uses the CaptionAwareProcessor to create chunks that preserve
    the relationship between captions and their associated content.

    Args:
        data_dir (str): Path to the directory containing documents
        chunk_size (int): Maximum size of each chunk in characters
        chunk_overlap (int): Number of characters to overlap between chunks

    Returns:
        list: List of Document objects with caption-aware chunking
    """
    from .caption_aware_processor import CaptionAwareProcessor

    documents = []
    processor = CaptionAwareProcessor(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Use Docling with default configuration
    doc_converter = DocumentConverter()

    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if not os.path.isfile(file_path):
            continue

        try:
            if file.endswith(".txt"):
                # Use LangChain TextLoader for simple text files
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            else:
                # Use Docling for all other formats
                result = doc_converter.convert(file_path)

                # Use caption-aware chunking
                chunks = processor.create_caption_centric_chunks(
                    result.document,
                    metadata={
                        "source": file_path,
                        "file_type": file.split(".")[-1],
                        "docling_metadata": result.document.origin.model_dump()
                        if hasattr(result.document, "origin") and result.document.origin
                        else {},
                    },
                )

                if chunks:
                    documents.extend(chunks)
                else:
                    # Fallback to regular markdown export if caption processing fails
                    text_content = result.document.export_to_markdown()
                    documents.append(
                        Document(
                            page_content=text_content,
                            metadata={
                                "source": file_path,
                                "file_type": file.split(".")[-1],
                                "chunking_method": "fallback",
                            },
                        )
                    )

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Fallback to basic text loading if Docling fails
            try:
                # Only try text fallback for text-based files, not binaries like PDFs
                if file_path.lower().endswith(
                    (".txt", ".md", ".py", ".json", ".xml", ".html")
                ):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={"source": file_path, "fallback": True},
                        )
                    )
                else:
                    print(f"Skipping {file_path} - not a text file and Docling failed")
            except Exception as e2:
                print(f"Failed to load {file_path} with fallback method: {e2}")

    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"Chunk {i + 1}: {chunk.page_content[:200]}...")
