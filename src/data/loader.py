import io
import logging
import os
import re
import warnings
from typing import Any, Dict, List

from docling.document_converter import DocumentConverter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from PIL import Image

logger = logging.getLogger(__name__)

# Suppress swig deprecation warnings from fitz library
warnings.filterwarnings("ignore", category=DeprecationWarning, module="importlib._bootstrap")

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract OCR not available. Install with: pip install pytesseract tesseract")

# Vision model setup for advanced image analysis
try:
    import ollama

    VISION_MODEL_AVAILABLE = True
    VISION_BACKEND = "ollama"
    logger.info("Ollama vision model available")
except ImportError:
    try:
        import transformers  # noqa: F401

        VISION_MODEL_AVAILABLE = True
        VISION_BACKEND = "transformers"
        logger.info("Transformers vision model dependencies available")
    except ImportError:
        VISION_MODEL_AVAILABLE = False
        VISION_BACKEND = None
        logger.warning("No vision model dependencies available. Install ollama or transformers")


def extract_text_with_ocr(pdf_path: str) -> str:  # noqa: C901
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
                pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))  # Higher scaling for better OCR
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                # OCR with multiple configurations optimized for German technical documents
                ocr_configs = [
                    (
                        "deu+eng",
                        "--psm 4 --oem 3",
                    ),  # Single column of text (better for technical docs)
                    (
                        "deu+eng",
                        "--psm 3 --oem 3",
                    ),  # Auto page segmentation with German+English
                    ("deu+eng", "--psm 6 --oem 3"),  # Uniform block of text
                    ("eng+deu", "--psm 4 --oem 3"),  # English first, single column
                ]

                # Enhanced character whitelist for technical German documents
                char_whitelist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜßabcdefghijklmnopqrstuvwxyzäöüß.,:;!?()-+*/=><[]{}@#$%&\\|_~`"
                enhanced_config = f"--psm 4 --oem 3 -c tessedit_char_whitelist={char_whitelist}"

                best_ocr_text = ""
                max_length = 0

                # Try enhanced configuration first
                try:
                    enhanced_text = pytesseract.image_to_string(
                        img, lang="deu+eng", config=enhanced_config
                    )
                    if len(enhanced_text.strip()) > 50:  # Minimum substantial content
                        best_ocr_text = enhanced_text
                        max_length = len(enhanced_text.strip())
                except Exception as e:
                    logger.debug(f"Enhanced OCR config failed: {e}")

                # Try multiple configurations if enhanced didn't work well
                for lang, config in ocr_configs:
                    try:
                        page_text = pytesseract.image_to_string(img, lang=lang, config=config)
                        if len(page_text.strip()) > max_length:
                            max_length = len(page_text.strip())
                            best_ocr_text = page_text
                    except Exception as config_error:
                        logger.debug(f"OCR config {lang} {config} failed: {config_error}")
                        continue

                # Combine existing text and OCR text
                combined_text = existing_text
                if best_ocr_text.strip():
                    if combined_text and not combined_text.endswith("\n"):
                        combined_text += "\n"
                    combined_text += best_ocr_text

                # Only add pages with substantial content
                if len(combined_text.strip()) > 20:  # Minimum content threshold
                    text_content.append(f"=== PAGE {page_num + 1} ===\n{combined_text}\n")
                    logger.debug(f"Processed page {page_num + 1}: {len(combined_text)} chars")

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

        logger.info(f"OCR processing {len(priority_pages)} priority pages out of {len(doc)} total")

        for page_num in priority_pages:
            try:
                page = doc.load_page(page_num)

                # Skip pages with too much text (likely already OCR'd by Docling)
                existing_text = page.get_text()
                if len(existing_text.strip()) > 500:  # Page already has substantial text
                    text_content.append(f"Page {page_num + 1} (existing text):\n{existing_text}\n")
                    continue

                # Get page as image for OCR
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better OCR
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                # Extract text with OCR
                page_text = pytesseract.image_to_string(img, lang="deu+eng")  # German + English

                if page_text.strip():  # Only add if OCR found text
                    text_content.append(f"Page {page_num + 1} (OCR):\n{page_text}\n")
                    logger.debug(f"OCR extracted {len(page_text)} chars from page {page_num + 1}")

            except Exception as page_error:
                logger.warning(f"OCR failed on page {page_num + 1}: {page_error}")
                continue

        doc.close()

        full_text = "\n".join(text_content)
        logger.info(f"OCR completed: {len(full_text)} characters from {len(priority_pages)} pages")
        return full_text

    except Exception as e:
        logger.error(f"OCR processing failed for {pdf_path}: {e}")
        return ""

    try:
        import fitz
        from PIL import Image

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
            page_text = pytesseract.image_to_string(img, lang="deu+eng")  # German + English
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
            pages_to_check.append(min(5, total_pages - 1))  # Page 6 or last page if shorter
        if total_pages > 10:
            pages_to_check.append(min(10, total_pages - 1))  # Page 11 or last page if shorter

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
    Advanced document processor with DeepSeek-OCR and multi-stage AI analysis.

    Pipeline:
    1. Document Input
    2. Docling Parser (baseline extraction)
    3. Quality Check → DeepSeek-OCR Fallback [if needed]
    4. Structure Analysis (phi3.5:3.8b for hierarchy + topics)
    5. Topic Classification (multi-strategy approach)
    6. Hierarchical Chunking (chapter-aware, token-based)
    7. Relevance Scoring (semantic + topic-aware)
    8. Embedding Generation (embeddinggemma:latest via Ollama)
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
        docling_content, docling_structure = self._extract_with_docling(pdf_path)
        results["extracted_content"] = docling_content
        if docling_structure:
            results["docling_structure"] = docling_structure

        # Stage 2: Quality check and vision fallback
        quality_score = self._assess_content_quality(docling_content, pdf_path)
        results["quality_metrics"]["baseline_quality"] = quality_score

        if quality_score < 0.7:  # Poor quality, use vision fallback
            results["processing_stages"].append("deepseek_ocr_processing")
            vision_content = self._extract_with_vision_model(pdf_path)
            if len(vision_content) > len(docling_content) * 1.2:
                results["extracted_content"] = vision_content
                results["processing_stages"].append("vision_improved")

        # Stage 3: Structure analysis
        results["processing_stages"].append("structure_analysis")

        # Use Docling structure if available, otherwise analyze content
        if "docling_structure" in results and results["docling_structure"].get("hierarchy"):
            structure_info = results["docling_structure"]
            logger.info(
                f"Using Docling structure: {len(structure_info.get('hierarchy', []))} items"
            )
        else:
            structure_info = self._analyze_document_structure(results["extracted_content"])
            logger.info(
                f"Using heuristic structure analysis: {len(structure_info.get('hierarchy', []))} items"
            )

        results["structure_analysis"] = structure_info
        results["chapters_detected"] = structure_info.get("estimated_chapters", 0)
        results["chapters"] = structure_info.get("sections", [])  # For compatibility

        # Add language detection
        try:
            from langdetect import detect

            detected_lang = detect(results["extracted_content"][:1000])  # Sample first 1000 chars
            results["language"] = detected_lang
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            results["language"] = "de"  # Default to German for technical docs

        # Stage 4: Topic classification
        results["processing_stages"].append("topic_classification")
        topics = self._classify_topics(results["extracted_content"], structure_info)
        results["topics"] = topics

        # Stage 5: Hierarchical chunking
        results["processing_stages"].append("hierarchical_chunking")
        from src.ai.pipeline.hierarchical_chunker import HierarchicalChunker

        # Clean and prepare content for chunking
        extracted_content = results["extracted_content"]

        # Clean OCR artifacts from vision content before chunking
        if "deepseek_ocr_processing" in results.get("processing_stages", []):
            logger.info("Cleaning vision OCR content before chunking")
            extracted_content = self._clean_vision_ocr_content(extracted_content)

        logger.info(f"Chunking content length: {len(extracted_content)}")
        logger.info(f"Content preview: {extracted_content[:200]}...")

        chunker = HierarchicalChunker()
        chunks = chunker.chunk_document(extracted_content, structure_info, pdf_path)
        results["chunks"] = chunks
        results["chunks_created"] = len(chunks)

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

    def _traverse_docling_tree(self, node, structure: Dict[str, Any], level: int, path: str):
        """Recursively traverse Docling document tree to extract structure."""
        try:
            if hasattr(node, "children"):
                section_count = 1
                for child in node.children:
                    if hasattr(child, "tag") and child.tag in [
                        "h1",
                        "h2",
                        "h3",
                        "section",
                        "chapter",
                    ]:
                        # Extract header text
                        header_text = ""
                        if hasattr(child, "children"):
                            for content_child in child.children:
                                if hasattr(content_child, "text"):
                                    header_text += content_child.text + " "
                                elif hasattr(content_child, "content"):
                                    header_text += str(content_child.content) + " "

                        header_text = header_text.strip()
                        if header_text:
                            structure["hierarchy"].append(
                                {
                                    "level": level,
                                    "path": (
                                        f"{path}.{section_count}"
                                        if level > 1
                                        else str(section_count)
                                    ),
                                    "title": header_text,
                                    "content_preview": "",
                                    "word_count": len(header_text.split()),
                                    "type": "chapter" if level == 1 else "section",
                                }
                            )
                            section_count += 1

                    # Recurse deeper
                    if level < 3:  # Limit depth
                        self._traverse_docling_tree(
                            child, structure, level + 1, f"{path}.{section_count - 1}"
                        )

        except Exception as e:
            logger.debug(f"Error traversing Docling tree: {e}")

    def _assess_content_quality(self, content: str, pdf_path: str) -> float:
        """Assess the quality of extracted content."""
        if not content:
            return 0.0

        # Quality metrics
        content_length = len(content)
        avg_words_per_page = content_length / max(1, self._get_pdf_page_count(pdf_path))

        # Check for structured elements
        has_headers = "##" in content or "#" in content
        has_lists = "- " in content or "* " in content or content.count("\n") > content_length * 0.1

        # Calculate quality score
        quality_score = min(
            1.0,
            (
                min(1.0, content_length / 10000) * 0.4  # Content length (up to 10k chars = 0.4)
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
            import base64

            import fitz
            from PIL import Image

            doc = fitz.open(pdf_path)
            vision_content = []

            logger.info(f"DeepSeek-OCR processing for {len(doc)} pages using {VISION_BACKEND}")

            # Process ALL pages for comprehensive content extraction
            # This ensures complete document coverage instead of selective sampling
            all_pages = list(range(len(doc)))

            logger.info(
                f"DeepSeek-OCR processing ALL {len(all_pages)} pages out of {len(doc)} total"
            )

            # Process pages in batches to manage memory and provide progress updates
            batch_size = 10  # Process 10 pages at a time
            total_pages = len(all_pages)

            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                batch_pages = all_pages[batch_start:batch_end]

                logger.info(
                    f"Processing page batch {batch_start // batch_size + 1}/{(total_pages + batch_size - 1) // batch_size} "
                    f"(pages {batch_start + 1}-{batch_end} of {total_pages})"
                )

                # Process pages in this batch
                for page_num in batch_pages:
                    try:
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))

                        # Convert to base64 for Ollama (JPEG format for DeepSeek-OCR compatibility)
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        jpeg_buffer = io.BytesIO()
                        img.save(jpeg_buffer, format="JPEG", quality=95)
                        img_base64 = base64.b64encode(jpeg_buffer.getvalue()).decode("utf-8")

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
                f"DeepSeek-OCR extracted {len(full_content)} characters from {len(vision_content)} pages"
            )
            logger.debug(f"Vision content preview: {full_content[:300]}...")
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
                response = self._query_ollama_vision(
                    img_data["image"], page_num, img_data["image_obj"]
                )
                if response:
                    return f"=== PAGE {page_num} ===\n{response}"

            elif VISION_BACKEND == "transformers":
                # Use transformers vision model (fallback)
                response = self._query_transformers_vision(img_data["image_obj"], page_num)
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
            logger.warning(f"OCR fallback also failed for page {img_data['page_num']}: {ocr_error}")

        return ""

    def _query_ollama_vision(
        self, image_base64: str, page_num: int, image_obj: Image.Image = None
    ) -> str:
        """Query DeepSeek-OCR model for German technical document analysis."""
        try:
            import ollama  # noqa: F401

            # Temporarily use Tesseract while DeepSeek-OCR image format issues are resolved
            if image_obj is not None:
                try:
                    import pytesseract

                    # Enhanced OCR with improved configuration for German technical text
                    char_whitelist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜßabcdefghijklmnopqrstuvwxyzäöüß.,:;!?()-+*/=><[]{}@#$%&\\|_~`"
                    custom_config = f"--psm 3 --oem 3 -c tessedit_char_whitelist={char_whitelist}"
                    text = pytesseract.image_to_string(
                        image_obj,
                        lang="deu+eng",
                        config=custom_config,
                    )

                    # Clean up common OCR errors
                    text = text.replace("|", "I")  # Common OCR error
                    text = text.replace("§", "S")  # Common OCR error

                    if text.strip():
                        return f"=== PAGE {page_num} (Tesseract OCR) ===\n{text.strip()}"
                    else:
                        return f"=== PAGE {page_num} (No Text Found) ==="
                except Exception as e:
                    logger.error(f"Tesseract OCR failed for page {page_num}: {e}")
                    return f"=== PAGE {page_num} (OCR Failed) ==="
            else:
                return f"=== PAGE {page_num} (OCR Fallback - No Image) ==="

            # TODO: Re-enable DeepSeek-OCR once image format issues are fixed
            # The code below is commented out until image format issues are resolved
            # response = ollama.chat(model="deepseek-ocr:latest", ...)
            # return processed content

        except Exception as e:
            logger.error(f"Ollama vision query failed for page {page_num}: {e}")
            return ""

    def _process_page_with_ocr(self, page_num: int, image_obj: Image.Image = None) -> str:
        """
        Process a single page using OCR with multiple fallback mechanisms and quality validation.
        """
        if image_obj is None:
            return f"=== PAGE {page_num} (No Image Available) ==="

        ocr_methods = [
            ("deepseek_ocr", self._try_deepseek_ocr),
            ("tesseract_enhanced", self._try_tesseract_enhanced),
            ("tesseract_basic", self._try_tesseract_basic),
            ("easyocr_fallback", self._try_easyocr_fallback),
        ]

        for method_name, ocr_function in ocr_methods:
            try:
                logger.debug(f"Trying OCR method: {method_name} for page {page_num}")
                result = ocr_function(image_obj, page_num)

                if result and self._validate_ocr_result(result):
                    logger.info(f"OCR method {method_name} succeeded for page {page_num}")
                    return result
                else:
                    logger.warning(
                        f"OCR method {method_name} produced invalid results for page {page_num}"
                    )

            except Exception as e:
                logger.warning(f"OCR method {method_name} failed for page {page_num}: {e}")
                continue

        # All OCR methods failed
        logger.error(f"All OCR methods failed for page {page_num}")
        return f"=== PAGE {page_num} (All OCR Methods Failed) ==="

    def _try_deepseek_ocr(self, image_obj: Image.Image, page_num: int) -> str:
        """Try DeepSeek-OCR via Ollama"""
        try:
            response = ollama.chat(
                model="deepseek-ocr:latest",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Extract all text from this image. Focus on technical content, preserve mathematical notation, and maintain document structure. Output clean, readable text without commentary.

INPUT: Image of page {page_num} from a technical document.

OUTPUT: Clean, structured text with proper German formatting. Preserve all technical content, mathematical notation, and document structure. Do not add commentary or explanations.""",
                        "images": [self._image_to_base64(image_obj)],
                    }
                ],
            )

            if response and "message" in response and "content" in response["message"]:
                content = response["message"]["content"].strip()
                if content:
                    return f"=== PAGE {page_num} ===\n{content}"

        except Exception as e:
            logger.debug(f"DeepSeek-OCR failed: {e}")

        return ""

    def _try_tesseract_enhanced(self, image_obj: Image.Image, page_num: int) -> str:
        """Try enhanced Tesseract OCR with German technical configuration"""
        try:
            import pytesseract

            # Enhanced OCR with improved configuration for German technical text
            char_whitelist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜßabcdefghijklmnopqrstuvwxyzäöüß.,:;!?()-+*/=><[]{}@#$%&\\|_~`"
            custom_config = f"--psm 3 --oem 3 -c tessedit_char_whitelist={char_whitelist}"
            text = pytesseract.image_to_string(
                image_obj,
                lang="deu+eng",
                config=custom_config,
            )

            # Clean up common OCR errors
            text = text.replace("|", "I")  # Common OCR error
            text = text.replace("§", "S")  # Common OCR error
            text = text.replace("©", "C")  # Common OCR error

            if text.strip():
                return f"=== PAGE {page_num} (Tesseract Enhanced) ===\n{text.strip()}"

        except Exception as e:
            logger.debug(f"Enhanced Tesseract OCR failed: {e}")

        return ""

    def _try_tesseract_basic(self, image_obj: Image.Image, page_num: int) -> str:
        """Try basic Tesseract OCR"""
        try:
            import pytesseract

            text = pytesseract.image_to_string(
                image_obj,
                lang="deu+eng",
            )

            if text.strip():
                return f"=== PAGE {page_num} (Tesseract Basic) ===\n{text.strip()}"

        except Exception as e:
            logger.debug(f"Basic Tesseract OCR failed: {e}")

        return ""

    def _try_easyocr_fallback(self, image_obj: Image.Image, page_num: int) -> str:
        """Try EasyOCR as final fallback"""
        try:
            import easyocr

            # Initialize reader (this can be cached for performance)
            reader = easyocr.Reader(["de", "en"])
            results = reader.readtext(image_obj)

            # Extract text from results
            text_parts = []
            for bbox, text, confidence in results:
                if confidence > 0.5:  # Only include reasonably confident results
                    text_parts.append(text)

            if text_parts:
                full_text = " ".join(text_parts)
                return f"=== PAGE {page_num} (EasyOCR) ===\n{full_text}"

        except Exception as e:
            logger.debug(f"EasyOCR failed: {e}")

        return ""

    def _validate_ocr_result(self, result: str) -> bool:
        """
        Validate OCR result quality.
        """
        if not result or len(result.strip()) < 10:
            return False

        # Check for excessive OCR artifacts
        artifact_indicators = ["=== page", "tesseract", "ocr failed", "no text found"]
        result_lower = result.lower()

        for indicator in artifact_indicators:
            if indicator in result_lower:
                # Allow some indicators but not too many
                if result_lower.count(indicator) > 2:
                    return False

        return True

    def _extract_with_docling(self, pdf_path: str) -> tuple[str, dict]:
        """Extract content and structure using Docling parser."""
        try:
            from docling.document_converter import DocumentConverter

            # Try with default settings first (Docling handles OCR automatically)
            doc_converter = DocumentConverter()
            result = doc_converter.convert(pdf_path)
            content = result.document.export_to_markdown()

            # Extract structure from Docling document
            structure = self._extract_docling_structure(result.document)

            return content, structure
        except Exception as e:
            logger.error(f"Docling extraction failed: {e}")
            return "", {}

    def _extract_docling_structure(self, docling_document) -> Dict[str, Any]:
        """Extract hierarchical structure from Docling document."""
        try:
            from docling.datamodel import SectionHeaderItem, TextItem, TitleItem

            structure = {
                "hierarchy": [],
                "sections": [],
                "estimated_chapters": 0,
                "headers_found": [],
                "content_sections": [],
            }

            # Extract hierarchical structure using iterate_items
            hierarchy = []
            headers_found = []
            level_counters = [0] * 7  # Index 0 unused, 1-6 for header levels

            # Track seen titles to avoid duplicates and prefer complete titles
            seen_titles = set()

            for item, level in docling_document.iterate_items():
                if isinstance(item, (SectionHeaderItem, TitleItem)):
                    # This is a header/section title
                    title = item.text.strip() if hasattr(item, "text") else str(item)
                    if title and len(title) < 100:  # Skip very long titles
                        # Skip titles that look like TOC entries (incomplete sentences)
                        if (
                            len(title) < 10
                            or title.endswith(",")
                            or title.endswith(" and")
                            or title.endswith(" to")
                            or title.endswith(" with")
                        ):
                            continue  # Skip likely TOC entries

                        # Skip if we already have a similar but longer title (prefer complete titles)
                        similar_found = False
                        for seen_title in seen_titles:
                            if title in seen_title and len(seen_title) > len(title) + 5:
                                similar_found = True
                                break
                        if similar_found:
                            continue

                        # Generate proper hierarchical path
                        if level <= 6:  # Ensure level is within bounds
                            level_counters[level] += 1
                            # Reset counters for deeper levels
                            for deeper_level in range(level + 1, 7):
                                level_counters[deeper_level] = 0

                            # Build path from level 1 to current level
                            path_parts = []
                            for lvl in range(1, level + 1):
                                path_parts.append(str(level_counters[lvl]))
                            path = ".".join(path_parts)
                        else:
                            path = title[:50]  # Fallback for invalid levels

                        chapter_data = {
                            "title": title,
                            "level": level,
                            "path": path,
                            "word_count": len(title.split()),
                            "content_preview": title,
                        }
                        hierarchy.append(chapter_data)
                        headers_found.append(title)
                        seen_titles.add(title)

                elif isinstance(item, TextItem):
                    # Check if this text item contains a chapter header that Docling missed
                    text_content = item.text.strip() if hasattr(item, "text") else ""
                    if text_content:
                        # Check for "Chapter X" patterns that might not be recognized as headers
                        chapter_match = re.match(
                            r"^(?:#+\s*)?Chapter\s+(\d+)(?:\s+(.+))?$",
                            text_content,
                            re.IGNORECASE,
                        )
                        if chapter_match:
                            # This is a chapter header - treat it as level 1
                            chapter_num = int(chapter_match.group(1))
                            chapter_title = (
                                chapter_match.group(2).strip()
                                if chapter_match.group(2)
                                else f"Chapter {chapter_num}"
                            )
                            full_title = f"Chapter {chapter_num}" + (
                                f" {chapter_title}"
                                if chapter_title != f"Chapter {chapter_num}"
                                else ""
                            )

                            # Skip if we already have this chapter
                            if full_title not in seen_titles:
                                # Use chapter number as the path for main chapters
                                path = str(chapter_num)

                                chapter_data = {
                                    "title": full_title,
                                    "level": 1,
                                    "path": path,
                                    "word_count": len(full_title.split()),
                                    "content_preview": full_title,
                                }
                                hierarchy.append(chapter_data)
                                headers_found.append(full_title)
                                seen_titles.add(full_title)
                                logger.debug(f"Detected missed chapter header: {full_title}")
                        else:
                            # This is regular content text - could be associated with the last header
                            if hierarchy and not hierarchy[-1].get("content_preview"):
                                # If the last header doesn't have content, add this as preview
                                content_preview = text_content[:200]
                                if content_preview:
                                    hierarchy[-1]["content_preview"] = content_preview

            structure["hierarchy"] = hierarchy
            structure["sections"] = hierarchy  # For compatibility
            structure["estimated_chapters"] = len(hierarchy)
            structure["headers_found"] = headers_found[:10]  # Limit to first 10

            logger.info(f"Extracted {len(hierarchy)} hierarchical items from Docling document")

            return structure

        except Exception as e:
            logger.warning(f"Failed to extract Docling structure: {e}")
            import traceback

            logger.warning(traceback.format_exc())

            # Fallback: try to extract structure from markdown content if available
            try:
                if hasattr(docling_document, "export_to_markdown"):
                    markdown_content = docling_document.export_to_markdown()
                    return self._extract_structure_from_markdown(markdown_content)
            except Exception as fallback_e:
                logger.warning(f"Fallback structure extraction also failed: {fallback_e}")

            return {
                "hierarchy": [],
                "sections": [],
                "estimated_chapters": 1,
                "headers_found": [],
                "content_sections": [],
            }

        except Exception as e:
            logger.warning(f"Failed to extract Docling structure: {e}")
            return {
                "hierarchy": [],
                "sections": [],
                "estimated_chapters": 1,
                "headers_found": [],
                "content_sections": [],
            }

    def _extract_structure_from_markdown(self, markdown_content: str) -> Dict[str, Any]:
        """Extract hierarchical structure from markdown content as fallback."""
        import re

        structure = {
            "hierarchy": [],
            "sections": [],
            "estimated_chapters": 0,
            "headers_found": [],
            "content_sections": [],
        }

        # Find all headers in markdown
        headers = []
        level_counters = [0] * 7  # Index 0 unused, 1-6 for header levels

        # Find all headers in markdown - handle both regular headers and "Chapter X" patterns
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        all_matches = []

        for match in header_pattern.finditer(markdown_content):
            level = len(match.group(1))
            title = match.group(2).strip()

            if title:
                # Check if this is a "Chapter X" style header
                chapter_match = re.match(r"Chapter\s+(\d+)(?:\s+(.+))?", title, re.IGNORECASE)
                if chapter_match:
                    # Extract chapter number and title
                    chapter_num = int(chapter_match.group(1))
                    chapter_title = (
                        chapter_match.group(2).strip()
                        if chapter_match.group(2)
                        else f"Chapter {chapter_num}"
                    )
                    # Use the chapter number as the path, not the header level
                    all_matches.append(
                        (match.start(), chapter_num, chapter_title, True)
                    )  # True = is chapter header
                else:
                    # Regular header
                    all_matches.append((match.start(), level, title, False))

        # Sort by position in document and process
        all_matches.sort(key=lambda x: x[0])

        for _, level, title in all_matches:
            if title:
                # Initialize counters if needed
                for lvl in range(1, level + 1):
                    if lvl not in level_counters:
                        level_counters[lvl] = 0

                # Increment current level counter
                level_counters[level] += 1

                # Reset counters for deeper levels when we encounter a new header at current level
                for deeper_level in range(level + 1, 7):
                    level_counters[deeper_level] = 0

                # Build path from level 1 to current level
                path_parts = []
                for lvl in range(1, level + 1):
                    path_parts.append(str(level_counters[lvl]))
                path = ".".join(path_parts)

                headers.append(
                    {
                        "title": title,
                        "level": level,
                        "path": path,
                        "word_count": len(title.split()),
                        "content_preview": title,
                    }
                )

        structure["hierarchy"] = headers
        structure["sections"] = headers
        structure["estimated_chapters"] = len(headers)
        structure["headers_found"] = [h["title"] for h in headers[:10]]

        logger.info(f"Extracted {len(headers)} headers from markdown fallback")

        return structure

    def _clean_vision_ocr_content(self, content: str) -> str:
        """
        Clean vision OCR content by removing page markers and artifacts.
        """
        if not content:
            return content

        # Remove page markers and OCR artifacts
        cleaned = re.sub(r"=== PAGE \d+ ===", "", content, flags=re.IGNORECASE)
        cleaned = re.sub(r"=== PAGE \d+ \(.*?\) ===", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"=== .*? ===", "", cleaned, flags=re.IGNORECASE)

        # Remove excessive whitespace and normalize
        cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)  # Multiple newlines to double
        cleaned = re.sub(r"\s+", " ", cleaned)  # Multiple spaces to single

        # Remove lines that are mostly OCR artifacts
        lines = cleaned.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip lines that are just page numbers or minimal content
            if re.match(r"^\d+$", line) and len(line) < 5:
                continue

            # Skip lines with excessive special characters
            special_chars = sum(1 for c in line if not c.isalnum() and not c.isspace())
            if len(line) > 0 and special_chars / len(line) > 0.5:
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

        # Check for meaningful content (should have some letters)
        letters = sum(1 for c in result if c.isalpha())
        if letters < 5:  # Too few letters
            return False

        # Check for reasonable text structure
        lines = result.split("\n")
        meaningful_lines = sum(1 for line in lines if len(line.strip()) > 10)

        return meaningful_lines >= 1  # At least one meaningful line

    def _tesseract_ocr_fallback(self, image: Image.Image, page_num: int) -> str:
        """Fallback to Tesseract OCR for reliable text extraction"""
        try:
            import pytesseract

            # Extract text with German + English support
            text = pytesseract.image_to_string(
                image,
                lang="deu+eng",
                config="--psm 6",  # Uniform block of text
            )

            if text.strip():
                return f"=== PAGE {page_num} (Tesseract OCR) ===\n{text.strip()}"
            else:
                return f"=== PAGE {page_num} (No Text Found) ==="

        except Exception as e:
            logger.error(f"Tesseract OCR fallback failed for page {page_num}: {e}")
            return f"=== PAGE {page_num} (OCR Failed) ==="

    def _query_transformers_vision(self, image: Image.Image, page_num: int) -> str:
        """Query transformers vision model (fallback)."""
        try:
            # Placeholder for transformers implementation
            logger.info(f"Transformers vision processing for page {page_num} (placeholder)")
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

        # Look for headers and structure patterns (improved for OCR text)
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 3:  # Skip very short lines
                continue

            # Detect potential headers (more flexible for OCR text)
            if len(line) < 200:  # Reasonable header length
                # Look for chapter/section indicators
                is_header = False
                level = 2  # Default to section level

                # Debug each line
                logger.debug(f"Checking line {i}: '{line}' (len={len(line)})")

                # Check for markdown headers (# Chapter 1, ## Overview, etc.)
                if line.startswith("# "):
                    is_header = True
                    level = 1  # Chapter level
                    logger.debug(f"  -> Matched markdown header level 1 (level {level})")
                elif line.startswith("## "):
                    is_header = True
                    level = 2  # Section level
                    logger.debug(f"  -> Matched markdown header level 2 (level {level})")
                elif line.startswith("### "):
                    is_header = True
                    level = 3  # Subsection level
                    logger.debug(f"  -> Matched markdown header level 3 (level {level})")

                    # Only detect explicit markdown headers to avoid false positives

                if is_header:
                    logger.debug(f"Found header: '{line}' (level {level})")
                    structure["headers_found"].append(
                        {
                            "text": line,
                            "line_number": i,
                            "level": level,
                            "position": i / len(lines),  # Relative position in document
                        }
                    )
                else:
                    logger.debug("  -> Not a header")

        # Group headers into chapters
        if structure["headers_found"]:
            # Improved chapter detection for numbered sections
            chapters = []
            current_chapter = None
            current_chapter_num = None

            for header in structure["headers_found"]:
                # Extract chapter number from header (e.g., "1.1" -> 1, "2.6.3" -> 2)
                # Simple extraction of first digit
                first_digit = None
                for char in header["text"]:
                    if char.isdigit():
                        first_digit = int(char)
                        break

                if first_digit is not None:
                    chapter_num = first_digit

                    if chapter_num != current_chapter_num:
                        # Start new chapter
                        if current_chapter:
                            chapters.append(current_chapter)

                        current_chapter = {
                            "title": header["text"],
                            "start_line": header["line_number"],
                            "headers": [header],
                            "level": 1,  # All main sections are level 1
                            "chapter_number": chapter_num,
                        }
                        current_chapter_num = chapter_num
                    else:
                        # Add to current chapter (same chapter number)
                        if current_chapter:
                            current_chapter["headers"].append(header)
                            # Update title to the first header in this chapter for consistency
                            # Keep the original title (first header encountered)

            if current_chapter:
                chapters.append(current_chapter)

            structure["hierarchy"] = chapters
            structure["sections"] = chapters  # Keep both for compatibility
            structure["estimated_chapters"] = len(chapters)

        # Fallback if no structure found
        if not structure["sections"]:
            # Create synthetic chapters based on content length
            content_length = len(content)
            num_chapters = max(3, min(12, content_length // 6000))  # 1 chapter per ~6k chars

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
        if not content or not content.strip():
            logger.warning("No content to chunk")
            return []

        chunk_size = 1000
        overlap = 200
        chunks = []

        # Simple chunking for now - split by page markers if present
        if "===" in content:
            # Split by page markers
            pages = content.split("===")
            current_chunk = ""

            for page in pages:
                if not page.strip():
                    continue

                # Check if adding this page would exceed chunk size
                if len(current_chunk) + len(page) > chunk_size and current_chunk:
                    # Create chunk
                    chunks.append(
                        {
                            "content": current_chunk.strip(),
                            "chapter": "auto",
                            "hierarchy_level": 1,
                            "chunk_type": "content",
                            "metadata": {"word_count": len(current_chunk.split())},
                        }
                    )
                    current_chunk = page
                else:
                    current_chunk += " " + page if current_chunk else page

            # Add final chunk
            if current_chunk.strip():
                chunks.append(
                    {
                        "content": current_chunk.strip(),
                        "chapter": "auto",
                        "hierarchy_level": 1,
                        "chunk_type": "content",
                        "metadata": {"word_count": len(current_chunk.split())},
                    }
                )
        else:
            # Fallback: simple character-based chunking
            for i in range(0, len(content), chunk_size - overlap):
                chunk_text = content[i : i + chunk_size]
                if chunk_text.strip():  # Only add non-empty chunks
                    chunks.append(
                        {
                            "content": chunk_text.strip(),
                            "chapter": "auto",
                            "hierarchy_level": 1,
                            "chunk_type": "content",
                            "metadata": {"word_count": len(chunk_text.split())},
                        }
                    )

        logger.info(f"Created {len(chunks)} chunks from {len(content)} characters")
        return chunks

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
        except (IOError, OSError, Exception):
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
                    vision_used = False  # Will be set to True if vision model is actually used

                    # Try Docling first for all PDFs
                    docling_text = ""
                    try:
                        result = doc_converter.convert(file_path)
                        docling_text = result.document.export_to_markdown()
                        text_content = docling_text
                        logger.info(f"Docling extracted {len(docling_text)} characters from {file}")
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
                                    if line.strip() and not line.strip().startswith("<!--")
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
                            image_tag_ratio > 0.05 or substantial_text_ratio < 0.2 or is_scanned
                        ):  # Poor Docling result or scanned PDF
                            logger.info(
                                f"Triggering vision model for {file}: scanned={is_scanned}, image_ratio={image_tag_ratio:.3f}, text_ratio={substantial_text_ratio:.3f}"
                            )
                            try:
                                advanced_processor = AdvancedDocumentProcessor()
                                vision_content = advanced_processor._extract_with_vision_model(
                                    file_path
                                )
                                logger.info(
                                    f"Vision model extracted {len(vision_content) if vision_content else 0} characters"
                                )

                                if vision_content and len(vision_content) > len(docling_text) * 1.2:
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
                            logger.error(f"All extraction methods failed for {file}: {e}")
                            text_content = "No content could be extracted from this document."

                    documents.append(
                        Document(
                            page_content=text_content,
                            metadata={
                                "source": file_path,
                                "file_type": "pdf",
                                "ocr_used": ocr_used,
                                "is_scanned": is_scanned_pdf(file_path) if not ocr_used else True,
                                "docling_metadata": (
                                    result.document.origin.model_dump()
                                    if "result" in locals()
                                    and hasattr(result.document, "origin")
                                    and result.document.origin
                                    else {}
                                ),
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
                                "docling_metadata": (
                                    result.document.origin.model_dump()
                                    if hasattr(result.document, "origin") and result.document.origin
                                    else {}
                                ),
                            },
                        )
                    )
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Fallback to basic text loading if Docling fails
            try:
                # Only try text fallback for text-based files, not binaries like PDFs
                if file_path.lower().endswith((".txt", ".md", ".py", ".json", ".xml", ".html")):
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


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"Chunk {i + 1}: {chunk.page_content[:200]}...")
