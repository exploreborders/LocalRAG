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
    from transformers import (
        Qwen2VLForConditionalGeneration,
        AutoTokenizer,
        AutoProcessor,
    )
    import torch

    VISION_MODEL_AVAILABLE = True
    logger.info("Vision model dependencies available")
except ImportError:
    VISION_MODEL_AVAILABLE = False
    logger.warning(
        "Vision model dependencies not available. Install with: pip install transformers torch"
    )


def extract_text_with_ocr(pdf_path: str) -> str:
    """
    Extract text from scanned PDF using OCR as fallback.
    Focuses on pages likely to contain table of contents and chapter information.

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
        text_content = []

        # For scanned PDFs, process comprehensively
        # Process first 50 pages (TOC and main content usually here)
        priority_pages = list(range(min(50, len(doc))))

        # Add every 10th page for broader coverage
        step = max(10, len(doc) // 20)
        for page_num in range(50, len(doc), step):
            priority_pages.append(page_num)

        # Add last 5 pages
        priority_pages.extend(range(max(0, len(doc) - 5), len(doc)))

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

        # Check multiple pages throughout the document
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

        logger.info(
            f"PDF {pdf_path}: {len(total_text.strip())} chars across {pages_checked} pages, ratio={text_ratio:.4f}, scanned={is_scanned}"
        )
        return is_scanned

    except Exception as e:
        logger.error(f"Failed to analyze PDF {pdf_path}: {e}")
        return False


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

    # Configure Docling with enhanced OCR for scanned PDFs
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # Enable OCR for scanned documents
    pipeline_options.do_table_structure = True  # Extract tables

    # Use multiple backends for better OCR support
    backends = [
        DoclingParseV2DocumentBackend,  # Advanced parsing with OCR
        PyPdfiumDocumentBackend,  # Fallback for general PDFs
    ]

    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: pipeline_options}
    )

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
                # Enhanced processing for PDFs with OCR fallback
                if file.endswith(".pdf"):
                    text_content = ""
                    ocr_used = False
                    docling_success = False
                    docling_text = ""

                    # First try Docling
                    try:
                        result = doc_converter.convert(file_path)
                        docling_text = result.document.export_to_markdown()
                        docling_success = True
                        logger.info(
                            f"Docling extracted {len(docling_text)} characters from {file}"
                        )
                    except Exception as e:
                        logger.warning(f"Docling failed for {file}: {e}")
                        docling_text = ""

                    # Always try OCR for PDFs to ensure comprehensive extraction
                    ocr_text = ""
                    try:
                        logger.info(f"Attempting OCR extraction for {file}")
                        ocr_text = extract_text_with_ocr(file_path)
                        logger.info(
                            f"OCR extracted {len(ocr_text)} characters from {file}"
                        )
                    except Exception as e:
                        logger.error(f"OCR failed for {file}: {e}")

                    # Choose the best extraction method
                    if (
                        ocr_text
                        and len(ocr_text.strip()) > len(docling_text.strip()) * 1.5
                    ):
                        # OCR has significantly more content
                        text_content = ocr_text
                        ocr_used = True
                        logger.info(f"Using OCR extraction (better content) for {file}")
                    elif docling_success and len(docling_text.strip()) > 100:
                        # Docling worked and has decent content
                        text_content = docling_text
                        ocr_used = False
                        logger.info(f"Using Docling extraction for {file}")
                    elif ocr_text:
                        # Fallback to OCR
                        text_content = ocr_text
                        ocr_used = True
                        logger.info(f"Using OCR extraction (fallback) for {file}")
                    else:
                        # Nothing worked
                        text_content = (
                            docling_text
                            or "No content could be extracted from this document."
                        )
                        logger.warning(f"No usable content extracted from {file}")

                    # Final check - if content is too short for a substantial PDF, force OCR
                    if len(text_content.strip()) < 500 and is_scanned_pdf(file_path):
                        logger.info(f"Forcing OCR for suspected scanned PDF {file}")
                        ocr_text = extract_text_with_ocr(file_path)
                        if ocr_text and len(ocr_text.strip()) > len(
                            text_content.strip()
                        ):
                            text_content = ocr_text
                            ocr_used = True
                            logger.info(f"OCR improved text extraction for {file}")

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
