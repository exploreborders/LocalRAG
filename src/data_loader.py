from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
import os

def load_documents(data_dir="data"):
    """
    Load documents from the specified directory using Docling.

    Supports multiple file formats: .txt, .pdf, .docx, .pptx, .xlsx, .html, .md, etc.

    Args:
        data_dir (str): Path to the directory containing documents

    Returns:
        list: List of Document objects with page_content and metadata
    """
    documents = []

    # Use default Docling configuration for best quality
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
                # Extract text content from Docling document
                text_content = result.document.export_to_markdown()
                documents.append(Document(
                    page_content=text_content,
                    metadata={
                        "source": file_path,
                        "file_type": file.split('.')[-1],
                        "docling_metadata": result.document.origin.model_dump() if hasattr(result.document, 'origin') and result.document.origin else {}
                    }
                ))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Fallback to basic text loading if Docling fails
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": file_path, "fallback": True}
                ))
            except:
                print(f"Failed to load {file_path} with fallback method")

    return documents

# Individual load functions removed - now using Docling for unified document processing

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks for better retrieval performance.

    Args:
        documents (list): List of Document objects to split
        chunk_size (int): Maximum size of each chunk in characters
        chunk_overlap (int): Number of characters to overlap between chunks

    Returns:
        list: List of Document objects representing text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


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
    processor = CaptionAwareProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Use default Docling configuration for best quality
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
                        "file_type": file.split('.')[-1],
                        "docling_metadata": result.document.origin.model_dump() if hasattr(result.document, 'origin') and result.document.origin else {}
                    }
                )

                if chunks:
                    documents.extend(chunks)
                else:
                    # Fallback to regular markdown export if caption processing fails
                    text_content = result.document.export_to_markdown()
                    documents.append(Document(
                        page_content=text_content,
                        metadata={
                            "source": file_path,
                            "file_type": file.split('.')[-1],
                            "chunking_method": "fallback"
                        }
                    ))

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Fallback to basic text loading if Docling fails
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": file_path, "fallback": True}
                ))
            except:
                print(f"Failed to load {file_path} with fallback method")

    return documents

if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"Chunk {i+1}: {chunk.page_content[:200]}...")