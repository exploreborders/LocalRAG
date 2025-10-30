from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import PyPDF2
from docx import Document as DocxDocument
from pptx import Presentation
import openpyxl

def load_documents(data_dir="data"):
    """
    Load documents from the data directory.
    Supports .txt, .pdf, .docx, .pptx, .xlsx files.
    """
    documents = []
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if file.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".pdf"):
            documents.extend(load_pdf(file_path))
        elif file.endswith(".docx"):
            documents.extend(load_docx(file_path))
        elif file.endswith(".pptx"):
            documents.extend(load_pptx(file_path))
        elif file.endswith(".xlsx"):
            documents.extend(load_xlsx(file_path))
        # Add more formats as needed
    return documents

def load_pdf(file_path):
    """Load PDF and return list of Documents."""
    documents = []
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        documents.append(Document(page_content=text, metadata={"source": file_path}))
    return documents

def load_docx(file_path):
    """Load DOCX and return list of Documents."""
    documents = []
    doc = DocxDocument(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    documents.append(Document(page_content=text, metadata={"source": file_path}))
    return documents

def load_pptx(file_path):
    """Load PPTX and return list of Documents."""
    documents = []
    prs = Presentation(file_path)
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    documents.append(Document(page_content=text, metadata={"source": file_path}))
    return documents

def load_xlsx(file_path):
    """Load XLSX and return list of Documents."""
    documents = []
    wb = openpyxl.load_workbook(file_path)
    text = ""
    for sheet in wb:
        for row in sheet.iter_rows(values_only=True):
            text += " ".join([str(cell) for cell in row if cell is not None]) + "\n"
    documents.append(Document(page_content=text, metadata={"source": file_path}))
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks for better retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"Chunk {i+1}: {chunk.page_content[:200]}...")