#!/usr/bin/env python3
"""
Documents Page - File Management and Library
"""

import streamlit as st
import os
import sys
from pathlib import Path
import time

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import system components
try:
    from src.data_loader import load_documents, split_documents
    from src.embeddings import create_embeddings, save_embeddings
    from src.vector_store import create_faiss_index, save_faiss_index
except ImportError:
    st.error("❌ Could not import RAG system components.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Local RAG - Documents",
    page_icon="📁",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .page-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .file-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
    .file-name {
        font-weight: bold;
        color: #2c3e50;
    }
    .file-size {
        color: #7f8c8d;
        font-size: 0.9rem;
    }
    .processing-status {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_supported_extensions():
    """Get list of supported file extensions"""
    return ['.txt', '.pdf', '.docx', '.pptx', '.xlsx']

def get_file_info(file_path):
    """Get file information"""
    path = Path(file_path)
    stat = path.stat()
    return {
        'name': path.name,
        'size': stat.st_size,
        'modified': stat.st_mtime,
        'extension': path.suffix.lower()
    }

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def list_documents():
    """List all documents in the data directory"""
    data_dir = Path("data")
    if not data_dir.exists():
        return []

    documents = []
    supported_ext = get_supported_extensions()

    for file_path in data_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_ext:
            info = get_file_info(file_path)
            documents.append(info)

    # Sort by modification time (newest first)
    documents.sort(key=lambda x: x['modified'], reverse=True)
    return documents

def process_uploaded_files(uploaded_files):
    """Process uploaded files"""
    if not uploaded_files:
        return

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    processed_count = 0

    for uploaded_file in uploaded_files:
        # Save file to data directory
        file_path = data_dir / uploaded_file.name

        # Handle duplicate names
        counter = 1
        while file_path.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            file_path = data_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        processed_count += 1

    if processed_count > 0:
        st.success(f"✅ Successfully uploaded {processed_count} file(s)")
        st.rerun()

def reprocess_documents():
    """Reprocess all documents to update embeddings and vector store"""
    try:
        with st.spinner("🔄 Reprocessing documents... This may take a while."):

            # Load and split documents
            docs = load_documents()
            chunks = split_documents(docs)

            if not chunks:
                st.warning("⚠️ No documents found to process")
                return

            # Create embeddings
            st.info("📊 Creating embeddings...")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [doc.page_content for doc in chunks]
            embeddings = model.encode(texts, show_progress_bar=True)

            # Save embeddings
            save_embeddings(embeddings, chunks)

            # Create and save FAISS index
            st.info("🔍 Building vector index...")
            index = create_faiss_index(embeddings)
            save_faiss_index(index)

            st.success(f"✅ Successfully processed {len(chunks)} document chunks from {len(docs)} files")

            # Update session state to force reinitialization
            if 'system_initialized' in st.session_state:
                st.session_state.system_initialized = False

    except Exception as e:
        st.error(f"❌ Failed to reprocess documents: {str(e)}")

def main():
    """Main page content"""
    st.markdown('<h1 class="page-header">📁 Document Management</h1>', unsafe_allow_html=True)
    st.markdown("Upload, manage, and process your document library")

    # File upload section
    st.markdown("### 📤 Upload Documents")
    st.markdown("Supported formats: TXT, PDF, DOCX, PPTX, XLSX")

    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx', 'pptx', 'xlsx']
    )

    if uploaded_files:
        if st.button("📤 Upload Files", type="primary"):
            process_uploaded_files(uploaded_files)

    st.markdown("---")

    # Document library
    st.markdown("### 📚 Document Library")

    documents = list_documents()

    if not documents:
        st.info("📭 No documents found. Upload some files to get started!")
    else:
        st.info(f"📊 Found {len(documents)} document(s)")

        # Processing controls
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🔄 Reprocess All Documents", type="secondary"):
                reprocess_documents()

        with col2:
            if st.button("🗑️ Clear All Documents", type="secondary"):
                # This would need more sophisticated handling
                st.warning("⚠️ Document deletion not implemented yet")

        # Document list
        for doc in documents:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.markdown(f"**{doc['name']}**")
                    st.caption(f"Size: {format_file_size(doc['size'])}")

                with col2:
                    st.caption(f"Modified: {time.strftime('%Y-%m-%d %H:%M', time.localtime(doc['modified']))}")

                with col3:
                    file_ext = doc['extension'].upper()
                    if file_ext == '.TXT':
                        st.markdown("📄 Text")
                    elif file_ext == '.PDF':
                        st.markdown("📕 PDF")
                    elif file_ext == '.DOCX':
                        st.markdown("📝 Word")
                    elif file_ext == '.PPTX':
                        st.markdown("📊 PowerPoint")
                    elif file_ext == '.XLSX':
                        st.markdown("📈 Excel")
                    else:
                        st.markdown("📄 File")

    # Processing status
    st.markdown("---")
    st.markdown("### 🔧 Processing Status")

    # Check if models exist
    models_dir = Path("models")
    embeddings_exist = (models_dir / "embeddings.pkl").exists()
    index_exists = (models_dir / "faiss_index.pkl").exists()

    col1, col2 = st.columns(2)

    with col1:
        if embeddings_exist:
            st.success("✅ Embeddings: Ready")
        else:
            st.warning("⚠️ Embeddings: Not found")

    with col2:
        if index_exists:
            st.success("✅ Vector Index: Ready")
        else:
            st.warning("⚠️ Vector Index: Not found")

    if not embeddings_exist or not index_exists:
        st.info("💡 Click 'Reprocess All Documents' to generate embeddings and vector index")

if __name__ == "__main__":
    main()