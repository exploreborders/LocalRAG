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
    from src.document_processor import DocumentProcessor
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
    """List all documents from database"""
    try:
        processor = DocumentProcessor()
        docs = processor.get_documents()

        documents = []
        for doc in docs:
            documents.append({
                'name': doc['filename'],
                'size': 0,  # Size not stored in DB
                'modified': doc['last_modified'],
                'extension': doc['filename'].split('.')[-1] if '.' in doc['filename'] else '',
                'detected_language': doc.get('detected_language', 'unknown'),
                'status': doc['status']
            })

        # Sort by modification time (newest first)
        documents.sort(key=lambda x: x['modified'], reverse=True)
        return documents
    except Exception as e:
        st.error(f"❌ Failed to load documents: {e}")
        return []

def process_uploaded_files(uploaded_files):
    """Process uploaded files"""
    if not uploaded_files:
        return

    import tempfile
    import os

    processor = DocumentProcessor()
    processed_count = 0

    # Use temporary directory for uploads to avoid cluttering data/
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for uploaded_file in uploaded_files:
            # Save file to temporary directory
            file_path = temp_path / uploaded_file.name

            # Handle duplicate names (unlikely in temp dir but safe)
            counter = 1
            while file_path.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                file_path = temp_path / f"{stem}_{counter}{suffix}"
                counter += 1

            # Save the file temporarily
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process the file
            try:
                processor.process_document(str(file_path))
                processed_count += 1
            except Exception as e:
                st.error(f"❌ Failed to process {uploaded_file.name}: {e}")
            # File is automatically cleaned up when temp directory exits

    if processed_count > 0:
        st.success(f"✅ Successfully uploaded and processed {processed_count} file(s)")
        st.rerun()

def reprocess_documents():
    """Reprocess all documents to update database and Elasticsearch with performance optimizations"""
    try:
        with st.spinner("🔄 Reprocessing documents with performance optimizations..."):
            processor = DocumentProcessor()
            processor.reprocess_all_documents(
                batch_size=5,  # Process in batches of 5
                use_parallel=True,  # Use parallel processing
                max_workers=4,  # Use up to 4 workers
                memory_limit_mb=500  # Memory limit
            )

            st.success("✅ Documents reprocessed with performance optimizations")

            # Update session state to force reinitialization
            if 'system_initialized' in st.session_state:
                st.session_state.system_initialized = False
    except Exception as e:
        st.error(f"❌ Reprocessing failed: {e}")



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
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("🔄 Reprocess Documents", type="secondary", use_container_width=True):
                reprocess_documents()

        with col2:
            if st.button("🗑️ Clear Documents", type="secondary", use_container_width=True):
                # This would need more sophisticated handling
                st.warning("⚠️ Document deletion not implemented yet")

        with col3:
            if st.button("🔄 Refresh", help="Refresh the document list"):
                st.rerun()



        # Document list
        for doc in documents:
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                with col1:
                    st.markdown(f"**{doc['name']}**")
                    st.caption(f"Size: {format_file_size(doc['size'])}")

                with col2:
                    st.caption(f"Modified: {time.strftime('%Y-%m-%d %H:%M', time.localtime(int(doc['modified'].timestamp())))}")

                with col3:
                    lang = doc.get('detected_language', 'unknown').upper()
                    if lang == 'EN':
                        st.markdown("🇺🇸 English")
                    elif lang == 'DE':
                        st.markdown("🇩🇪 German")
                    elif lang == 'FR':
                        st.markdown("🇫🇷 French")
                    elif lang == 'ES':
                        st.markdown("🇪🇸 Spanish")
                    elif lang == 'IT':
                        st.markdown("🇮🇹 Italian")
                    elif lang == 'UNKNOWN':
                        st.markdown("❓ Unknown")
                    else:
                        st.markdown(f"🌍 {lang}")

                with col4:
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

    # Get available models and check status for each
    available_models = ["nomic-ai/nomic-embed-text-v1.5"]

    st.markdown("**Available Embedding Models:**")
    for model in available_models:
        safe_model_name = model.replace('/', '_').replace('-', '_')
        embeddings_file = Path(f"models/embeddings_{safe_model_name}.pkl")
        index_file = Path(f"models/faiss_index_{safe_model_name}.pkl")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.write(f"**{model}**")

        with col2:
            if embeddings_file.exists():
                st.success("✅ Embeddings")
            else:
                st.warning("⚠️ No Embeddings")

        with col3:
            if index_file.exists():
                st.success("✅ Index")
            else:
                st.warning("⚠️ No Index")

    # Overall status
    st.markdown("---")
    has_any_embeddings = any(
        Path(f"models/embeddings_{model.replace('/', '_').replace('-', '_')}.pkl").exists()
        for model in available_models
    )

    if not has_any_embeddings:
        st.info("💡 Click 'Reprocess Documents' to generate embeddings and vector index")
    else:
        st.success("✅ Documents have been processed with embeddings")

if __name__ == "__main__":
    main()