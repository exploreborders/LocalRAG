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
                'status': doc['status']
            })

        # Sort by modification time (newest first)
        documents.sort(key=lambda x: x['modified'], reverse=True)
        return documents
    except Exception as e:
        st.error(f"❌ Failed to load documents: {e}")
        return []

def process_uploaded_files(uploaded_files, selected_model="nomic-ai/nomic-embed-text-v1.5"):
    """Process uploaded files"""
    if not uploaded_files:
        return

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    processor = DocumentProcessor()
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

        # Process the file
        try:
            processor.process_document(str(file_path), selected_model)
            processed_count += 1
        except Exception as e:
            st.error(f"❌ Failed to process {uploaded_file.name}: {e}")

    if processed_count > 0:
        st.success(f"✅ Successfully uploaded and processed {processed_count} file(s)")
        st.rerun()

def reprocess_documents(selected_model="nomic-ai/nomic-embed-text-v1.5"):
    """Reprocess all documents to update database and Elasticsearch"""
    try:
        with st.spinner("🔄 Processing documents..."):
            processor = DocumentProcessor()
            processor.process_directory("data", selected_model)

            st.success("✅ Documents processed and stored in database")

            # Update session state to force reinitialization
            if 'system_initialized' in st.session_state:
                st.session_state.system_initialized = False
    except Exception as e:
        st.error(f"❌ Processing failed: {e}")

def batch_process_documents(selected_models):
    """Process documents with multiple models in batch"""
    try:
        with st.spinner("🔄 Batch processing documents... This may take a while."):

            # Load and split documents once
            docs = load_documents()
            chunks = split_documents(docs)

            if not chunks:
                st.warning("⚠️ No documents found to process")
                return

            progress_bar = st.progress(0)
            status_text = st.empty()

            total_models = len(selected_models)
            processed_models = 0

            for model_name in selected_models:
                status_text.text(f"Processing with {model_name}...")

                # Check if already processed (smart caching)
                from src.embeddings import get_documents_hash
                current_hash = get_documents_hash(chunks)

                safe_model_name = model_name.replace('/', '_').replace('-', '_')
                embeddings_file = f"models/embeddings_{safe_model_name}.pkl"

                needs_processing = True
                if os.path.exists(embeddings_file):
                    try:
                        _, _, stored_hash = load_embeddings(model_name)
                        if stored_hash == current_hash:
                            st.info(f"✅ {model_name} already up to date")
                            needs_processing = False
                    except Exception:
                        pass

                if needs_processing:
                    # Create embeddings
                    from sentence_transformers import SentenceTransformer
                    import torch
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model = SentenceTransformer(model_name, device=device)
                    texts = [doc.page_content for doc in chunks]
                    embeddings = model.encode(texts, show_progress_bar=False)

                    # Save embeddings and index
                    save_embeddings(embeddings, chunks, model_name)
                    index = create_faiss_index(embeddings)
                    save_faiss_index(index, model_name)

                    st.success(f"✅ Processed {len(chunks)} chunks with {model_name}")

                processed_models += 1
                progress_bar.progress(processed_models / total_models)

            progress_bar.empty()
            status_text.empty()

            st.success(f"✅ Batch processing completed! Processed documents with {len(selected_models)} models")

            # Update session state to force reinitialization
            if 'system_initialized' in st.session_state:
                st.session_state.system_initialized = False

    except Exception as e:
        st.error(f"❌ Batch processing failed: {str(e)}")

def main():
    """Main page content"""
    st.markdown('<h1 class="page-header">📁 Document Management</h1>', unsafe_allow_html=True)
    st.markdown("Upload, manage, and process your document library")

    # File upload section
    st.markdown("### 📤 Upload Documents")
    st.markdown("Supported formats: TXT, PDF, DOCX, PPTX, XLSX")

    # Get available embedding models for upload
    from utils.session_manager import get_available_embedding_models
    available_models = get_available_embedding_models()
    upload_model = st.selectbox(
        "Embedding Model for Upload",
        available_models,
        help="Choose which embedding model to use for processing uploaded documents"
    )

    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx', 'pptx', 'xlsx']
    )

    if uploaded_files:
        if st.button("📤 Upload Files", type="primary"):
            process_uploaded_files(uploaded_files, upload_model)

    st.markdown("---")

    # Document library
    st.markdown("### 📚 Document Library")

    documents = list_documents()

    if not documents:
        st.info("📭 No documents found. Upload some files to get started!")
    else:
        st.info(f"📊 Found {len(documents)} document(s)")

        # Processing controls
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        # Get available embedding models
        from utils.session_manager import get_available_embedding_models
        available_models = get_available_embedding_models()

        # Debug: Show available models
        if st.checkbox("🔍 Debug: Show available models", value=False):
            st.write("Available models:", available_models)
            st.write("Number of models:", len(available_models))

        with col4:
            if st.button("🔄 Refresh Models", help="Refresh the list of available models"):
                st.rerun()

        with col1:
            selected_model = st.selectbox(
                "Embedding Model",
                available_models,
                help="Choose which embedding model to use for processing documents"
            )

        with col2:
            if st.button("🔄 Reprocess Documents", type="secondary", use_container_width=True):
                reprocess_documents(selected_model)

        with col3:
            if st.button("🗑️ Clear Documents", type="secondary", use_container_width=True):
                # This would need more sophisticated handling
                st.warning("⚠️ Document deletion not implemented yet")

        # Batch processing section
        st.markdown("### 🔄 Batch Processing")
        st.markdown("Process documents with multiple models simultaneously")

        # Get available models
        available_models = get_available_embedding_models()

        if len(available_models) > 1:
            selected_batch_models = st.multiselect(
                "Select models for batch processing",
                available_models,
                default=[available_models[0]],
                help="Choose multiple models to process documents with"
            )

            if st.button("🚀 Process with Selected Models", type="primary", use_container_width=True):
                if selected_batch_models:
                    batch_process_documents(selected_batch_models)
                else:
                    st.warning("Please select at least one model")
        else:
            st.info("📝 Need multiple embedding models for batch processing")

        # Document list
        for doc in documents:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.markdown(f"**{doc['name']}**")
                    st.caption(f"Size: {format_file_size(doc['size'])}")

                with col2:
                    st.caption(f"Modified: {time.strftime('%Y-%m-%d %H:%M', time.localtime(int(doc['modified'].timestamp())))}")

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

    # Get available models and check status for each
    available_models = get_available_embedding_models()

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
        st.info("💡 Select a model above and click 'Reprocess Documents' to generate embeddings and vector index")
    else:
        st.success("✅ At least one model has processed embeddings")

if __name__ == "__main__":
    main()