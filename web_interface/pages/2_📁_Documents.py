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
    from src.upload_processor import UploadProcessor
    from src.document_processor import DocumentProcessor
    from src.database.models import SessionLocal, Document
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

    .feature-highlight {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
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
    """
    Process uploaded files with integrated structure extraction and chapter creation.

    Args:
        uploaded_files: List of uploaded file objects from Streamlit
    """
    if not uploaded_files:
        return

    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    file_status = st.empty()

    def progress_callback(filename, progress, message):
        """Update progress display"""
        status_text.text(f"Processing {filename}: {message}")
        progress_bar.progress(int(progress))

    # Initialize upload processor with progress callback
    processor = UploadProcessor(progress_callback=progress_callback)

    try:
        status_text.text("Starting upload process...")
        progress_bar.progress(5)

        # Process files with enhanced upload processor
        results = processor.upload_files(
            uploaded_files,
            data_dir="data",
            use_parallel=len(uploaded_files) > 1,
            max_workers=min(4, len(uploaded_files))
        )

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Display results
        if results['successful_uploads'] > 0:
            st.success(f"✅ Successfully processed {results['successful_uploads']} file(s)")
            st.info(f"📊 Created {results['total_chunks']} chunks and {results['total_chapters']} chapters")

            # Show detailed results
            with st.expander("📋 Processing Details", expanded=False):
                for result in results['file_results']:
                    if result['success']:
                        st.write(f"✅ {result['filename']}: {result['chunks_created']} chunks, {result['chapters_created']} chapters")
                    else:
                        st.error(f"❌ {result['filename']}: {result.get('error', 'Unknown error')}")

        if results['failed_uploads'] > 0:
            st.warning(f"⚠️ {results['failed_uploads']} file(s) failed to process")
            with st.expander("❌ Errors", expanded=False):
                for error in results['errors']:
                    st.write(error)

        # Refresh the page to show updated document list
        if results['successful_uploads'] > 0:
            st.rerun()

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Upload failed: {e}")
    finally:
        processor.db.close()

def reprocess_documents():
    """
    Reprocess all existing documents with enhanced structure extraction.

    This function now uses the new UploadProcessor for better chapter extraction
    and parallel processing capabilities.
    """
    try:
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(filename, progress, message):
            """Update progress display"""
            status_text.text(f"Reprocessing {filename}: {message}")
            progress_bar.progress(int(progress))

        # Use UploadProcessor for reprocessing with progress tracking
        processor = UploadProcessor(progress_callback=progress_callback)

        status_text.text("🔄 Starting reprocessing with enhanced structure extraction...")

        # Get all existing documents
        docs = processor.db.query(Document).all()

        if not docs:
            st.warning("⚠️ No documents found to reprocess")
            return

        st.info(f"📊 Reprocessing {len(docs)} document(s) with chapter extraction...")

        successful_reprocess = 0
        total_chunks = 0
        total_chapters = 0

        for i, doc in enumerate(docs):
            try:
                status_text.text(f"🔄 Reprocessing {doc.filename}...")
                progress_bar.progress(int((i / len(docs)) * 100))

                # Reprocess with enhanced processor
                result = processor.process_single_file(doc.filepath, doc.filename, doc.file_hash)

                if result['success']:
                    successful_reprocess += 1
                    total_chunks += result['chunks_created']
                    total_chapters += result['chapters_created']
                else:
                    st.warning(f"⚠️ Failed to reprocess {doc.filename}: {result['error']}")

            except Exception as e:
                st.error(f"❌ Error reprocessing {doc.filename}: {e}")

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        if successful_reprocess > 0:
            st.success(f"✅ Successfully reprocessed {successful_reprocess} document(s)")
            st.info(f"📊 Created {total_chunks} chunks and {total_chapters} chapters")

            # Update session state to force reinitialization
            if 'system_initialized' in st.session_state:
                st.session_state.system_initialized = False
        else:
            st.warning("⚠️ No documents were successfully reprocessed")

    except Exception as e:
        st.error(f"❌ Reprocessing failed: {e}")
    finally:
        if 'processor' in locals():
            processor.db.close()

# Tag and category management not yet implemented
# def get_tag_manager():
#     """Get tag manager instance"""
#     db = SessionLocal()
#     return TagManager(db)

# def get_category_manager():
#     """Get category manager instance"""
#     db = SessionLocal()
#     return CategoryManager(db)

# Tag and category management not yet implemented
# def manage_tags():
#     """Tag management interface"""
#     st.markdown("### 🏷️ Tag Management")
#     st.info("🏷️ Tag management coming soon!")

# def manage_categories():
#     """Category management interface"""
#     st.markdown("### 📂 Category Management")
#     st.info("📂 Category management coming soon!")

# def display_category_tree(category, cat_manager, level=0):
#     """Recursively display category tree"""
#     pass

# AI enrichment not yet implemented
# def ai_enrich_documents():
#     """AI-powered document enrichment interface"""
#     st.markdown("### 🤖 AI Document Enrichment")
#     st.info("🤖 AI enrichment coming soon!")



def main():
    """Main page content"""
    st.markdown('<h1 class="page-header">📁 Document Management</h1>', unsafe_allow_html=True)
    st.markdown("Upload, manage, and process your document library")

    # File upload section
    st.markdown("### 📤 Upload Documents")

    # Feature highlights
    st.markdown("""
    <div class="feature-highlight">
        <h4>🚀 Enhanced Upload Features</h4>
        <ul>
            <li>📋 <strong>Automatic Structure Extraction</strong> - Chapters, sections, and metadata</li>
            <li>🏷️ <strong>Chapter-Aware Processing</strong> - Hierarchical document organization</li>
            <li>⚡ <strong>Parallel Processing</strong> - Fast upload for multiple files</li>
            <li>🔍 <strong>Immediate Searchability</strong> - No separate processing step needed</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Supported formats:** TXT, PDF, DOCX, PPTX, XLSX")



    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx', 'pptx', 'xlsx'],
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.markdown(f"**📋 Selected {len(uploaded_files)} file(s):**")
        for file in uploaded_files:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"📄 {file.name}")
            with col2:
                st.write(f"{len(file.getbuffer()):,} bytes")
            with col3:
                st.write(file.type if file.type else "Unknown")

        if st.button("🚀 Process & Upload Files", type="primary", use_container_width=True):
            process_uploaded_files(uploaded_files)

    st.markdown("---")

    # Organization management
    st.markdown("### 📚 Document Library")

    documents = list_documents()

    if not documents:
        st.info("📭 No documents found. Upload some files to get started!")
    else:
        st.info(f"📊 Found {len(documents)} document(s)")

        # Processing controls
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("🔄 Reprocess with Structure Extraction", type="secondary", use_container_width=True):
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
                    detected_lang = doc.get('detected_language')
                    lang = (detected_lang if detected_lang is not None else 'unknown').upper()
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

    # Note: Tag, category, and AI enrichment features coming soon
    st.markdown("---")
    st.info("🏷️ Tag management, 📂 category organization, and 🤖 AI enrichment features are coming soon!")

    # Processing status
    st.markdown("---")
    st.markdown("### 🔧 Processing Status")

    # Check database and Elasticsearch status
    try:
        processor = DocumentProcessor()

        # Check database connectivity and document count
        docs = processor.get_documents()
        total_docs = len(docs)
        processed_docs = len([doc for doc in docs if doc['status'] == 'processed'])

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown("**Database Status**")

        with col2:
            if total_docs > 0:
                st.success(f"✅ {total_docs} Documents")
            else:
                st.warning("⚠️ No Documents")

        with col3:
            if processed_docs > 0:
                st.success(f"✅ {processed_docs} Processed")
            else:
                st.warning("⚠️ None Processed")

        # Check Elasticsearch connectivity
        st.markdown("**Vector Search Status**")
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown("**Elasticsearch**")

        with col2:
            try:
                # Check if ES is accessible
                es_info = processor.es.info()
                st.success("✅ Connected")
            except Exception:
                st.error("❌ Not Connected")

        with col3:
            try:
                # Check if rag_vectors index exists and has documents
                index_stats = processor.es.indices.stats(index="rag_vectors")
                doc_count = index_stats['indices']['rag_vectors']['total']['docs']['count']
                if doc_count > 0:
                    st.success(f"✅ {doc_count} Vectors")
                else:
                    st.warning("⚠️ No Vectors")
            except Exception:
                st.warning("⚠️ Index Missing")

    except Exception as e:
        st.error(f"❌ System Status Check Failed: {e}")

    # Overall status
    st.markdown("---")
    try:
        has_processed_docs = processed_docs > 0 if 'processed_docs' in locals() else False
        has_vectors = False
        try:
            index_stats = processor.es.indices.stats(index="rag_vectors")
            doc_count = index_stats['indices']['rag_vectors']['total']['docs']['count']
            has_vectors = doc_count > 0
        except:
            pass

        if has_processed_docs and has_vectors:
            st.success("✅ System is ready for queries")
        elif total_docs > 0:
            st.info("💡 Click 'Reprocess Documents' to generate embeddings and vector index")
        else:
            st.info("📭 Upload some documents to get started")
    except:
        st.info("💡 Click 'Reprocess Documents' to generate embeddings and vector index")

if __name__ == "__main__":
    main()