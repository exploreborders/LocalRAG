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
    from src.database.models import (
        SessionLocal, Document, DocumentChunk, DocumentChapter, DocumentEmbedding,
        DocumentTopic, DocumentTagAssignment, DocumentCategoryAssignment
    )
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
    """List all documents from database with enhanced AI metadata"""
    try:
        db = SessionLocal()
        docs = db.query(Document).all()

        documents = []
        for doc in docs:
            # Get chapter and chunk counts
            chapter_count = len(doc.chapters) if doc.chapters else 0
            chunk_count = len(doc.chunks) if doc.chunks else 0

            documents.append({
                'id': doc.id,
                'name': doc.filename,
                'size': 0,  # Size not stored in DB
                'modified': doc.last_modified,
                'extension': doc.filename.split('.')[-1] if '.' in doc.filename else '',
                'detected_language': doc.detected_language or 'unknown',
                'status': doc.status,
                # AI-enriched metadata
                'document_summary': doc.document_summary,
                'key_topics': doc.key_topics,
                'reading_time_minutes': doc.reading_time_minutes,
                'author': doc.author,
                'publication_date': doc.publication_date,
                # Structure info
                'chapter_count': chapter_count,
                'chunk_count': chunk_count,
                'has_chapters': chapter_count > 0,
                'has_topics': doc.key_topics is not None and len(doc.key_topics) > 0
            })

        db.close()

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

def clear_all_documents():
    """
    Clear all documents from the system with confirmation.

    This function will:
    1. Delete all documents from database
    2. Clear Elasticsearch indices
    3. Clear Redis cache
    4. Remove physical files
    """
    try:
        # Get document count for confirmation
        db = SessionLocal()
        doc_count = db.query(Document).count()
        db.close()

        if doc_count == 0:
            st.info("ℹ️ No documents to clear")
            return

        # Confirmation dialog
        st.markdown("---")
        st.error("⚠️ **DANGER ZONE**")
        st.warning(f"You are about to delete **{doc_count} document(s)** and all associated data. This action cannot be undone!")

        col1, col2 = st.columns([1, 1])
        with col1:
            confirm_clear = st.button("🗑️ YES, DELETE ALL DOCUMENTS", type="primary", use_container_width=True)
        with col2:
            cancel_clear = st.button("❌ Cancel", use_container_width=True)

        if cancel_clear:
            st.session_state.show_clear_dialog = False
            st.rerun()

        if confirm_clear:
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("🗑️ Starting document deletion...")
                progress_bar.progress(10)

                # Clear Redis cache first
                try:
                    from src.cache.redis_cache import RedisCache
                    cache = RedisCache()
                    # Clear all cache patterns
                    cleared_llm = cache.clear_pattern("llm:*")
                    cleared_meta = cache.clear_pattern("doc_meta:*")
                    status_text.text(f"🗑️ Cleared Redis cache ({cleared_llm + cleared_meta} entries)...")
                    progress_bar.progress(20)
                except Exception as e:
                    st.warning(f"⚠️ Could not clear Redis cache: {e}")

                # Clear Elasticsearch indices
                try:
                    from elasticsearch import Elasticsearch
                    es = Elasticsearch(hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}], verify_certs=False)
                    if es.ping():
                        # Delete all rag_* indices
                        indices = es.cat.indices(index="rag_*", format="json")
                        for idx in indices:
                            index_name = idx['index']
                            es.indices.delete(index=index_name, ignore_unavailable=True)
                        status_text.text("🗑️ Cleared Elasticsearch indices...")
                        progress_bar.progress(40)
                    else:
                        st.warning("⚠️ Elasticsearch not available, skipping index cleanup")
                except Exception as e:
                    st.warning(f"⚠️ Could not clear Elasticsearch: {e}")

                # Clear database (manual cascade delete for safety)
                db = SessionLocal()
                try:
                    # Get file paths before deletion for cleanup
                    docs = db.query(Document).all()
                    file_paths = [doc.filepath for doc in docs if doc.filepath and os.path.exists(doc.filepath)]

                    # Delete related records in correct order (reverse dependency order)
                    # Delete embeddings (references chunks)
                    db.query(DocumentEmbedding).delete()

                    # Delete chunks (references documents)
                    db.query(DocumentChunk).delete()

                    # Delete chapters (references documents, may have self-references)
                    db.query(DocumentChapter).delete()

                    # Delete topic relationships
                    db.query(DocumentTopic).delete()

                    # Delete tag relationships
                    db.query(DocumentTagAssignment).delete()

                    # Delete category relationships
                    db.query(DocumentCategoryAssignment).delete()

                    # Finally delete documents
                    db.query(Document).delete()

                    db.commit()

                    status_text.text("🗑️ Cleared database records...")
                    progress_bar.progress(70)

                    # Remove physical files
                    deleted_files = 0
                    for file_path in file_paths:
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                deleted_files += 1
                        except Exception as e:
                            st.warning(f"⚠️ Could not delete file {file_path}: {e}")

                    status_text.text(f"🗑️ Removed {deleted_files} physical files...")
                    progress_bar.progress(90)

                except Exception as e:
                    db.rollback()
                    raise e
                finally:
                    db.close()

                # Clear progress indicators
                progress_bar.progress(100)
                progress_bar.empty()
                status_text.empty()

                # Success message
                st.success("✅ All documents and data have been successfully deleted!")
                st.info("🔄 The system is now ready for new document uploads.")

                # Reset dialog state
                st.session_state.show_clear_dialog = False

                # Force page refresh
                st.rerun()

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Failed to clear documents: {e}")
                st.error("Please check system logs for details.")

    except Exception as e:
        st.error(f"❌ Error accessing database: {e}")

def show_clear_documents_dialog():
    """
    Show the clear documents confirmation dialog.
    """
    # Check if we should show the dialog
    if 'show_clear_dialog' not in st.session_state:
        st.session_state.show_clear_dialog = False

    # Trigger dialog from button
    if st.button("🗑️ Clear Documents", type="secondary", use_container_width=True):
        st.session_state.show_clear_dialog = True
        st.rerun()

    # Show dialog if triggered
    if st.session_state.show_clear_dialog:
        clear_all_documents()

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
            show_clear_documents_dialog()

        with col3:
            if st.button("🔄 Refresh", help="Refresh the document list"):
                st.rerun()

        # Document list with enhanced display
        for doc in documents:
            with st.expander(f"📄 {doc['name']}", expanded=False):
                # Header row
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                with col1:
                    st.markdown(f"**{doc['name']}**")
                    if doc.get('author'):
                        st.caption(f"👤 Author: {doc['author']}")
                    if doc.get('publication_date'):
                        st.caption(f"📅 Published: {doc['publication_date']}")

                with col2:
                    st.caption(f"Modified: {time.strftime('%Y-%m-%d %H:%M', time.localtime(int(doc['modified'].timestamp())))}")

                with col3:
                    detected_lang = doc.get('detected_language', 'unknown').upper()
                    if detected_lang == 'EN':
                        st.markdown("🇺🇸 English")
                    elif detected_lang == 'DE':
                        st.markdown("🇩🇪 German")
                    elif detected_lang == 'FR':
                        st.markdown("🇫🇷 French")
                    elif detected_lang == 'ES':
                        st.markdown("🇪🇸 Spanish")
                    elif detected_lang == 'IT':
                        st.markdown("🇮🇹 Italian")
                    else:
                        st.markdown(f"🌍 {detected_lang}")

                with col4:
                    status = doc['status']
                    if status == 'processed':
                        st.markdown("✅ Processed")
                    elif status == 'processing':
                        st.markdown("🔄 Processing")
                    elif status == 'failed':
                        st.markdown("❌ Failed")
                    else:
                        st.markdown("📤 Uploaded")

                # Enhanced metadata section
                if doc.get('document_summary') or doc.get('key_topics') or doc.get('reading_time_minutes'):
                    st.markdown("---")
                    st.markdown("### 🤖 AI-Enriched Metadata")

                    meta_col1, meta_col2, meta_col3 = st.columns([2, 1, 1])

                    with meta_col1:
                        if doc.get('document_summary'):
                            st.markdown("**📝 Summary:**")
                            st.info(doc['document_summary'][:200] + "..." if len(doc['document_summary']) > 200 else doc['document_summary'])

                        if doc.get('key_topics'):
                            st.markdown("**🏷️ Key Topics:**")
                            topics = doc['key_topics'][:5]  # Show first 5 topics
                            st.write(", ".join(topics))

                    with meta_col2:
                        if doc.get('reading_time_minutes'):
                            st.metric("⏱️ Reading Time", f"{doc['reading_time_minutes']} min")

                        if doc.get('chapter_count', 0) > 0:
                            st.metric("📚 Chapters", doc['chapter_count'])

                    with meta_col3:
                        if doc.get('chunk_count', 0) > 0:
                            st.metric("📦 Chunks", doc['chunk_count'])

                        # Processing status indicators
                        indicators = []
                        if doc.get('has_chapters'):
                            indicators.append("📖 Structured")
                        if doc.get('has_topics'):
                            indicators.append("🏷️ Topics")
                        if indicators:
                            st.markdown("**Features:**")
                            for indicator in indicators:
                                st.caption(indicator)

                # Document structure preview (if available)
                if doc.get('chapter_count', 0) > 0:
                    st.markdown("---")
                    st.markdown("### 📑 Document Structure")
                    try:
                        db = SessionLocal()
                        chapters = db.query(DocumentChapter).filter(DocumentChapter.document_id == doc['id']).order_by(DocumentChapter.chapter_path).limit(10).all()
                        db.close()

                        if chapters:
                            for chapter in chapters:
                                level_indent = "  " * (chapter.level - 1)
                                st.caption(f"{level_indent}📄 {chapter.chapter_title} ({chapter.word_count} words)")
                            if len(chapters) == 10:
                                st.caption("... and more chapters")
                    except Exception as e:
                        st.caption(f"Could not load chapter structure: {e}")

    # System capabilities status
    st.markdown("---")
    st.markdown("### 🎯 System Capabilities")

    capabilities_col1, capabilities_col2, capabilities_col3 = st.columns(3)

    with capabilities_col1:
        st.success("✅ **AI Enrichment Active**")
        st.caption("Documents get automatic summaries, topics, and reading time estimates during upload")

    with capabilities_col2:
        st.info("🔄 **Tag System Ready**")
        st.caption("Database supports document tagging - UI coming in future update")

    with capabilities_col3:
        st.info("🔄 **Category System Ready**")
        st.caption("Hierarchical categorization available - UI coming in future update")

    st.markdown("**🚀 Current Features:**")
    st.markdown("""
    - 🤖 **AI-Powered Upload**: Automatic enrichment during document processing
    - 📚 **Hierarchical Structure**: Chapter-aware document organization
    - 🔍 **Advanced Search**: Vector + keyword hybrid search
    - 📊 **Rich Analytics**: Comprehensive system monitoring
    - 🗑️ **Safe Management**: Complete document deletion with cleanup
    """)

    # Enhanced processing status
    st.markdown("---")
    st.markdown("### 🔧 System Status & Analytics")

    # Check database and Elasticsearch status
    try:
        db = SessionLocal()

        # Get comprehensive document statistics
        total_docs = db.query(Document).count()
        processed_docs = db.query(Document).filter(Document.status == 'processed').count()
        total_chunks = db.query(DocumentChunk).count()
        total_chapters = db.query(DocumentChapter).count()

        # AI enrichment statistics
        docs_with_summary = db.query(Document).filter(Document.document_summary.isnot(None)).count()
        docs_with_topics = db.query(Document).filter(Document.key_topics.isnot(None)).count()

        db.close()

        # Status overview
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            if total_docs > 0:
                st.metric("📄 Documents", total_docs)
            else:
                st.warning("⚠️ No Documents")

        with col2:
            if processed_docs > 0:
                st.metric("✅ Processed", f"{processed_docs}/{total_docs}")
            else:
                st.warning("⚠️ None Processed")

        with col3:
            if total_chunks > 0:
                st.metric("📦 Chunks", total_chunks)
            else:
                st.info("📦 0 Chunks")

        with col4:
            if total_chapters > 0:
                st.metric("📚 Chapters", total_chapters)
            else:
                st.info("📚 0 Chapters")

        # AI enrichment status
        st.markdown("**🤖 AI Enrichment Status**")
        ai_col1, ai_col2, ai_col3 = st.columns([1, 1, 1])

        with ai_col1:
            if docs_with_summary > 0:
                st.metric("📝 Summaries", f"{docs_with_summary}/{total_docs}")
            else:
                st.info("📝 0 Summaries")

        with ai_col2:
            if docs_with_topics > 0:
                st.metric("🏷️ Topics", f"{docs_with_topics}/{total_docs}")
            else:
                st.info("🏷️ 0 Topics")

        with ai_col3:
            ai_ready = docs_with_summary + docs_with_topics
            if ai_ready > 0:
                st.metric("🎯 AI Ready", f"{ai_ready}/{total_docs}")
            else:
                st.info("🎯 0 AI Ready")

        # Check Elasticsearch connectivity
        st.markdown("**🔍 Vector Search Status**")
        try:
            from elasticsearch import Elasticsearch
            es = Elasticsearch(hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}], verify_certs=False)
            if es.ping():
                # Get index stats
                indices = es.cat.indices(index="rag_*", format="json")
                if indices:
                    total_vectors = sum(int(idx.get('docs.count', 0)) for idx in indices)
                    st.success(f"✅ Connected - {total_vectors} vectors indexed")
                else:
                    st.warning("⚠️ Connected but no indices found")
            else:
                st.error("❌ Not responding")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    except Exception as e:
        st.error(f"❌ Failed to check system status: {e}")

if __name__ == "__main__":
    main()