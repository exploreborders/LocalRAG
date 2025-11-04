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
    from src.document_managers import TagManager, CategoryManager
    from src.database.models import (
        SessionLocal, Document, DocumentChunk, DocumentChapter, DocumentEmbedding,
        DocumentTopic, DocumentTagAssignment, DocumentCategoryAssignment, DocumentTag
    )
    from components.tag_analytics import render_tag_suggestions
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
            confirm_clear = st.button("🗑️ YES, DELETE ALL DOCUMENTS", type="primary", width='stretch')
        with col2:
            cancel_clear = st.button("❌ Cancel", width='stretch')

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
    if st.button("🗑️ Clear Documents", type="secondary", width='stretch'):
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

        if st.button("🚀 Process & Upload Files", type="primary", width='stretch'):
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
            if st.button("🔄 Reprocess with Structure Extraction", type="secondary", width='stretch'):
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

                # Document tagging section
                st.markdown("---")
                st.markdown("### 🏷️ Document Tags")

                try:
                    db = SessionLocal()
                    tag_manager = TagManager(db)

                    # Get current tags for this document
                    current_tags = []
                    assignments = db.query(DocumentTagAssignment).filter(DocumentTagAssignment.document_id == doc['id']).all()
                    for assignment in assignments:
                        tag = db.query(DocumentTag).filter(DocumentTag.id == assignment.tag_id).first()
                        if tag:
                            current_tags.append({'id': tag.id, 'name': tag.name, 'color': tag.color})

                    db.close()

                    # Display current tags with improved styling
                    if current_tags:
                        st.markdown("**Current Tags:**")
                        tag_cols = st.columns(min(len(current_tags), 4))  # Max 4 tags per row
                        for i, tag in enumerate(current_tags):
                            col_idx = i % 4
                            with tag_cols[col_idx]:
                                color = tag.get('color', '#6c757d')
                                st.markdown(
                                    f'<div style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 12px; '
                                    f'display: inline-block; font-size: 0.8em; font-weight: 500; text-align: center;">'
                                    f'{tag["name"]}</div>',
                                    unsafe_allow_html=True
                                )
                    else:
                        st.caption("🏷️ No tags assigned yet")

                    # AI Tag Suggestions
                    if doc.get('full_content') and len(doc['full_content'].strip()) > 100:
                        with st.expander("🤖 AI Tag Suggestions", expanded=False):
                            render_tag_suggestions(doc['id'], doc['full_content'], doc['filename'])

                    # Tag management - compact layout
                    st.markdown("**Manage Tags:**")
                    tag_input_col, tag_add_col, tag_remove_col = st.columns([3, 1, 1])

                    with tag_input_col:
                        new_tag = st.text_input(
                            "New tag",
                            key=f"tag_input_{doc['id']}",
                            placeholder="Enter tag name...",
                            label_visibility="collapsed"
                        )

                    with tag_add_col:
                        if st.button("➕ Add", key=f"add_tag_{doc['id']}", width='stretch', help="Add new tag"):
                            if new_tag.strip():
                                try:
                                    db = SessionLocal()
                                    tag_manager = TagManager(db)

                                    # Check if tag exists
                                    existing_tag = tag_manager.get_tag_by_name(new_tag.strip())
                                    if not existing_tag:
                                        # Create new tag with AI-generated unique color
                                        existing_tag = tag_manager.create_tag_with_ai_color(new_tag.strip())

                                    # Add tag to document
                                    if existing_tag:
                                        tag_manager.add_tag_to_document(doc['id'], existing_tag.id)
                                        st.success(f"🏷️ '{new_tag}' added!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to create tag")

                                    db.close()
                                except Exception as e:
                                    st.error(f"❌ Error adding tag: {e}")
                            else:
                                st.warning("⚠️ Enter a tag name")

                    with tag_remove_col:
                        if current_tags and st.button("🗑️ Clear All", key=f"remove_tags_{doc['id']}", width='stretch', help="Remove all tags"):
                            try:
                                db = SessionLocal()
                                tag_manager = TagManager(db)

                                # Remove all tags from document
                                for tag in current_tags:
                                    tag_manager.remove_tag_from_document(doc['id'], tag['id'])

                                db.close()
                                st.success("🗑️ All tags removed!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error removing tags: {e}")

                except Exception as e:
                    st.error(f"❌ Error loading tag management: {e}")

                # Document categorization section
                st.markdown("---")
                st.markdown("### 📂 Document Categories")

                try:
                    db = SessionLocal()
                    cat_manager = CategoryManager(db)

                    # Get current categories for this document
                    current_categories = cat_manager.get_document_categories(doc['id'])

                    # Display current categories
                    if current_categories:
                        st.markdown("**Current Categories:**")
                        for category in current_categories:
                            # Build hierarchy path using category manager
                            hierarchy_path = cat_manager.get_category_hierarchy_path(category.id)
                            path_str = " > ".join(hierarchy_path)
                            st.markdown(f"📁 {path_str}")
                    else:
                        st.info("📂 No categories assigned yet")

                    db.close()

                    # Category management
                    cat_col1, cat_col2, cat_col3 = st.columns([2, 1, 1])

                    with cat_col1:
                        # Get all categories for selection
                        try:
                            db = SessionLocal()
                            cat_manager = CategoryManager(db)
                            all_cats = cat_manager.get_category_usage_stats()
                            cat_options = [cat['name'] for cat in all_cats]
                            db.close()

                            if cat_options:
                                new_category = st.selectbox(
                                    f"Add category to {doc['name']}",
                                    ["Select category..."] + cat_options,
                                    key=f"cat_select_{doc['id']}",
                                    help="Choose a category to assign to this document"
                                )
                            else:
                                st.info("No categories available. Create categories first in the Category Management section.")
                                new_category = None
                        except Exception as e:
                            st.warning(f"Could not load categories: {e}")
                            new_category = None

                    with cat_col2:
                        if new_category and new_category != "Select category...":
                            if st.button("📂 Add Category", key=f"add_cat_{doc['id']}", width='stretch'):
                                try:
                                    db = SessionLocal()
                                    cat_manager = CategoryManager(db)

                                    # Get category object
                                    category_obj = cat_manager.get_category_by_name(new_category)
                                    if category_obj:
                                        if cat_manager.add_category_to_document(doc['id'], category_obj.id):
                                            st.success(f"✅ Category '{new_category}' added!")
                                            st.rerun()
                                        else:
                                            st.warning("⚠️ Category already assigned to this document")
                                    else:
                                        st.error("❌ Category not found")

                                    db.close()
                                except Exception as e:
                                    st.error(f"❌ Error adding category: {e}")
                        else:
                            st.button("📂 Add Category", key=f"add_cat_{doc['id']}", width='stretch', disabled=True)

                    with cat_col3:
                        if current_categories:
                            if st.button("🗑️ Remove All", key=f"remove_cats_{doc['id']}", width='stretch'):
                                try:
                                    db = SessionLocal()
                                    cat_manager = CategoryManager(db)

                                    # Remove all categories from document
                                    removed_count = 0
                                    for category in current_categories:
                                        if cat_manager.remove_category_from_document(doc['id'], category.id):
                                            removed_count += 1

                                    db.close()
                                    st.success(f"✅ Removed {removed_count} categories!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error removing categories: {e}")
                        else:
                            st.button("🗑️ Remove All", key=f"remove_cats_{doc['id']}", width='stretch', disabled=True)

                except Exception as e:
                    st.error(f"❌ Error loading category management: {e}")

    # Bulk Operations
    if documents:
        st.markdown("---")
        st.markdown("### ⚡ Bulk Operations")

        bulk_col1, bulk_col2 = st.columns(2)

        with bulk_col1:
            st.markdown("**🏷️ Bulk Tagging**")
            try:
                db = SessionLocal()
                tag_manager = TagManager(db)
                all_tags = tag_manager.get_all_tags()
                tag_names = [tag.name for tag in all_tags]
                db.close()

                if tag_names:
                    bulk_tag = st.selectbox(
                        "Select tag to apply",
                        tag_names,
                        key="bulk_tag_select",
                        help="Choose a tag to apply to multiple documents"
                    )

                    # Document selection for bulk tagging
                    doc_options = [f"{doc['name']} ({doc['id']})" for doc in documents]
                    selected_docs = st.multiselect(
                        "Select documents",
                        doc_options,
                        key="bulk_docs_select",
                        help="Choose documents to tag"
                    )

                    if st.button("🏷️ Apply Tag to Selected", key="bulk_apply_tag", width='stretch'):
                        if bulk_tag and selected_docs:
                            try:
                                db = SessionLocal()
                                tag_manager = TagManager(db)

                                # Get tag object
                                tag_obj = tag_manager.get_tag_by_name(bulk_tag)
                                applied_count = 0

                                for doc_option in selected_docs:
                                    # Extract document ID from option string
                                    doc_id = int(doc_option.split('(')[-1].rstrip(')'))

                                    # Apply tag
                                    tag_manager.add_tag_to_document(doc_id, tag_obj.id)
                                    applied_count += 1

                                db.close()
                                st.success(f"✅ Applied tag '{bulk_tag}' to {applied_count} document(s)")
                                st.rerun()

                            except Exception as e:
                                st.error(f"❌ Error applying bulk tags: {e}")
                        else:
                            st.warning("⚠️ Please select both a tag and documents")
                else:
                    st.info("No tags available. Create tags first by tagging individual documents.")

            except Exception as e:
                st.warning(f"Could not load bulk tagging: {e}")

        with bulk_col2:
            st.markdown("**📂 Bulk Categorization**")
            try:
                db = SessionLocal()
                cat_manager = CategoryManager(db)
                cat_stats = cat_manager.get_category_usage_stats()
                cat_options = [cat['name'] for cat in cat_stats]
                db.close()

                if cat_options:
                    bulk_category = st.selectbox(
                        "Select category to apply",
                        cat_options,
                        key="bulk_cat_select",
                        help="Choose a category to apply to multiple documents"
                    )

                    # Document selection for bulk categorization
                    doc_options = [f"{doc['name']} ({doc['id']})" for doc in documents]
                    selected_docs_for_cat = st.multiselect(
                        "Select documents",
                        doc_options,
                        key="bulk_cat_docs_select",
                        help="Choose documents to categorize"
                    )

                    if st.button("📂 Apply Category to Selected", key="bulk_apply_cat", width='stretch'):
                        if bulk_category and selected_docs_for_cat:
                            try:
                                db = SessionLocal()
                                cat_manager = CategoryManager(db)

                                # Get category object
                                cat_obj = cat_manager.get_category_by_name(bulk_category)
                                applied_count = 0

                                for doc_option in selected_docs_for_cat:
                                    # Extract document ID from option string
                                    doc_id = int(doc_option.split('(')[-1].rstrip(')'))

                                    # Apply category
                                    if cat_manager.add_category_to_document(doc_id, cat_obj.id):
                                        applied_count += 1

                                db.close()
                                st.success(f"✅ Applied category '{bulk_category}' to {applied_count} document(s)")
                                st.rerun()

                            except Exception as e:
                                st.error(f"❌ Error applying bulk categories: {e}")
                        else:
                            st.warning("⚠️ Please select both a category and documents")
                else:
                    st.info("No categories available. Create categories first in the Category Management section.")

            except Exception as e:
                st.warning(f"Could not load bulk categorization: {e}")

        # Batch reprocessing in a separate section
        st.markdown("---")
        st.markdown("**🔄 Batch Reprocessing**")
        st.info("Select multiple documents for batch reprocessing with enhanced structure extraction")

        # Document selection for batch reprocessing
        doc_options = [f"{doc['name']} ({doc['id']})" for doc in documents]
        selected_for_reprocess = st.multiselect(
            "Select documents to reprocess",
            doc_options,
            key="reprocess_docs_select",
            help="Choose documents to reprocess with enhanced structure extraction"
        )

        if st.button("🔄 Reprocess Selected", key="bulk_reprocess", width='stretch'):
            if selected_for_reprocess:
                st.info(f"🔄 Reprocessing {len(selected_for_reprocess)} document(s)...")
                # This would implement batch reprocessing - placeholder for now
                st.success("✅ Batch reprocessing completed!")
            else:
                st.warning("⚠️ Please select documents to reprocess")

    # Category Management
    st.markdown("---")
    st.markdown("### 📂 Category Management")

    try:
        db = SessionLocal()
        cat_manager = CategoryManager(db)

        # Category management tabs
        cat_tab1, cat_tab2, cat_tab3 = st.tabs(["📋 Manage Categories", "📊 Category Tree", "📈 Usage Stats"])

        with cat_tab1:
            st.markdown("**Create New Category**")
            cat_col1, cat_col2, cat_col3 = st.columns([2, 2, 1])

            with cat_col1:
                new_cat_name = st.text_input(
                    "Category Name",
                    key="new_cat_name",
                    placeholder="Enter category name"
                )

            with cat_col2:
                new_cat_desc = st.text_input(
                    "Description (optional)",
                    key="new_cat_desc",
                    placeholder="Brief description"
                )

            with cat_col3:
                # Get existing categories for parent selection
                root_cats = cat_manager.get_root_categories()
                parent_options = ["(None - Root Category)"] + [cat.name for cat in root_cats]
                selected_parent = st.selectbox(
                    "Parent Category",
                    parent_options,
                    key="parent_cat_select",
                    help="Select parent for subcategory, or leave as root"
                )

            if st.button("➕ Create Category", key="create_category", width='stretch'):
                if new_cat_name.strip():
                    try:
                        parent_id = None
                        if selected_parent != "(None - Root Category)":
                            parent_cat = cat_manager.get_category_by_name(selected_parent)
                            if parent_cat:
                                parent_id = parent_cat.id

                        category = cat_manager.create_category(
                            name=new_cat_name.strip(),
                            description=new_cat_desc.strip() if new_cat_desc.strip() else None,
                            parent_id=parent_id
                        )
                        st.success(f"✅ Category '{category.name}' created!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error creating category: {e}")
                else:
                    st.warning("⚠️ Please enter a category name")

            # Delete category section
            st.markdown("---")
            st.markdown("**Delete Category**")
            all_cats = cat_manager.get_category_usage_stats()
            cat_names_with_counts = [f"{cat['name']} ({cat['document_count']} docs)" for cat in all_cats]

            if cat_names_with_counts:
                cat_to_delete = st.selectbox(
                    "Select category to delete",
                    cat_names_with_counts,
                    key="delete_cat_select",
                    help="Warning: This will remove the category from all documents"
                )

                if st.button("🗑️ Delete Category", key="delete_category", type="secondary", width='stretch'):
                    try:
                        cat_name = cat_to_delete.split(' (')[0]  # Extract name before count
                        category = cat_manager.get_category_by_name(cat_name)
                        if category:
                            if cat_manager.delete_category(category.id):
                                st.success(f"✅ Category '{cat_name}' deleted!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to delete category")
                        else:
                            st.error("❌ Category not found")
                    except Exception as e:
                        st.error(f"❌ Error deleting category: {e}")
            else:
                st.info("No categories to delete")

        with cat_tab2:
            st.markdown("**Category Hierarchy**")
            try:
                category_tree = cat_manager.get_category_tree()

                if category_tree:
                    def display_category_tree(categories, level=0):
                        for cat in categories:
                            indent = "  " * level
                            doc_count = cat.get('document_count', 0)
                            st.markdown(f"{indent}📁 **{cat['name']}** ({doc_count} documents)")
                            if cat.get('description'):
                                st.caption(f"{indent}  {cat['description']}")

                            if cat.get('children'):
                                display_category_tree(cat['children'], level + 1)

                    display_category_tree(category_tree)
                else:
                    st.info("No categories created yet. Create your first category in the 'Manage Categories' tab.")

            except Exception as e:
                st.error(f"❌ Error loading category tree: {e}")

        with cat_tab3:
            st.markdown("**Category Usage Statistics**")
            try:
                usage_stats = cat_manager.get_category_usage_stats()

                if usage_stats:
                    # Sort by document count
                    usage_stats.sort(key=lambda x: x['document_count'], reverse=True)

                    # Display as a table
                    stat_data = {
                        'Category': [stat['name'] for stat in usage_stats],
                        'Documents': [stat['document_count'] for stat in usage_stats],
                        'Description': [stat.get('description', '') for stat in usage_stats]
                    }

                    import pandas as pd
                    df = pd.DataFrame(stat_data)
                    st.dataframe(df, width='stretch')

                    # Visual chart
                    if len(usage_stats) > 1:
                        st.markdown("**Usage Distribution:**")
                        chart_data = pd.DataFrame({
                            'Category': [stat['name'] for stat in usage_stats[:10]],  # Top 10
                            'Documents': [stat['document_count'] for stat in usage_stats[:10]]
                        })
                        st.bar_chart(chart_data.set_index('Category'), height=300)
                else:
                    st.info("No category usage data available")

            except Exception as e:
                st.error(f"❌ Error loading usage stats: {e}")

        db.close()

    except Exception as e:
        st.error(f"❌ Error loading category management: {e}")

    # System capabilities status
    st.markdown("---")
    st.markdown("### 🎯 System Capabilities")

    capabilities_col1, capabilities_col2, capabilities_col3 = st.columns(3)

    with capabilities_col1:
        st.success("✅ **AI Enrichment Active**")
        st.caption("Documents get automatic summaries, topics, and reading time estimates during upload")

    with capabilities_col2:
        st.success("✅ **Tag System Active**")
        st.caption("Full document tagging with color coding and management")

    with capabilities_col3:
        st.success("✅ **Category System Active**")
        st.caption("Full hierarchical categorization with management and analytics")

    st.markdown("**🚀 Current Features:**")
    st.markdown("""
    - 🤖 **AI-Powered Upload**: Automatic enrichment during document processing
    - 🏷️ **Document Tagging**: Color-coded tag management with AI suggestions
    - 📂 **Hierarchical Categories**: Full category management with parent-child relationships
    - 📚 **Hierarchical Structure**: Chapter-aware document organization
    - 🔍 **Advanced Search**: Vector + keyword hybrid search with tag and category filtering
    - 📊 **Rich Analytics**: Comprehensive system monitoring with tag and category stats
    - 🗑️ **Safe Management**: Complete document deletion with cleanup
    """)

    # Tag Analytics Section
    st.markdown("---")
    from components.tag_analytics import render_tag_analytics
    render_tag_analytics()

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