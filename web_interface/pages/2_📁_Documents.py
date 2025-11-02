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
    from src.document_managers import TagManager, CategoryManager
    from src.ai_enrichment import AIEnrichmentService
    from src.database.models import SessionLocal
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

def get_tag_manager():
    """Get tag manager instance"""
    db = SessionLocal()
    return TagManager(db)

def get_category_manager():
    """Get category manager instance"""
    db = SessionLocal()
    return CategoryManager(db)

def manage_tags():
    """Tag management interface"""
    st.markdown("### 🏷️ Tag Management")

    tag_manager = get_tag_manager()

    # Create new tag
    with st.expander("➕ Create New Tag"):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            new_tag_name = st.text_input("Tag Name", key="new_tag_name")
        with col2:
            new_tag_color = st.color_picker("Color", "#FF5733", key="new_tag_color")
        with col3:
            new_tag_desc = st.text_input("Description (optional)", key="new_tag_desc")

        if st.button("Create Tag", key="create_tag"):
            if new_tag_name:
                try:
                    tag = tag_manager.create_tag(new_tag_name, new_tag_color, new_tag_desc or None)
                    st.success(f"✅ Created tag '{tag.name}'")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to create tag: {e}")
            else:
                st.error("❌ Tag name is required")

    # List existing tags
    tags = tag_manager.get_all_tags()
    if tags:
        st.markdown("**Existing Tags:**")
        for tag in tags:
            col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
            with col1:
                st.markdown(f"🏷️ **{tag.name}**")
            with col2:
                st.markdown(f"🎨 {tag.color}")
            with col3:
                st.markdown(f"📝 {tag.description or 'No description'}")
            with col4:
                if st.button("🗑️", key=f"delete_tag_{tag.id}", help="Delete tag"):
                    try:
                        if tag_manager.delete_tag(tag.id):
                            st.success(f"✅ Deleted tag '{tag.name}'")
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete tag")
                    except Exception as e:
                        st.error(f"❌ Failed to delete tag: {e}")
    else:
        st.info("📭 No tags created yet")

def manage_categories():
    """Category management interface"""
    st.markdown("### 📂 Category Management")

    cat_manager = get_category_manager()

    # Create new category
    with st.expander("➕ Create New Category"):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            new_cat_name = st.text_input("Category Name", key="new_cat_name")
        with col2:
            new_cat_desc = st.text_input("Description (optional)", key="new_cat_desc")
        with col3:
            # Get existing categories for parent selection
            existing_cats = cat_manager.get_root_categories()
            parent_options = ["None (Root Category)"] + [cat.name for cat in existing_cats]
            selected_parent = st.selectbox("Parent Category", parent_options, key="new_cat_parent")

        if st.button("Create Category", key="create_cat"):
            if new_cat_name:
                try:
                    parent_id = None
                    if selected_parent != "None (Root Category)":
                        parent_cat = cat_manager.get_category_by_name(selected_parent)
                        parent_id = parent_cat.id if parent_cat else None

                    cat = cat_manager.create_category(new_cat_name, new_cat_desc or None, parent_id)
                    st.success(f"✅ Created category '{cat.name}'")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to create category: {e}")
            else:
                st.error("❌ Category name is required")

    # List existing categories
    root_cats = cat_manager.get_root_categories()
    if root_cats:
        st.markdown("**Category Tree:**")
        for cat in root_cats:
            display_category_tree(cat, cat_manager, level=0)
    else:
        st.info("📭 No categories created yet")

def display_category_tree(category, cat_manager, level=0):
    """Recursively display category tree"""
    indent = "  " * level
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        st.markdown(f"{indent}📂 **{category.name}**")
    with col2:
        st.markdown(f"{indent}📝 {category.description or 'No description'}")
    with col3:
        if st.button("🗑️", key=f"delete_cat_{category.id}", help="Delete category"):
            try:
                if cat_manager.delete_category(category.id):
                    st.success(f"✅ Deleted category '{category.name}'")
                    st.rerun()
                else:
                    st.error("❌ Failed to delete category")
            except Exception as e:
                st.error(f"❌ Failed to delete category: {e}")

    # Display children
    children = cat_manager.get_category_children(category.id)
    for child in children:
        display_category_tree(child, cat_manager, level + 1)

def ai_enrich_documents():
    """AI-powered document enrichment interface"""
    st.markdown("### 🤖 AI Document Enrichment")
    st.markdown("Use AI to automatically generate summaries, tags, and topics for your documents.")

    try:
        enrichment_service = AIEnrichmentService()

        # Get unenriched documents
        db = SessionLocal()
        enriched_docs = db.query(Document).filter(
            Document.custom_fields.op('->>')('ai_enriched') == 'true'
        ).count()

        total_docs = db.query(Document).count()
        unenriched_docs = total_docs - enriched_docs
        db.close()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", total_docs)
        with col2:
            st.metric("AI Enriched", enriched_docs)
        with col3:
            st.metric("Pending Enrichment", unenriched_docs)

        # Enrichment options
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🔄 Enrich All Unenriched Documents", type="primary", use_container_width=True):
                with st.spinner("🤖 AI enrichment in progress..."):
                    try:
                        db = SessionLocal()
                        unenriched = db.query(Document).filter(
                            ~Document.custom_fields.op('->>')('ai_enriched').isnot(None) |
                            (Document.custom_fields.op('->>')('ai_enriched') != 'true')
                        ).all()
                        doc_ids = [doc.id for doc in unenriched]
                        db.close()

                        if doc_ids:
                            results = enrichment_service.batch_enrich_documents(doc_ids)
                            st.success(f"✅ Enriched {results['successful']} documents, {results['failed']} failed")
                        else:
                            st.info("ℹ️ All documents are already enriched")
                    except Exception as e:
                        st.error(f"❌ Enrichment failed: {e}")

        with col2:
            if st.button("🔄 Re-enrich All Documents", type="secondary", use_container_width=True):
                with st.spinner("🤖 Re-enrichment in progress..."):
                    try:
                        db = SessionLocal()
                        all_docs = db.query(Document).all()
                        doc_ids = [doc.id for doc in all_docs]
                        db.close()

                        if doc_ids:
                            results = enrichment_service.batch_enrich_documents(doc_ids, force=True)
                            st.success(f"✅ Re-enriched {results['successful']} documents, {results['failed']} failed")
                        else:
                            st.info("ℹ️ No documents to enrich")
                    except Exception as e:
                        st.error(f"❌ Re-enrichment failed: {e}")

        # Manual enrichment for specific document
        st.markdown("---")
        st.markdown("**Manual Enrichment**")

        # Get document list for selection
        try:
            processor = DocumentProcessor()
            docs = processor.get_documents()

            if docs:
                doc_options = ["Select a document..."] + [f"{doc['filename']} (ID: {doc['id']})" for doc in docs]
                selected_doc = st.selectbox("Select document to enrich", doc_options)

                if selected_doc != "Select a document...":
                    # Extract document ID
                    import re
                    match = re.search(r'\(ID: (\d+)\)', selected_doc)
                    if match:
                        doc_id = int(match.group(1))

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🤖 Enrich This Document", key=f"enrich_{doc_id}"):
                                with st.spinner("🤖 Enriching document..."):
                                    result = enrichment_service.enrich_document(doc_id)
                                    if result['success']:
                                        st.success("✅ Document enriched successfully!")

                                        # Show enrichment results
                                        enrichment = result['enrichment']
                                        st.markdown("**Generated Summary:**")
                                        st.info(enrichment.get('summary', 'No summary generated'))

                                        if enrichment.get('tags'):
                                            st.markdown("**Suggested Tags:**")
                                            st.write(", ".join(enrichment['tags']))

                                        if enrichment.get('topics'):
                                            st.markdown("**Extracted Topics:**")
                                            st.write(", ".join(enrichment['topics']))
                                    else:
                                        st.error(f"❌ Enrichment failed: {result['error']}")

                        with col2:
                            if st.button("🔄 Re-enrich (Force)", key=f"reenrich_{doc_id}"):
                                with st.spinner("🤖 Re-enriching document..."):
                                    result = enrichment_service.enrich_document(doc_id, force=True)
                                    if result['success']:
                                        st.success("✅ Document re-enriched successfully!")
                                    else:
                                        st.error(f"❌ Re-enrichment failed: {result['error']}")
            else:
                st.info("📭 No documents available for enrichment")

        except Exception as e:
            st.error(f"❌ Failed to load documents: {e}")

    except Exception as e:
        st.error(f"❌ AI enrichment service unavailable: {e}")
        st.info("💡 Make sure Ollama is running and the LLM model is available for AI features.")



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

    # Organization management
    tab1, tab2, tab3 = st.tabs(["📚 Documents", "🏷️ Tags", "📂 Categories"])

    with tab1:
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

    with tab2:
        manage_tags()

    with tab3:
        manage_categories()

    with st.expander("🤖 AI Enrichment", expanded=False):
        ai_enrich_documents()

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