"""
Query interface components for the Local RAG Web Interface
"""

from datetime import datetime
from typing import Any, Dict

import streamlit as st


def render_query_input():
    """Render the main query input interface"""
    st.markdown("### 🔍 Ask a Question")

    # Query input area
    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "Enter your question:",
            placeholder="What is Retrieval-Augmented Generation?",
            key="query_input",
            label_visibility="collapsed",
            value=st.session_state.get("current_query", ""),
        )

    with col2:
        mode_options = (
            ["topic-aware", "rag"]
            if st.session_state.get("rag_available", False)
            else ["topic-aware"]
        )
        mode_labels = {"topic-aware": "🎯 Smart Search", "rag": "🤖 Full RAG"}

        def format_mode(x):
            return mode_labels.get(x, str(x))

        mode = st.selectbox(
            "Mode",
            mode_options,
            format_func=format_mode,
            help="Smart Search uses intelligent topic relevance boosting. Full RAG generates AI answers with source citations.",
            key="query_mode",
        )

    # Advanced filters
    filters = render_advanced_filters()

    # Update session state
    st.session_state.current_query = query
    st.session_state.current_filters = filters

    return query, mode, filters


def render_advanced_filters() -> Dict[str, Any]:
    """Render advanced search filters with faceted search support"""
    filters = {}

    # Get current query for faceted search
    current_query = st.session_state.get("current_query", "")

    with st.expander("🔧 Advanced Filters", expanded=False):
        # Search mode selection
        search_mode = st.radio(
            "Search Mode",
            ["Standard Search", "Advanced Query", "Faceted Search"],
            help="Choose search mode: Standard (simple), Advanced (boolean operators), Faceted (dynamic filters)",
        )

        if search_mode == "Advanced Query":
            render_advanced_query_help()

        col1, col2, col3 = st.columns(3)

        with col1:
            # Tag filter with facets
            if search_mode == "Faceted Search" and current_query:
                render_faceted_tag_filter(current_query, filters)
            else:
                render_basic_tag_filter(filters)

        with col2:
            # Category filter with facets
            if search_mode == "Faceted Search" and current_query:
                render_faceted_category_filter(current_query, filters)
            else:
                render_basic_category_filter(filters)

        with col3:
            # Language filter
            render_language_filter(filters)

        # Date range filter
        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input(
                "From Date",
                value=None,
                help="Only include documents uploaded after this date",
            )
            if date_from:
                filters["date_from"] = date_from.isoformat()

        with col2:
            date_to = st.date_input(
                "To Date",
                value=None,
                help="Only include documents uploaded before this date",
            )
            if date_to:
                filters["date_to"] = date_to.isoformat()

        # Author filter
        author = st.text_input(
            "Author",
            placeholder="Filter by document author",
            help="Only search documents by this author",
        )
        if author.strip():
            filters["author"] = author.strip()

        # File type filter
        file_types = ["pdf", "txt", "md", "docx", "html", "json", "xml"]
        selected_file_types = st.multiselect(
            "File Types",
            file_types,
            help="Filter by document file types",
        )
        if selected_file_types:
            filters["file_types"] = selected_file_types

        # Hybrid search weights
        if search_mode in ["Standard Search", "Faceted Search"]:
            col1, col2 = st.columns(2)
            with col1:
                vector_weight = st.slider(
                    "Vector Similarity Weight",
                    0.0,
                    1.0,
                    0.7,
                    help="Weight for semantic similarity search",
                )
                filters["vector_weight"] = vector_weight

            with col2:
                bm25_weight = st.slider(
                    "Keyword Search Weight",
                    0.0,
                    1.0,
                    0.3,
                    help="Weight for keyword-based search",
                )
                filters["bm25_weight"] = bm25_weight

    # Store search mode in filters
    filters["search_mode"] = search_mode.lower().replace(" ", "_")

    return filters


def render_advanced_query_help():
    """Render help text for advanced query syntax"""
    st.markdown(
        """
    **Advanced Query Syntax:**
    - **Boolean operators**: `AND`, `OR`, `NOT`
    - **Field search**: `title:topic`, `tags:machine learning`
    - **Phrase search**: `"exact phrase"`
    - **Wildcards**: `analys*`, `test~` (fuzzy)
    - **Exclusion**: `-exclude_this`
    - **Examples**:
        - `machine learning AND deep learning`
        - `title:"neural networks" OR tags:AI`
        - `author:Smith NOT draft`
    """
    )


def render_faceted_tag_filter(query: str, filters: Dict[str, Any]):
    """Render tag filter with faceted search data"""
    try:
        from src.core.advanced_search import HybridSearchEngine

        search_engine = HybridSearchEngine()
        facets = search_engine.get_search_facets(query)

        tag_options = [f"{tag['value']} ({tag['count']})" for tag in facets.get("tags", [])]
        selected_display = st.multiselect(
            "Filter by Tags (with counts)",
            tag_options,
            help="Tags available in current search results",
        )

        # Extract tag names from display
        selected_tags = [tag.split(" (")[0] for tag in selected_display]
        if selected_tags:
            filters["tags"] = selected_tags

    except Exception as e:
        st.warning(f"Faceted tag filtering unavailable: {e}")
        render_basic_tag_filter(filters)


def render_basic_tag_filter(filters: Dict[str, Any]):
    """Render basic tag filter"""
    try:
        from src.core.tagging.tag_manager import TagManager
        from src.database.models import SessionLocal

        db = SessionLocal()
        tag_manager = TagManager(db)
        tags = tag_manager.get_all_tags()
        tag_names = [tag.name for tag in tags]
        selected_tags = st.multiselect(
            "Filter by Tags",
            tag_names,
            help="Only search documents with these tags",
        )
        if selected_tags:
            filters["tags"] = selected_tags
        db.close()
    except Exception:
        st.warning("Tag filtering unavailable")


def render_faceted_category_filter(query: str, filters: Dict[str, Any]):
    """Render category filter with faceted search data"""
    try:
        from src.core.advanced_search import HybridSearchEngine

        search_engine = HybridSearchEngine()
        facets = search_engine.get_search_facets(query)

        cat_options = [f"{cat['value']} ({cat['count']})" for cat in facets.get("categories", [])]
        selected_display = st.multiselect(
            "Filter by Categories (with counts)",
            cat_options,
            help="Categories available in current search results",
        )

        # Extract category names from display
        selected_cats = [cat.split(" (")[0] for cat in selected_display]
        if selected_cats:
            filters["categories"] = selected_cats

    except Exception as e:
        st.warning(f"Faceted category filtering unavailable: {e}")
        render_basic_category_filter(filters)


def render_basic_category_filter(filters: Dict[str, Any]):
    """Render basic category filter"""
    try:
        from src.core.categorization.category_manager import CategoryManager
        from src.database.models import SessionLocal

        db = SessionLocal()
        cat_manager = CategoryManager(db)
        root_cats = cat_manager.get_root_categories()
        cat_names = [cat.name for cat in root_cats]
        selected_cats = st.multiselect(
            "Filter by Categories",
            cat_names,
            help="Only search documents in these categories",
        )
        if selected_cats:
            filters["categories"] = selected_cats
        db.close()
    except Exception:
        st.warning("Category filtering unavailable")


def render_language_filter(filters: Dict[str, Any]):
    """Render language filter"""
    languages = ["en", "de", "fr", "es", "it", "pt", "nl", "sv", "pl", "zh", "ja", "ko"]
    lang_labels = {
        "en": "🇺🇸 English",
        "de": "🇩🇪 German",
        "fr": "🇫🇷 French",
        "es": "🇪🇸 Spanish",
        "it": "🇮🇹 Italian",
        "pt": "🇵🇹 Portuguese",
        "nl": "🇳🇱 Dutch",
        "sv": "🇸🇪 Swedish",
        "pl": "🇵🇱 Polish",
        "zh": "🇨🇳 Chinese",
        "ja": "🇯🇵 Japanese",
        "ko": "🇰🇷 Korean",
    }
    selected_lang = st.selectbox(
        "Language",
        ["All"] + languages,
        format_func=lambda x: (
            lang_labels.get(x, x.upper() if x else "unknown") if x != "All" else "🌍 All Languages"
        ),
        help="Filter by document language",
    )
    if selected_lang != "All":
        filters["detected_language"] = selected_lang


def render_submit_button(query, mode):
    """Render the submit button and handle query processing"""
    disabled = not query.strip() or not st.session_state.get("system_initialized", False)

    if st.button("🔎 Search", type="primary", disabled=disabled):
        if not query.strip():
            st.warning("⚠️ Please enter a question")
            return False
        elif not st.session_state.get("system_initialized", False):
            st.warning("⚠️ Please initialize the system first")
            return False
        else:
            return True
    return False


def render_processing_status():
    """Render processing status during query execution"""
    if st.session_state.get("processing_time", 0) > 0:
        st.info(f"⏱️ Last query processed in {st.session_state.processing_time:.2f} seconds")


def render_query_history():
    """Render query history in sidebar"""
    st.sidebar.markdown("### 📚 Query History")

    history = st.session_state.get("query_history", [])

    if not history:
        st.sidebar.info("No queries yet")
        return

    # Show last 5 queries
    for i, item in enumerate(history[:5]):
        timestamp = item.get("timestamp", datetime.now())
        query_text = item.get("query", "")[:40]
        if len(item.get("query", "")) > 40:
            query_text += "..."

        mode_icon = "📄" if item.get("mode") == "retrieval" else "🤖"

        with st.sidebar.expander(f"{mode_icon} {query_text}", expanded=(i == 0)):
            st.write(f"**Mode:** {item.get('mode', 'unknown').title()}")
            st.write(f"**Time:** {timestamp.strftime('%H:%M:%S')}")
            if "processing_time" in item:
                st.write(f"**Duration:** {item['processing_time']:.2f}s")


def clear_query_history():
    """Clear the query history"""
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.query_history = []
        st.sidebar.success("History cleared!")
        st.rerun()
