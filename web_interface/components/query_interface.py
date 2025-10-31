"""
Query interface components for the Local RAG Web Interface
"""

import streamlit as st
from datetime import datetime

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
            value=st.session_state.get('current_query', '')
        )

    with col2:
        mode_options = ["retrieval", "rag"] if st.session_state.get('rag_available', False) else ["retrieval"]
        mode_labels = {
            "retrieval": "📄 Retrieval Only",
            "rag": "🤖 Full RAG"
        }

        mode = st.selectbox(
            "Mode",
            mode_options,
            format_func=lambda x: mode_labels.get(x, x),
            help="Retrieval only searches documents. Full RAG generates AI answers.",
            key="query_mode"
        )

    # Update session state
    st.session_state.current_query = query

    return query, mode

def render_submit_button(query, mode):
    """Render the submit button and handle query processing"""
    disabled = not query.strip() or not st.session_state.get('system_initialized', False)

    if st.button(
        "🔎 Search",
        type="primary",
        use_container_width=True,
        disabled=disabled
    ):
        if not query.strip():
            st.warning("⚠️ Please enter a question")
            return False
        elif not st.session_state.get('system_initialized', False):
            st.warning("⚠️ Please initialize the system first")
            return False
        else:
            return True
    return False

def render_processing_status():
    """Render processing status during query execution"""
    if st.session_state.get('processing_time', 0) > 0:
        st.info(f"⏱️ Last query processed in {st.session_state.processing_time:.2f} seconds")

def render_query_history():
    """Render query history in sidebar"""
    st.sidebar.markdown("### 📚 Query History")

    history = st.session_state.get('query_history', [])

    if not history:
        st.sidebar.info("No queries yet")
        return

    # Show last 5 queries
    for i, item in enumerate(history[:5]):
        timestamp = item.get('timestamp', datetime.now())
        query_text = item.get('query', '')[:40]
        if len(item.get('query', '')) > 40:
            query_text += "..."

        mode_icon = "📄" if item.get('mode') == 'retrieval' else "🤖"

        with st.sidebar.expander(f"{mode_icon} {query_text}", expanded=(i==0)):
            st.write(f"**Mode:** {item.get('mode', 'unknown').title()}")
            st.write(f"**Time:** {timestamp.strftime('%H:%M:%S')}")
            if 'processing_time' in item:
                st.write(f"**Duration:** {item['processing_time']:.2f}s")

def clear_query_history():
    """Clear the query history"""
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.query_history = []
        st.sidebar.success("History cleared!")
        st.rerun()