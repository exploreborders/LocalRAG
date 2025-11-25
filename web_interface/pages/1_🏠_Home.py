#!/usr/bin/env python3
"""
Home Page - Main Query Interface
"""

import streamlit as st
import time
import os
import sys
from pathlib import Path

# Find project root (two levels up from /web_interface/pages/)
ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "src"
WEB = ROOT / "web_interface"

for p in (SRC, WEB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Import system components
try:
    from src.core.retrieval import (
        DatabaseRetriever,
        RAGPipelineDB,
        format_results_db,
        format_answer_db,
    )
except ImportError:
    st.error(
        "❌ Could not import RAG system components. Please ensure you're running from the project root."
    )
    st.stop()

# Import web interface components (after path setup)
try:
    from web_interface.utils.session_manager import (
        load_settings,
        update_settings,
        initialize_session_state,
        add_query_to_history,
    )
    from web_interface.components.query_interface import (
        render_query_input,
        render_submit_button,
    )
    from web_interface.components.results_display import render_results
except ImportError as e:
    st.error(f"❌ Could not import web interface components: {e}")
    st.stop()

# Page configuration
st.set_page_config(page_title="Local RAG - Home", page_icon="🏠", layout="wide")

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .welcome-text {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_system(show_ui=True):
    """Initialize the RAG system components

    Args:
        show_ui (bool): Whether to show UI feedback during initialization
    """
    try:
        if show_ui:
            with st.spinner("🔄 Initializing Local RAG System..."):
                result = _do_initialization()
            if result["success"]:
                st.success(
                    f"✅ System initialized successfully with {result['embedding_model']} embeddings!"
                )
            if result["rag_error"]:
                st.warning(f"⚠️ RAG mode unavailable: {result['rag_error']}")
        else:
            result = _do_initialization()

        return result

    except Exception as e:
        if show_ui:
            st.error(f"❌ Failed to initialize system: {str(e)}")
        st.session_state.system_initialized = False
        return {"success": False, "error": str(e)}


def _do_initialization():
    """Core initialization logic without UI"""
    import warnings
    import logging

    # Suppress common PyTorch warnings that are usually harmless
    warnings.filterwarnings("ignore", message=".*torch.classes.*")
    warnings.filterwarnings("ignore", message=".*Examining the path of torch.classes.*")

    # Reduce logging verbosity for cleaner output
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    # Get configured models from settings
    from web_interface.utils.session_manager import load_settings

    settings = load_settings()
    embedding_model = settings.get("retrieval", {}).get(
        "embedding_model", "nomic-ai/nomic-embed-text-v1.5"
    )
    llm_model = settings.get("generation", {}).get("model", "llama2")
    cache_enabled = settings.get("cache", {}).get("enabled", True)
    cache_settings = settings.get("cache", {})

    # Initialize retriever with configured embedding model
    st.session_state.retriever = DatabaseRetriever(embedding_model)

    # Try to initialize RAG pipeline with configured LLM (may fail if Ollama not running)
    rag_error = None
    try:
        st.session_state.rag_pipeline = RAGPipelineDB(
            embedding_model,
            llm_model,
            cache_enabled=cache_enabled,
            cache_settings=cache_settings,
        )
        st.session_state.rag_available = True
    except Exception as e:
        st.session_state.rag_pipeline = None
        st.session_state.rag_available = False
        rag_error = str(e)

    st.session_state.system_initialized = True

    return {"success": True, "embedding_model": embedding_model, "rag_error": rag_error}


def process_query(query, mode="topic-aware", filters=None):
    """
    Process a query using the selected mode and return results.
    """
    # Ensure session state is initialized
    if "query_history" not in st.session_state:
        initialize_session_state()

    start_time = time.time()

    try:
        if mode == "topic-aware":
            if st.session_state.retriever is None:
                raise Exception("Retriever not initialized")

            results = st.session_state.retriever.retrieve_with_topic_boost(
                query, top_k=3, filters=filters or {}
            )
            formatted_results = format_results_db(results)

            # Store results for display
            st.session_state.current_results = {
                "type": "topic-aware",
                "query": query,
                "results": results,
                "formatted": formatted_results,
                "filters": filters,
            }

        elif mode == "rag":
            if st.session_state.rag_pipeline is None:
                raise Exception(
                    "RAG pipeline not available. Please ensure Ollama is running."
                )

            result = st.session_state.rag_pipeline.query(
                query, top_k=3, filters=filters or {}
            )
            formatted_answer = format_answer_db(result["answer"])

        # Store results for display
        st.session_state.current_results = {
            "type": "rag",
            "query": query,
            "result": result,
            "formatted": formatted_answer,
            "filters": filters,
            "processing_info": {
                "ocr_used": getattr(st.session_state, "last_upload_ocr_used", False),
                "is_scanned": getattr(
                    st.session_state, "last_upload_is_scanned", False
                ),
            },
        }

        # Add to query history
        processing_time = time.time() - start_time
        add_query_to_history(
            query, mode, processing_time, st.session_state.current_results
        )

        st.session_state.processing_time = processing_time

        return True

    except Exception as e:
        st.error(f"❌ Query processing failed: {str(e)}")
        return False


def main():
    """Main page content"""
    # Initialize session state first
    initialize_session_state()

    st.markdown(
        '<h1 class="page-header">🏠 Local RAG - Ask Questions</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("Ask questions about your documents and get AI-powered answers")

    # Auto-initialize system if not already done
    if not st.session_state.get("system_initialized", False):
        initialize_system(show_ui=True)
        # Continue to render the interface below

    # Success message
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ Retriever Active")
    with col2:
        if st.session_state.get("rag_available", False):
            st.success("✅ RAG Pipeline Active")
        else:
            st.info("ℹ️ RAG Pipeline Offline")

    st.markdown("---")

    # Query interface
    query, mode, filters = render_query_input()

    # Submit button and processing
    if render_submit_button(query, mode):
        with st.spinner("🔄 Processing query..."):
            success = process_query(query.strip(), mode, filters)
            if success:
                st.success("✅ Query processed successfully!")
                st.rerun()

    # Display results
    render_results()


if __name__ == "__main__":
    main()
