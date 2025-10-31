"""
Session management utilities for the Local RAG Web Interface
"""

import streamlit as st
from datetime import datetime
import yaml
import os

def load_settings():
    """Load settings from YAML file"""
    settings_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_settings.yaml')
    try:
        with open(settings_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Could not load settings: {e}")
        return {}

def initialize_session_state():
    """Initialize all session state variables"""
    settings = load_settings()

    # System state
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'rag_available' not in st.session_state:
        st.session_state.rag_available = False

    # Query state
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = 0

    # History
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    # Settings
    if 'settings' not in st.session_state:
        st.session_state.settings = settings

    # UI state
    if 'sidebar_expanded' not in st.session_state:
        st.session_state.sidebar_expanded = True

def update_settings(new_settings):
    """Update session settings"""
    st.session_state.settings.update(new_settings)

def add_query_to_history(query, mode, processing_time, results=None):
    """Add a query to the history"""
    history_item = {
        'query': query,
        'mode': mode,
        'timestamp': datetime.now(),
        'processing_time': processing_time,
        'results': results
    }

    st.session_state.query_history.insert(0, history_item)

    # Keep only the configured maximum
    max_history = st.session_state.settings.get('system', {}).get('max_query_history', 10)
    st.session_state.query_history = st.session_state.query_history[:max_history]

def clear_history():
    """Clear query history"""
    st.session_state.query_history = []

def get_available_embedding_models():
    """Get list of available sentence-transformers models"""
    # Common embedding models to check - focus on the most commonly used ones
    candidate_models = [
        "all-MiniLM-L6-v2",  # Most common and well-tested
        "all-mpnet-base-v2",  # Good performance
        "paraphrase-multilingual-mpnet-base-v2"  # Multilingual support
    ]

    available_models = []

    try:
        from sentence_transformers import SentenceTransformer
        import threading

        def test_model(model_name, results, index):
            """Test if a model can be loaded within timeout"""
            try:
                # Try to load model with a short timeout
                # This will download if not cached, but should be fast for cached models
                model = SentenceTransformer(model_name, device='cpu')
                results[index] = model_name
                del model  # Clean up
            except Exception:
                results[index] = None

        # Test models with timeout (since downloading can take time)
        threads = []
        results = [None] * len(candidate_models)

        for i, model_name in enumerate(candidate_models):
            thread = threading.Thread(target=test_model, args=(model_name, results, i))
            thread.daemon = True
            threads.append(thread)
            thread.start()

        # Wait for all threads with timeout
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout per model

        # Collect successful results
        for result in results:
            if result is not None:
                available_models.append(result)

    except ImportError:
        st.warning("sentence-transformers not available. Using default model list.")
        return ["all-MiniLM-L6-v2"]
    except Exception as e:
        st.warning(f"Error checking embedding models: {e}")
        return ["all-MiniLM-L6-v2"]

    # Always include at least the default model
    if not available_models:
        available_models = ["all-MiniLM-L6-v2"]

    return available_models

def get_system_status():
    """Get current system status"""
    return {
        'initialized': st.session_state.system_initialized,
        'retriever_active': st.session_state.retriever is not None,
        'rag_available': st.session_state.rag_available,
        'total_queries': len(st.session_state.query_history)
    }