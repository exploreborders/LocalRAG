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
    # Currently only one model is supported for multilingual embeddings
    # nomic-ai/nomic-embed-text-v1.5 is the only model that supports all 12 languages
    return ["nomic-ai/nomic-embed-text-v1.5"]

def get_system_status():
    """Get current system status"""
    return {
        'initialized': st.session_state.system_initialized,
        'retriever_active': st.session_state.retriever is not None,
        'rag_available': st.session_state.rag_available,
        'total_queries': len(st.session_state.query_history)
    }