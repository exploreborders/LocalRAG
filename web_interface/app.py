#!/usr/bin/env python3
"""
Local RAG System - Web Interface Landing Page
Main entry point for the multipage Streamlit application
"""

import streamlit as st
import warnings
import logging
import sys
from pathlib import Path

# Suppress common warnings that are usually harmless
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*Examining the path of torch.classes.*")
warnings.filterwarnings("ignore", message=".*huggingface/tokenizers.*")
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")
warnings.filterwarnings("ignore", message=".*does not exist.*")

# Reduce logging verbosity for cleaner output
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("huggingface").setLevel(logging.WARNING)

# Add src to path for imports
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
WEB = ROOT / "web_interface"

# Add all necessary paths
for path in [ROOT, SRC, WEB]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Also ensure PYTHONPATH is set
import os

current_pythonpath = os.environ.get("PYTHONPATH", "")
paths_to_add = [str(ROOT), str(SRC), str(WEB)]
for path in paths_to_add:
    if path not in current_pythonpath:
        if current_pythonpath:
            current_pythonpath = f"{path}:{current_pythonpath}"
        else:
            current_pythonpath = path
os.environ["PYTHONPATH"] = current_pythonpath


def check_system_setup():
    """Check if the RAG system is properly set up."""
    issues = []

    # Check if we're in virtual environment
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    if not in_venv:
        issues.append("Not running in virtual environment")

    # Check required packages
    required_packages = [
        "streamlit",
        "sqlalchemy",
        "sentence_transformers",
        "docling",
        "psycopg2",
        "elasticsearch",
        "redis",
    ]

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Missing package: {package}")

    # Check database connectivity
    try:
        from src.database.models import SessionLocal

        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
    except Exception as e:
        issues.append(f"Database connection failed: {e}")

    return issues


# Page configuration
st.set_page_config(
    page_title="Local RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .welcome-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .welcome-subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        text-align: center;
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .feature-title {
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .feature-description {
        color: #7f8c8d;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    """Main landing page"""
    st.markdown(
        '<h1 class="welcome-header">🤖 Local RAG System</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="welcome-subtitle">Retrieval-Augmented Generation for local document analysis</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Quick start section
    st.markdown("### 🚀 Quick Start")
    st.markdown("Navigate using the sidebar to explore different features:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🏠 Home**: Ask questions and get answers
        - Natural language queries
        - Retrieval-only or AI-powered responses
        - Query history and performance tracking
        """)

        st.markdown("""
        **📁 Documents**: Manage your knowledge base
        - Upload new documents (PDF, DOCX, TXT, etc.)
        - View document library
        - Reprocess documents for updates
        """)

    with col2:
        st.markdown("""
        **⚙️ Settings**: Configure system parameters
        - Adjust retrieval and generation settings
        - Choose embedding models and parameters
        - Customize interface preferences
        """)

        st.markdown("""
        **📊 Analytics**: Monitor performance
        - Query statistics and trends
        - System health metrics
        - Performance analytics and exports
        """)

    st.markdown("---")

    # Feature highlights
    st.markdown("### ✨ Key Features")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Smart Search</div>
            <div class="feature-description">Advanced retrieval using vector similarity</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI Generation</div>
            <div class="feature-description">Ollama-powered answer generation</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="feature-card">
            <div class="feature-icon">📁</div>
            <div class="feature-title">Multi-Format</div>
            <div class="feature-description">Support for PDF, DOCX, XLSX, PPTX, TXT</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
        <div class="feature-card">
            <div class="feature-icon">🔒</div>
            <div class="feature-title">Local & Private</div>
            <div class="feature-description">All processing happens locally</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Getting started guide
    st.markdown("### 📚 Getting Started")

    with st.expander("🔧 System Setup"):
        st.markdown("""
        1. **System auto-initializes** when you first visit the Home page
        2. **Upload documents** via the Documents page
        3. **Reprocess documents** to create embeddings
        4. **Start asking questions!**
        """)

    with st.expander("🤖 RAG Mode Setup"):
        st.markdown("""
        For AI-powered answers, you'll need Ollama:
        1. Install Ollama from https://ollama.ai
        2. Run `ollama serve` in a terminal
        3. Pull a model: `ollama pull llama2`
        4. The system will automatically detect RAG availability
        """)

    with st.expander("📊 Performance Tips"):
        st.markdown("""
        - **Faster responses**: Reduce chunk size or k-value
        - **Better accuracy**: Increase k-value or use larger chunks
        - **Memory efficiency**: Use smaller embedding models
        - **Batch processing**: Upload multiple documents at once
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "*Built with Streamlit, LangChain, FAISS, and Ollama • [View Source](https://github.com/your-repo)*"
    )


if __name__ == "__main__":
    main()
