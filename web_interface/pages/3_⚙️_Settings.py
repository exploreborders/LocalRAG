#!/usr/bin/env python3
"""
Settings Page - Configuration Options
"""

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

# Find project root (two levels up from /web_interface/pages/)
ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "src"
WEB = ROOT / "web_interface"

for p in (SRC, WEB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Import utilities
from web_interface.utils.session_manager import (
    initialize_session_state,
    load_settings,
    update_settings,
)

# Page configuration
st.set_page_config(page_title="Local RAG - Settings", page_icon="⚙️", layout="wide")

# Custom CSS
st.markdown(
    """
<style>
    .page-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .setting-group {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    .setting-title {
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .setting-description {
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def save_settings_to_file(settings):
    """Save settings to YAML file and update Streamlit config"""
    import yaml

    settings_path = os.path.join(os.path.dirname(__file__), "..", "config", "default_settings.yaml")
    streamlit_config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", ".streamlit", "config.toml"
    )

    try:
        # Save our app settings
        with open(settings_path, "w") as f:
            yaml.dump(settings, f, default_flow_style=False, sort_keys=False)

        # Update Streamlit theme config
        theme = settings.get("interface", {}).get("theme", "light")
        update_streamlit_theme(streamlit_config_path, theme)

        return True
    except Exception as e:
        st.error(f"Failed to save settings: {e}")
        return False


def get_installed_ollama_models():
    """Get list of installed Ollama models"""
    try:
        # Run ollama list command
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:  # Skip header line
                # Parse model names from output
                models = []
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if parts:
                            model_name = parts[0].split(":")[0]  # Remove :latest tag
                            if model_name not in models:
                                models.append(model_name)
                return models if models else ["llama2"]  # Default fallback
        else:
            st.warning("Could not connect to Ollama. Using default model list.")
            return ["llama2"]

    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
        subprocess.SubprocessError,
    ) as e:
        st.warning(f"Ollama not available: {e}. Using default model.")
        return ["llama2"]


def get_available_embedding_models():
    # Only support embeddinggemma:latest via Ollama
    return ["embeddinggemma:latest"]


def update_streamlit_theme(config_path, theme):
    """Update Streamlit theme configuration"""
    try:
        # Read current config
        with open(config_path, "r") as f:
            content = f.read()

        # Update theme base
        if theme == "dark":
            # Dark theme colors
            theme_config = '''[theme]

# The preset Streamlit theme that your custom theme inherits from.
# One of "light" or "dark".
base = "dark"

# Primary accent color for interactive elements.
primaryColor = "#1f77b4"

# Background color for the main content area.
backgroundColor = "#0e1117"

# Background color for sidebar and most interactive widgets.
secondaryBackgroundColor = "#262730"

# Color used for almost all text.
textColor = "#fafafa"

# Font family for all text in the app, except code blocks. One of "sans serif",
# "serif", or "monospace".
font = "sans serif"'''
        else:
            # Light theme colors
            theme_config = '''[theme]

# The preset Streamlit theme that your custom theme inherits from.
# One of "light" or "dark".
base = "light"

# Primary accent color for interactive elements.
primaryColor = "#1f77b4"

# Background color for the main content area.
backgroundColor = "#ffffff"

# Background color for sidebar and most interactive widgets.
secondaryBackgroundColor = "#f8f9fa"

# Color used for almost all text.
textColor = "#262730"

# Font family for all text in the app, except code blocks. One of "sans serif",
# "serif", or "monospace".
font = "sans serif"'''

        # Replace theme section
        import re

        pattern = r"\[theme\].*?(?=\[|\Z)"
        new_content = re.sub(pattern, theme_config, content, flags=re.DOTALL)

        # Write back
        with open(config_path, "w") as f:
            f.write(new_content)

    except Exception as e:
        st.warning(f"Could not update Streamlit theme: {e}")


def reset_settings():
    """Reset settings to defaults"""
    # This would load default settings and save them
    st.info("⚠️ Reset functionality not implemented yet")


def main():
    """Main page content"""
    st.markdown('<h1 class="page-header">⚙️ Settings</h1>', unsafe_allow_html=True)
    st.markdown("Configure your Local RAG system parameters")

    # Load current settings
    settings = load_settings()

    # Track if settings have changed
    settings_changed = False

    # Generation Settings
    st.markdown("### 🤖 Generation Settings")
    with st.container():
        st.markdown('<div class="setting-group">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=settings.get("generation", {}).get("temperature", 0.7),
                step=0.1,
                help="Controls randomness in AI responses (0.0 = deterministic, 1.0 = very random)",
            )

            max_tokens = st.slider(
                "Max Tokens",
                min_value=100,
                max_value=2000,
                value=settings.get("generation", {}).get("max_tokens", 500),
                step=50,
                help="Maximum length of AI-generated responses",
            )

        with col2:
            ollama_host = st.text_input(
                "Ollama Host",
                value=settings.get("generation", {}).get("ollama_host", "http://localhost:11434"),
                help="URL where Ollama server is running",
            )

            # Get installed models dynamically
            installed_models = get_installed_ollama_models()
            current_model = settings.get("generation", {}).get("model", "llama2")

            # Ensure installed_models is a list
            if not isinstance(installed_models, list) or not installed_models:
                installed_models = ["llama2"]

            # Ensure current model is in the list, otherwise use first available
            if current_model not in installed_models:
                current_model = installed_models[0]

            model_name = st.selectbox(
                "LLM Model",
                installed_models,
                index=(
                    installed_models.index(current_model)
                    if current_model in installed_models
                    else 0
                ),
                help=f"Available Ollama models: {', '.join(installed_models)}",
            )

        # Update settings if changed
        if (
            temperature != settings.get("generation", {}).get("temperature", 0.7)
            or max_tokens != settings.get("generation", {}).get("max_tokens", 500)
            or ollama_host
            != settings.get("generation", {}).get("ollama_host", "http://localhost:11434")
            or model_name != settings.get("generation", {}).get("model", "llama2")
        ):
            settings_changed = True
            settings["generation"] = {
                "model": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "ollama_host": ollama_host,
                "ollama_port": 11434,
            }

        st.markdown("</div>", unsafe_allow_html=True)

    # Retrieval Settings
    st.markdown("### 🔍 Retrieval Settings")
    with st.container():
        st.markdown('<div class="setting-group">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            chunk_size = st.slider(
                "Chunk Size",
                min_value=500,
                max_value=2000,
                value=settings.get("retrieval", {}).get("chunk_size", 1000),
                step=100,
                help="Size of text chunks for embedding",
            )

            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=settings.get("retrieval", {}).get("chunk_overlap", 200),
                step=50,
                help="Overlap between text chunks",
            )

        with col2:
            k_retrieval = st.slider(
                "Retrieval Count (k)",
                min_value=1,
                max_value=10,
                value=settings.get("retrieval", {}).get("k_retrieval", 3),
                help="Number of documents to retrieve for each query",
            )

            # Get available embedding models
            available_embedding_models = get_available_embedding_models()
            current_embedding_model = settings.get("retrieval", {}).get(
                "embedding_model", "embeddinggemma:latest"
            )

            embedding_model = st.selectbox(
                "Embedding Model",
                available_embedding_models,
                index=(
                    available_embedding_models.index(current_embedding_model)
                    if current_embedding_model in available_embedding_models
                    else 0
                ),
                help="Embedding model (uses Ollama backend)",
            )

            # Embedding backend is fixed to Ollama for embeddinggemma:latest
            embedding_backend = "ollama"

        # Update settings if changed
        if (
            chunk_size != settings.get("retrieval", {}).get("chunk_size", 1000)
            or chunk_overlap != settings.get("retrieval", {}).get("chunk_overlap", 200)
            or k_retrieval != settings.get("retrieval", {}).get("k_retrieval", 3)
            or embedding_model
            != settings.get("retrieval", {}).get("embedding_model", "embeddinggemma:latest")
        ):
            settings_changed = True
            settings["retrieval"] = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "k_retrieval": k_retrieval,
                "embedding_model": embedding_model,
                "embedding_backend": "ollama",  # Fixed backend
            }

        st.markdown("</div>", unsafe_allow_html=True)

    # Interface Settings
    st.markdown("### 🎨 Interface Settings")
    with st.container():
        st.markdown('<div class="setting-group">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            max_results = st.slider(
                "Max Results Display",
                min_value=1,
                max_value=10,
                value=settings.get("interface", {}).get("max_results_display", 5),
                help="Maximum number of results to display",
            )

            max_history = st.slider(
                "Query History Size",
                min_value=5,
                max_value=50,
                value=settings.get("system", {}).get("max_query_history", 10),
                help="Number of queries to keep in history",
            )

        with col2:
            theme = st.selectbox(
                "Theme",
                ["light", "dark"],
                index=["light", "dark"].index(settings.get("interface", {}).get("theme", "light")),
                help="UI theme - requires app restart to take effect",
            )

        # Update settings if changed
        if max_results != settings.get("interface", {}).get(
            "max_results_display", 5
        ) or theme != settings.get("interface", {}).get("theme", "light"):
            settings_changed = True
            settings["interface"] = {"theme": theme, "max_results_display": max_results}

        if max_history != settings.get("system", {}).get("max_query_history", 10):
            settings_changed = True
            settings["system"] = settings.get("system", {})
            settings["system"]["max_query_history"] = max_history

        st.markdown("</div>", unsafe_allow_html=True)

    # Cache Settings
    st.markdown("### 🚀 Cache Settings")
    with st.container():
        st.markdown('<div class="setting-group">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            cache_enabled = st.checkbox(
                "Enable LLM Response Caching",
                value=settings.get("cache", {}).get("enabled", True),
                help="Cache LLM responses to improve performance for repeated queries",
            )

            cache_ttl = st.slider(
                "Cache TTL (hours)",
                min_value=1,
                max_value=168,  # 1 week
                value=settings.get("cache", {}).get("ttl_hours", 24),
                help="How long to keep cached responses (in hours)",
            )

        with col2:
            # Cache status and controls
            try:
                if (
                    hasattr(st.session_state, "rag_pipeline")
                    and st.session_state.rag_pipeline
                    and st.session_state.rag_pipeline.cache
                ):
                    cache_stats = st.session_state.rag_pipeline.cache.get_stats()
                    st.metric("Cached Responses", cache_stats.get("total_keys", 0))
                    st.metric("Memory Used", cache_stats.get("memory_used", "unknown"))
                    st.metric("Hit Rate", f"{cache_stats.get('hit_rate', 0):.1%}")

                    # Clear cache button
                    if st.button("🗑️ Clear Cache", type="secondary"):
                        cleared = st.session_state.rag_pipeline.cache.clear_pattern("llm:*")
                        st.success(f"✅ Cleared {cleared} cached responses")
                        st.rerun()
                else:
                    st.info("ℹ️ Cache is disabled or not configured")
            except Exception as e:
                st.error(f"❌ Cache status error: {e}")
            except Exception as e:
                st.error(f"❌ Cache status error: {e}")

        # Update settings if changed
        if cache_enabled != settings.get("cache", {}).get(
            "enabled", True
        ) or cache_ttl != settings.get("cache", {}).get("ttl_hours", 24):
            settings_changed = True
            settings["cache"] = {"enabled": cache_enabled, "ttl_hours": cache_ttl}

        st.markdown("</div>", unsafe_allow_html=True)

    # Action buttons
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if settings_changed:
            if st.button("💾 Save Settings", type="primary"):
                if save_settings_to_file(settings):
                    update_settings(settings)
                    st.success("✅ Settings saved successfully!")

                    # Check if embedding model changed and warn about reprocessing
                    old_embedding_model = (
                        st.session_state.get("settings", {})
                        .get("retrieval", {})
                        .get("embedding_model", "nomic-ai/nomic-embed-text-v1.5")
                    )
                    current_embedding_model = settings.get("retrieval", {}).get(
                        "embedding_model", "nomic-ai/nomic-embed-text-v1.5"
                    )
                    if current_embedding_model != old_embedding_model:
                        st.warning(
                            f"⚠️ **Embedding model changed from {old_embedding_model} to {current_embedding_model}**"
                        )
                        st.info(
                            "📄 Go to the Documents page and click 'Reprocess Documents' to create embeddings with the new model"
                        )

                    if theme != settings.get("interface", {}).get("theme", "light"):
                        st.info(
                            "🔄 **Theme change requires app restart** - Please restart the application to apply the new theme"
                        )
                    else:
                        st.info("🔄 Some changes may require a page refresh")
                else:
                    st.error("❌ Failed to save settings")
        else:
            st.button("💾 Save Settings", type="secondary", disabled=True)

    with col2:
        if st.button("🔄 Reset to Defaults", type="secondary"):
            reset_settings()

    with col3:
        if st.button("📊 Export Settings", type="secondary"):
            st.download_button(
                label="Download",
                data=str(settings).replace("'", '"'),
                file_name="rag_settings.json",
                mime="application/json",
            )

    # Current settings info
    with st.expander("📋 Current Settings (JSON)"):
        st.json(settings)


if __name__ == "__main__":
    main()
