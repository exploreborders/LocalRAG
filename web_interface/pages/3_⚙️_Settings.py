#!/usr/bin/env python3
"""
Settings Page - Configuration Options
"""

import streamlit as st
import os
import sys
import subprocess
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import utilities
from utils.session_manager import load_settings, update_settings

# Page configuration
st.set_page_config(
    page_title="Local RAG - Settings",
    page_icon="⚙️",
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
""", unsafe_allow_html=True)

def save_settings_to_file(settings):
    """Save settings to YAML file and update Streamlit config"""
    import yaml

    settings_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_settings.yaml')
    streamlit_config_path = os.path.join(os.path.dirname(__file__), '..', '..', '.streamlit', 'config.toml')

    try:
        # Save our app settings
        with open(settings_path, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False, sort_keys=False)

        # Update Streamlit theme config
        theme = settings.get('interface', {}).get('theme', 'light')
        update_streamlit_theme(streamlit_config_path, theme)

        return True
    except Exception as e:
        st.error(f"Failed to save settings: {e}")
        return False

def get_installed_ollama_models():
    """Get list of installed Ollama models"""
    try:
        # Run ollama list command
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Skip header line
                # Parse model names from output
                models = []
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if parts:
                            model_name = parts[0].split(':')[0]  # Remove :latest tag
                            if model_name not in models:
                                models.append(model_name)
                return models if models else ["llama2"]  # Default fallback
        else:
            st.warning("Could not connect to Ollama. Using default model list.")
            return ["llama2"]

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
        st.warning(f"Ollama not available: {e}. Using default model.")
        return ["llama2"]

def get_available_embedding_models():
    """Get list of available sentence-transformers models"""
    # Common embedding models to check - focus on the most commonly used ones
    candidate_models = [
        "nomic-ai/nomic-embed-text-v1.5"  # Primary model
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
        return ["nomic-ai/nomic-embed-text-v1.5"]
    except Exception as e:
        st.warning(f"Error checking embedding models: {e}")
        return ["nomic-ai/nomic-embed-text-v1.5"]

    # Always include at least the default model
    if not available_models:
        available_models = ["nomic-ai/nomic-embed-text-v1.5"]

    return available_models

def update_streamlit_theme(config_path, theme):
    """Update Streamlit theme configuration"""
    try:
        # Read current config
        with open(config_path, 'r') as f:
            content = f.read()

        # Update theme base
        if theme == 'dark':
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
        pattern = r'\[theme\].*?(?=\[|\Z)'
        new_content = re.sub(pattern, theme_config, content, flags=re.DOTALL)

        # Write back
        with open(config_path, 'w') as f:
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
                value=settings.get('generation', {}).get('temperature', 0.7),
                step=0.1,
                help="Controls randomness in AI responses (0.0 = deterministic, 1.0 = very random)"
            )

            max_tokens = st.slider(
                "Max Tokens",
                min_value=100,
                max_value=2000,
                value=settings.get('generation', {}).get('max_tokens', 500),
                step=50,
                help="Maximum length of AI-generated responses"
            )

        with col2:
            ollama_host = st.text_input(
                "Ollama Host",
                value=settings.get('generation', {}).get('ollama_host', 'http://localhost:11434'),
                help="URL where Ollama server is running"
            )

            # Get installed models dynamically
            installed_models = get_installed_ollama_models()
            current_model = settings.get('generation', {}).get('model', 'llama2')

            # Ensure current model is in the list, otherwise use first available
            if current_model not in installed_models:
                current_model = installed_models[0] if installed_models else 'llama2'

            model_name = st.selectbox(
                "LLM Model",
                installed_models,
                index=installed_models.index(current_model) if current_model in installed_models else 0,
                help=f"Available Ollama models: {', '.join(installed_models)}"
            )

        # Update settings if changed
        if (temperature != settings.get('generation', {}).get('temperature', 0.7) or
            max_tokens != settings.get('generation', {}).get('max_tokens', 500) or
            ollama_host != settings.get('generation', {}).get('ollama_host', 'http://localhost:11434') or
            model_name != settings.get('generation', {}).get('model', 'llama2')):
            settings_changed = True
            settings['generation'] = {
                'model': model_name,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'ollama_host': ollama_host,
                'ollama_port': 11434
            }

        st.markdown('</div>', unsafe_allow_html=True)

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
                value=settings.get('interface', {}).get('max_results_display', 5),
                help="Maximum number of results to display"
            )

            max_history = st.slider(
                "Query History Size",
                min_value=5,
                max_value=50,
                value=settings.get('system', {}).get('max_query_history', 10),
                help="Number of queries to keep in history"
            )

        with col2:
            theme = st.selectbox(
                "Theme",
                ["light", "dark"],
                index=["light", "dark"].index(
                    settings.get('interface', {}).get('theme', 'light')
                ),
                help="UI theme - requires app restart to take effect"
            )

        # Update settings if changed
        if (max_results != settings.get('interface', {}).get('max_results_display', 5) or
            theme != settings.get('interface', {}).get('theme', 'light')):
            settings_changed = True
            settings['interface'] = {
                'theme': theme,
                'max_results_display': max_results
            }

        if max_history != settings.get('system', {}).get('max_query_history', 10):
            settings_changed = True
            settings['system'] = settings.get('system', {})
            settings['system']['max_query_history'] = max_history

        st.markdown('</div>', unsafe_allow_html=True)

    # Action buttons
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if settings_changed:
            if st.button("💾 Save Settings", type="primary", use_container_width=True):
                if save_settings_to_file(settings):
                    update_settings(settings)
                    st.success("✅ Settings saved successfully!")

                    # Check if embedding model changed and warn about reprocessing
                    old_embedding_model = st.session_state.get('settings', {}).get('retrieval', {}).get('embedding_model', 'nomic-ai/nomic-embed-text-v1.5')
                    if embedding_model != old_embedding_model:
                        st.warning(f"⚠️ **Embedding model changed from {old_embedding_model} to {embedding_model}**")
                        st.info("📄 Go to the Documents page and click 'Reprocess Documents' to create embeddings with the new model")

                    if theme != settings.get('interface', {}).get('theme', 'light'):
                        st.info("🔄 **Theme change requires app restart** - Please restart the application to apply the new theme")
                    else:
                        st.info("🔄 Some changes may require a page refresh")
                else:
                    st.error("❌ Failed to save settings")
        else:
            st.button("💾 Save Settings", type="secondary", use_container_width=True, disabled=True)

    with col2:
        if st.button("🔄 Reset to Defaults", type="secondary", use_container_width=True):
            reset_settings()

    with col3:
        if st.button("📊 Export Settings", type="secondary", use_container_width=True):
            st.download_button(
                label="Download",
                data=str(settings).replace("'", '"'),
                file_name="rag_settings.json",
                mime="application/json"
            )

    # Current settings info
    with st.expander("📋 Current Settings (JSON)"):
        st.json(settings)

if __name__ == "__main__":
    main()