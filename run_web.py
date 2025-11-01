#!/usr/bin/env python3
"""
Convenience script to run the Local RAG web interface
"""

import subprocess
import sys
import os

def main():
    """
    Run the Streamlit web interface for the Local RAG System.

    Checks for virtual environment and required dependencies before
    launching the web application on localhost:8501.
    """
    # Set environment variables to suppress warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Check if we're in the virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment not detected. Please activate 'rag_env' first:")
        print("   source rag_env/bin/activate")
        print()

    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not found. Please install it:")
        print("   pip install streamlit")
        sys.exit(1)

    # Run the web interface
    web_app_path = os.path.join(os.path.dirname(__file__), 'web_interface', 'app.py')

    if not os.path.exists(web_app_path):
        print(f"❌ Web interface not found at: {web_app_path}")
        sys.exit(1)

    print("🚀 Starting Local RAG Web Interface...")
    print("📱 Open your browser to: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print()

    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', web_app_path], check=True)
    except KeyboardInterrupt:
        print("\n👋 Web interface stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start web interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()