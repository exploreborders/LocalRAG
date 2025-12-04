#!/usr/bin/env python3
"""
Convenience script to run the Local RAG web interface
"""

import os
import subprocess
import sys


def check_dependencies():
    """Check if required dependencies are installed."""
    missing_deps = []

    required_deps = [
        "streamlit",
        "sqlalchemy",
        "sentence_transformers",
        "docling",
        "psycopg2",
        "elasticsearch",
        "redis",
        "torch",
        "networkx",
        "plotly",
        "wordcloud",
        "matplotlib",
    ]

    for dep in required_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)

    return missing_deps


def setup_package_path():
    """Set up the Python path so that relative imports work."""
    import sys
    from pathlib import Path

    # Get the project root
    project_root = Path(__file__).resolve().parent

    # Add the project root to Python path so we can import src
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Also add src directly
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Set PYTHONPATH environment variable
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if current_pythonpath:
        os.environ["PYTHONPATH"] = (
            f"{str(project_root)}:{str(src_path)}:{current_pythonpath}"
        )
    else:
        os.environ["PYTHONPATH"] = f"{str(project_root)}:{str(src_path)}"


def main():
    """
    Run the Streamlit web interface for the Local RAG System.

    Checks for virtual environment and required dependencies before
    launching the web application on localhost:8501.
    """
    # Set up package path first
    setup_package_path()

    # Set environment variables to suppress warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Check if we're in the virtual environment
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    if not in_venv:
        print("⚠️  Warning: Virtual environment not detected.")
        print("   Please activate the virtual environment first:")
        print("   source rag_env/bin/activate")
        print("   Then run: python run_web.py")
        print()

    # Check for missing dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print()
        print("   Install with: pip install -r requirements.txt")
        print()
        sys.exit(1)

    # Check if streamlit is installed
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("❌ Streamlit not found. Please install it:")
        print("   pip install streamlit")
        sys.exit(1)

    # Run the web interface
    web_app_path = os.path.join(os.path.dirname(__file__), "web_interface", "app.py")

    if not os.path.exists(web_app_path):
        print(f"❌ Web interface not found at: {web_app_path}")
        sys.exit(1)

    print("🚀 Starting Local RAG Web Interface...")
    print("📱 Open your browser to: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print()

    # Create a wrapper script that sets up the environment properly
    wrapper_script = f"""
import sys
import os
from pathlib import Path

# Set up paths
project_root = Path("{os.path.dirname(os.path.abspath(__file__))}")
src_path = project_root / "src"

# Add to sys.path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

# Set environment variables
os.environ["PYTHONPATH"] = f"{{project_root}}:{{src_path}}"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import and run streamlit
import streamlit.web.cli as st_cli
import sys

# Replace sys.argv to run the app
sys.argv = ["streamlit", "run", "{web_app_path}"]
st_cli.main()
"""

    # Write wrapper script to a temporary file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper_script)
        wrapper_path = f.name

    try:
        # Run the wrapper script
        subprocess.run([sys.executable, wrapper_path], check=True)
    except KeyboardInterrupt:
        print("\n👋 Web interface stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start web interface: {e}")
    finally:
        # Clean up temporary file
        try:
            os.unlink(wrapper_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
