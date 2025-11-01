#!/usr/bin/env python3
"""
Automated setup script for Local RAG system.
This script handles all initialization steps automatically.
"""

import os
import sys
import subprocess
import time
import shutil
from pathlib import Path

def run_command(cmd, description, cwd=None, timeout=300):
    """Run a command with proper error handling."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print("✅ Python version OK")

    # Check if virtual environment exists
    if not os.path.exists('rag_env'):
        print("❌ Virtual environment not found. Run: python -m venv rag_env")
        return False
    print("✅ Virtual environment found")

    return True

def setup_databases():
    """Set up databases using Docker."""
    print("\n🏗️ Setting up databases...")

    # Check if Docker is running
    if not run_command("docker info", "Checking Docker status"):
        print("❌ Docker not available. Please start Docker and try again.")
        return False

    # Start databases
    if not run_command("docker-compose up -d", "Starting databases with Docker Compose"):
        return False

    # Wait for databases to be ready
    print("⏳ Waiting for databases to be ready...")
    time.sleep(10)

    return True

def initialize_databases():
    """Initialize databases with schema and data."""
    print("\n🗄️ Initializing databases...")

    # Run database setup
    if not run_command("python setup_databases.py docker", "Setting up database connections"):
        return False

    # Run migrations
    if not run_command("python scripts/migrate_to_db.py", "Running database migrations"):
        return False

    if not run_command("python scripts/migrate_add_language.py", "Adding language support"):
        return False

    # Setup OpenSearch
    if not run_command("python src/database/opensearch_setup.py", "Setting up OpenSearch"):
        return False

    return True

def download_spacy_models():
    """Download required spaCy language models."""
    print("\n📥 Downloading spaCy language models...")

    models = [
        'de_core_news_sm',
        'fr_core_news_sm',
        'es_core_news_sm',
        'it_core_news_sm',
        'pt_core_news_sm',
        'nl_core_news_sm',
        'sv_core_news_sm',
        'pl_core_news_sm',
        'zh_core_web_sm',
        'ja_core_news_sm',
        'ko_core_news_sm'
    ]

    for model in models:
        if not run_command(f"python -m spacy download {model}", f"Downloading {model}", timeout=120):
            print(f"⚠️ Failed to download {model}, continuing...")

    return True

def run_tests():
    """Run the test suite to verify everything works."""
    print("\n🧪 Running tests...")

    if not run_command("python tests/test_system.py", "Running system tests"):
        return False

    return True

def create_startup_script():
    """Create a simple startup script."""
    startup_script = """#!/bin/bash
# Local RAG System Startup Script

echo "🚀 Starting Local RAG System..."

# Check if databases are running
if ! docker-compose ps | grep -q "Up"; then
    echo "📦 Starting databases..."
    docker-compose up -d
    sleep 10
fi

# Start the web interface
echo "🌐 Starting web interface..."
streamlit run web_interface/app.py --server.port 8501 --server.address 0.0.0.0

echo "✅ Local RAG System started!"
echo "🌐 Open http://localhost:8501 in your browser"
"""

    with open('start.sh', 'w') as f:
        f.write(startup_script)

    # Make executable
    os.chmod('start.sh', 0o755)

    print("✅ Created startup script: start.sh")
    return True

def main():
    """Main setup function."""
    print("🤖 Local RAG System - Automated Setup")
    print("=" * 50)

    # Check if we're in the right directory
    if not os.path.exists('requirements.txt') or not os.path.exists('web_interface'):
        print("❌ Please run this script from the Local RAG project root directory")
        return 1

    # Check dependencies
    if not check_dependencies():
        return 1

    # Setup databases
    if not setup_databases():
        return 1

    # Initialize databases
    if not initialize_databases():
        return 1

    # Download language models
    if not download_spacy_models():
        return 1

    # Run tests
    if not run_tests():
        print("⚠️ Tests failed, but setup completed. You may need to troubleshoot.")

    # Create startup script
    create_startup_script()

    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETE!")
    print("=" * 50)
    print("Your Local RAG system is now ready to use!")
    print("\n🚀 Quick start:")
    print("  ./start.sh              # Start everything")
    print("  # OR")
    print("  streamlit run web_interface/app.py  # Start web interface only")
    print("\n🌐 Then open: http://localhost:8501")
    print("\n📚 The system will auto-initialize on first use!")

    return 0

if __name__ == "__main__":
    sys.exit(main())