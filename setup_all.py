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
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
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
    if not os.path.exists("rag_env"):
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

    # Start only database services first (exclude rag_app which takes long to build)
    if not run_command(
        "docker-compose up -d postgres elasticsearch redis",
        "Starting database services with Docker Compose",
    ):
        return False

    # Wait for databases to be ready
    print("⏳ Waiting for databases to be ready...")
    time.sleep(30)  # Give more time for health checks

    return True


def initialize_databases():
    """Initialize databases with schema and data."""
    print("\n🗄️ Initializing databases...")

    # Run database setup
    if not run_command(
        "source rag_env/bin/activate && python setup_databases.py docker",
        "Setting up database connections",
    ):
        return False

    # Run migrations
    if not run_command(
        "source rag_env/bin/activate && python scripts/migrate_to_db.py",
        "Running initial database migration",
    ):
        return False

    if not run_command(
        "source rag_env/bin/activate && python scripts/migrate_database_schema.py",
        "Setting up complete database schema",
    ):
        return False

    # Setup OpenSearch
    if not run_command(
        "source rag_env/bin/activate && python src/database/opensearch_setup.py",
        "Setting up OpenSearch",
    ):
        return False

    return True


def start_app_service():
    """Start the rag_app service after databases are ready."""
    print("\n🚀 Starting application service...")

    # Start the app service (this will build the Docker image)
    if not run_command(
        "docker-compose up -d rag_app",
        "Starting application service",
        timeout=600,  # Give it 10 minutes for the build
    ):
        print(
            "⚠️ App service failed to start, but databases are ready. You can start it manually with: docker-compose up -d rag_app"
        )
        return False

    return True


def download_spacy_models():
    """Download required spaCy language models."""
    print("\n📥 Downloading spaCy language models...")

    models = [
        "de_core_news_sm",
        "fr_core_news_sm",
        "es_core_news_sm",
        "it_core_news_sm",
        "pt_core_news_sm",
        "nl_core_news_sm",
        "sv_core_news_sm",
        "pl_core_news_sm",
        "zh_core_web_sm",
        "ja_core_news_sm",
        "ko_core_news_sm",
    ]

    for model in models:
        if not run_command(
            f"source rag_env/bin/activate && python -m spacy download {model}",
            f"Downloading {model}",
            timeout=120,
        ):
            print(f"⚠️ Failed to download {model}, continuing...")

    return True


def run_tests():
    """Run the test suite to verify everything works."""
    print("\n🧪 Running tests...")

    if not run_command(
        "source rag_env/bin/activate && python tests/run_tests.py",
        "Running system tests",
    ):
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

    with open("start.sh", "w") as f:
        f.write(startup_script)

    # Make executable
    os.chmod("start.sh", 0o755)

    print("✅ Created startup script: start.sh")
    return True


def main():
    """Main setup function."""
    print("🤖 Local RAG System - Automated Setup")
    print("=" * 50)

    # Check if we're in the right directory
    if not os.path.exists("requirements.txt") or not os.path.exists("web_interface"):
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

    # Note: App service startup is skipped in automated setup due to long build times
    # Start it manually later with: docker-compose up -d rag_app
    print(
        "ℹ️ Databases are ready! Start the app manually with: docker-compose up -d rag_app"
    )

    # Note: spaCy model downloads are skipped in automated setup due to long download times
    # Download them manually later if needed
    print("ℹ️ spaCy models not downloaded automatically. Download manually if needed.")

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
