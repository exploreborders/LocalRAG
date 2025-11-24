#!/usr/bin/env python3
"""
Setup check script for Local RAG Web Interface
Run this before starting the web interface to ensure everything is properly configured.
"""

import sys
import os
import subprocess
from pathlib import Path


def check_virtual_environment():
    """Check if we're in the correct virtual environment."""
    print("🔍 Checking virtual environment...")

    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )

    if in_venv:
        print("✅ Virtual environment is active")
        return True
    else:
        print("❌ Not in virtual environment")
        print("   Please activate it with:")
        print("   source rag_env/bin/activate")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")

    required_deps = [
        ("streamlit", "Streamlit web interface"),
        ("sqlalchemy", "Database ORM"),
        ("sentence_transformers", "Text embeddings"),
        ("docling", "Document processing"),
        ("psycopg2", "PostgreSQL driver"),
        ("elasticsearch", "Search engine client"),
        ("redis", "Cache backend"),
        ("torch", "Machine learning framework"),
    ]

    missing_deps = []
    for dep, description in required_deps:
        try:
            __import__(dep)
            print(f"✅ {dep} - {description}")
        except ImportError:
            print(f"❌ {dep} - {description}")
            missing_deps.append(dep)

    if missing_deps:
        print(f"\n❌ Missing {len(missing_deps)} dependencies")
        print("   Install with: pip install -r requirements.txt")
        return False

    print("✅ All dependencies installed")
    return True


def check_database_connection():
    """Check if database connection works."""
    print("🔍 Checking database connection...")

    # Set environment variables for database connection
    os.environ.setdefault("POSTGRES_HOST", "localhost")
    os.environ.setdefault("POSTGRES_PORT", "5432")
    os.environ.setdefault("POSTGRES_DB", "rag_system")
    os.environ.setdefault("POSTGRES_USER", "christianhein")

    try:
        import sqlalchemy
        from sqlalchemy import create_engine, text

        # Create connection string
        db_url = f"postgresql://{os.environ['POSTGRES_USER']}@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"

        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) as count FROM documents"))
            count = result.scalar()
            print(f"✅ Database connected - {count} documents found")
            return True

    except ImportError:
        print("❌ SQLAlchemy not available")
        return False
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("   Make sure PostgreSQL is running and properly configured")
        return False


def check_services():
    """Check if required services are running."""
    print("🔍 Checking services...")

    services = [
        ("PostgreSQL", "localhost", 5432),
        ("Elasticsearch", "localhost", 9200),
        ("Redis", "localhost", 6379),
    ]

    all_running = True
    for name, host, port in services:
        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()

            if result == 0:
                print(f"✅ {name} running on {host}:{port}")
            else:
                print(f"❌ {name} not accessible on {host}:{port}")
                all_running = False
        except Exception as e:
            print(f"❌ Error checking {name}: {e}")
            all_running = False

    return all_running


def main():
    """Run all setup checks."""
    print("🚀 Local RAG Web Interface Setup Check")
    print("=" * 50)

    checks = [
        check_virtual_environment,
        check_dependencies,
        check_database_connection,
        check_services,
    ]

    results = []
    for check in checks:
        result = check()
        results.append(result)
        print()

    passed = sum(results)
    total = len(results)

    print("=" * 50)
    if passed == total:
        print("🎉 All checks passed! Ready to start web interface.")
        print()
        print("Start the web interface with:")
        print("python run_web.py")
        return 0
    else:
        print(f"❌ {total - passed} checks failed. Please fix the issues above.")
        print()
        print("Common fixes:")
        print("1. Activate virtual environment: source rag_env/bin/activate")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Start services: docker-compose up -d postgres elasticsearch redis")
        print("4. Run setup: python setup_databases.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
