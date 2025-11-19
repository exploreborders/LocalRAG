#!/usr/bin/env python3
"""
Database setup helper script for Local RAG system.

This script helps you set up and manage PostgreSQL and Elasticsearch databases
for both local development and Docker environments.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and print status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def setup_docker_databases():
    """Set up databases using Docker Compose."""
    print("🚀 Setting up databases with Docker Compose...")
    print("This will start PostgreSQL (with pgvector) and Elasticsearch containers.")

    if not Path("docker-compose.yml").exists():
        print("❌ docker-compose.yml not found in current directory")
        return False

    # Start databases
    if not run_command(
        "docker-compose up -d postgres elasticsearch",
        "Starting PostgreSQL and Elasticsearch containers",
    ):
        return False

    print("\n⏳ Waiting for databases to be ready...")
    print("PostgreSQL will be available at: localhost:5432")
    print("Elasticsearch will be available at: localhost:9200")

    # Check if databases are ready
    import time

    time.sleep(10)  # Give containers time to start

    # Test PostgreSQL connection
    try:
        import psycopg2

        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="rag_system",
            user="christianhein",
            password="",
        )
        conn.close()
        print("✅ PostgreSQL connection successful")
    except ImportError:
        print("⚠️ psycopg2 not available, skipping PostgreSQL check")
    except Exception as e:
        print(f"⚠️ PostgreSQL connection check failed: {e}")

    # Test Elasticsearch connection
    try:
        import requests

        response = requests.get("http://localhost:9200/_cluster/health", timeout=5)
        if response.status_code == 200:
            print("✅ Elasticsearch connection successful")
        else:
            print(f"⚠️ Elasticsearch returned status {response.status_code}")
    except ImportError:
        print("⚠️ requests not available, skipping Elasticsearch check")
    except Exception as e:
        print(f"⚠️ Elasticsearch connection check failed: {e}")

    print("\n🎉 Databases are ready!")
    print("Run the following commands to complete setup:")
    print("  python scripts/migrate_to_db.py")
    print("  python src/database/opensearch_setup.py")

    return True


def setup_local_databases():
    """Provide instructions for local database setup."""
    print("🏠 Local Database Setup Instructions:")
    print()
    print("1. Install PostgreSQL locally:")
    print("   - macOS: brew install postgresql")
    print("   - Ubuntu: sudo apt install postgresql postgresql-contrib")
    print("   - Or download from: https://www.postgresql.org/download/")
    print()
    print("2. Start PostgreSQL service:")
    print("   - macOS: brew services start postgresql")
    print("   - Linux: sudo systemctl start postgresql")
    print()
    print("3. Create database and user:")
    print("   createdb rag_system")
    print("   createuser christianhein  # (use your username)")
    print()
    print("4. Install pgvector extension:")
    print("   - Follow instructions at: https://github.com/pgvector/pgvector")
    print()
    print("5. Install Elasticsearch:")
    print(
        '   docker run -d -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" elasticsearch:8.11.0'
    )
    print()
    print("6. Update .env file with your local settings")
    print("7. Run: python scripts/migrate_to_db.py")
    print("8. Run: python src/database/opensearch_setup.py")


def stop_docker_databases():
    """Stop Docker databases."""
    print("🛑 Stopping Docker databases...")
    run_command("docker-compose down", "Stopping database containers")


def main():
    if len(sys.argv) < 2:
        print("Usage: python setup_databases.py <command>")
        print("Commands:")
        print("  docker    - Set up databases using Docker Compose")
        print("  local     - Show instructions for local setup")
        print("  stop      - Stop Docker databases")
        return

    command = sys.argv[1].lower()

    if command == "docker":
        setup_docker_databases()
    elif command == "local":
        setup_local_databases()
    elif command == "stop":
        stop_docker_databases()
    else:
        print(f"Unknown command: {command}")
        print("Use 'docker', 'local', or 'stop'")


if __name__ == "__main__":
    main()
