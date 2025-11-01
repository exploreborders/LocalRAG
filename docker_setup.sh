#!/bin/bash
# Docker-based setup for Local RAG System
# This script provides a fully containerized setup

set -e

echo "🐳 Local RAG System - Docker Setup"
echo "==================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

echo "✅ Docker environment detected"

# Build the application container
echo "🏗️ Building Local RAG container..."
docker build -t local-rag .

# Start all services
echo "🚀 Starting all services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."
sleep 30

# Run initialization inside container
echo "🔧 Running database initialization..."
docker-compose exec app python setup_databases.py docker
docker-compose exec app python scripts/migrate_to_db.py
docker-compose exec app python scripts/migrate_add_language.py
docker-compose exec app python src/database/opensearch_setup.py

echo ""
echo "🎉 Docker setup complete!"
echo "=========================="
echo "🌐 Web interface: http://localhost:8501"
echo "📊 Analytics: http://localhost:8501 (switch to Analytics page)"
echo ""
echo "To stop: docker-compose down"
echo "To restart: docker-compose up -d"
echo "To view logs: docker-compose logs -f"