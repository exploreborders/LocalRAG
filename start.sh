#!/bin/bash
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
