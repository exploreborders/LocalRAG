# Local RAG System

A local Retrieval-Augmented Generation system built with Python, PostgreSQL, Elasticsearch, and Ollama.

## Features

- **Database-Backed Storage**: Documents and chunks stored in PostgreSQL with pgvector
- **Vector Search**: High-performance similarity search using Elasticsearch with dense vectors
- **Hybrid Retrieval**: Combine vector similarity with BM25 text search
- **Advanced Document Processing**: Docling-powered parsing with layout awareness and table extraction
- **Multi-Format Support**: Load documents from .txt, .pdf, .docx, .pptx, .xlsx files with unified processing
- **Multilingual Support**: Automatic language detection and processing for English, German, French, and Spanish
- **Ollama Integration**: Local LLM generation with context from retrieved documents
- **Web Interface**: Modern Streamlit UI for querying, document management, and analytics
- **Scalable Architecture**: Designed for production use with proper database indexing

## Setup

1. **Clone or navigate to the project directory**

2. **Create virtual environment:**
   ```bash
   python3 -m venv rag_env
   source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up databases:**
    - **Option 1 - Docker Compose (Recommended):**
      ```bash
      python setup_databases.py docker
      ```
      Or manually: `docker-compose up -d`

      This starts both PostgreSQL (with pgvector) and Elasticsearch with proper configuration.

    - **Option 2 - Local Setup:**
      ```bash
      python setup_databases.py local
      ```
      Follow the printed instructions for local PostgreSQL and Elasticsearch installation.

    - Update `.env` file with database credentials (see comments in `.env` for Docker vs local settings)

5. **Initialize databases:**
    ```bash
    python scripts/migrate_to_db.py  # Process documents and create chunks/embeddings
    python src/database/opensearch_setup.py  # Set up Elasticsearch indices
    ```

6. **Install and setup Ollama:**
   - Download from https://ollama.ai
   - Pull a model: `ollama pull llama2`
   - Start server: `ollama serve`

## Usage

### Web Interface (Recommended):
```bash
python run_web.py
```
Or directly:
```bash
streamlit run web_interface/app.py
```
Then open http://localhost:8501 in your browser for a comprehensive multipage experience with:
- **🏠 Home**: Query interface with RAG and retrieval-only modes
- **📁 Documents**: Upload and manage documents with automatic processing
- **⚙️ Settings**: Configure generation parameters and interface options
- **📊 Analytics**: Monitor system performance and query statistics

### Command Line Interface:
```bash
python -m src.app
```

Choose between:
- **Retrieval only**: Search for relevant documents
- **Full RAG**: Get AI-generated answers with context

### Testing:
```bash
python test_system.py  # Run system tests
python test_performance.py  # Performance benchmarking
```

## Project Structure

```
LocalRAG/
├── src/
│   ├── __init__.py
│   ├── app.py              # CLI interface
│   ├── data_loader.py      # Document loading and chunking
│   ├── document_processor.py # Database document processing
│   ├── embeddings.py       # Embedding creation utilities
│   ├── rag_pipeline_db.py  # RAG pipeline with database
│   ├── retrieval_db.py     # Database-backed retrieval
│   └── database/
│       ├── models.py       # SQLAlchemy models
│       ├── opensearch_setup.py # Elasticsearch configuration
│       └── schema.sql      # Database schema
├── web_interface/
│   ├── app.py              # Main Streamlit app
│   ├── pages/              # Individual pages
│   └── components/         # Reusable UI components
├── scripts/
│   └── migrate_to_db.py    # Database migration script
├── setup_databases.py      # Database setup helper script
├── test_system.py          # System tests
├── test_performance.py     # Performance benchmarks
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Docker services configuration
├── plan.md                 # Implementation plan
└── README.md               # This file
```

## Architecture

- **Document Processing**: Documents are parsed using Docling for superior text extraction, then chunked and embedded using nomic-embed-text-v1.5
- **Storage**: Chunks and metadata stored in PostgreSQL, embeddings indexed in Elasticsearch
- **Retrieval**: Hybrid search combining vector similarity and BM25 text search
- **Generation**: Context from retrieved documents fed to Ollama LLMs for answer generation

## Requirements

- Python 3.8+
- Docker (recommended for databases) OR:
  - PostgreSQL with pgvector extension
  - Elasticsearch 8.x
- Ollama for local LLM inference

## Implementation Status

See `plan.md` for detailed implementation progress. The system is fully operational with database-backed storage and vector search.