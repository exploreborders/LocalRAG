# AGENTS.md

## Build/Lint/Test Commands
- Setup: `python -m venv rag_env && source rag_env/bin/activate && pip install -r requirements.txt && python -m spacy download de_core_news_sm`
- Set up databases: `python setup_databases.py docker` (or `docker-compose up -d`)
- Initialize databases: `python scripts/migrate_to_db.py && python scripts/migrate_add_language.py && python src/database/opensearch_setup.py`
- Run all tests: `python test_system.py`
- Run single test: `python -c "from test_system import test_retrieval; test_retrieval()"`
- Performance tests: `python test_performance.py`
- Run web interface: `python run_web.py` (or `streamlit run web_interface/app.py`)
- Run CLI app: `python -m src.app`
- Process data: `python -m src.embeddings`
- Reprocess documents (for language detection): Click "🔄 Reprocess Documents" in web interface (uses batch processing, parallelization, and memory management)
- Stop databases: `docker-compose down`

## Document Processing Commands
- Process documents with optimizations: `python scripts/migrate_to_db.py` (uses optimized batch processing)
- Process with custom settings: `python -c "from src.document_processor import DocumentProcessor; p = DocumentProcessor(); p.process_existing_documents(batch_size=10, use_parallel=True, max_workers=4)"`

## Performance Optimization Settings
- **Batch Processing**: Documents processed in configurable batches (default: 5)
- **Parallel Processing**: Multi-worker processing for large document sets (default: 4 workers, SQLAlchemy-safe)
- **Memory Management**: Automatic memory monitoring with configurable limits (default: 500MB)
- **Pipeline Optimization**: Using default docling options (OCR/table processing enabled due to v2.60.0 backend bug)
- **Converter Reuse**: Single DocumentConverter instance reused across operations
- **Smart Routing**: Separate optimized paths for text files vs. complex documents
- **Serialization Safety**: Workers use only serializable data to avoid SQLAlchemy pickling issues

## Environment Configuration
- **Local Development**: Use `.env` with localhost settings (currently active)
- **Docker Deployment**: Environment variables are set automatically in docker-compose.yml
- **Database Switching**: Comment/uncomment sections in `.env` based on deployment method

## Code Style Guidelines
- **Imports**: Group stdlib/third-party/local, sort alphabetically, use relative imports
- **Formatting**: 4 spaces, 88 char limit, double quotes, docstrings for public APIs
- **Naming**: snake_case functions/vars, PascalCase classes, UPPER_SNAKE_CASE constants
- **Types**: Type hints on parameters/returns when beneficial for clarity
- **Error Handling**: try/except with specific exceptions, validate inputs, context managers
- **Best Practices**: PEP 8, descriptive names, single responsibility, unit tests, no global state

## Document Processing
- **Unified Loading**: Docling integration for high-quality document parsing (PDF, DOCX, PPTX, XLSX)
- **Quality Processing**: Uses Docling for all document types including reprocessing for better text extraction
- **Fallback Support**: PyPDF2 fallback for PDFs if Docling fails
- **Text Files**: Direct loading for simple .txt files

## Model Requirements
- **Default model: nomic-ai/nomic-embed-text-v1.5**: Primary embedding model for all operations (requires einops)
- **Smart Caching**: Document hash comparison prevents reprocessing
- **Batch Processing**: Efficient document processing with single model