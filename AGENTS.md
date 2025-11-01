# AGENTS.md

## Build/Lint/Test Commands
- Quick Setup: `python setup_all.py` (automated setup of everything)
- Docker Setup: `./docker_setup.sh` (fully containerized setup)
- Manual Setup: `python -m venv rag_env && source rag_env/bin/activate && pip install -r requirements.txt && python -m spacy download de_core_news_sm fr_core_news_sm es_core_news_sm it_core_news_sm pt_core_news_sm nl_core_news_sm sv_core_news_sm pl_core_news_sm zh_core_web_sm ja_core_news_sm ko_core_news_sm`
- Set up databases: `python setup_databases.py docker` (or `docker-compose up -d`)
- Initialize databases: `python scripts/migrate_to_db.py && python scripts/migrate_add_language.py && python src/database/opensearch_setup.py`
- Run all tests: `python tests/run_all_tests.py` (runs all test files)
- Run system tests: `python tests/test_system.py`
- Run single test: `python -c "from tests.test_system import test_retrieval; test_retrieval()"`
- Performance tests: `python tests/test_performance.py`
- Cache tests: `python tests/test_cache.py` (test Redis caching functionality)
- Cache performance: `python tests/test_cache_performance.py` (measure cache speedup)
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

## Redis Cache Management
- **Cache Status**: View cache metrics in Analytics dashboard (total keys, memory usage, hit rate)
- **Cache Settings**: Configure cache TTL and enable/disable in Settings page
- **Clear Cache**: Use "🗑️ Clear Cache" button in Settings to manually clear all cached responses
- **Cache Performance**: 172.5x speedup demonstrated for repeated queries (3.45s → 0.02s)
- **Redis Monitoring**: `docker-compose exec redis redis-cli` for direct Redis access
- **Cache Keys**: Pattern `llm:*` for LLM response cache entries

## Multilingual Support
- **Language Detection**: Automatic language detection using `langdetect` for 12 supported languages
- **Text Preprocessing**: Language-specific spaCy models for proper tokenization and sentence segmentation
- **LLM Responses**: Language-aware prompt templates ensure responses in the user's query language
- **Source Citations**: LLM answers include references to source documents ([Source 1: filename.pdf])
- **Supported Languages**: English, German, French, Spanish, Italian, Portuguese, Dutch, Swedish, Polish, Chinese, Japanese, Korean

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
- **Default model: nomic-ai/nomic-embed-text-v1.5**: Primary multilingual embedding model for all operations (requires einops)
- **Language Support**: 12 languages with automatic detection and processing
- **spaCy Models**: 11 language-specific models for text preprocessing (de, fr, es, it, pt, nl, sv, pl, zh, ja, ko)
- **Smart Caching**: Document hash comparison prevents reprocessing
- **Batch Processing**: Efficient document processing with single model