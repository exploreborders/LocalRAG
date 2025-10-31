# AGENTS.md

## Build/Lint/Test Commands
- Setup: `python -m venv rag_env && source rag_env/bin/activate && pip install -r requirements.txt`
- Run all tests: `python test_system.py`
- Run single test: `python -c "from test_system import test_retrieval; test_retrieval()"`
- Run web interface: `streamlit run web_interface/app.py`
- Run CLI app: `python -m src.app`
- Process data: `python -m src.embeddings`
- Set up databases: `docker-compose up -d` (PostgreSQL + Elasticsearch)
- Run migration: `python scripts/migrate_to_db.py`
- Set up indices: `python src/database/opensearch_setup.py`
- Performance tests: `python test_performance.py`
- Stop databases: `docker-compose down`

## Code Style Guidelines
- **Imports**: Group stdlib/third-party/local, sort alphabetically, use relative imports
- **Formatting**: 4 spaces, 88 char limit, double quotes, docstrings for public APIs
- **Naming**: snake_case functions/vars, PascalCase classes, UPPER_SNAKE_CASE constants
- **Types**: Type hints on parameters/returns when beneficial for clarity
- **Error Handling**: try/except with specific exceptions, validate inputs, context managers
- **Best Practices**: PEP 8, descriptive names, single responsibility, unit tests, no global state

## Document Processing
- **Unified Loading**: Docling integration for better document parsing (PDF, DOCX, PPTX, XLSX)
- **Fallback Support**: Graceful fallback to basic parsers if Docling fails
- **Text Files**: Direct loading for simple .txt files

## Model Requirements
- **Default model: nomic-ai/nomic-embed-text-v1.5**: Primary embedding model for all operations (requires einops)
- **Smart Caching**: Document hash comparison prevents reprocessing
- **Batch Processing**: Efficient document processing with single model