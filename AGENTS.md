# AGENTS.md

## Build/Lint/Test Commands
- Setup: `python -m venv rag_env && source rag_env/bin/activate && pip install -r requirements.txt`
- Run all tests: `python test_system.py`
- Run single test: `python -c "from test_system import test_retrieval; test_retrieval()"`
- Run web interface: `python run_web.py` or `streamlit run web_interface/app.py` (multipage app)
- Run CLI app: `python -m src.app`
- Process data: `python -m src.embeddings` (uses nomic-embed-text-v1.5)
- Test retrieval: `python -m src.retrieval`
- Set up databases: PostgreSQL running, OpenSearch via Docker (see plan.md)
- Run migration: `python scripts/migrate_to_db.py` (after databases are set up)
- Set up Elasticsearch indices: `python src/database/opensearch_setup.py`
- Run performance tests: `python test_performance.py`
- Deploy with Docker: `docker-compose up -d` (requires Docker Desktop)
- **Model Requirements**: Requires `einops` package for nomic-embed-text-v1.5

## Single Model Support

- **Only nomic-embed-text-v1.5**: The system now uses only the nomic-ai/nomic-embed-text-v1.5 model for all embeddings
- **Smart Caching**: Document hash comparison prevents unnecessary reprocessing
- **Batch Processing**: Process documents with the single supported model
- **Fixed Configuration**: No more model selection - always uses nomic-embed-text-v1.5

## Code Style Guidelines
- **Imports**: Group stdlib/third-party/local, sort alphabetically, use relative imports (`from .module import Class`)
- **Formatting**: 4 spaces indentation, 88 char line limit, double quotes, docstrings for public functions/classes
- **Naming**: snake_case for vars/functions, PascalCase for classes, UPPER_SNAKE_CASE for constants
- **Types**: Use type hints for function parameters and return values when clarity benefits
- **Error Handling**: try/except blocks, specific exceptions, validate inputs, use context managers for files
- **Best Practices**: PEP 8, descriptive names, single responsibility functions, add unit tests, avoid global state