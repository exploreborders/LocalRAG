# AGENTS.md

## Build/Lint/Test Commands
- Quick Setup: `python setup_all.py`
- Docker Setup: `./docker_setup.sh && docker-compose up -d`
- Manual Setup: `python -m venv rag_env && source rag_env/bin/activate && pip install -r requirements.txt`
- Run all tests: `python tests/run_all_tests.py`
- Run single test: `python -c "from tests.test_analytics import test_analytics_metrics; test_analytics_metrics()"`
- Run web interface: `python run_web.py` or `streamlit run web_interface/app.py --server.port 8501`
- Run CLI app: `python -m src.app`
- Performance testing: `python performance_test.py`
- Build Docker: `docker build -t local-rag .`
- Reprocess documents: `python reprocess_documents.py` (forces AI enrichment update)

## Code Style Guidelines
- **Imports**: Group stdlib/third-party/local alphabetically, use relative imports for local modules
- **Formatting**: 4 spaces indentation, double quotes for strings, no trailing whitespace
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE_CASE for constants
- **Types**: Use type hints on parameters/returns when beneficial, Optional for nullable types
- **Error Handling**: try/except with specific exceptions, validate inputs early, use context managers
- **Documentation**: Docstrings for public APIs/classes, inline comments for complex logic
- **Best Practices**: PEP 8 compliance, descriptive names, single responsibility, no global state, avoid magic numbers

## Document Structure Guidelines
- **Chapter Extraction**: Only extract substantive section headings, filter out:
  - Generic terms like "Note:", "See also:", "Table of contents:"
  - Single words or short phrases like "Wobei:", "Example:"
  - Code snippets, references, or citations
  - Very short lines (< 5 characters) or symbol-only lines
- **Table of Contents**: Automatically detect and parse markdown table format TOCs (priority method)
- **LLM Analysis**: Use phi3.5 model for documents without clear TOCs
- **Hierarchy**: Properly maintain chapter levels (1, 1.1, 1.1.1, etc.) from structured documents
- **Model**: Use llama3.2:latest for LLM-based analysis (better JSON generation than phi3.5)