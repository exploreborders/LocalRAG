# AGENTS.md

**Build/Lint/Test**
- Quick Setup: `python setup_all.py`
- Docker Setup: `./docker_setup.sh && docker-compose up -d`
- Manual Setup: `python -m venv rag_env && source rag_env/bin/activate && pip install -r requirements.txt`
- Run all tests: `pytest -q`
- Run single test: `pytest tests/unit/test_core/test_document_processor.py::TestDocumentProcessor::test_process_document -q`
- Run with coverage: `pytest --cov=src --cov-report=term-missing`
- Run unit tests only: `pytest -m unit`
- Run web interface: `python run_web.py`
- Run CLI: `python -m src.interfaces.cli`
- Build Docker: `docker build -t local-rag .`
- Reprocess documents: `python -m src.core.reprocess_documents`

**Code Style**
- Imports: stdlib/third-party/local alphabetically; absolute imports preferred; isort with black profile
- Formatting: black (100 chars, py312 target); 4 spaces; double quotes; no trailing spaces
- Naming: snake_case for funcs/vars/methods; PascalCase for classes; UPPER_SNAKE for constants
- Types: type hints on all params/returns; Optional/Union as needed; mypy strict mode
- Errors: specific exceptions; input validation; context managers; proper logging
- Linting: flake8 (max-complexity=10, ignores E203,E501,W503,F821)
- Docs/Logging: Google/NumPy docstrings for public APIs; `logger = logging.getLogger(__name__)`