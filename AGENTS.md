# AGENTS.md

**Build/Lint/Test Commands**
- Setup: `python setup_all.py` (quick) or `pip install -r requirements.txt`
- Docker: `./docker_setup.sh && docker-compose up -d`
- Run all tests: `pytest -q`
- Run single test: `pytest tests/unit/test_core/test_document_processor.py::TestDocumentProcessor::test_process_document -q`
- Coverage: `pytest --cov=src --cov-report=term-missing`
- Unit tests only: `pytest -m unit`
- Quality checks: `python scripts/check_quality.py`
- Web interface: `python run_web.py`
- CLI: `python -m src.interfaces.cli`

**Code Style Guidelines**
- **Imports**: stdlib/third-party/local alphabetically; absolute imports preferred; isort with black profile (line_length=100, multi_line_output=3)
- **Formatting**: black (100 chars, py312 target); 4 spaces; double quotes; no trailing spaces
- **Naming**: snake_case for functions/vars/methods; PascalCase for classes; UPPER_SNAKE for constants
- **Types**: type hints on all params/returns; Optional/Union as needed; mypy strict mode (python_version=3.12)
- **Errors**: specific exceptions; input validation; context managers; proper logging
- **Linting**: flake8 (max-complexity=10, ignores E203,E501,W503,F821,C901)
- **Docs**: Google/NumPy docstrings for public APIs; `logger = logging.getLogger(__name__)`