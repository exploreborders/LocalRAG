# AGENTS.md

**Build/Lint/Test**
- Quick Setup: `python setup_all.py`
- Docker Setup: `./docker_setup.sh && docker-compose up -d`
- Manual Setup: `python -m venv rag_env && source rag_env/bin/activate && pip install -r requirements.txt`
- Run all tests: `pytest -q`
- Run single test: `pytest tests/<module>.py::<TestClass|function> -q`
- Run basic tests: `python tests/run_tests.py`
- Run web interface: `python run_web.py`
- Run CLI: `python -m src.interfaces.cli`
- Build Docker: `docker build -t local-rag .`
- Reprocess documents: `python -m src.core.reprocess_documents`

**Code Style**
- Imports: stdlib/third-party/local alphabetically; absolute imports; relative for locals
- Formatting: 4 spaces; 100 chars; double quotes; no trailing spaces
- Naming: snake_case for funcs/vars/methods; PascalCase for classes; UPPER_SNAKE for constants
- Types: type hints on all params/returns; Optional/Union as needed
- Errors: specific exceptions; input validation; context managers
- Docs/Logging: docstrings for public APIs; Google/NumPy; `logger = logging.getLogger(__name__)`