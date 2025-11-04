# AGENTS.md

## Build/Lint/Test Commands
- Quick Setup: `python setup_all.py`
- Docker Setup: `./docker_setup.sh && docker-compose up -d`
- Manual Setup: `python -m venv rag_env && source rag_env/bin/activate && pip install -r requirements.txt`
- Run all tests: `python tests/run_tests.py`
- Run single test: `python -c "import sys; sys.path.insert(0, 'src'); from tests.test_file import test_function; test_function()"`
- Run web interface: `python run_web.py`
- Run CLI app: `python -m src.interfaces.cli`
- Build Docker: `docker build -t local-rag .`
- Reprocess documents: `python -m src.core.reprocess_documents`

## Code Style Guidelines
- **Imports**: Group stdlib/third-party/local alphabetically, relative imports for local modules
- **Formatting**: 4 spaces indentation, double quotes for strings, no trailing whitespace
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE_CASE for constants
- **Types**: Type hints on parameters/returns, Optional for nullable types
- **Error Handling**: try/except with specific exceptions, validate inputs early, use context managers
- **Documentation**: Docstrings for public APIs/classes, inline comments for complex logic
- **Best Practices**: PEP 8 compliance, descriptive names, single responsibility, no global state
- **Logging**: Use `logger = logging.getLogger(__name__)` for module-level logging