# AGENTS.md

## Build/Lint/Test Commands
- Quick Setup: `python setup_all.py`
- Docker Setup: `./docker_setup.sh`
- Manual Setup: `python -m venv rag_env && source rag_env/bin/activate && pip install -r requirements.txt`
- Run all tests: `python tests/run_all_tests.py`
- Run single test: `python -c "from tests.test_system import test_retrieval; test_retrieval()"`
- Run web interface: `streamlit run web_interface/app.py`
- Run CLI app: `python -m src.app`

## Code Style Guidelines
- **Imports**: Group stdlib/third-party/local imports alphabetically, use relative imports for local modules
- **Formatting**: 4 spaces indentation, 88 char line limit, double quotes for strings
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE_CASE for constants
- **Types**: Use type hints on parameters/returns when beneficial for clarity
- **Error Handling**: try/except with specific exceptions, validate inputs, use context managers
- **Documentation**: Docstrings for public APIs and classes, use logging for debugging
- **Best Practices**: PEP 8 compliance, descriptive names, single responsibility principle, no global state