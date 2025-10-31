# AGENTS.md

## Build/Lint/Test Commands
- Setup: `python -m venv rag_env && source rag_env/bin/activate && pip install -r requirements.txt`
- Run all tests: `python test_system.py`
- Run single test: `python -c "from test_system import test_retrieval; test_retrieval()"`
- Run web interface: `python run_web.py` or `streamlit run web_interface/app.py` (multipage app)
- Run CLI app: `python -m src.app`
- Process data: `python -m src.embeddings`
- Test retrieval: `python -m src.retrieval`

## Code Style Guidelines
- **Imports**: Group stdlib/third-party/local, sort alphabetically, use relative imports (`from .module import Class`)
- **Formatting**: 4 spaces indentation, 88 char line limit, double quotes, docstrings for public functions/classes
- **Naming**: snake_case for vars/functions, PascalCase for classes, UPPER_SNAKE_CASE for constants
- **Types**: Use type hints for function parameters and return values when clarity benefits
- **Error Handling**: try/except blocks, specific exceptions, validate inputs, use context managers for files
- **Best Practices**: PEP 8, descriptive names, single responsibility functions, add unit tests, avoid global state