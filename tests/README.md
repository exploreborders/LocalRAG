# Testing Guide for LocalRAG

## Overview

This document provides guidance for writing and running unit tests in the LocalRAG project. The project uses pytest with comprehensive mocking to enable isolated unit testing, with 409+ unit tests covering the entire codebase.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests (isolated, fast)
│   ├── __init__.py
│   ├── test_utils/          # Utility function tests
│   ├── test_core/           # Core business logic tests
│   └── test_models/         # Database model tests
├── integration/             # Integration tests (full system)
└── fixtures/                # Test data and fixtures
```

## Running Tests

### Basic Commands

```bash
# Run all unit tests
pytest tests/unit/

# Run all integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_utils/test_error_handler.py

# Run specific test class/method
pytest tests/unit/test_utils/test_error_handler.py::TestSafeExecute::test_successful_execution

# Run tests in parallel (requires pytest-xdist)
pytest -n auto tests/unit/
```

### Coverage Requirements

- **Unit tests**: 80%+ coverage (currently exceeding)
- **Integration tests**: Key user workflows
- **Coverage report**: `htmlcov/index.html`

## Writing Unit Tests

### Test Organization

```python
import pytest
from unittest.mock import Mock, MagicMock

class TestClassName:
    """Test class for specific functionality."""

    def test_method_name(self, mock_dependency):
        """Test specific behavior."""
        # Arrange
        # Act
        # Assert
```

### Mocking Guidelines

1. **Mock External Dependencies**: Database, LLM, file system, network calls
2. **Use Fixtures**: Common mocks in `conftest.py`
3. **Test Behavior**: Focus on what the code does, not implementation details
4. **Descriptive Names**: Test names should explain what they're testing

### Available Fixtures

**Database & Storage:**
- `mock_db_session`: Mock SQLAlchemy session
- `mock_elasticsearch`: Mock Elasticsearch client
- `mock_redis_cache`: Mock Redis cache

**AI/ML Services:**
- `mock_ollama_client`: Mock Ollama LLM client
- `mock_sentence_transformer`: Mock embedding model

**System:**
- `mock_file_system`: Mock file operations
- `mock_requests`: Mock HTTP requests

**Test Data:**
- `sample_document_data`: Sample document structure
- `sample_chunk_data`: Sample document chunk
- `sample_query`: Sample query data

### Example Test

```python
import pytest
from src.utils.error_handler import safe_execute, ValidationError

class TestSafeExecute:
    """Test the safe_execute function."""

    def test_successful_execution(self):
        """Test successful function execution."""
        result = safe_execute(lambda: 42)
        assert result == 42

    def test_exception_with_default(self):
        """Test exception handling with default return value."""
        result = safe_execute(lambda: 1/0, default_return=0)
        assert result == 0

    def test_specific_exception_catching(self):
        """Test catching specific exception types."""
        def failing_func():
            raise ValidationError("test")

        result = safe_execute(
            failing_func,
            error_types=ValidationError,
            default_return="caught"
        )
        assert result == "caught"
```

## Best Practices

### Test Isolation
- Each test should be independent
- Use mocks to avoid external dependencies
- Clean up test data between tests

### Test Naming
- `test_<method_name>_<scenario>_<expected_result>`
- Example: `test_validate_filename_dangerous_raises_error`

### Assertions
- Use specific assertions (`assert result == expected`)
- Check types when relevant (`assert isinstance(result, dict)`)
- Test error conditions with `pytest.raises()`

### Mocking Patterns
```python
# Mock a method return value
mock_client.return_value = "expected response"

# Mock a method to raise an exception
mock_client.side_effect = ValueError("test error")

# Mock multiple calls
mock_client.side_effect = ["first", "second", ValueError("third")]
```

## CI/CD Integration

### GitHub Actions
```yaml
- name: Run unit tests
  run: pytest tests/unit/ -v --cov=src --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Pre-commit Hooks
```bash
# Run tests before commit
pytest tests/unit/ --tb=short
```

## Adding New Tests

1. **Identify testable components** in your code
2. **Create test file** in appropriate directory
3. **Write tests** following the patterns above
4. **Run tests** to ensure they pass
5. **Check coverage** to ensure adequate testing

## Common Patterns

### Testing Functions with Dependencies
```python
def test_function_with_db_dependency(mock_db_session):
    # Mock database calls
    mock_db_session.query.return_value.filter.return_value.all.return_value = []
    
    result = my_function()
    assert result == expected
```

### Testing Error Conditions
```python
def test_function_error_handling():
    with pytest.raises(ValidationError):
        my_function(invalid_input)
```

### Testing Async Functions
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await my_async_function()
    assert result == expected
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure test files are in the correct directory structure
2. **Mock Not Working**: Check that you're mocking the right object/method
3. **Fixture Not Found**: Ensure fixtures are defined in `conftest.py`
4. **Coverage Low**: Add more test cases or check coverage exclusions

### Debugging Tests
```bash
# Run with verbose output
pytest -v -s

# Stop on first failure
pytest --tb=short --maxfail=1

# Run specific failing test
pytest tests/unit/test_specific.py::TestClass::test_method -xvs
```