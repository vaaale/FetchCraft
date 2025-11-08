# RAG Framework Test Suite

This directory contains the comprehensive test suite for the RAG Framework.

## Test Files

### `test_retriever.py`
Tests for retriever functionality including:
- Basic retrieval operations
- Top-k parameter overriding
- Configuration updates
- SymNode parent resolution
- Direct retriever creation
- Async retrieval aliases

### `test_node_persistence.py`
Tests for node persistence and storage:
- Node property persistence
- Parent-child relationships
- Chunk-specific properties
- Metadata storage and retrieval
- Node relationship preservation

### `test_symnode.py`
Tests for SymNode functionality:
- SymNode creation and validation
- Parent resolution in vector index
- Multiple parent resolution
- Deduplication of same parents
- Symbolic node requirements

### `conftest.py`
Pytest configuration and shared fixtures:
- Mock embeddings
- Qdrant client fixtures
- Vector store and index fixtures
- Sample data fixtures
- Asyncio event loop configuration

### `test_suite.py`
Test suite runner with multiple execution modes:
- Run all tests
- Run specific test suites
- Run with coverage reports
- Continue on errors mode

## Running Tests

### Run All Tests
```bash
# Using pytest directly
python -m pytest src/tests/ -v

# Using test suite
python src/tests/test_suite.py all
```

### Run Specific Test Suite
```bash
# Using pytest
python -m pytest src/tests/test_retriever.py -v

# Using test suite
python src/tests/test_suite.py retriever
```

### Run with Coverage
```bash
# Using pytest
python -m pytest src/tests/ --cov=fetchcraft --cov-report=html

# Using test suite
python src/tests/test_suite.py coverage
```

### Continue on Errors
```bash
# Using pytest
python -m pytest src/tests/ -v --maxfail=0

# Using test suite
python src/tests/test_suite.py all --continue
```

## Test Options

### Available Test Suite Options
- `all` - Run all tests (default)
- `retriever` - Run only retriever tests
- `node_persistence` - Run only node persistence tests
- `symnode` - Run only SymNode tests
- `coverage` - Run all tests with coverage report

### Pytest Markers
- `@pytest.mark.asyncio` - Async tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.integration` - Integration tests

## Test Structure

```
src/tests/
├── README.md                 # This file
├── conftest.py              # Pytest configuration and fixtures
├── test_suite.py            # Test suite runner
├── test_retriever.py        # Retriever tests
├── test_node_persistence.py # Node persistence tests
└── test_symnode.py          # SymNode tests
```

## Writing New Tests

### Using Fixtures

```python
import pytest


@pytest.mark.asyncio
async def test_my_feature(vector_index, mock_embeddings, sample_nodes):
    """Test description."""
    # Use fixtures directly
    await vector_index.insert_nodes(sample_nodes)
    # ... test logic
```

### Creating Custom Fixtures

Add to `conftest.py`:

```python
@pytest.fixture
def my_custom_fixture():
    """Custom fixture description."""
    # Setup
    resource = create_resource()
    yield resource
    # Teardown
    resource.cleanup()
```

### Async Tests

All async tests must be marked with `@pytest.mark.asyncio`:

```python
@pytest.mark.asyncio
async def test_async_operation():
    """Test async operation."""
    result = await some_async_function()
    assert result is not None
```

## Requirements

Install test dependencies:

```bash
pip install pytest pytest-asyncio pytest-cov
```

## CI/CD Integration

For GitHub Actions or similar CI/CD:

```yaml
- name: Run tests
  run: |
    python -m pytest src/tests/ -v --cov=fetchcraft --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Best Practices

1. **Keep tests isolated** - Each test should be independent
2. **Use fixtures** - Reuse common setup code via fixtures
3. **Test one thing** - Each test should verify one specific behavior
4. **Clear naming** - Test names should describe what they test
5. **Assertions** - Use descriptive assertion messages
6. **Mock external dependencies** - Use MockEmbeddings for API calls
7. **Clean up** - Use fixtures with teardown for resource cleanup

## Debugging Tests

### Run a single test
```bash
python -m pytest src/tests/test_retriever.py::test_basic_retriever -v
```

### Run with detailed output
```bash
python -m pytest src/tests/ -v -s
```

### Run with pdb debugger
```bash
python -m pytest src/tests/ --pdb
```

### Show print statements
```bash
python -m pytest src/tests/ -v -s --capture=no
```

## Coverage Reports

After running with coverage, view the HTML report:

```bash
python -m pytest src/tests/ --cov=fetchcraft --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain >80% code coverage
4. Add docstrings to test functions
5. Update this README if adding new test files
