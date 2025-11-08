.PHONY: help test test-all test-retriever test-node test-symnode test-cov test-fast clean install lint format

help:
	@echo "RAG Framework - Available commands:"
	@echo ""
	@echo "  make test          - Run all tests"
	@echo "  make test-all      - Run all tests with verbose output"
	@echo "  make test-retriever- Run retriever tests only"
	@echo "  make test-node     - Run node persistence tests only"
	@echo "  make test-symnode  - Run SymNode tests only"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make test-fast     - Run tests, stop on first failure"
	@echo "  make clean         - Clean up generated files"
	@echo "  make install       - Install development dependencies"
	@echo "  make lint          - Run code linters"
	@echo "  make format        - Format code with black"

test:
	python -m pytest src/tests/ -v

test-all:
	python -m pytest src/tests/ -v -s

test-retriever:
	python -m pytest src/tests/test_retriever.py -v

test-node:
	python -m pytest src/tests/test_node_persistence.py -v

test-symnode:
	python -m pytest src/tests/test_symnode.py -v

test-cov:
	python -m pytest src/tests/ --cov=rag_framework --cov-report=html --cov-report=term-missing
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

test-fast:
	python -m pytest src/tests/ -x -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf dist/
	rm -rf build/

install:
	pip install -e .
	pip install pytest pytest-asyncio pytest-cov black ruff

lint:
	ruff check src/

format:
	black src/ --line-length 100
