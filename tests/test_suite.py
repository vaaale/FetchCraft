"""
Test Suite for RAG Framework

This module provides a comprehensive test suite for running all tests.
Can be run with: python -m pytest src/tests/test_suite.py -v
"""

import pytest
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))


def run_all_tests():
    """Run all tests in the test suite."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", "src/tests/", "-v", "--tb=short"],
        capture_output=False
    )
    return result.returncode


def run_specific_suite(suite_name: str):
    """
    Run a specific test suite.
    
    Args:
        suite_name: Name of the test file (without .py extension)
                   Options: 'test_retriever', 'test_node_persistence', 'test_symnode'
    """
    import subprocess
    test_file = f"src/tests/{suite_name}.py"
    result = subprocess.run(
        ["python", "-m", "pytest", test_file, "-v", "--tb=short"],
        capture_output=False
    )
    return result.returncode


class TestSuite:
    """Test suite class for organizing all tests."""
    
    @staticmethod
    def test_retriever():
        """Run all retriever tests."""
        return pytest.main([
            "src/tests/test_retriever.py",
            "-v",
            "--tb=short"
        ])
    
    @staticmethod
    def test_node_persistence():
        """Run node persistence tests."""
        return pytest.main([
            "src/tests/test_node_persistence.py",
            "-v",
            "--tb=short"
        ])
    
    @staticmethod
    def test_symnode():
        """Run SymNode tests."""
        pytest.main([
            "src/tests/test_symnode.py",
            "-v",
            "--tb=short"
        ])
    
    @staticmethod
    def run_all():
        """Run all tests."""
        return pytest.main([
            "src/tests/",
            "-v",
            "--tb=short",
            "-x"  # Stop on first failure
        ])
    
    @staticmethod
    def run_all_continue():
        """Run all tests without stopping on failures."""
        return pytest.main([
            "src/tests/",
            "-v",
            "--tb=short"
        ])
    
    @staticmethod
    def run_with_coverage():
        """Run all tests with coverage report."""
        return pytest.main([
            "src/tests/",
            "-v",
            "--cov=fetchcraft",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAG Framework tests")
    parser.add_argument(
        "suite",
        nargs="?",
        default="all",
        choices=["all", "retriever", "node_persistence", "symnode", "coverage"],
        help="Which test suite to run"
    )
    parser.add_argument(
        "--continue",
        dest="continue_on_error",
        action="store_true",
        help="Continue running tests even if some fail"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("RAG Framework Test Suite")
    print("="*60)
    print()
    
    suite = TestSuite()
    
    if args.suite == "all":
        if args.continue_on_error:
            exit_code = suite.run_all_continue()
        else:
            exit_code = suite.run_all()
    elif args.suite == "retriever":
        exit_code = suite.test_retriever()
    elif args.suite == "node_persistence":
        exit_code = suite.test_node_persistence()
    elif args.suite == "symnode":
        exit_code = suite.test_symnode()
    elif args.suite == "coverage":
        exit_code = suite.run_with_coverage()
    else:
        exit_code = -1
    
    print()
    print("="*60)
    if exit_code == 0:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*60)
    
    # sys.exit(exit_code)
