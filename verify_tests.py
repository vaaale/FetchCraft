#!/usr/bin/env python3
"""
Verify that the test suite is properly configured and can run.
"""

import sys
import subprocess
from pathlib import Path


def check_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    path = Path(file_path)
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"{status} {file_path}")
    return exists


def check_pytest_installed() -> bool:
    """Check if pytest is installed."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ pytest is installed: {result.stdout.strip()}")
            return True
        else:
            print("✗ pytest is not installed")
            return False
    except Exception as e:
        print(f"✗ Error checking pytest: {e}")
        return False


def check_pytest_asyncio_installed() -> bool:
    """Check if pytest-asyncio is installed."""
    try:
        import pytest_asyncio
        print(f"✓ pytest-asyncio is installed")
        return True
    except ImportError:
        print("✗ pytest-asyncio is not installed")
        return False


def count_tests() -> int:
    """Count the number of tests."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "src/tests/", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            # Look for line like "14 tests collected"
            for line in lines:
                if "test" in line and "collected" in line:
                    count = line.split()[0]
                    print(f"✓ Found {count} tests")
                    return int(count)
        print("✗ Could not count tests")
        return 0
    except Exception as e:
        print(f"✗ Error counting tests: {e}")
        return 0


def main():
    """Main verification function."""
    print("=" * 60)
    print("Test Suite Verification")
    print("=" * 60)
    print()
    
    all_checks_passed = True
    
    # Check required files
    print("Checking required files...")
    required_files = [
        "pytest.ini",
        "Makefile",
        "src/tests/README.md",
        "src/tests/conftest.py",
        "src/tests/test_suite.py",
        "src/tests/test_retriever.py",
        "src/tests/test_node_persistence.py",
        "src/tests/test_symnode.py",
    ]
    
    for file_path in required_files:
        if not check_file_exists(file_path):
            all_checks_passed = False
    
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_pytest_installed():
        all_checks_passed = False
        print("  Install with: pip install pytest")
    
    if not check_pytest_asyncio_installed():
        all_checks_passed = False
        print("  Install with: pip install pytest-asyncio")
    
    print()
    
    # Count tests
    print("Counting tests...")
    test_count = count_tests()
    if test_count == 0:
        all_checks_passed = False
    
    print()
    print("=" * 60)
    
    if all_checks_passed:
        print("✓ Test suite is properly configured!")
        print()
        print("Run tests with:")
        print("  python -m pytest src/tests/ -v")
        print("  python src/tests/test_suite.py all")
        print("  make test")
        return 0
    else:
        print("✗ Test suite has issues. Please fix the problems above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
