#!/usr/bin/env python3
"""
Basic test runner for the Local RAG system.
Run with: python tests/run_tests.py
"""

import sys
import os
from pathlib import Path

def run_tests():
    """Run all tests in the tests directory."""
    print("🧪 Local RAG Test Runner")
    print("=" * 40)

    # Add src to path
    project_root = Path(__file__).parent.parent
    src_path = project_root / 'src'
    sys.path.insert(0, str(src_path))

    # Find all test files
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob('test_*.py'))

    if not test_files:
        print("ℹ️  No test files found. Create test files in tests/ directory.")
        print("   Example: tests/test_basic.py")
        return

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for test_file in test_files:
        print(f"\n📋 Running {test_file.name}")
        try:
            # Import and run the test module
            module_name = test_file.stem
            spec = importlib.util.spec_from_file_location(module_name, test_file)
            if spec and spec.loader:
                test_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(test_module)

                # Run test functions
                for attr_name in dir(test_module):
                    if attr_name.startswith('test_'):
                        test_func = getattr(test_module, attr_name)
                        if callable(test_func):
                            try:
                                test_func()
                                print(f"  ✅ {attr_name}")
                                passed_tests += 1
                            except Exception as e:
                                print(f"  ❌ {attr_name}: {e}")
                                failed_tests += 1
                            total_tests += 1

        except Exception as e:
            print(f"  ❌ Failed to load {test_file.name}: {e}")
            failed_tests += 1

    print(f"\n📊 Test Results: {passed_tests}/{total_tests} passed")
    if failed_tests > 0:
        print(f"❌ {failed_tests} tests failed")
        return 1
    else:
        print("🎉 All tests passed!")
        return 0

if __name__ == "__main__":
    import importlib.util
    sys.exit(run_tests())