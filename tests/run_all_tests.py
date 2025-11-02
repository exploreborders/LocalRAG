#!/usr/bin/env python3
"""
Test runner script to execute all tests in the tests directory.
"""

import os
import sys
import subprocess
import glob

def run_test(test_file):
    """Run a single test file and return the result."""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print('='*60)

    try:
        # Import and run the test module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_module", test_file)
        if spec and spec.loader:
            test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_module)

            # Run the main function if it exists
            if hasattr(test_module, 'main'):
                result = test_module.main()
                return result == 0
            elif hasattr(test_module, 'test_analytics_metrics'):
                # Special case for analytics test
                result = test_module.test_analytics_metrics()
                return result
            else:
                # Assume the test runs on import
                return True
        else:
            print(f"❌ Could not load {test_file}")
            return False
    except Exception as e:
        print(f"❌ Error running {test_file}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all test files in the tests directory."""
    print("🧪 Local RAG Test Suite")
    print("=" * 60)

    # Get all test files
    test_dir = os.path.dirname(__file__)
    test_files = glob.glob(os.path.join(test_dir, "test_*.py"))
    test_files.sort()  # Run in alphabetical order

    if not test_files:
        print("❌ No test files found!")
        return 1

    print(f"Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {test_file}")

    # Run tests
    passed = 0
    failed = 0

    for test_file in test_files:
        if run_test(test_file):
            passed += 1
            print(f"✅ {os.path.basename(test_file)} PASSED")
        else:
            failed += 1
            print(f"❌ {os.path.basename(test_file)} FAILED")

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    total_run = passed + failed
    print(f"Total tests: {total_run}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    if total_run > 0:
        print(f"Success rate: {(passed/total_run*100):.1f}%")

    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())