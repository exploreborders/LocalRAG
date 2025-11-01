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
        result = subprocess.run([sys.executable, test_file],
                              capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"❌ {test_file} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running {test_file}: {e}")
        return False

def main():
    """Run all test files in the tests directory."""
    print("🧪 Local RAG Test Suite")
    print("=" * 60)

    # Get all test files
    test_files = glob.glob("test_*.py")
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
            print(f"✅ {test_file} PASSED")
        else:
            failed += 1
            print(f"❌ {test_file} FAILED")

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    print(f"Total tests: {len(test_files)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/len(test_files)*100):.1f}%")

    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())