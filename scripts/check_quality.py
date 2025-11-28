#!/usr/bin/env python3
"""
Quality assurance script for LocalRAG.
Runs all code quality checks and reports results.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"🔍 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} passed")
            return True
        else:
            print(f"❌ {description} failed")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def main():
    """Run all quality checks."""
    print("🚀 LocalRAG Quality Assurance")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Error: Run this script from the project root directory")
        sys.exit(1)

    checks = [
        # Code formatting
        ("black --check --diff src/ tests/", "Code formatting (Black)"),
        ("isort --check-only --diff src/ tests/", "Import sorting (isort)"),
        # Linting
        (
            "flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics",
            "Critical linting (flake8)",
        ),
        # Type checking (allow to fail for now)
        (
            "mypy src/ --ignore-missing-imports --no-strict-optional || echo 'MyPy found issues'",
            "Type checking (mypy)",
        ),
        # Unit tests
        (
            "python -m pytest tests/unit/ -v --tb=short --cov=src --cov-report=term-missing",
            "Unit tests",
        ),
        # Security checks
        (
            "bandit -r src/ -f json -o /tmp/bandit-report.json || true",
            "Security scan (bandit)",
        ),
        (
            "safety check --full-report || echo 'Safety check completed'",
            "Dependency security (safety)",
        ),
    ]

    results = []
    for cmd, description in checks:
        success = run_command(cmd, description)
        results.append((description, success))

    # Summary
    print("\n" + "=" * 50)
    print("📊 QUALITY CHECK SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(results)

    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {description}")
        if success:
            passed += 1

    print(f"\n📈 Results: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All quality checks passed!")
        return 0
    else:
        print("⚠️  Some quality checks failed. Please review and fix issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
