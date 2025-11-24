#!/usr/bin/env python3
"""
Tests for retrieval functionality including hybrid search and error handling.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from utils.error_handler import ValidationError


def test_retrieval_validation():
    """Test input validation in retrieval components."""
    print("🧪 Testing retrieval input validation...")

    try:
        # Test invalid question
        from core.retrieval import RAGPipelineDB

        # Mock the dependencies to avoid actual initialization
        with patch("core.retrieval.DatabaseRetriever") as mock_retriever_class:
            mock_retriever = Mock()
            mock_retriever_class.return_value = mock_retriever

            pipeline = RAGPipelineDB()

            # Test empty question
            try:
                pipeline.query("", top_k=5)
                print("  ❌ Should have raised ValidationError for empty question")
                return False
            except ValidationError:
                print("  ✅ Correctly validated empty question")

            # Test invalid hybrid_alpha
            try:
                pipeline.query("test question", hybrid_alpha=1.5)
                print(
                    "  ❌ Should have raised ValidationError for invalid hybrid_alpha"
                )
                return False
            except ValidationError:
                print("  ✅ Correctly validated invalid hybrid_alpha")

            # Test negative hybrid_alpha
            try:
                pipeline.query("test question", hybrid_alpha=-0.1)
                print(
                    "  ❌ Should have raised ValidationError for negative hybrid_alpha"
                )
                return False
            except ValidationError:
                print("  ✅ Correctly validated negative hybrid_alpha")

        print("✅ Validation tests passed!")
        return True

    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False


def test_hybrid_search_parameter():
    """Test hybrid_alpha parameter functionality."""
    print("🧪 Testing hybrid search parameter...")

    try:
        from core.retrieval import DatabaseRetriever

        # Test valid hybrid_alpha values
        retriever = DatabaseRetriever(hybrid_alpha=0.8)
        assert retriever.hybrid_alpha == 0.8, "Failed to set hybrid_alpha"

        retriever = DatabaseRetriever(hybrid_alpha=0.0)  # Vector only
        assert retriever.hybrid_alpha == 0.0, "Failed to set vector-only hybrid_alpha"

        retriever = DatabaseRetriever(hybrid_alpha=1.0)  # BM25 only
        assert retriever.hybrid_alpha == 1.0, "Failed to set BM25-only hybrid_alpha"

        print("✅ Hybrid search parameter tests passed!")
        return True

    except Exception as e:
        print(f"❌ Hybrid search parameter test failed: {e}")
        return False


def test_error_handler_integration():
    """Test that error handler is properly integrated."""
    print("🧪 Testing error handler integration...")

    try:
        from core.retrieval import DatabaseRetriever

        retriever = DatabaseRetriever()
        assert hasattr(retriever, "error_handler"), (
            "DatabaseRetriever should have error_handler"
        )

        # Test error handler has expected methods
        assert hasattr(retriever.error_handler, "handle_error"), (
            "Error handler should have handle_error method"
        )
        assert hasattr(retriever.error_handler, "get_error_stats"), (
            "Error handler should have get_error_stats method"
        )

        print("✅ Error handler integration tests passed!")
        return True

    except Exception as e:
        print(f"❌ Error handler integration test failed: {e}")
        return False


def test_cache_key_generation():
    """Test cache key generation with hybrid_alpha."""
    print("🧪 Testing cache key generation...")

    try:
        from core.retrieval import RAGPipelineDB

        # Mock dependencies
        with patch("core.retrieval.DatabaseRetriever") as mock_retriever_class:
            mock_retriever = Mock()
            mock_retriever_class.return_value = mock_retriever

            pipeline = RAGPipelineDB()

            # Test cache key generation
            key1 = pipeline._generate_cache_key("test question", 5, None, "en", 0.7)
            key2 = pipeline._generate_cache_key("test question", 5, None, "en", 0.8)
            key3 = pipeline._generate_cache_key(
                "different question", 5, None, "en", 0.7
            )

            # Keys should be different when hybrid_alpha differs
            assert key1 != key2, "Cache keys should differ with different hybrid_alpha"

            # Keys should be different when question differs
            assert key1 != key3, "Cache keys should differ with different questions"

            # Keys should be same when parameters are identical
            key1_duplicate = pipeline._generate_cache_key(
                "test question", 5, None, "en", 0.7
            )
            assert key1 == key1_duplicate, (
                "Identical parameters should produce same cache key"
            )

        print("✅ Cache key generation tests passed!")
        return True

    except Exception as e:
        print(f"❌ Cache key generation test failed: {e}")
        return False


if __name__ == "__main__":
    tests = [
        test_retrieval_validation,
        test_hybrid_search_parameter,
        test_error_handler_integration,
        test_cache_key_generation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n📊 Test Results: {passed}/{total} passed")
    if passed == total:
        print("🎉 All retrieval tests passed!")
        sys.exit(0)
    else:
        print(f"❌ {total - passed} tests failed")
        sys.exit(1)
