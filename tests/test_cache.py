#!/usr/bin/env python3
"""
Test script for Redis cache functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def test_redis_connection():
    """Test Redis cache connectivity"""
    try:
        from src.cache.redis_cache import RedisCache
        cache = RedisCache(host="localhost", port=6379)
        assert cache.health_check()
        print("✅ Redis connection test passed")
        return True
    except Exception as e:
        print(f"❌ Redis connection test failed: {e}")
        return False

def test_cache_operations():
    """Test basic cache operations"""
    try:
        from src.cache.redis_cache import RedisCache
        cache = RedisCache(host="localhost", port=6379)

        # Test data
        test_key = "test:key"
        test_data = {"answer": "test response", "timestamp": "2024-01-01"}

        # Test set
        assert cache.set(test_key, test_data)
        print("✅ Cache set operation passed")

        # Test get
        retrieved = cache.get(test_key)
        assert retrieved is not None
        assert retrieved["answer"] == "test response"
        print("✅ Cache get operation passed")

        # Test delete
        assert cache.delete(test_key)
        assert cache.get(test_key) is None
        print("✅ Cache delete operation passed")

        return True
    except Exception as e:
        print(f"❌ Cache operations test failed: {e}")
        return False

def test_rag_cache_integration():
    """Test RAG pipeline cache integration"""
    try:
        from src.rag_pipeline_db import RAGPipelineDB

        # Create pipeline (cache should be disabled if Redis not available)
        pipeline = RAGPipelineDB()

        # Check cache status
        cache_enabled = pipeline.cache_enabled
        print(f"ℹ️  Cache enabled: {cache_enabled}")

        if cache_enabled:
            # Test cache stats
            stats = pipeline.get_cache_stats()
            print(f"ℹ️  Cache stats: {stats}")

            # Test cache key generation
            cache_key = pipeline.generate_cache_key(
                "test query", [], "en", "llama2", 0.7, 500
            )
            assert cache_key.startswith("llm:")
            print("✅ Cache key generation passed")

        print("✅ RAG cache integration test passed")
        return True
    except Exception as e:
        print(f"❌ RAG cache integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Redis Cache Tests")
    print("=" * 40)

    tests = [
        test_redis_connection,
        test_cache_operations,
        test_rag_cache_integration
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print("=" * 40)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All cache tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some cache tests failed")
        sys.exit(1)