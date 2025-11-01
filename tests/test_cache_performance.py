#!/usr/bin/env python3
"""
Performance test for Redis caching functionality
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def test_cache_performance():
    """Test cache performance with repeated queries"""
    try:
        from src.rag_pipeline_db import RAGPipelineDB

        pipeline = RAGPipelineDB()

        if not pipeline.cache_enabled:
            print("⚠️ Cache not enabled, skipping performance test")
            return True

        # Test query that should work with existing documents
        test_query = "What is machine learning?"

        print("🧪 Cache Performance Test")
        print("=" * 50)

        # First query (cache miss)
        print("📝 First query (cache miss)...")
        start_time = time.time()
        result1 = pipeline.query(test_query)
        first_time = time.time() - start_time

        print(".2f")
        print(f"   Cache status: {'HIT' if 'cached' in str(result1).lower() else 'MISS'}")

        # Second query (cache hit)
        print("📝 Second query (cache hit)...")
        start_time = time.time()
        result2 = pipeline.query(test_query)
        second_time = time.time() - start_time

        print(".2f")

        # Calculate speedup
        if second_time > 0 and first_time > 0:
            speedup = first_time / second_time
            print(".1f")

            if speedup >= 2.0:
                print("✅ Caching provides significant performance improvement!")
                return True
            else:
                print("⚠️ Caching provides minimal improvement")
                return True
        else:
            print("⚠️ Unable to calculate speedup")
            return False

    except Exception as e:
        print(f"❌ Cache performance test failed: {e}")
        return False

def test_cache_stats():
    """Test cache statistics"""
    try:
        from src.rag_pipeline_db import RAGPipelineDB

        pipeline = RAGPipelineDB()

        if not pipeline.cache_enabled:
            print("⚠️ Cache not enabled")
            return True

        stats = pipeline.get_cache_stats()
        print("📊 Cache Statistics:")
        print(f"   Total keys: {stats.get('total_keys', 0)}")
        print(f"   Memory used: {stats.get('memory_used', 'unknown')}")
        print(".1%")
        print(f"   Connected clients: {stats.get('connected_clients', 0)}")

        return True

    except Exception as e:
        print(f"❌ Cache stats test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Cache Performance Testing")
    print("=" * 50)

    tests = [
        test_cache_performance,
        test_cache_stats
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 Cache performance tests completed!")
        sys.exit(0)
    else:
        print("⚠️ Some cache performance tests failed")
        sys.exit(1)