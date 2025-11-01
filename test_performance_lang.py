#!/usr/bin/env python3
"""
Performance test for language detection.
"""

import time
from src.rag_pipeline_db import RAGPipelineDB

def test_detection_performance():
    """Test the performance of language detection."""
    pipeline = RAGPipelineDB()

    # Test queries of different lengths and complexities
    test_queries = [
        "What is AI?",  # Short English
        "Was bedeutet Deep Learning?",  # Short German
        "Qu'est-ce que l'intelligence artificielle et comment fonctionne-t-elle dans le domaine de l'apprentissage automatique?",  # Long French
        "Wie funktioniert maschinelles Lernen und welche Algorithmen werden dabei verwendet?",  # Long German
        "人工智能是什么，它如何在机器学习领域工作？",  # Chinese
        "機械学習とは何ですか、そしてそれはどのように機能しますか？",  # Japanese
        "",  # Empty
        "hi",  # Very short
        "What bedeutet machine learning?",  # Mixed
    ]

    print("Language Detection Performance Test")
    print("=" * 50)

    times = []
    for query in test_queries:
        start_time = time.time()
        detected_lang = pipeline.detect_query_language(query)
        end_time = time.time()

        duration_ms = (end_time - start_time) * 1000
        times.append(duration_ms)
        print(f"Query: '{query[:50]}...' -> {detected_lang} ({duration_ms:.2f}ms)")

    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)

    print(f"\nPerformance Results:")
    print(f"Average detection time: {avg_time:.2f}ms")
    print(f"Max detection time: {max_time:.2f}ms")
    print(f"Min detection time: {min_time:.2f}ms")

    # Performance criteria: should be under 50ms average
    if avg_time < 50:
        print("✅ Performance acceptable (< 50ms average)")
        return True
    else:
        print("❌ Performance too slow (>= 50ms average)")
        return False

def test_batch_performance():
    """Test performance with multiple queries in sequence."""
    pipeline = RAGPipelineDB()

    # Generate 100 test queries
    base_queries = [
        "What is machine learning?",
        "Was bedeutet Deep Learning?",
        "Qu'est-ce que l'IA?",
        "Qué es el aprendizaje automático?",
        "Cosa significa intelligenza artificiale?",
    ]

    test_queries = base_queries * 20  # 100 queries total

    print(f"\nBatch Performance Test ({len(test_queries)} queries)")
    print("=" * 50)

    start_time = time.time()
    for query in test_queries:
        pipeline.detect_query_language(query)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_query = (total_time / len(test_queries)) * 1000

    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per query: {avg_time_per_query:.2f}ms")
    print(f"Queries per second: {len(test_queries) / total_time:.1f}")

    # Batch performance criteria: should be under 10ms per query
    if avg_time_per_query < 10:
        print("✅ Batch performance acceptable (< 10ms per query)")
        return True
    else:
        print("❌ Batch performance too slow (>= 10ms per query)")
        return False

if __name__ == "__main__":
    single_perf_ok = test_detection_performance()
    batch_perf_ok = test_batch_performance()

    print("\n" + "=" * 50)
    print("PERFORMANCE TEST RESULTS:")
    print(f"Single Query Performance: {'PASS' if single_perf_ok else 'FAIL'}")
    print(f"Batch Performance: {'PASS' if batch_perf_ok else 'FAIL'}")

    if single_perf_ok and batch_perf_ok:
        print("🎉 All performance tests passed!")
    else:
        print("❌ Some performance tests failed.")