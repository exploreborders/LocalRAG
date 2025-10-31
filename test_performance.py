#!/usr/bin/env python3
"""
Performance benchmarking for RAG system.
Tests the database-backed system with Elasticsearch vector search.
"""

import time
import numpy as np
import sys

# Add src to path
sys.path.append('src')

from src.retrieval_db import DatabaseRetriever
from src.rag_pipeline_db import RAGPipelineDB

def benchmark_retrieval():
    """
    Benchmark document retrieval performance.

    Measures query latency and result quality for different
    types of queries and model configurations.
    """
    print("🔍 Benchmarking Retrieval Performance")
    print("=" * 50)

    # Test queries
    test_queries = [
        "What is machine learning?",
        "How does natural language processing work?",
        "Explain the concept of embeddings",
        "What are the benefits of RAG systems?",
        "How to implement document search?"
    ]

    # Models to test
    models = ["nomic-ai/nomic-embed-text-v1.5"]

    results = {}

    for model in models:
        print(f"\n📊 Testing model: {model}")
        results[model] = {}

        # Initialize retrievers
        try:
            old_retriever = OldRetriever(model)
            old_available = True
        except Exception as e:
            print(f"⚠️ Old retriever failed: {e}")
            old_available = False

        try:
            new_retriever = NewRetriever(model)
            new_available = True
        except Exception as e:
            print(f"⚠️ New retriever failed: {e}")
            new_available = False

        if not old_available and not new_available:
            print("❌ Both retrievers unavailable")
            continue

        # Test each query
        for query in test_queries:
            print(f"  Query: '{query[:50]}...'")

            # Old system
            if old_available:
                start_time = time.time()
                old_results = old_retriever.retrieve(query, k=5)
                old_time = time.time() - start_time
                print(".3f")

            # New system
            if new_available:
                start_time = time.time()
                new_results = new_retriever.retrieve(query, top_k=5)
                new_time = time.time() - start_time
                print(".3f")

                if old_available:
                    speedup = old_time / new_time if new_time > 0 else float('inf')
                    print(".2f")

def benchmark_rag():
    """Benchmark RAG pipeline performance"""
    print("\n🤖 Benchmarking RAG Pipeline Performance")
    print("=" * 50)

    test_queries = [
        "What is machine learning?",
        "How does RAG improve LLM responses?"
    ]

    model = "nomic-ai/nomic-embed-text-v1.5"
    llm_model = "llama2"

    for query in test_queries:
        print(f"\nQuery: '{query}'")

        # Old RAG
        try:
            old_rag = OldRAG(llm_model)
            start_time = time.time()
            old_result = old_rag.query(query)
            old_time = time.time() - start_time
            print(".3f")
        except Exception as e:
            print(f"⚠️ Old RAG failed: {e}")
            old_time = None

        # New RAG
        try:
            new_rag = NewRAG(model, llm_model)
            start_time = time.time()
            new_result = new_rag.query(query, top_k=3)
            new_time = time.time() - start_time
            print(".3f")
        except Exception as e:
            print(f"⚠️ New RAG failed: {e}")
            new_time = None

        if old_time and new_time:
            speedup = old_time / new_time
            print(".2f")

def benchmark_load():
    """Benchmark load testing with concurrent queries"""
    print("\n⚡ Load Testing with Concurrent Queries")
    print("=" * 50)

    import concurrent.futures
    import threading

    test_queries = [
        "What is AI?",
        "Explain neural networks",
        "How do transformers work?",
        "What is vector search?",
        "Benefits of RAG"
    ] * 5  # 25 queries

    model = "nomic-ai/nomic-embed-text-v1.5"

    # Test new system
    try:
        retriever = NewRetriever(model)

        def query_worker(query):
            start_time = time.time()
            results = retriever.retrieve(query, top_k=3)
            end_time = time.time()
            return end_time - start_time, len(results)

        print(f"Testing {len(test_queries)} concurrent queries...")

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(query_worker, query) for query in test_queries]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        total_time = time.time() - start_time
        avg_time = np.mean([r[0] for r in results])
        total_results = sum([r[1] for r in results])

        print(".3f")
        print(".3f")
        print(".1f")
        print(f"✅ Total results returned: {total_results}")

    except Exception as e:
        print(f"❌ Load test failed: {e}")

def benchmark_storage():
    """Benchmark storage and indexing"""
    print("\n💾 Storage and Indexing Benchmark")
    print("=" * 50)

    try:
        from document_processor import DocumentProcessor
        processor = DocumentProcessor()

        # Get stats
        docs = processor.get_documents()
        print(f"📊 Documents in database: {len(docs)}")

        if docs:
            doc = processor.get_chunks(docs[0]['id'])
            print(f"📄 Chunks for first document: {len(doc)}")

        # Elasticsearch stats
        es = processor.es
        doc_stats = es.count(index="rag_documents")
        vector_stats = es.count(index="rag_vectors")

        print(f"📋 Documents in ES: {doc_stats['count']}")
        print(f"🔍 Vectors in ES: {vector_stats['count']}")

    except Exception as e:
        print(f"❌ Storage benchmark failed: {e}")

def main():
    """Run all benchmarks"""
    print("🚀 RAG System Performance Benchmark")
    print("Comparing old FAISS-based vs new database-backed system")
    print("=" * 60)

    try:
        benchmark_storage()
        benchmark_retrieval()
        benchmark_rag()
        benchmark_load()

        print("\n✅ Benchmarking completed!")

    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()