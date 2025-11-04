#!/usr/bin/env python3
"""
Performance test script for the enhanced Local RAG system.
Tests query latency and context expansion improvements.
"""

import sys
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.rag_pipeline_db import RAGPipelineDB
from src.database.models import SessionLocal

def test_query_performance():
    """Test query performance with the enhanced system."""
    print("🚀 Enhanced RAG System Performance Test")
    print("=" * 50)

    # Initialize RAG pipeline
    rag_pipeline = RAGPipelineDB(llm_model="llama3.2:latest")

    # Test queries
    test_queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does deep learning work?",
        "What are the differences between AI and machine learning?"
    ]

    results = []

    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")

        start_time = time.time()
        try:
            # Run the enhanced query
            result = rag_pipeline.query(query, top_k=3)
            latency = time.time() - start_time

            # Analyze results
            retrieved_docs = result.get('retrieved_documents', [])
            context_length = sum(len(doc.get('content', '')) for doc in retrieved_docs)
            answer_length = len(result.get('answer', ''))

            # Count relationships used (rough estimate)
            all_content = ' '.join(doc.get('content', '') for doc in retrieved_docs)
            relationships_used = all_content.count('related') + all_content.count('co-occurs')

            results.append({
                'query': query,
                'latency': latency,
                'context_length': context_length,
                'answer_length': answer_length,
                'relationships_used': relationships_used,
                'success': True
            })

            print(".2f")
            print(f"  📏 Context length: {context_length} chars")
            print(f"  🤖 Answer length: {answer_length} chars")
            print(f"  🔗 Relationships used: {relationships_used}")

        except Exception as e:
            latency = time.time() - start_time
            print(f"  ❌ Failed: {e}")
            results.append({
                'query': query,
                'latency': latency,
                'success': False,
                'error': str(e)
            })

    # Calculate averages
    successful_results = [r for r in results if r['success']]

    if successful_results:
        avg_latency = sum(r['latency'] for r in successful_results) / len(successful_results)
        avg_context = sum(r['context_length'] for r in successful_results) / len(successful_results)
        avg_answer = sum(r['answer_length'] for r in successful_results) / len(successful_results)
        total_relationships = sum(r['relationships_used'] for r in successful_results)

        print("\n📊 PERFORMANCE SUMMARY")
        print("=" * 50)
        print(".2f")
        print(f"📏 Average context length: {avg_context:.0f} chars")
        print(f"🤖 Average answer length: {avg_answer:.0f} chars")
        print(f"🔗 Total relationships leveraged: {total_relationships}")
        print(f"✅ Successful queries: {len(successful_results)}/{len(test_queries)}")

        # Performance assessment
        print("\n🎯 PERFORMANCE ASSESSMENT")
        print("=" * 50)

        if avg_latency < 2.0:
            print("✅ Query latency: EXCELLENT (< 2s)")
        elif avg_latency < 5.0:
            print("✅ Query latency: GOOD (< 5s)")
        else:
            print("⚠️ Query latency: NEEDS OPTIMIZATION (> 5s)")

        if avg_context > 2000:
            print("✅ Context expansion: EXCELLENT (> 2000 chars)")
        elif avg_context > 1000:
            print("✅ Context expansion: GOOD (> 1000 chars)")
        else:
            print("⚠️ Context expansion: LIMITED (< 1000 chars)")

        if total_relationships > 5:
            print("✅ Knowledge graph utilization: EXCELLENT (> 5 relationships)")
        elif total_relationships > 2:
            print("✅ Knowledge graph utilization: GOOD (> 2 relationships)")
        else:
            print("⚠️ Knowledge graph utilization: LIMITED (< 3 relationships)")

    else:
        print("❌ No successful queries - system may need troubleshooting")

if __name__ == "__main__":
    test_query_performance()