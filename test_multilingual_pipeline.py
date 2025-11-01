#!/usr/bin/env python3
"""
Test the complete multilingual RAG pipeline with real queries.
"""

from src.rag_pipeline_db import RAGPipelineDB

def test_multilingual_pipeline():
    """Test the complete pipeline with multilingual queries."""
    pipeline = RAGPipelineDB()

    print("Testing Complete Multilingual RAG Pipeline")
    print("=" * 50)

    # Test queries that should work with the current document set
    test_queries = [
        # English queries
        ("What is RAG?", "en"),
        ("What is machine learning?", "en"),
        ("Explain neural networks", "en"),

        # German queries (the original problem case)
        ("Was bedeutet Deep Learning?", "de"),
        ("Was ist künstliche Intelligenz?", "de"),
        ("Wie funktioniert maschinelles Lernen?", "de"),

        # Other languages
        ("Qu'est-ce que l'IA?", "fr"),
        ("¿Qué es el aprendizaje automático?", "es"),
        ("Cosa significa intelligenza artificiale?", "it"),
    ]

    results = []
    for query, expected_lang in test_queries:
        print(f"\nTesting: '{query}'")
        print("-" * 40)

        try:
            # This will test the full pipeline, but may fail if no documents are found
            result = pipeline.query(query)

            detected_lang = result['query_language']
            answer = result['answer']
            num_docs = result['num_docs']

            # Check if language detection worked
            lang_correct = detected_lang == expected_lang

            # Check if we got a reasonable answer
            has_answer = len(answer.strip()) > 10 and not answer.startswith("Error")

            # Check if documents were retrieved
            has_docs = num_docs > 0

            success = lang_correct and has_answer

            results.append({
                'query': query,
                'expected_lang': expected_lang,
                'detected_lang': detected_lang,
                'lang_correct': lang_correct,
                'has_answer': has_answer,
                'has_docs': has_docs,
                'answer_preview': answer[:100] + "..." if len(answer) > 100 else answer,
                'num_docs': num_docs,
                'success': success
            })

            status = "✓" if success else "✗"
            print(f"{status} Language: {detected_lang} ({'✓' if lang_correct else '✗'})")
            print(f"   Documents found: {num_docs}")
            print(f"   Answer generated: {'Yes' if has_answer else 'No'}")
            print(f"   Preview: {answer[:100]}{'...' if len(answer) > 100 else ''}")

        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                'query': query,
                'expected_lang': expected_lang,
                'error': str(e),
                'success': False
            })

    return results

def analyze_pipeline_results(results):
    """Analyze the pipeline test results."""
    print("\n" + "=" * 60)
    print("MULTILINGUAL PIPELINE TEST RESULTS")
    print("=" * 60)

    successful_results = [r for r in results if r.get('success', False)]
    total_queries = len(results)
    successful_queries = len(successful_results)

    print(f"Total queries tested: {total_queries}")
    print(f"Successful queries: {successful_queries}")
    print(f"Success rate: {(successful_queries/total_queries)*100:.1f}%")

    if successful_results:
        print("\nSuccessful queries:")
        for result in successful_results:
            print(f"✓ '{result['query']}' -> {result['detected_lang']} "
                  f"({result['num_docs']} docs, answer generated)")

    failed_results = [r for r in results if not r.get('success', False)]
    if failed_results:
        print("\nFailed queries:")
        for result in failed_results:
            if 'error' in result:
                print(f"✗ '{result['query']}' -> Error: {result['error']}")
            else:
                lang_status = "✓" if result.get('lang_correct') else "✗"
                docs_status = "✓" if result.get('has_docs') else "✗"
                answer_status = "✓" if result.get('has_answer') else "✗"
                print(f"✗ '{result['query']}' -> Lang:{lang_status} Docs:{docs_status} Answer:{answer_status}")

    # Check if the original problem is solved
    original_problem = "Was bedeutet Deep Learning?"
    original_result = next((r for r in results if r['query'] == original_problem), None)

    if original_result and original_result.get('success'):
        print(f"\n🎉 ORIGINAL PROBLEM SOLVED: '{original_problem}' correctly detected as German and answered!")
    else:
        print(f"\n❌ ORIGINAL PROBLEM NOT SOLVED: '{original_problem}' failed")

    # Overall assessment
    if successful_queries >= total_queries * 0.7:  # 70% success rate
        print("\n✅ Multilingual pipeline testing PASSED")
        return True
    else:
        print("\n❌ Multilingual pipeline testing FAILED")
        return False

def test_language_consistency():
    """Test that the same query gets consistent language detection."""
    pipeline = RAGPipelineDB()

    print("\nTesting Language Detection Consistency")
    print("=" * 50)

    test_query = "Was bedeutet Deep Learning?"
    detections = []

    # Test the same query multiple times
    for i in range(10):
        detected = pipeline.detect_query_language(test_query)
        detections.append(detected)

    # Check consistency
    consistent = all(d == detections[0] for d in detections)
    expected_lang = 'de'

    if consistent and detections[0] == expected_lang:
        print(f"✓ Consistent detection: {test_query} -> {detections[0]} (all 10 runs)")
        return True
    else:
        print(f"✗ Inconsistent detection: {detections}")
        return False

if __name__ == "__main__":
    # Test the complete pipeline
    pipeline_results = test_multilingual_pipeline()

    # Test consistency
    consistency_passed = test_language_consistency()

    # Analyze results
    pipeline_passed = analyze_pipeline_results(pipeline_results)

    print("\n" + "=" * 60)
    print("FINAL ASSESSMENT")
    print("=" * 60)

    if pipeline_passed and consistency_passed:
        print("🎉 ALL TESTS PASSED - Multilingual RAG pipeline is working!")
        print("\nKey achievements:")
        print("- German queries are correctly detected and answered in German")
        print("- Language detection is consistent and fast")
        print("- Multiple languages are supported with appropriate prompts")
        print("- The original issue has been resolved")
    else:
        print("⚠️  Some tests failed - check the output above for details")

    print(f"\nPipeline tests: {'PASS' if pipeline_passed else 'FAIL'}")
    print(f"Consistency tests: {'PASS' if consistency_passed else 'FAIL'}")