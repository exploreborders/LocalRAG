#!/usr/bin/env python3
"""
Detailed edge case testing for language detection.
"""

from src.rag_pipeline_db import RAGPipelineDB

def test_problematic_cases():
    """Test the specific problematic cases identified."""
    pipeline = RAGPipelineDB()

    print("Testing Problematic Edge Cases")
    print("=" * 50)

    # Cases that were failing in previous tests
    problematic_cases = [
        # Chinese detection issue
        ("什么是机器学习？", "zh", "Chinese characters not detected properly"),

        # German queries detected as Dutch
        ("Deep learning bedeutet?", "de", "German verb at end detected as Dutch"),
        ("bedeutet deep learning", "de", "German verb at start detected as Dutch"),

        # Mixed language queries
        ("What bedeutet machine learning?", "en", "Mixed English/German should default to English"),
        ("Wie funktioniert AI?", "de", "German with English acronym"),
        ("What is künstliche Intelligenz?", "en", "English with German umlaut word"),
    ]

    results = []
    for query, expected, description in problematic_cases:
        detected = pipeline.detect_query_language(query)
        success = detected == expected
        results.append((query, expected, detected, success, description))

        status = "✓" if success else "✗"
        print(f"{status} '{query}' -> Expected: {expected}, Got: {detected}")
        print(f"   Reason: {description}")
        print()

    return results

def test_ambiguous_queries():
    """Test queries that could be ambiguous."""
    pipeline = RAGPipelineDB()

    print("Testing Ambiguous Queries")
    print("=" * 50)

    ambiguous_cases = [
        # Very short queries
        ("AI", "en", "Single acronym"),
        ("ML", "en", "Machine Learning acronym"),
        ("RAG", "en", "RAG acronym"),

        # Numbers and symbols
        ("What is 2+2?", "en", "Math query"),
        ("¿Qué?", "es", "Spanish question mark only"),
        ("Was?", "de", "German question word"),

        # Technical terms that appear in multiple languages
        ("neural network", "en", "Technical term"),
        ("neuronales Netzwerk", "de", "German technical term"),
        ("red neuronal", "es", "Spanish technical term"),

        # Empty or whitespace
        ("", "en", "Empty string"),
        ("   ", "en", "Whitespace only"),
        ("\t\n", "en", "Tabs and newlines"),
    ]

    results = []
    for query, expected, description in ambiguous_cases:
        detected = pipeline.detect_query_language(query)
        success = detected == expected
        results.append((query, expected, detected, success, description))

        status = "✓" if success else "✗"
        print(f"{status} '{query}' -> Expected: {expected}, Got: {detected}")
        print(f"   Reason: {description}")
        print()

    return results

def test_mixed_language_scenarios():
    """Test realistic mixed language scenarios."""
    pipeline = RAGPipelineDB()

    print("Testing Mixed Language Scenarios")
    print("=" * 50)

    mixed_cases = [
        # Code switching
        ("I want to know what bedeutet 'machine learning'", "en", "English sentence with German word"),
        ("Wie funktioniert the neural network?", "de", "German sentence with English terms"),
        ("¿Qué es deep learning en español?", "es", "Spanish sentence with English term"),

        # Multilingual technical queries
        ("What is the difference between SVM and SVM?", "en", "Repeated technical terms"),
        ("Explain CNN convolutional neural network", "en", "Technical acronym expansion"),

        # Queries with proper names
        ("What did Albert Einstein discover?", "en", "English with German name"),
        ("Wer war Nikola Tesla?", "de", "German with Serbian name"),
    ]

    results = []
    for query, expected, description in mixed_cases:
        detected = pipeline.detect_query_language(query)
        success = detected == expected
        results.append((query, expected, detected, success, description))

        status = "✓" if success else "✗"
        print(f"{status} '{query}' -> Expected: {expected}, Got: {detected}")
        print(f"   Reason: {description}")
        print()

    return results

def analyze_results(all_results):
    """Analyze the test results."""
    print("EDGE CASE TEST RESULTS ANALYSIS")
    print("=" * 50)

    total_tests = len(all_results)
    passed_tests = sum(1 for _, _, _, success, _ in all_results if success)
    accuracy = (passed_tests / total_tests) * 100

    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Accuracy: {accuracy:.1f}%")

    # Group failures by expected language
    failures_by_lang = {}
    for query, expected, detected, success, desc in all_results:
        if not success:
            if expected not in failures_by_lang:
                failures_by_lang[expected] = []
            failures_by_lang[expected].append((query, detected, desc))

    if failures_by_lang:
        print("\nFAILURES BY EXPECTED LANGUAGE:")
        for lang, failures in failures_by_lang.items():
            print(f"\n{lang.upper()}: {len(failures)} failures")
            for query, detected, desc in failures:
                print(f"  '{query}' -> {detected} ({desc})")

    # Performance criteria: at least 75% accuracy on edge cases
    if accuracy >= 75:
        print("\n✅ Edge case testing PASSED (≥75% accuracy)")
        return True
    else:
        print("\n❌ Edge case testing FAILED (<75% accuracy)")
        return False

if __name__ == "__main__":
    print("Detailed Edge Case Testing for Language Detection")
    print("=" * 60)

    results1 = test_problematic_cases()
    results2 = test_ambiguous_queries()
    results3 = test_mixed_language_scenarios()

    all_results = results1 + results2 + results3
    passed = analyze_results(all_results)

    if passed:
        print("\n🎉 Edge case testing completed successfully!")
    else:
        print("\n⚠️  Edge case testing completed with issues to address.")