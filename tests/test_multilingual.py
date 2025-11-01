#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""
Test script for multilingual language detection and RAG pipeline.
"""

from src.rag_pipeline_db import RAGPipelineDB

def test_language_detection():
    """Test language detection for all supported languages."""
    pipeline = RAGPipelineDB()

    # Test queries in different languages
    test_queries = {
        'en': "What is machine learning?",
        'de': "Was bedeutet Deep Learning?",
        'fr': "Qu'est-ce que l'apprentissage automatique?",
        'es': "Qué es el aprendizaje automático?",
        'it': "Cosa significa apprendimento profondo?",
        'pt': "O que é aprendizado de máquina?",
        'nl': "Wat is machinaal leren?",
        'sv': "Vad är maskininlärning?",
        'pl': "Co to jest uczenie maszynowe?",
        'zh': "什么是机器学习？",
        'ja': "機械学習とは何ですか？",
        'ko': "머신러닝이란 무엇인가요?"
    }

    print("Testing Language Detection:")
    print("=" * 50)

    correct_detections = 0
    total_tests = len(test_queries)

    for expected_lang, query in test_queries.items():
        detected_lang = pipeline.detect_query_language(query)
        status = "✓" if detected_lang == expected_lang else "✗"
        print(f"{status} Expected: {expected_lang}, Detected: {detected_lang} - '{query}'")

        if detected_lang == expected_lang:
            correct_detections += 1

    accuracy = (correct_detections / total_tests) * 100
    print(f"\nAccuracy: {correct_detections}/{total_tests} ({accuracy:.1f}%)")

    return accuracy >= 60  # Pass if accuracy is at least 60%

def test_edge_cases():
    """Test edge cases for language detection."""
    pipeline = RAGPipelineDB()

    edge_cases = [
        ("", "en"),  # Empty query
        ("hi", "en"),  # Very short
        ("What is AI?", "en"),  # Mixed technical terms
        ("Was bedeutet AI?", "de"),  # German with English term
        ("Comment ça marche?", "fr"),  # French with contractions
        ("¿Cómo funciona?", "es"),  # Spanish with question marks
        ("Wie funktioniert das?", "de"),  # German question
        ("Deep learning bedeutet?", "de"),  # German verb at end
        ("bedeutet deep learning", "de"),  # German verb at start
        ("What bedeutet machine learning?", "en"),  # Mixed languages
    ]

    print("\nTesting Edge Cases:")
    print("=" * 50)

    correct_detections = 0
    total_tests = len(edge_cases)

    for query, expected_lang in edge_cases:
        detected_lang = pipeline.detect_query_language(query)
        status = "✓" if detected_lang == expected_lang else "✗"
        print(f"{status} Expected: {expected_lang}, Detected: {detected_lang} - '{query}'")

        if detected_lang == expected_lang:
            correct_detections += 1

    accuracy = (correct_detections / total_tests) * 100
    print(f"\nEdge Case Accuracy: {correct_detections}/{total_tests} ({accuracy:.1f}%)")

    return accuracy >= 70  # Pass if accuracy is at least 70%

def test_full_pipeline():
    """Test the complete RAG pipeline with multilingual queries."""
    pipeline = RAGPipelineDB()

    test_queries = [
        ("What is RAG?", "en"),
        ("Was ist RAG?", "de"),
        ("Qu'est-ce que le RAG?", "fr"),
    ]

    print("\nTesting Full RAG Pipeline:")
    print("=" * 50)

    for query, expected_lang in test_queries:
        result = pipeline.query(query)
        detected_lang = result['query_language']
        answer_preview = result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']

        status = "✓" if detected_lang == expected_lang else "✗"
        print(f"{status} Query: '{query}'")
        print(f"   Detected Language: {detected_lang} (Expected: {expected_lang})")
        print(f"   Answer Preview: {answer_preview}")
        print(f"   Retrieved Docs: {result['num_docs']}")
        print()

if __name__ == "__main__":
    print("Multilingual RAG Pipeline Testing")
    print("=" * 50)

    # Test language detection
    lang_test_passed = test_language_detection()

    # Test edge cases
    edge_test_passed = test_edge_cases()

    # Test full pipeline
    test_full_pipeline()

    print("\nTest Results:")
    print(f"Language Detection: {'PASS' if lang_test_passed else 'FAIL'}")
    print(f"Edge Cases: {'PASS' if edge_test_passed else 'FAIL'}")

    if lang_test_passed and edge_test_passed:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed. Check the output above.")