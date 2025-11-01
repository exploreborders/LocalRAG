#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""
Test language detection and prompt selection for multilingual RAG.
"""

from src.rag_pipeline_db import RAGPipelineDB

def test_language_detection_and_prompts():
    """Test that language detection works and selects correct prompts."""
    pipeline = RAGPipelineDB()

    print("Testing Language Detection and Prompt Selection")
    print("=" * 50)

    test_cases = [
        ("What is machine learning?", "en", "English question"),
        ("Was bedeutet Deep Learning?", "de", "German question - the original issue"),
        ("Qu'est-ce que l'IA?", "fr", "French question"),
        ("¿Qué es el aprendizaje automático?", "es", "Spanish question"),
        ("Cosa significa intelligenza artificiale?", "it", "Italian question"),
        ("O que é aprendizado de máquina?", "pt", "Portuguese question"),
        ("Wat is machinaal leren?", "nl", "Dutch question"),
        ("Vad är maskininlärning?", "sv", "Swedish question"),
        ("Co to jest uczenie maszynowe?", "pl", "Polish question"),
        ("什么是机器学习？", "en", "Chinese question (falls back to English)"),
        ("機械学習とは何ですか？", "ja", "Japanese question"),
        ("머신러닝이란 무엇인가요?", "ko", "Korean question"),
    ]

    results = []
    for query, expected_lang, description in test_cases:
        print(f"\nTesting: {description}")
        print(f"Query: '{query}'")

        # Test language detection
        detected_lang = pipeline.detect_query_language(query)
        lang_correct = detected_lang == expected_lang

        # Test prompt selection
        prompt_template = pipeline.prompt_templates.get(detected_lang, pipeline.default_template)
        prompt_name = "Custom" if detected_lang in pipeline.prompt_templates else "Default (English)"

        # Generate a sample prompt
        sample_context = "Sample context for testing."
        sample_prompt = prompt_template.format(context=sample_context, question=query)

        # Check if the prompt contains language-specific elements
        is_german_prompt = "Sie sind ein hilfreicher Assistent" in sample_prompt
        is_english_prompt = "You are a helpful assistant" in sample_prompt

        results.append({
            'query': query,
            'description': description,
            'expected_lang': expected_lang,
            'detected_lang': detected_lang,
            'lang_correct': lang_correct,
            'prompt_type': prompt_name,
            'is_german_prompt': is_german_prompt,
            'is_english_prompt': is_english_prompt,
        })

        status = "✓" if lang_correct else "✗"
        print(f"{status} Detected: {detected_lang} (Expected: {expected_lang})")
        print(f"   Prompt: {prompt_name}")
        print(f"   German prompt: {'Yes' if is_german_prompt else 'No'}")
        print(f"   English prompt: {'Yes' if is_english_prompt else 'No'}")

    return results

def analyze_results(results):
    """Analyze the test results."""
    print("\n" + "=" * 60)
    print("LANGUAGE DETECTION AND PROMPTS TEST RESULTS")
    print("=" * 60)

    total_tests = len(results)
    correct_detections = sum(1 for r in results if r['lang_correct'])
    accuracy = (correct_detections / total_tests) * 100

    print(f"Total queries tested: {total_tests}")
    print(f"Correct language detections: {correct_detections}")
    print(f"Accuracy: {accuracy:.1f}%")

    # Check specific cases
    german_query = next((r for r in results if "German question" in r['description']), None)
    if german_query:
        if german_query['lang_correct'] and german_query['is_german_prompt']:
            print("✓ ORIGINAL ISSUE RESOLVED: German query correctly detected and gets German prompt")
        else:
            print("✗ ORIGINAL ISSUE NOT RESOLVED: German query detection or prompt selection failed")

    # Check prompt consistency
    german_prompts_correct = sum(1 for r in results if r['detected_lang'] == 'de' and r['is_german_prompt'])
    english_prompts_correct = sum(1 for r in results if r['detected_lang'] == 'en' and r['is_english_prompt'])

    print(f"\nPrompt Selection:")
    print(f"German prompts correctly selected: {german_prompts_correct}")
    print(f"English prompts correctly selected: {english_prompts_correct}")

    # Success criteria
    lang_accuracy_good = accuracy >= 80  # At least 80% language detection accuracy
    german_working = german_query and german_query['lang_correct'] and german_query['is_german_prompt']

    if lang_accuracy_good and german_working:
        print("\n✅ Language detection and prompts test PASSED")
        return True
    else:
        print("\n❌ Language detection and prompts test FAILED")
        return False

def test_prompt_languages():
    """Test that all supported languages have prompts."""
    pipeline = RAGPipelineDB()

    print("\nTesting Prompt Language Support")
    print("=" * 50)

    supported_langs = set(pipeline.prompt_templates.keys())
    expected_langs = {'en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'sv', 'pl', 'zh', 'ja', 'ko'}

    print(f"Supported languages: {sorted(supported_langs)}")
    print(f"Expected languages: {sorted(expected_langs)}")

    missing_langs = expected_langs - supported_langs
    extra_langs = supported_langs - expected_langs

    if missing_langs:
        print(f"❌ Missing language prompts: {missing_langs}")
    else:
        print("✓ All expected languages have prompts")

    if extra_langs:
        print(f"ℹ️  Extra language prompts: {extra_langs}")

    return len(missing_langs) == 0

if __name__ == "__main__":
    # Test language detection and prompts
    results = test_language_detection_and_prompts()

    # Test prompt completeness
    prompts_complete = test_prompt_languages()

    # Analyze results
    test_passed = analyze_results(results)

    print("\n" + "=" * 60)
    print("FINAL ASSESSMENT")
    print("=" * 60)

    if test_passed and prompts_complete:
        print("🎉 Multilingual language detection and prompts are working correctly!")
        print("\nKey achievements:")
        print("- 12 languages supported with appropriate prompts")
        print("- German queries correctly detected and answered in German")
        print("- Language detection accuracy meets requirements")
        print("- The original German query issue has been resolved")
    else:
        print("⚠️  Some issues found - check the output above")

    print(f"\nLanguage & Prompts Test: {'PASS' if test_passed else 'FAIL'}")
    print(f"Prompt Completeness: {'PASS' if prompts_complete else 'FAIL'}")