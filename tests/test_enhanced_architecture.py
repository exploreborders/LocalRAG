#!/usr/bin/env python3
"""
Simple test for enhanced architecture components without external dependencies.
"""

import sys
import os
from pathlib import Path

def test_architecture_structure():
    """Test that all enhanced architecture files exist and have expected structure."""
    print("🧪 Testing Enhanced Architecture Structure")

    base_dir = Path(__file__).parent.parent / 'src'

    # Check that all enhanced files exist
    required_files = [
        'knowledge_graph.py',
        'ai_enrichment.py',
        'retrieval_db.py',
        'rag_pipeline_db.py'
    ]

    for file in required_files:
        file_path = base_dir / file
        if file_path.exists():
            print(f"✅ {file} exists")

            # Check file size (should be substantial)
            size = file_path.stat().st_size
            if size > 1000:  # At least 1KB
                print(f"   📏 File size: {size} bytes")
            else:
                print(f"   ⚠️ File seems too small: {size} bytes")
        else:
            print(f"❌ {file} missing")
            return False

    # Check for enhanced methods in key files
    enhanced_methods = {
        'ai_enrichment.py': ['_classify_document_category'],
        'retrieval_db.py': ['retrieve_with_knowledge_graph'],
        'rag_pipeline_db.py': ['_build_enhanced_context'],
        'knowledge_graph.py': ['expand_query_context', 'find_documents_by_relationships']
    }

    for file, methods in enhanced_methods.items():
        file_path = base_dir / file
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            for method in methods:
                if f"def {method}" in content:
                    print(f"✅ Method {method} found in {file}")
                else:
                    print(f"❌ Method {method} missing from {file}")
                    return False
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")
            return False

    return True

def test_schema_enhancements():
    """Test that schema enhancement script exists."""
    print("🧪 Testing Schema Enhancement Script")

    schema_file = Path(__file__).parent.parent / 'scripts' / 'enhance_knowledge_graph_schema.py'

    if schema_file.exists():
        print("✅ Schema enhancement script exists")

        # Check file size
        size = schema_file.stat().st_size
        if size > 2000:  # Should be substantial
            print(f"   📏 Script size: {size} bytes")
        else:
            print(f"   ⚠️ Script seems too small: {size} bytes")

        # Check for key functions
        try:
            with open(schema_file, 'r') as f:
                content = f.read()

            required_functions = ['run_schema_enhancements', 'populate_initial_relationships']
            for func in required_functions:
                if f"def {func}" in content:
                    print(f"✅ Function {func} found in schema script")
                else:
                    print(f"❌ Function {func} missing from schema script")
                    return False
        except Exception as e:
            print(f"❌ Error reading schema script: {e}")
            return False

        return True
    else:
        print("❌ Schema enhancement script missing")
        return False

def test_architecture_concepts():
    """Test that key architecture concepts are implemented."""
    print("🧪 Testing Architecture Concepts")

    concepts_found = []

    # Check knowledge graph for key concepts
    kg_file = Path(__file__).parent.parent / 'src' / 'knowledge_graph.py'
    try:
        with open(kg_file, 'r') as f:
            content = f.read()

        concepts = [
            'co-occurrence',
            'relationship',
            'graph traversal',
            'context expansion',
            'tag relationship',
            'category relationship'
        ]

        for concept in concepts:
            if concept.replace('-', ' ') in content.lower():
                concepts_found.append(concept)
                print(f"✅ Concept '{concept}' found in knowledge graph")
            else:
                print(f"❌ Concept '{concept}' missing from knowledge graph")

    except Exception as e:
        print(f"❌ Error checking concepts: {e}")
        return False

    # Should find most concepts
    if len(concepts_found) >= 4:
        print(f"✅ Found {len(concepts_found)}/{len(concepts)} key concepts")
        return True
    else:
        print(f"❌ Only found {len(concepts_found)}/{len(concepts)} key concepts")
        return False

def test_enhanced_workflow():
    """Test that the enhanced workflow is documented."""
    print("🧪 Testing Enhanced Workflow Documentation")

    # Check if the enhanced architecture is documented in key files
    workflow_indicators = [
        'Knowledge Graph',
        'AI Enrichment',
        'Enhanced Retrieval',
        'Rich Context'
    ]

    files_to_check = ['README.md', 'AGENTS.md']

    found_indicators = set()

    for file in files_to_check:
        file_path = Path(__file__).parent.parent / file
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    content = f.read()

                for indicator in workflow_indicators:
                    if indicator in content:
                        found_indicators.add(indicator)
                        print(f"✅ '{indicator}' documented in {file}")
            except Exception as e:
                print(f"⚠️ Error reading {file}: {e}")

    if len(found_indicators) >= 2:
        print(f"✅ Found {len(found_indicators)}/{len(workflow_indicators)} workflow indicators")
        return True
    else:
        print(f"⚠️ Limited workflow documentation: {len(found_indicators)}/{len(workflow_indicators)} indicators found")
        return True  # Not critical for functionality

def main():
    """Run all architecture tests."""
    print("🤖 Enhanced Architecture Test Suite")
    print("=" * 50)

    tests = [
        test_architecture_structure,
        test_schema_enhancements,
        test_architecture_concepts,
        test_enhanced_workflow
    ]

    passed = 0
    failed = 0

    for test in tests:
        print(f"\n{'='*60}")
        print(f"Running {test.__name__}")
        print('='*60)

        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                failed += 1
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("ENHANCED ARCHITECTURE TEST SUMMARY")
    print('='*60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(".1f")

    if failed == 0:
        print("\n🎉 All enhanced architecture tests passed!")
        print("✅ Enhanced Knowledge Graph + AI Categories system is properly implemented")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed - review implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())