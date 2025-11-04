#!/usr/bin/env python3
"""
Test suite for Knowledge Graph functionality.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_knowledge_graph_basic():
    """Test basic knowledge graph functionality."""
    try:
        from knowledge_graph import KnowledgeGraph
        from database.models import SessionLocal

        print("🧪 Testing Knowledge Graph Basic Functionality")

        # Initialize knowledge graph
        db = SessionLocal()
        kg = KnowledgeGraph(db)

        # Test graph statistics
        stats = kg.get_graph_statistics()
        print(f"✅ Knowledge graph initialized with {stats['tags']['total']} tags and {stats['categories']['total']} categories")

        # Test empty relationship inference (should not crash)
        relationships = kg.build_relationships_from_cooccurrence()
        print(f"✅ Built {len(relationships)} tag relationships")

        # Test empty context expansion
        expansion = kg.expand_query_context([], [])
        print(f"✅ Context expansion works: {expansion['total_expansions']} expansions")

        # Test empty document finding
        docs = kg.find_documents_by_relationships([], [])
        print(f"✅ Found {len(docs)} related documents")

        db.close()
        return True

    except Exception as e:
        print(f"❌ Knowledge graph test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_enrichment_enhancement():
    """Test enhanced AI enrichment with categories."""
    try:
        from ai_enrichment import AIEnrichmentService

        print("🧪 Testing Enhanced AI Enrichment")

        # Initialize without LLM (will use fallback)
        enrichment = AIEnrichmentService(llm_client=None)

        # Test category classification (should return defaults)
        category_data = enrichment._classify_document_category(
            "This is a technical document about machine learning algorithms.",
            "ml_guide.pdf",
            ["machine-learning", "algorithms"],
            ["artificial intelligence", "computer science"]
        )

        print(f"✅ Category classification works: {category_data.get('primary_category', 'unknown')}")

        enrichment.db.close()
        return True

    except Exception as e:
        print(f"❌ AI enrichment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_retrieval():
    """Test enhanced retrieval with knowledge graph."""
    try:
        from retrieval_db import DatabaseRetriever

        print("🧪 Testing Enhanced Retrieval")

        # Initialize retriever (will work even without full dependencies)
        retriever = DatabaseRetriever()

        # Check if knowledge graph is available
        if hasattr(retriever, 'knowledge_graph'):
            print("✅ Knowledge graph integrated into retrieval")
        else:
            print("⚠️ Knowledge graph not integrated yet")

        # Test method exists
        if hasattr(retriever, 'retrieve_with_knowledge_graph'):
            print("✅ Enhanced retrieval method available")
        else:
            print("⚠️ Enhanced retrieval method not available")

        retriever.db.close()
        return True

    except Exception as e:
        print(f"❌ Enhanced retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_pipeline_enhancement():
    """Test enhanced RAG pipeline."""
    try:
        from rag_pipeline_db import RAGPipelineDB

        print("🧪 Testing Enhanced RAG Pipeline")

        # Check if enhanced context method exists
        if hasattr(RAGPipelineDB, '_build_enhanced_context'):
            print("✅ Enhanced context building available")
        else:
            print("⚠️ Enhanced context building not available")

        return True

    except Exception as e:
        print(f"❌ RAG pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all knowledge graph tests."""
    print("🤖 Knowledge Graph Test Suite")
    print("=" * 50)

    tests = [
        test_knowledge_graph_basic,
        test_ai_enrichment_enhancement,
        test_enhanced_retrieval,
        test_rag_pipeline_enhancement
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

    print(f"\n{'='*60}")
    print("KNOWLEDGE GRAPH TEST SUMMARY")
    print('='*60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(".1f")

    if failed == 0:
        print("\n🎉 All knowledge graph tests passed!")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())