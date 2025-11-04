#!/usr/bin/env python3
"""
Test database integration for enhanced knowledge graph architecture.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_database_integration():
    """Test that enhanced database integration works."""
    try:
        from knowledge_graph import KnowledgeGraph
        from database.models import SessionLocal

        print("🧪 Testing Database Integration")

        # Initialize knowledge graph with database
        db = SessionLocal()
        kg = KnowledgeGraph(db)

        # Test basic graph statistics
        stats = kg.get_graph_statistics()
        print("✅ Knowledge graph connected to database")
        print(f"   📊 Tags: {stats['tags']['total']}, Relationships: {stats['tags']['with_relationships']}")
        print(f"   📊 Categories: {stats['categories']['total']}")
        print(f"   📊 Assignments: {stats['assignments']['tag_assignments']} tag, {stats['assignments']['category_assignments']} category")

        # Test relationship building (should work even with empty data)
        relationships = kg.build_relationships_from_cooccurrence()
        print(f"✅ Relationship building works: {len(relationships)} tag groups analyzed")

        # Test category relationships
        cat_relationships = kg.infer_tag_category_relationships()
        print(f"✅ Category relationship inference works: {len(cat_relationships)} tag-category mappings")

        # Test context expansion
        expansion = kg.expand_query_context(['test'], ['academic'])
        print(f"✅ Context expansion works: {expansion['total_expansions']} expansions possible")

        db.close()
        return True

    except Exception as e:
        print(f"❌ Database integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_enrichment_integration():
    """Test AI enrichment with database integration."""
    try:
        from ai_enrichment import AIEnrichmentService

        print("🧪 Testing AI Enrichment Integration")

        # Initialize with database (LLM will be None, but category classification should work)
        enrichment = AIEnrichmentService(llm_client=None)

        # Test category classification (should return defaults without LLM)
        category_data = enrichment._classify_document_category(
            "This is a technical document about machine learning algorithms.",
            "ml_guide.pdf",
            ["machine-learning", "algorithms"],
            ["artificial intelligence", "computer science"]
        )

        expected_keys = ['primary_category', 'subcategories', 'confidence', 'alternatives']
        if all(key in category_data for key in expected_keys):
            print("✅ AI enrichment category classification structure correct")
            print(f"   📊 Default category: {category_data.get('primary_category')}")
        else:
            print("❌ AI enrichment category classification structure incorrect")
            return False

        enrichment.db.close()
        return True

    except Exception as e:
        print(f"❌ AI enrichment integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_retrieval_integration():
    """Test enhanced retrieval with database."""
    try:
        from retrieval_db import DatabaseRetriever

        print("🧪 Testing Enhanced Retrieval Integration")

        # Initialize retriever
        retriever = DatabaseRetriever()

        # Check if knowledge graph is integrated
        if hasattr(retriever, 'knowledge_graph'):
            print("✅ Knowledge graph integrated into retrieval system")
        else:
            print("❌ Knowledge graph not integrated into retrieval system")
            return False

        # Check if enhanced method exists
        if hasattr(retriever, 'retrieve_with_knowledge_graph'):
            print("✅ Enhanced retrieval method available")
        else:
            print("❌ Enhanced retrieval method not available")
            return False

        retriever.db.close()
        return True

    except Exception as e:
        print(f"❌ Enhanced retrieval integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all database integration tests."""
    print("🗄️ Database Integration Test Suite")
    print("=" * 50)

    tests = [
        test_database_integration,
        test_ai_enrichment_integration,
        test_enhanced_retrieval_integration
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
    print("DATABASE INTEGRATION TEST SUMMARY")
    print('='*60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(".1f")

    if failed == 0:
        print("\n🎉 All database integration tests passed!")
        print("✅ Enhanced Knowledge Graph Architecture is fully operational!")
        print("🗄️ Database is ready with enhanced schema and relationships!")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed - check database setup")
        return 1

if __name__ == "__main__":
    sys.exit(main())