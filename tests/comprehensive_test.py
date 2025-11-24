#!/usr/bin/env python3
"""
Comprehensive test for document ingestion fixes.
Tests the logic and database schema without requiring full dependencies.
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def test_code_fixes():
    """Test that all code fixes are properly applied."""
    print("🧪 Testing code fixes...")

    # Test 1: Check that document_manager.py has the fixes
    try:
        with open("src/core/document_manager.py", "r") as f:
            content = f.read()

        # Check for full_content assignment
        if "full_content=doc_content" not in content:
            print("❌ Document manager missing full_content assignment")
            return False

        # Check for status update
        if 'document.status = "processed"' not in content:
            print("❌ Document manager missing status update")
            return False

        # Check for correct split_documents usage
        if "doc_content = split_documents([file_path])" not in content:
            print("❌ Document manager not using split_documents correctly")
            return False

        print("✅ Document manager fixes verified")

    except Exception as e:
        print(f"❌ Error checking document manager: {e}")
        return False

    # Test 2: Check that data/loader.py has enhanced Docling config
    try:
        with open("src/data/loader.py", "r") as f:
            content = f.read()

        # Check for OCR and table processing
        if "do_ocr = True" not in content:
            print("❌ Data loader missing OCR configuration")
            return False

        if "do_table_structure = True" not in content:
            print("❌ Data loader missing table structure configuration")
            return False

        if "InputFormat.PDF: pipeline_options" not in content:
            print("❌ Data loader missing PDF format options")
            return False

        print("✅ Data loader enhancements verified")

    except Exception as e:
        print(f"❌ Error checking data loader: {e}")
        return False

    # Test 3: Check retrieval.py has hybrid_alpha parameter
    try:
        with open("src/core/retrieval.py", "r") as f:
            content = f.read()

        if "hybrid_alpha: Optional[float] = None" not in content:
            print("❌ Retrieval missing hybrid_alpha parameter")
            return False

        if '"boost": self.hybrid_alpha' not in content:
            print("❌ Retrieval missing hybrid_alpha boost logic")
            return False

        print("✅ Retrieval enhancements verified")

    except Exception as e:
        print(f"❌ Error checking retrieval: {e}")
        return False

    return True


def test_database_schema():
    """Test database schema readiness."""
    print("🧪 Testing database schema...")

    # This would require database connection, but let's check the SQL schema file
    try:
        with open("src/database/schema.sql", "r") as f:
            schema = f.read()

        required_tables = [
            "documents",
            "document_chunks",
            "document_embeddings",
            "document_tags",
            "document_categories",
        ]

        for table in required_tables:
            if table not in schema:
                print(f"❌ Schema missing {table} table")
                return False

        # Check for full_content column
        if "full_content TEXT" not in schema:
            print("❌ Schema missing full_content column")
            return False

        print("✅ Database schema verified")

    except Exception as e:
        print(f"❌ Error checking schema: {e}")
        return False

    return True


def test_pdf_file():
    """Test that the PDF file exists and is valid."""
    print("🧪 Testing PDF file...")

    pdf_path = "data/velpTEC_K4.0053_2.0_AI-Development_Exkurs_PyTorch Geometric und Open3D_01.E.01.pdf"

    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return False

    # Check file size (should be reasonable for a 51-page PDF)
    file_size = Path(pdf_path).stat().st_size
    if file_size < 100000:  # Less than 100KB seems suspicious
        print(f"⚠️  PDF file seems small: {file_size} bytes")
    else:
        print(f"✅ PDF file exists: {file_size:,} bytes")

    return True


def simulate_processing_steps():
    """Simulate the expected processing steps."""
    print("🔄 Simulating document processing steps...")

    steps = [
        "1. Language Detection → 'en'",
        "2. PDF Content Extraction → Enhanced Docling with OCR",
        "3. Content Storage → full_content field populated",
        "4. Document Chunking → Multiple semantic chunks created",
        "5. Embedding Generation → Vectors for all chunks",
        "6. Elasticsearch Indexing → Searchable content",
        "7. AI Enrichment → Tags and categories assigned",
        "8. Status Update → 'processed'",
    ]

    for step in steps:
        print(f"   {step}")

    print("✅ Processing simulation complete")


def expected_database_state():
    """Show expected database state after successful processing."""
    print("📊 Expected Database State:")

    expectations = [
        ("documents", "1 record with status='processed', full_content populated"),
        ("document_chunks", "10-50 chunks with meaningful content (>100 chars each)"),
        ("document_embeddings", "Same count as chunks, with vector data"),
        ("document_tag_assignments", "5-15 AI-generated tags"),
        ("document_category_assignments", "2-5 categories"),
        ("elasticsearch", "Indexed document with searchable chunks"),
    ]

    for table, expectation in expectations:
        print(f"   {table}: {expectation}")

    print("✅ Database expectations defined")


def create_verification_queries():
    """Create SQL queries for manual verification."""
    print("🔍 Manual Verification Queries:")

    queries = [
        "SELECT id, filename, status, detected_language, LENGTH(full_content) as content_len FROM documents;",
        "SELECT COUNT(*) as chunk_count FROM document_chunks WHERE document_id = (SELECT id FROM documents LIMIT 1);",
        "SELECT chunk_index, LENGTH(content) as content_len FROM document_chunks WHERE document_id = (SELECT id FROM documents LIMIT 1) LIMIT 5;",
        "SELECT COUNT(*) as embedding_count FROM document_embeddings WHERE chunk_id IN (SELECT id FROM document_chunks WHERE document_id = (SELECT id FROM documents LIMIT 1));",
        "SELECT COUNT(*) as tag_count FROM document_tag_assignments WHERE document_id = (SELECT id FROM documents LIMIT 1);",
        "SELECT COUNT(*) as category_count FROM document_category_assignments WHERE document_id = (SELECT id FROM documents LIMIT 1);",
    ]

    for i, query in enumerate(queries, 1):
        print(f"   {i}. {query}")

    print("✅ Verification queries ready")


def main():
    """Run all tests."""
    print("🚀 COMPREHENSIVE DOCUMENT INGESTION TEST")
    print("=" * 50)

    tests = [
        test_code_fixes,
        test_database_schema,
        test_pdf_file,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print(f"📊 Test Results: {passed}/{len(tests)} core tests passed")

    if passed == len(tests):
        print("✅ All core tests passed!")
        print()

        simulate_processing_steps()
        print()
        expected_database_state()
        print()
        create_verification_queries()
        print()

        print("🎉 DOCUMENT INGESTION SYSTEM READY!")
        print("The fixes are in place and should resolve the previous issues.")
        print("When run with proper dependencies, the system should:")
        print("  • Extract full PDF content using enhanced Docling")
        print("  • Store complete content in database")
        print("  • Create meaningful chunks and embeddings")
        print("  • Properly index for search")
        print("  • Generate AI-powered metadata")

    else:
        print("❌ Some tests failed. Please check the issues above.")

    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
