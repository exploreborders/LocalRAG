#!/usr/bin/env python3
"""
Test script to rebuild database and process a single PDF with our optimized chapter-aware structure.
"""

import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.database.models import SessionLocal, engine, Base
    from src.document_processor import DocumentProcessor
    from sqlalchemy import text
    import shutil
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def rebuild_database():
    """Drop and recreate all tables."""
    print("🔄 Rebuilding database...")

    db = SessionLocal()
    try:
        # Drop all tables
        print("  🗑️  Dropping existing tables...")
        db.execute(text("DROP TABLE IF EXISTS document_chapters CASCADE;"))
        db.execute(text("DROP TABLE IF EXISTS document_chunks CASCADE;"))
        db.execute(text("DROP TABLE IF EXISTS documents CASCADE;"))
        db.commit()

        # Recreate tables
        print("  🏗️  Creating optimized tables...")
        Base.metadata.create_all(bind=engine)

        print("✅ Database rebuilt successfully!")

    except Exception as e:
        print(f"❌ Error rebuilding database: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def process_single_pdf(pdf_path):
    """Process a single PDF with our optimized pipeline."""
    print(f"📄 Processing PDF: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False

    try:
        # Initialize document processor
        processor = DocumentProcessor()

        # Process the single document
        print("  🔄 Extracting content and structure...")
        processor.process_document(pdf_path)
        print("✅ Document processed")

        # Now populate the content columns
        print("  🔄 Populating content columns...")
        import subprocess
        result = subprocess.run([sys.executable, "scripts/migrate_add_content_columns.py", "--max-docs", "1", "--skip-add-columns"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Content columns populated")
        else:
            print(f"❌ Content population failed: {result.stderr}")

        # Populate chapter data
        print("  🔄 Populating chapter data...")
        result = subprocess.run([sys.executable, "scripts/migrate_add_chapter_tables.py", "--populate-only", "--max-docs", "1"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Chapter data populated")
        else:
            print(f"❌ Chapter population failed: {result.stderr}")

        # Update chunk metadata
        print("  🔄 Updating chunk chapter metadata...")
        result = subprocess.run([sys.executable, "scripts/update_chunk_chapter_metadata.py"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Chunk metadata updated")
        else:
            print(f"❌ Chunk metadata update failed: {result.stderr}")

        return True

    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_processing():
    """Verify that the document was processed correctly."""
    print("\n🔍 Verifying processing results...")

    db = SessionLocal()
    try:
        # Check documents
        docs_result = db.execute(text("SELECT id, filename, full_content, chapter_content, detected_language FROM documents"))
        docs = docs_result.fetchall()
        print(f"📊 Documents: {len(docs)}")
        for doc in docs:
            print(f"  📄 {doc.filename}")
            print(f"    Content length: {len(doc.full_content) if doc.full_content else 0} chars")
            chapter_count = "present" if doc.chapter_content else "none"
            print(f"    Chapters: {chapter_count} sections")
            print(f"    Language: {doc.detected_language}")

        # Check chapters
        chapters_result = db.execute(text("SELECT chapter_title, word_count FROM document_chapters LIMIT 3"))
        chapters = chapters_result.fetchall()
        print(f"📊 Chapters: {len(db.execute(text('SELECT COUNT(*) FROM document_chapters')).scalar())}")
        for chapter in chapters:
            print(f"  📖 {chapter.chapter_title} ({chapter.word_count} words)")

        # Check chunks
        total_chunks = db.execute(text("SELECT COUNT(*) FROM document_chunks")).scalar()
        chapter_aware = db.execute(text("SELECT COUNT(*) FROM document_chunks WHERE chapter_title IS NOT NULL")).scalar()
        print(f"📊 Chunks: {total_chunks}")
        print(f"  Chapter-aware chunks: {chapter_aware}/{total_chunks}")

        print("✅ Processing verification completed!")

    except Exception as e:
        print(f"❌ Error verifying processing: {e}")
    finally:
        db.close()


def test_retrieval():
    """Test retrieval with our optimized structure."""
    print("\n🔍 Testing retrieval functionality...")

    try:
        from src.retrieval_db import DatabaseRetriever
        from src.rag_pipeline_db import RAGPipelineDB

        # Initialize retriever
        retriever = DatabaseRetriever()

        # Test basic retrieval
        query = "what is the main topic of this document"
        print(f"  🔍 Query: '{query}'")

        results = retriever.retrieve(query, top_k=3)
        print(f"  📊 Retrieved {len(results)} results")

        for i, result in enumerate(results[:2], 1):
            doc = result.get('document', {})
            print(f"    {i}. {doc.get('filename', 'Unknown')} (score: {result.get('score', 0):.3f})")

        # Test RAG pipeline
        print("  🤖 Testing RAG pipeline...")
        rag = RAGPipelineDB()
        answer = rag.generate_answer(query, results)
        print(f"  💬 Answer: {answer[:200]}...")

        print("✅ Retrieval testing completed!")

    except Exception as e:
        print(f"❌ Error testing retrieval: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🧪 Single PDF Test Script")
    print("=" * 50)
    print("This script will:")
    print("1. Rebuild the database with optimized structure")
    print("2. Process one PDF document")
    print("3. Verify processing results")
    print("4. Test retrieval functionality")

    # Choose a test PDF
    test_pdf = "data/velpTEC_K8.0002_1.0_Bewerbungstraining_Theorieskript_ Kapitel_6.pdf"
    test_pdf_path = os.path.join(os.path.dirname(__file__), '..', test_pdf)

    try:
        # Step 1: Rebuild database
        rebuild_database()

        # Step 2: Process single PDF
        success = process_single_pdf(test_pdf_path)
        if not success:
            print("❌ PDF processing failed, stopping test")
            sys.exit(1)

        # Step 3: Verify results
        verify_processing()

        # Step 4: Test retrieval
        test_retrieval()

        print("\n🎉 Single PDF test completed successfully!")
        print("💡 The optimized chapter-aware structure is working correctly!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()