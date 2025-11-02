#!/usr/bin/env python3
"""
Script to populate metadata fields for existing documents using AI enrichment.

This script will enrich all existing documents that haven't been AI-enriched yet
with author, reading_time, and custom_fields (summary, topics, etc.).
"""

import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.ai_enrichment import AIEnrichmentService
    from src.database.models import SessionLocal, Document
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)

def populate_existing_documents():
    """
    Populate metadata for all existing documents using AI enrichment.
    """
    print("🤖 Starting AI enrichment for existing documents...")

    # Initialize AI enrichment service
    enrichment_service = AIEnrichmentService()

    # Get all documents
    db = SessionLocal()
    try:
        documents = db.query(Document).all()
        total_docs = len(documents)

        print(f"📊 Found {total_docs} documents to process")

        # Filter out already enriched documents
        unenriched_docs = []
        for doc in documents:
            if not doc.custom_fields or doc.custom_fields.get('ai_enriched') != True:
                unenriched_docs.append(doc)

        print(f"🎯 {len(unenriched_docs)} documents need enrichment")

        if not unenriched_docs:
            print("✅ All documents are already enriched!")
            return

        # Process one by one to avoid overwhelming the LLM and for better error handling
        successful = 0
        failed = 0

        for i, doc in enumerate(unenriched_docs):
            print(f"🔄 Processing document {i+1}/{len(unenriched_docs)}: {doc.filename[:50]}...")

            try:
                result = enrichment_service.enrich_document(doc.id, force=False)
                if result['success']:
                    successful += 1
                    print(f"  ✅ Enriched successfully")
                else:
                    failed += 1
                    print(f"  ❌ Failed: {result['error']}")

            except Exception as e:
                print(f"  ❌ Exception: {e}")
                failed += 1

        print("\n📈 Enrichment Summary:")
        print(f"  ✅ Successfully enriched: {successful}")
        print(f"  ❌ Failed: {failed}")
        print(f"  📊 Total processed: {successful + failed}")

        if successful > 0:
            print("\n🎉 Documents are now enriched with AI-generated metadata!")
            print("   - Automatic summaries")
            print("   - Topic extraction")
            print("   - Reading time estimates")
            print("   - Smart tagging suggestions")

    except Exception as e:
        print(f"❌ Error during enrichment: {e}")
        raise
    finally:
        db.close()

def show_sample_enriched_documents():
    """
    Show a sample of enriched documents to verify the results.
    """
    print("\n📋 Sample of Enriched Documents:")
    print("-" * 50)

    db = SessionLocal()
    try:
        # Get a few enriched documents
        enriched_docs = db.query(Document).filter(
            Document.custom_fields.op('->>')('ai_enriched') == 'true'
        ).limit(3).all()

        if not enriched_docs:
            print("No enriched documents found yet.")
            return

        for doc in enriched_docs:
            print(f"📄 {doc.filename}")
            print(f"   Author: {doc.author or 'Not set'}")
            print(f"   Reading Time: {doc.reading_time or 'Not estimated'} minutes")

            if doc.custom_fields:
                summary = doc.custom_fields.get('summary', 'No summary')
                topics = doc.custom_fields.get('topics', [])
                word_count = doc.custom_fields.get('word_count', 0)

                print(f"   Word Count: {word_count}")
                print(f"   Topics: {', '.join(topics) if topics else 'None extracted'}")
                print(f"   Summary: {summary[:100]}{'...' if len(summary) > 100 else ''}")
            print()

    except Exception as e:
        print(f"❌ Error showing samples: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    print("🚀 Document Metadata Population Script")
    print("=" * 50)

    populate_existing_documents()
    show_sample_enriched_documents()

    print("\n✨ Script completed!")
    print("💡 You can now use the advanced search filters and see enriched metadata in the web interface.")