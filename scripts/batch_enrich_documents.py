#!/usr/bin/env python3
"""
Batch AI enrichment script for existing documents.
Processes all documents in the database with AI enrichment.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from database.models import SessionLocal, Document
from ai.enrichment import AIEnrichmentService

def main():
    print("🤖 Batch AI Enrichment for Existing Documents")
    print("=" * 50)

    # Initialize services
    db = SessionLocal()
    enrichment_service = AIEnrichmentService()

    try:
        # Get all documents
        documents = db.query(Document).all()
        print(f"Found {len(documents)} documents to process")

        successful = 0
        failed = 0

        for i, doc in enumerate(documents, 1):
            print(f"\n📄 Processing document {i}/{len(documents)}: {doc.filename[:50]}...")

            # Check if already enriched
            if doc.document_summary and doc.key_topics:
                print("  ⏭️  Already enriched, skipping")
                successful += 1
                continue

            # Enrich document
            result = enrichment_service.enrich_document(doc.id, force=False)

            if result['success']:
                print("  ✅ Successfully enriched")
                successful += 1
            else:
                print(f"  ❌ Failed: {result['error']}")
                failed += 1

        print("\n" + "=" * 50)
        print("🎉 Batch enrichment completed!")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Total processed: {len(documents)}")

    except Exception as e:
        print(f"❌ Error during batch enrichment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    main()