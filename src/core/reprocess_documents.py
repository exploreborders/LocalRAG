#!/usr/bin/env python3
"""
Reprocess existing documents with enhanced chunking and chapter metadata.
"""

import sys
from pathlib import Path

# Add parent src directory to path if running as standalone script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from database.models import SessionLocal, Document
from src.core.document_manager import UploadProcessor


def reprocess_documents():
    """Reprocess existing documents with enhanced chunking."""
    print("🔄 Reprocessing existing documents with enhanced chunking...")
    print("=" * 60)

    db = SessionLocal()
    processor = UploadProcessor()

    try:
        # Get all documents
        documents = db.query(Document).all()
        print(f"Found {len(documents)} documents to reprocess")

        for i, doc in enumerate(documents, 1):
            print(
                f"\n📄 Reprocessing document {i}/{len(documents)}: {doc.filename[:50]}..."
            )

            # Check if document file exists
            if not Path(doc.filepath).exists():
                print(f"  ⚠️  File not found: {doc.filepath}")
                continue

            try:
                # Reprocess the document with forced AI enrichment
                result = processor.process_single_file(
                    doc.filepath, doc.filename, doc.file_hash, force_enrichment=True
                )

                if result["success"]:
                    print(
                        f"  ✅ Successfully reprocessed: {result.get('chunks_created', 0)} chunks"
                    )
                else:
                    print(
                        f"  ❌ Failed to reprocess: {result.get('error', 'Unknown error')}"
                    )

            except Exception as e:
                print(f"  ❌ Error reprocessing: {e}")

        print("\n" + "=" * 60)
        print("🎉 Document reprocessing completed!")

    except Exception as e:
        print(f"❌ Error during reprocessing: {e}")
        import traceback

        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    reprocess_documents()
