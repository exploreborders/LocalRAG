#!/usr/bin/env python3
"""
Reprocess existing documents with enhanced chunking and chapter metadata.
"""

import sys
from pathlib import Path

# Add parent src directory to path if running as standalone script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.processing.document_processor import DocumentProcessor
from src.core.processing.upload_processor import UploadProcessor
from src.database.models import (
    Document,
    DocumentChapter,
    DocumentChunk,
    DocumentEmbedding,
    SessionLocal,
)
from src.core.embeddings import create_embeddings


def reprocess_documents():
    """Reprocess existing documents with enhanced chunking."""
    print("🔄 Reprocessing existing documents with enhanced chunking...")
    print("=" * 60)

    db = None
    try:
        db = SessionLocal()
        processor = UploadProcessor(embedding_model="embeddinggemma:latest")

        # Get all documents
        documents = db.query(Document).all()
        print(f"Found {len(documents)} documents to reprocess")

        for i, doc in enumerate(documents, 1):
            print(
                f"\n📄 Reprocessing document {i}/{len(documents)}: {doc.filename[:50]}..."
            )

            try:
                # Check if we have stored content to reprocess with
                if not doc.full_content:
                    print(f"  ⚠️  No stored content for document: {doc.filename}")
                    continue

                # Simple reprocessing: just re-chunk the existing content
                doc_processor = DocumentProcessor(db)

                # Detect chapters from the stored content
                all_chapters = doc_processor._detect_all_chapters(doc.full_content)
                print(f"    Detected {len(all_chapters)} chapters")

                # Process the document with stored content to get chunks
                processing_result = doc_processor._process_document_standard(
                    doc.filepath, doc.filename, content=doc.full_content
                )

                if not processing_result.get("success"):
                    print(
                        f"  ❌ Failed to process content: {processing_result.get('error', 'Unknown error')}"
                    )
                    continue

                chunks = processing_result.get("chunks", [])
                print(f"    Created {len(chunks)} chunks")

                # Create processing result for reprocessing
                processing_result = {
                    "extracted_content": doc.full_content,
                    "chapters": all_chapters,
                    "chunks": chunks,
                    "chapters_detected": len(all_chapters),
                }

                # Reprocess with the new chunking
                result = processor.reprocess_existing_document(
                    doc, processing_result, doc.filepath
                )

                if result["success"]:
                    print(
                        f"  ✅ Successfully reprocessed: {result.get('chunks_created', 0)} chunks, {len(all_chapters)} chapters"
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
        if db is not None:
            db.close()


if __name__ == "__main__":
    reprocess_documents()
