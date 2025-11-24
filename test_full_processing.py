#!/usr/bin/env python3
"""
Test full document processing with chapter detection.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from core.document_manager import DocumentProcessor
    from database.models import SessionLocal, Document, DocumentChapter

    pdf_path = "data/velpTEC_K4.0053_2.0_AI-Development_Exkurs_PyTorch Geometric und Open3D_01.E.01.pdf"

    print("🧪 Testing full document processing...")

    # Create processor
    processor = DocumentProcessor()

    # Process the document
    result = processor.process_document(pdf_path)

    if result.get("success"):
        print("✅ Document processing successful!")
        print(f"   Document ID: {result['document_id']}")
        print(f"   Chunks created: {result['chunks_count']}")
        print(f"   Language detected: {result['language']}")

        # Check if chapters were stored
        db = SessionLocal()
        try:
            doc = (
                db.query(Document).filter(Document.id == result["document_id"]).first()
            )
            if doc:
                chapters = (
                    db.query(DocumentChapter)
                    .filter(DocumentChapter.document_id == doc.id)
                    .all()
                )
                print(f"   Chapters stored: {len(chapters)}")
                if chapters:
                    print("   First 5 chapters:")
                    for i, chapter in enumerate(chapters[:5], 1):
                        print(
                            f"     {i}. {chapter.chapter_title} (path: {chapter.chapter_path})"
                        )

                print(f"   Document language in DB: {doc.detected_language}")
                print(f"   Document status: {doc.status}")

        finally:
            db.close()

    else:
        print(f"❌ Document processing failed: {result.get('error', 'Unknown error')}")

    print("🎉 Full document processing test completed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
