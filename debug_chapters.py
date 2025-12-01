#!/usr/bin/env python3
"""
Debug script to check what chapter data is stored in the database
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from database.models import SessionLocal, Document, DocumentChapter


def check_database_chapters():
    """Check what chapter data is stored in the database."""
    db = SessionLocal()

    try:
        # Get all documents
        documents = db.query(Document).all()
        print(f"Found {len(documents)} documents in database")

        for doc in documents:
            print(f"\nDocument: {doc.filename} (ID: {doc.id})")
            print(f"Status: {doc.status}")
            print(f"Content length: {len(doc.full_content) if doc.full_content else 0}")

            # Get chapters for this document
            chapters = (
                db.query(DocumentChapter)
                .filter(DocumentChapter.document_id == doc.id)
                .order_by(DocumentChapter.chapter_path)
                .all()
            )
            print(f"Chapters: {len(chapters)}")

            for i, chapter in enumerate(chapters[:10]):  # Show first 10
                print(
                    f"  {i + 1}. Level {chapter.level}: '{chapter.chapter_title}' ({chapter.word_count} words)"
                )
                if chapter.content and len(chapter.content.strip()) > 0:
                    preview = chapter.content.strip()[:100].replace("\n", " ")
                    print(f"      Content: {preview}...")

            if len(chapters) > 10:
                print(f"  ... and {len(chapters) - 10} more chapters")

    except Exception as e:
        print(f"Error checking database: {e}")
        import traceback

        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    check_database_chapters()
