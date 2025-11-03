#!/usr/bin/env python3
"""
Script to populate author information for existing documents in the database.

This script processes all documents that don't have author information
and attempts to extract author data using Docling.
"""

import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.document_processor import DocumentProcessor
    from src.database.models import SessionLocal, Document, DocumentChunk
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)
from dotenv import load_dotenv

load_dotenv()


def populate_authors(limit=None, dry_run=False):
    """
    Populate author information for documents that don't have it.

    Args:
        limit: Maximum number of documents to process (None for all)
        dry_run: If True, only show what would be done without making changes
    """
    db = SessionLocal()
    processor = DocumentProcessor()
    processor.db = db  # Use the same session

    try:
        # Get all documents without author information
        query = db.query(Document).filter(
            (Document.author.is_(None)) | (Document.author == '')
        )

        if limit:
            query = query.limit(limit)

        documents_without_authors = query.all()

        if not documents_without_authors:
            print("✅ No documents found that need author information.")
            return

        print(f"📝 Found {len(documents_without_authors)} documents without author information.")
        if limit:
            print(f"🔢 Processing limit: {limit} documents")

        if dry_run:
            print("🔍 DRY RUN MODE - No changes will be made")
            for doc in documents_without_authors[:5]:  # Show first 5
                print(f"  Would process: {doc.filename}")
            return

        successful = 0
        failed = 0

        for i, doc in enumerate(documents_without_authors, 1):
            if not os.path.exists(doc.filepath):
                print(f"⚠️ [{i}/{len(documents_without_authors)}] File not found: {doc.filepath}")
                failed += 1
                continue

            try:
                print(f"🔄 [{i}/{len(documents_without_authors)}] Processing {doc.filename}...")

                # Try to extract author using Docling
                from docling.document_converter import DocumentConverter
                doc_converter = DocumentConverter()
                result = doc_converter.convert(doc.filepath)

                author = processor.extract_author_from_docling(result.document)

                if author:
                    if not dry_run:
                        doc.author = author
                    print(f"✅ Extracted author: {author}")
                    successful += 1
                else:
                    print(f"❌ No author found in {doc.filename}")
                    failed += 1

            except Exception as e:
                print(f"❌ Error processing {doc.filename}: {e}")
                failed += 1

        # Commit all changes
        if not dry_run:
            db.commit()
            print(f"\n💾 Changes committed to database")

        print(f"\n📊 Summary:")
        print(f"✅ Successfully populated authors for {successful} documents")
        print(f"❌ Failed to populate authors for {failed} documents")

    except Exception as e:
        print(f"❌ Error during author population: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Populate author information for documents")
    parser.add_argument("--limit", type=int, help="Maximum number of documents to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")

    args = parser.parse_args()

    print("🚀 Starting author population script...")
    if args.dry_run:
        print("🔍 Running in DRY RUN mode")
    if args.limit:
        print(f"🔢 Processing limit: {args.limit} documents")

    # Populate authors using Docling extraction
    populate_authors(limit=args.limit, dry_run=args.dry_run)

    print("🎉 Author population completed!")