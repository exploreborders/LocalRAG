#!/usr/bin/env python3
"""
Migration script to add content columns to the documents table.

This script adds structured content columns to help LLMs get better context:
- full_content: Complete document content
- chapter_content: Content organized by chapters/subchapters
- toc_content: Table of contents structure
"""

import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.database.models import SessionLocal, Document, engine
    from sqlalchemy import text
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc.document import DoclingDocument
    from typing import Optional, Dict, List, Any
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def add_content_columns():
    """
    Add content columns to the documents table.
    """
    print("🔄 Adding content columns to documents table...")

    # SQL to add new columns
    alter_statements = [
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_type VARCHAR(50);",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'uploaded';",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS full_content TEXT;",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS chapter_content JSONB;",  # Store chapters as JSON
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS toc_content JSONB;",  # Store table of contents as JSON
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_structure JSONB;",  # Store document structure metadata
        # Convert existing JSON columns to JSONB if needed
        "ALTER TABLE documents ALTER COLUMN chapter_content TYPE JSONB USING chapter_content::JSONB;",
        "ALTER TABLE documents ALTER COLUMN toc_content TYPE JSONB USING toc_content::JSONB;",
        "ALTER TABLE documents ALTER COLUMN content_structure TYPE JSONB USING content_structure::JSONB;",
    ]

    db = SessionLocal()
    try:
        for statement in alter_statements:
            print(f"Executing: {statement}")
            db.execute(text(statement))

        db.commit()
        print("✅ Content columns added successfully!")

        # Create indexes for performance
        index_statements = [
            "CREATE INDEX IF NOT EXISTS idx_documents_full_content ON documents USING gin(to_tsvector('german', full_content));",
            "CREATE INDEX IF NOT EXISTS idx_documents_chapter_content ON documents USING gin(chapter_content jsonb_ops);",
        ]

        for statement in index_statements:
            print(f"Creating index: {statement}")
            db.execute(text(statement))

        db.commit()
        print("✅ Indexes created successfully!")

    except Exception as e:
        print(f"❌ Error adding columns: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def populate_content_columns(batch_size=5):
    """
    Populate content columns for existing documents using Docling extraction.
    Processes documents in batches to avoid timeouts.
    """
    print("🔄 Populating content columns for existing documents...")

    from docling.document_converter import DocumentConverter
    from docling_core.types.doc.document import DoclingDocument

    extractor = DocumentConverter()
    db = SessionLocal()

    try:
        # Get documents that don't have content yet
        documents = db.query(Document).filter(
            Document.full_content.is_(None)
        ).all()

        total_docs = len(documents)
        print(f"📊 Found {total_docs} documents to process (batch size: {batch_size})")

        successful = 0
        failed = 0

        for i, doc in enumerate(documents, 1):
            if not os.path.exists(doc.filepath):
                print(f"⚠️ [{i}/{total_docs}] File not found: {doc.filepath}")
                failed += 1
                continue

            print(f"🔄 [{i}/{total_docs}] Processing {doc.filename[:50]}...")

            # Skip non-PDF files for now (they may cause issues)
            if not doc.filename.lower().endswith('.pdf'):
                print(f"  ⏭️  Skipping non-PDF file: {doc.filename}")
                failed += 1
                continue

            # Skip problematic documents that cause timeouts
            problematic_files = [
                "velpTEC_K2.0038_Digital Automations mit KI Prozessoptimierung_Theorieskript.pdf"
            ]
            if doc.filename in problematic_files:
                print(f"  ⏭️  Skipping problematic file: {doc.filename}")
                failed += 1
                continue

            try:
                # Extract content using Docling
                result = extractor.convert(doc.filepath)
                docling_doc = result.document

                # Extract full content
                full_content = docling_doc.export_to_markdown()

                # Extract structured content by chapters
                chapter_content = extract_chapter_structure(docling_doc)

                # Extract table of contents
                toc_content = extract_table_of_contents(docling_doc)

                # Extract document structure metadata
                content_structure = extract_content_structure(docling_doc)

                # Update document
                doc.full_content = full_content
                doc.chapter_content = chapter_content
                doc.toc_content = toc_content
                doc.content_structure = content_structure

                print(f"  ✅ Extracted {len(full_content)} chars, {len(chapter_content) if chapter_content else 0} chapters")

                successful += 1

                # Commit after each successful document
                db.commit()
                print(f"  💾 Committed document")

            except Exception as e:
                print(f"  ❌ Failed: {e}")
                failed += 1

        # Final commit
        db.commit()
        print("\n💾 Final commit completed")
        print("\n📈 Content Population Summary:")
        print(f"  ✅ Successfully processed: {successful}")
        print(f"  ❌ Failed: {failed}")
        print(f"  📊 Total processed: {successful + failed}")

    except Exception as e:
        print(f"❌ Error during content population: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def extract_chapter_structure(doc: DoclingDocument) -> Optional[Dict[str, Any]]:
    """
    Extract content organized by chapters and subchapters.
    """
    chapters = {}

    try:
        # Get all text items
        if hasattr(doc, 'texts'):
            current_chapter = None
            current_subchapter = None
            chapter_content = []
            subchapter_content = []

            for item in doc.texts:
                if hasattr(item, 'label') and hasattr(item, 'text'):
                    if item.label == 'section_header':
                        # Save previous chapter if exists
                        if current_chapter and chapter_content:
                            if current_subchapter:
                                if current_chapter not in chapters:
                                    chapters[current_chapter] = {}
                                chapters[current_chapter][current_subchapter] = '\n'.join(subchapter_content)
                                subchapter_content = []
                                current_subchapter = None
                            else:
                                chapters[current_chapter] = '\n'.join(chapter_content)

                        # Start new chapter
                        current_chapter = item.text.strip()
                        chapter_content = []
                        current_subchapter = None

                    elif item.label in ['list_item', 'paragraph', 'text'] and item.text.strip():
                        # Add content to current chapter/subchapter
                        if current_subchapter:
                            subchapter_content.append(item.text.strip())
                        elif current_chapter:
                            chapter_content.append(item.text.strip())

            # Save final chapter
            if current_chapter:
                if current_subchapter and subchapter_content:
                    if current_chapter not in chapters:
                        chapters[current_chapter] = {}
                    chapters[current_chapter][current_subchapter] = '\n'.join(subchapter_content)
                elif chapter_content:
                    chapters[current_chapter] = '\n'.join(chapter_content)

    except Exception as e:
        print(f"Warning: Could not extract chapter structure: {e}")

    return chapters if chapters else None


def extract_table_of_contents(doc: DoclingDocument) -> Optional[List[Dict[str, Any]]]:
    """
    Extract table of contents from document structure.
    """
    toc = []

    try:
        if hasattr(doc, 'texts'):
            for item in doc.texts:
                if hasattr(item, 'label') and item.label == 'section_header' and hasattr(item, 'text'):
                    toc.append({
                        'title': item.text.strip(),
                        'level': 1  # Could be enhanced to detect sub-levels
                    })

    except Exception as e:
        print(f"Warning: Could not extract table of contents: {e}")

    return toc if toc else None


def extract_content_structure(doc: DoclingDocument) -> dict:
    """
    Extract document structure metadata.
    """
    structure = {
        'total_pages': doc.num_pages() if hasattr(doc, 'num_pages') and callable(getattr(doc, 'num_pages')) else None,
        'total_text_items': len(doc.texts) if hasattr(doc, 'texts') else 0,
        'total_tables': len(doc.tables) if hasattr(doc, 'tables') else 0,
        'total_images': len(doc.pictures) if hasattr(doc, 'pictures') else 0,
        'has_headers': False,
        'has_lists': False,
        'has_tables': len(doc.tables) > 0 if hasattr(doc, 'tables') else False,
        'has_images': len(doc.pictures) > 0 if hasattr(doc, 'pictures') else False,
    }

    # Analyze text items for structure
    if hasattr(doc, 'texts'):
        for item in doc.texts:
            if hasattr(item, 'label'):
                if item.label == 'section_header':
                    structure['has_headers'] = True
                elif item.label in ['list_item', 'ordered_list', 'unordered_list']:
                    structure['has_lists'] = True

    return structure


def show_content_samples():
    """
    Show samples of documents with structured content.
    """
    print("\n📋 Sample Documents with Structured Content:")
    print("-" * 60)

    db = SessionLocal()
    try:
        # Get a few documents with content
        docs_with_content = db.query(Document).filter(
            Document.full_content.isnot(None)
        ).limit(3).all()

        if not docs_with_content:
            print("No documents with structured content found yet.")
            return

        for doc in docs_with_content:
            print(f"📄 {doc.filename}")
            print(f"   Content Length: {len(doc.full_content) if doc.full_content else 0} characters")

            if doc.chapter_content:
                chapters = list(doc.chapter_content.keys()) if isinstance(doc.chapter_content, dict) else []
                print(f"   Chapters: {len(chapters)}")
                if chapters:
                    print(f"   Chapter Titles: {chapters[:3]}...")  # Show first 3

            if doc.toc_content:
                toc_items = len(doc.toc_content) if isinstance(doc.toc_content, list) else 0
                print(f"   TOC Items: {toc_items}")

            if doc.content_structure:
                structure = doc.content_structure
                print(f"   Structure: {structure.get('total_pages', 'N/A')} pages, "
                      f"{structure.get('total_tables', 0)} tables, "
                      f"{structure.get('total_images', 0)} images")

            print()

    except Exception as e:
        print(f"❌ Error showing samples: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Content Columns Migration Script")
    parser.add_argument("--max-docs", type=int, default=5,
                       help="Maximum number of documents to process (default: 5)")
    parser.add_argument("--skip-add-columns", action="store_true",
                       help="Skip adding columns (if already added)")
    parser.add_argument("--skip-populate", action="store_true",
                       help="Skip content population")
    parser.add_argument("--skip-samples", action="store_true",
                       help="Skip showing samples")

    args = parser.parse_args()

    print("🚀 Content Columns Migration Script")
    print("=" * 50)
    print(f"Processing up to {args.max_docs} documents per run")

    try:
        # Step 1: Add columns (unless skipped)
        if not args.skip_add_columns:
            add_content_columns()
        else:
            print("⏭️  Skipping column addition")

        # Step 2: Populate content (unless skipped)
        if not args.skip_populate:
            populate_content_columns(batch_size=args.max_docs)
        else:
            print("⏭️  Skipping content population")

        # Step 3: Show samples (unless skipped)
        if not args.skip_samples:
            show_content_samples()
        else:
            print("⏭️  Skipping sample display")

        print("\n✨ Migration step completed!")
        print("💡 Run again with --skip-add-columns to continue processing more documents")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()