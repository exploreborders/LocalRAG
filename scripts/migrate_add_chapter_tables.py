#!/usr/bin/env python3
"""
Migration script to add chapter-aware tables for enhanced LLM context retrieval.

This script adds:
- Chapter metadata columns to document_chunks table
- New document_chapters table for hierarchical retrieval
- Indexes for performance
"""

import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.database.models import SessionLocal, engine
    from sqlalchemy import text
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def add_chapter_columns():
    """
    Add chapter-aware columns to document_chunks table.
    """
    print("🔄 Adding chapter columns to document_chunks table...")

    # SQL to add new columns to document_chunks
    alter_statements = [
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS chapter_title VARCHAR(255);",
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS chapter_path VARCHAR(500);",
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS section_type VARCHAR(50);",
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS content_relevance FLOAT;",
    ]

    db = SessionLocal()
    try:
        for statement in alter_statements:
            print(f"Executing: {statement}")
            db.execute(text(statement))

        db.commit()
        print("✅ Chapter columns added successfully!")

        # Create indexes for performance
        index_statements = [
            "CREATE INDEX IF NOT EXISTS idx_chunks_chapter_title ON document_chunks(chapter_title);",
            "CREATE INDEX IF NOT EXISTS idx_chunks_chapter_path ON document_chunks(chapter_path);",
            "CREATE INDEX IF NOT EXISTS idx_chunks_section_type ON document_chunks(section_type);",
            "CREATE INDEX IF NOT EXISTS idx_chunks_content_relevance ON document_chunks(content_relevance);",
        ]

        for statement in index_statements:
            print(f"Creating index: {statement}")
            db.execute(text(statement))

        db.commit()
        print("✅ Chunk indexes created successfully!")

    except Exception as e:
        print(f"❌ Error adding chapter columns: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def create_chapter_table():
    """
    Create the document_chapters table for hierarchical retrieval.
    """
    print("🔄 Creating document_chapters table...")

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS document_chapters (
        id SERIAL PRIMARY KEY,
        document_id INTEGER NOT NULL REFERENCES documents(id),
        chapter_title VARCHAR(255) NOT NULL,
        chapter_path VARCHAR(100) NOT NULL,
        content TEXT NOT NULL,
        embedding JSONB,
        embedding_model VARCHAR(100),
        word_count INTEGER DEFAULT 0,
        section_type VARCHAR(50) DEFAULT 'chapter',
        parent_chapter_id INTEGER REFERENCES document_chapters(id),
        level INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    db = SessionLocal()
    try:
        print("Creating document_chapters table...")
        db.execute(text(create_table_sql))

        # Convert JSON to JSONB if needed
        alter_json_sql = "ALTER TABLE document_chapters ALTER COLUMN embedding TYPE JSONB USING embedding::JSONB;"
        print("Converting embedding column to JSONB...")
        db.execute(text(alter_json_sql))

        # Create indexes
        index_statements = [
            "CREATE INDEX IF NOT EXISTS idx_chapters_document_id ON document_chapters(document_id);",
            "CREATE INDEX IF NOT EXISTS idx_chapters_chapter_path ON document_chapters(chapter_path);",
            "CREATE INDEX IF NOT EXISTS idx_chapters_section_type ON document_chapters(section_type);",
            "CREATE INDEX IF NOT EXISTS idx_chapters_parent_id ON document_chapters(parent_chapter_id);",
            "CREATE INDEX IF NOT EXISTS idx_chapters_level ON document_chapters(level);",
            # GIN index for embedding similarity search
            "CREATE INDEX IF NOT EXISTS idx_chapters_embedding ON document_chapters USING gin(embedding jsonb_ops);",
        ]

        for statement in index_statements:
            print(f"Creating index: {statement}")
            db.execute(text(statement))

        db.commit()
        print("✅ Document chapters table created successfully!")

    except Exception as e:
        print(f"❌ Error creating chapters table: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def populate_chapter_data(batch_size=5):
    """
    Populate chapter data from existing document content.
    """
    print("🔄 Populating chapter data from existing documents...")

    from src.database.models import Document, DocumentChapter
    from sentence_transformers import SentenceTransformer
    import json

    # Initialize embedding model
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    db = SessionLocal()

    try:
        # Get documents that have chapter_content but no chapters yet
        documents = db.query(Document).filter(
            Document.chapter_content.isnot(None),
            ~Document.id.in_(
                db.query(DocumentChapter.document_id.distinct())
            )
        ).all()

        total_docs = len(documents)
        print(f"📊 Found {total_docs} documents to process for chapters")

        successful = 0

        for i, doc in enumerate(documents[:batch_size], 1):  # Limit to batch_size
            print(f"🔄 [{i}/{min(batch_size, total_docs)}] Processing chapters for {doc.filename[:50]}...")

            try:
                if not doc.chapter_content:
                    continue

                # Process each chapter
                chapters_created = 0
                for chapter_path, chapter_data in doc.chapter_content.items():
                        # Truncate chapter_path if too long
                        truncated_path = chapter_path[:95] if len(chapter_path) > 95 else chapter_path

                        if isinstance(chapter_data, dict):
                            # Handle nested chapters (subchapters)
                            for subchapter_title, subchapter_content in chapter_data.items():
                                full_path = f"{truncated_path}.{subchapter_title}"
                                full_path = full_path[:95] if len(full_path) > 95 else full_path

                                chapter = DocumentChapter(
                                    document_id=doc.id,
                                    chapter_title=subchapter_title[:250],  # Truncate title too
                                    chapter_path=full_path,
                                    content=subchapter_content if isinstance(subchapter_content, str) else str(subchapter_content),
                                    word_count=len((subchapter_content if isinstance(subchapter_content, str) else str(subchapter_content)).split()),
                                    section_type='subsection',
                                    level=2
                                )
                                db.add(chapter)
                                chapters_created += 1
                        else:
                            # Simple chapter
                            chapter = DocumentChapter(
                                document_id=doc.id,
                                chapter_title=chapter_path[:250],  # Truncate title
                                chapter_path=truncated_path,
                                content=chapter_data if isinstance(chapter_data, str) else str(chapter_data),
                                word_count=len((chapter_data if isinstance(chapter_data, str) else str(chapter_data)).split()),
                                section_type='chapter',
                                level=1
                            )
                            db.add(chapter)
                            chapters_created += 1

                # Generate embeddings for chapters (batch process)
                if chapters_created > 0:
                    db.flush()  # Get IDs for new chapters

                    # Get chapters for this document
                    doc_chapters = db.query(DocumentChapter).filter(
                        DocumentChapter.document_id == doc.id
                    ).all()

                    # Generate embeddings
                    contents = [ch.content for ch in doc_chapters]
                    embeddings = model.encode(contents)
                    embeddings = [emb.tolist() for emb in embeddings]  # Convert to list

                    # Update chapters with embeddings
                    for ch, emb in zip(doc_chapters, embeddings):
                        ch.embedding = emb
                        ch.embedding_model = "nomic-ai/nomic-embed-text-v1.5"

                db.commit()
                successful += 1
                print(f"  ✅ Created {chapters_created} chapters with embeddings")

            except Exception as e:
                print(f"  ❌ Failed to process chapters: {e}")
                db.rollback()
                continue

        print(f"\n📈 Chapter Population Summary:")
        print(f"  ✅ Successfully processed: {successful}")
        print(f"  📊 Total documents processed: {successful}")

    except Exception as e:
        print(f"❌ Error during chapter population: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chapter Tables Migration Script")
    parser.add_argument("--populate-only", action="store_true",
                       help="Only populate chapter data (skip table creation)")
    parser.add_argument("--max-docs", type=int, default=5,
                       help="Maximum number of documents to process when populating")

    args = parser.parse_args()

    print("🚀 Chapter Tables Migration Script")
    print("=" * 50)

    try:
        if args.populate_only:
            print("Populating chapter data only...")
            populate_chapter_data(batch_size=args.max_docs)
        else:
            print("This script will:")
            print("1. Add chapter columns to document_chunks table")
            print("2. Create document_chapters table")
            print("3. Populate chapter data from existing documents")

            # Step 1: Add chapter columns
            add_chapter_columns()

            # Step 2: Create chapters table
            create_chapter_table()

            # Step 3: Populate chapter data (with limit)
            populate_chapter_data(batch_size=args.max_docs)

        print("\n✨ Migration completed!")
        print("💡 Documents now have chapter-aware structure for better LLM context!")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()