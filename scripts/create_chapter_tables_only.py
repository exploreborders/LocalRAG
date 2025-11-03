#!/usr/bin/env python3
"""
Simple script to create chapter tables without data population.
"""

import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.database.models import SessionLocal
    from sqlalchemy import text
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def add_chapter_columns():
    """Add chapter-aware columns to document_chunks table."""
    print("🔄 Adding chapter columns to document_chunks table...")

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

        # Create indexes
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
    """Create the document_chapters table."""
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

        # Create indexes
        index_statements = [
            "CREATE INDEX IF NOT EXISTS idx_chapters_document_id ON document_chapters(document_id);",
            "CREATE INDEX IF NOT EXISTS idx_chapters_chapter_path ON document_chapters(chapter_path);",
            "CREATE INDEX IF NOT EXISTS idx_chapters_section_type ON document_chapters(section_type);",
            "CREATE INDEX IF NOT EXISTS idx_chapters_parent_id ON document_chapters(parent_chapter_id);",
            "CREATE INDEX IF NOT EXISTS idx_chapters_level ON document_chapters(level);",
            "CREATE INDEX IF NOT EXISTS idx_chapters_embedding ON document_chapters USING gin(embedding);",
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


if __name__ == "__main__":
    print("🚀 Chapter Tables Creation Script")
    print("=" * 50)
    print("This script will create the database structure for chapter-aware retrieval.")

    try:
        add_chapter_columns()
        create_chapter_table()

        print("\n✨ Tables created successfully!")
        print("💡 Run populate_chapter_data.py separately to populate chapter data")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()