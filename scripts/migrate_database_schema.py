#!/usr/bin/env python3
"""
Comprehensive database schema migration script.

This script applies all necessary database schema changes for the Local RAG system,
including advanced document management, caption-aware chunking, and performance optimizations.
Run this after setting up the database connection.
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def migrate_database_schema():
    """
    Apply all necessary database schema changes.
    """
    # Database connection
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", 5432),
        dbname=os.getenv("POSTGRES_DB", "rag_system"),
        user=os.getenv("POSTGRES_USER", "christianhein"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )

    cursor = conn.cursor()

    try:
        print("🔄 Applying database schema migrations...")

        # Add language support
        print("  📝 Adding language support...")
        cursor.execute("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS detected_language VARCHAR(10);
        """)

        # Add advanced document management features
        print("  🏷️ Adding advanced document management...")
        cursor.execute("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS author VARCHAR(255),
            ADD COLUMN IF NOT EXISTS reading_time INTEGER,
            ADD COLUMN IF NOT EXISTS custom_fields JSONB;
        """)

        # Create document_tags table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_tags (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE,
                color VARCHAR(7),
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create document_categories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_categories (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                description TEXT,
                parent_id INTEGER REFERENCES document_categories(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create association tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_tags_association (
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                tag_id INTEGER REFERENCES document_tags(id) ON DELETE CASCADE,
                PRIMARY KEY (document_id, tag_id)
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_categories_association (
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                category_id INTEGER REFERENCES document_categories(id) ON DELETE CASCADE,
                PRIMARY KEY (document_id, category_id)
            );
        """)

        # Add caption-aware chunking columns
        print("  📦 Adding caption-aware chunking support...")
        cursor.execute("""
            ALTER TABLE document_chunks
            ADD COLUMN IF NOT EXISTS chunk_type VARCHAR(50),
            ADD COLUMN IF NOT EXISTS has_captions BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS caption_text TEXT,
            ADD COLUMN IF NOT EXISTS caption_line INTEGER,
            ADD COLUMN IF NOT EXISTS context_lines VARCHAR(50);
        """)

        # Add performance indexes
        print("  ⚡ Adding performance indexes...")
        cursor.execute("""
            -- Core indexes
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id_index ON document_chunks(document_id, chunk_index);
            CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON document_chunks(created_at);
            CREATE INDEX IF NOT EXISTS idx_documents_modified ON documents(last_modified);
            CREATE INDEX IF NOT EXISTS idx_documents_language ON documents(detected_language);
            CREATE INDEX IF NOT EXISTS idx_documents_processed ON documents(status) WHERE status = 'processed';

            -- Caption-aware indexes
            CREATE INDEX IF NOT EXISTS idx_document_chunks_chunk_type ON document_chunks(chunk_type);
            CREATE INDEX IF NOT EXISTS idx_document_chunks_has_captions ON document_chunks(has_captions);
            CREATE INDEX IF NOT EXISTS idx_document_chunks_caption_line ON document_chunks(caption_line);

            -- Tagging and categorization indexes
            CREATE INDEX IF NOT EXISTS idx_document_tags_name ON document_tags(name);
            CREATE INDEX IF NOT EXISTS idx_document_categories_name ON document_categories(name);
            CREATE INDEX IF NOT EXISTS idx_document_categories_parent ON document_categories(parent_category_id);
            CREATE INDEX IF NOT EXISTS idx_document_tags_assoc_doc ON document_tags_association(document_id);
            CREATE INDEX IF NOT EXISTS idx_document_tags_assoc_tag ON document_tags_association(tag_id);
            CREATE INDEX IF NOT EXISTS idx_document_categories_assoc_doc ON document_categories_association(document_id);
            CREATE INDEX IF NOT EXISTS idx_document_categories_assoc_cat ON document_categories_association(category_id);
        """)

        conn.commit()
        print("✅ Database schema migration completed successfully!")
        print(
            "✅ Added language support, advanced document management, caption-aware chunking, and performance indexes"
        )

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    print("🚀 Starting comprehensive database schema migration...")
    migrate_database_schema()
    print("🎉 Database schema migration completed!")
