#!/usr/bin/env python3
"""
Migration script to add advanced document management features.
Adds tagging, categorization, and enhanced metadata support.
Run this after updating the models to support Phase 18 features.
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def migrate_advanced_document_management():
    """
    Add advanced document management tables and columns.
    """
    # Database connection
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", 5432),
        dbname=os.getenv("POSTGRES_DB", "rag_system"),
        user=os.getenv("POSTGRES_USER", "christianhein"),
        password=os.getenv("POSTGRES_PASSWORD", "")
    )

    cursor = conn.cursor()

    try:
        # Add enhanced metadata columns to documents table
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

        # Create indexes for better performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_tags_name ON document_tags(name);
            CREATE INDEX IF NOT EXISTS idx_document_categories_name ON document_categories(name);
            CREATE INDEX IF NOT EXISTS idx_document_categories_parent ON document_categories(parent_id);
            CREATE INDEX IF NOT EXISTS idx_document_tags_assoc_doc ON document_tags_association(document_id);
            CREATE INDEX IF NOT EXISTS idx_document_tags_assoc_tag ON document_tags_association(tag_id);
            CREATE INDEX IF NOT EXISTS idx_document_categories_assoc_doc ON document_categories_association(document_id);
            CREATE INDEX IF NOT EXISTS idx_document_categories_assoc_cat ON document_categories_association(category_id);
        """)

        conn.commit()
        print("✅ Successfully added advanced document management features")
        print("✅ Created document_tags and document_categories tables")
        print("✅ Added association tables and indexes")
        print("✅ Enhanced documents table with author, reading_time, and custom_fields")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    print("Starting advanced document management migration...")
    migrate_advanced_document_management()
    print("Migration completed!")