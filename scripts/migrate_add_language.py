#!/usr/bin/env python3
"""
Migration script to add detected_language column to documents table.
Run this after updating the schema to support multilingual features.
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def add_language_column():
    """
    Add detected_language column to the documents table.
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
        # Add the detected_language column if it doesn't exist
        cursor.execute("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS detected_language VARCHAR(10);
        """)

        # Update existing records to have 'unknown' as default
        cursor.execute("""
            UPDATE documents
            SET detected_language = 'unknown'
            WHERE detected_language IS NULL;
        """)

        conn.commit()
        print("✅ Successfully added detected_language column to documents table")
        print("✅ Updated existing records with 'unknown' language")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    print("Starting language column migration...")
    add_language_column()
    print("Migration completed!")