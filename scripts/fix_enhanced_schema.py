#!/usr/bin/env python3
"""
Fix the enhanced database schema by adding missing columns and cleaning up.

This script adds the missing AI-enriched columns and fixes schema inconsistencies.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Text, TIMESTAMP, ForeignKey, func, JSON, Date
from sqlalchemy.dialects.postgresql import JSONB

def get_database_url():
    """Get database URL from environment variables."""
    return os.getenv(
        "DATABASE_URL",
        f"postgresql://christianhein:@localhost:5432/rag_system"
    )

def add_missing_columns(engine):
    """Add missing columns to existing tables."""
    print("Adding missing columns to enhanced schema...")

    with engine.connect() as conn:
        # Add missing columns to documents table
        missing_doc_cols = [
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_summary TEXT;",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS key_topics JSONB;",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS reading_time_minutes INTEGER;",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS author VARCHAR(255);",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS publication_date DATE;",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS custom_metadata JSONB;",
        ]

        # Add missing columns to document_chunks table
        missing_chunk_cols = [
            "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS primary_topic VARCHAR(100);",
            "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS topic_relevance REAL;",
            "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;",
        ]

        # Add missing columns to document_chapters table
        missing_chapter_cols = [
            "ALTER TABLE document_chapters ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;",
        ]

        # Fix processing_jobs table (rename job_metadata to metadata)
        processing_fixes = [
            "ALTER TABLE processing_jobs ADD COLUMN IF NOT EXISTS metadata JSONB;",
            "UPDATE processing_jobs SET metadata = job_metadata WHERE metadata IS NULL AND job_metadata IS NOT NULL;",
            "ALTER TABLE processing_jobs DROP COLUMN IF EXISTS job_metadata;",
        ]

        all_alterations = missing_doc_cols + missing_chunk_cols + missing_chapter_cols + processing_fixes

        for sql in all_alterations:
            try:
                conn.execute(text(sql))
                print(f"✅ Executed: {sql[:50]}...")
            except Exception as e:
                print(f"⚠️  Failed: {sql[:50]}... - {e}")

        conn.commit()
        print("✅ Missing columns added")

def add_missing_indexes(engine):
    """Add missing performance indexes."""
    print("Adding missing performance indexes...")

    with engine.connect() as conn:
        missing_indexes = [
            # Missing chunk indexes
            "CREATE INDEX IF NOT EXISTS idx_chunks_topic ON document_chunks(primary_topic);",
            "CREATE INDEX IF NOT EXISTS idx_chunks_relevance ON document_chunks(content_relevance);",
            "CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON document_chunks(created_at);",

            # Missing chapter indexes
            "CREATE INDEX IF NOT EXISTS idx_chapters_created_at ON document_chapters(created_at);",

            # Processing job indexes (with corrected column name)
            "CREATE INDEX IF NOT EXISTS idx_jobs_progress ON processing_jobs(progress);",
            "CREATE INDEX IF NOT EXISTS idx_jobs_metadata ON processing_jobs USING GIN (metadata);",
        ]

        for sql in missing_indexes:
            try:
                conn.execute(text(sql))
                print(f"✅ Created index: {sql.split('ON')[1].split('(')[0].strip()}")
            except Exception as e:
                print(f"⚠️  Index creation failed: {e}")

        conn.commit()
        print("✅ Missing indexes added")

def clean_extra_tables(engine):
    """Remove extra/unused tables."""
    print("Cleaning up extra tables...")

    with engine.connect() as conn:
        extra_tables = ["test_table"]

        for table in extra_tables:
            try:
                # Check if table exists and drop it
                result = conn.execute(text(f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table}')"))
                exists = result.fetchone()[0]

                if exists:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE;"))
                    print(f"✅ Dropped unused table: {table}")
                else:
                    print(f"ℹ️  Table {table} already removed")

            except Exception as e:
                print(f"⚠️  Failed to drop {table}: {e}")

        conn.commit()
        print("✅ Extra tables cleaned up")

def verify_schema_fix(engine):
    """Verify that the schema fixes were successful."""
    print("Verifying schema fixes...")

    from sqlalchemy import inspect
    inspector = inspect(engine)

    # Check key tables have expected columns
    critical_checks = {
        'documents': ['document_summary', 'key_topics', 'reading_time_minutes', 'author', 'publication_date', 'custom_metadata'],
        'document_chunks': ['primary_topic', 'topic_relevance', 'created_at'],
        'document_chapters': ['created_at'],
        'processing_jobs': ['metadata']
    }

    all_good = True

    for table, expected_cols in critical_checks.items():
        if table not in inspector.get_table_names():
            print(f"❌ Table missing: {table}")
            all_good = False
            continue

        actual_cols = [col['name'] for col in inspector.get_columns(table)]
        missing = set(expected_cols) - set(actual_cols)

        if missing:
            print(f"❌ {table}: Still missing {list(missing)}")
            all_good = False
        else:
            print(f"✅ {table}: All expected columns present")

    # Check extra tables are gone
    tables = inspector.get_table_names()
    if 'test_table' in tables:
        print("❌ test_table still exists")
        all_good = False
    else:
        print("✅ Extra tables removed")

    return all_good

def main():
    """Main schema fix function."""
    print("🔧 Enhanced Schema Fix Script")
    print("=" * 40)

    try:
        # Get database connection
        database_url = get_database_url()
        print(f"Connecting to database...")

        engine = create_engine(database_url)

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection successful")

        # Apply fixes
        add_missing_columns(engine)
        add_missing_indexes(engine)
        clean_extra_tables(engine)

        # Verify
        if verify_schema_fix(engine):
            print("\n" + "=" * 40)
            print("🎉 Schema fix completed successfully!")
            print("✅ All missing columns added")
            print("✅ All indexes created")
            print("✅ Extra tables removed")
            print("✅ Database schema now fully aligned with enhanced AI system")
        else:
            print("\n⚠️  Some issues remain - please check the output above")

    except Exception as e:
        print(f"❌ Schema fix failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()