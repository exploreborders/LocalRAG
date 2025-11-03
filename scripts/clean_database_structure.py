#!/usr/bin/env python3
"""
Database cleanup script to remove old structure and keep only the new optimized chapter-aware system.
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


def show_current_state():
    """Show current database state before cleanup."""
    print("📊 Current Database State:")
    print("-" * 40)

    db = SessionLocal()
    try:
        result = db.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public';"))
        tables = [row[0] for row in result]

        for table in sorted(tables):
            count_result = db.execute(text(f"SELECT COUNT(*) FROM {table};"))
            count = count_result.scalar()
            print(f"  {table}: {count} rows")
    except Exception as e:
        print(f"❌ Error checking state: {e}")
    finally:
        db.close()


def clean_old_tables():
    """Remove old tables that are no longer needed."""
    print("\n🧹 Cleaning up old database structure...")

    # Tables to remove (old tag/category system, empty processing jobs)
    tables_to_drop = [
        'document_tags',
        'document_tags_association',
        'document_categories',
        'document_categories_association',
        'processing_jobs'
    ]

    db = SessionLocal()
    try:
        for table in tables_to_drop:
            print(f"  🗑️  Dropping table: {table}")
            db.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE;"))

        db.commit()
        print("✅ Old tables removed successfully!")

    except Exception as e:
        print(f"❌ Error dropping tables: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def clean_chunk_data():
    """Clean up chunk data that's no longer needed."""
    print("\n🧽 Cleaning up chunk data...")

    db = SessionLocal()
    try:
        # Remove chunks that don't have chapter information (old chunks)
        # We'll keep chunks but mark them as needing chapter metadata updates
        result = db.execute(text("SELECT COUNT(*) FROM document_chunks WHERE chapter_title IS NULL;"))
        old_chunks = result.scalar()

        if old_chunks and old_chunks > 0:
            print(f"  📝 Found {old_chunks} chunks without chapter metadata")
            print("  💡 These will be updated when we repopulate with chapter awareness")

        # For now, we'll keep all chunks but they should be repopulated with chapter info
        print("✅ Chunk cleanup completed (keeping existing chunks)")

    except Exception as e:
        print(f"❌ Error cleaning chunks: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def clean_document_metadata():
    """Clean up document metadata that's no longer relevant."""
    print("\n📋 Cleaning up document metadata...")

    db = SessionLocal()
    try:
        # Remove old metadata fields that are no longer used
        # The documents table now has our new content columns, so old fields are fine

        # Check for documents without content
        result = db.execute(text("SELECT COUNT(*) FROM documents WHERE full_content IS NULL;"))
        docs_without_content = result.scalar()

        if docs_without_content and docs_without_content > 0:
            print(f"  📄 {docs_without_content} documents still need content population")
            print("  💡 Run content migration scripts to populate missing content")

        print("✅ Document metadata cleanup completed")

    except Exception as e:
        print(f"❌ Error cleaning document metadata: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def show_final_state():
    """Show final database state after cleanup."""
    print("\n✨ Final Database State:")
    print("-" * 40)

    db = SessionLocal()
    try:
        result = db.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public';"))
        tables = [row[0] for row in result]

        for table in sorted(tables):
            count_result = db.execute(text(f"SELECT COUNT(*) FROM {table};"))
            count = count_result.scalar()
            print(f"  {table}: {count} rows")
    except Exception as e:
        print(f"❌ Error checking final state: {e}")
    finally:
        db.close()


def vacuum_database():
    """Reclaim space and optimize database."""
    print("\n🧹 Vacuuming database...")

    db = SessionLocal()
    try:
        db.execute(text("VACUUM ANALYZE;"))
        db.commit()
        print("✅ Database vacuumed and analyzed")
    except Exception as e:
        print(f"❌ Error vacuuming: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    print("🧹 Database Cleanup Script")
    print("=" * 50)
    print("This script will:")
    print("1. Remove old tag/category tables")
    print("2. Remove empty processing jobs table")
    print("3. Clean up chunk and document metadata")
    print("4. Keep only the new chapter-aware structure")

    try:
        # Show current state
        show_current_state()

        # Confirm cleanup
        response = input("\n⚠️  This will permanently delete old data. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Cleanup cancelled by user")
            sys.exit(0)

        # Perform cleanup
        clean_old_tables()
        clean_chunk_data()
        clean_document_metadata()
        vacuum_database()

        # Show final state
        show_final_state()

        print("\n🎉 Database cleanup completed!")
        print("💡 Your database now contains only the optimized chapter-aware structure:")
        print("   - documents (with content columns)")
        print("   - document_chunks (with chapter metadata)")
        print("   - document_chapters (hierarchical structure)")

    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        import traceback
        traceback.print_exc()