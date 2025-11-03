#!/usr/bin/env python3
"""
Remove unused columns from database tables to optimize the chapter-aware structure.
"""

import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.database.models import SessionLocal
    from sqlalchemy import text, inspect
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def show_current_columns():
    """Show current table structures."""
    print("📊 Current Table Structures:")
    print("=" * 50)

    # Use psql command to show structure
    import subprocess
    try:
        result = subprocess.run([
            'psql', '-h', 'localhost', '-U', 'christianhein', '-d', 'rag_system',
            '-c', '\\d documents; \\d document_chunks; \\d document_chapters;'
        ], capture_output=True, text=True, env={'PGPASSWORD': ''})

        print(result.stdout)
    except Exception as e:
        print(f"❌ Error checking columns: {e}")
        print("Showing known structure instead:")
        print("\nDOCUMENTS: id, filename, filepath, file_hash, content_type, upload_date, last_modified, status, detected_language, author, reading_time, custom_fields, full_content, chapter_content, toc_content, content_structure")
        print("DOCUMENT_CHUNKS: id, document_id, chunk_index, content, embedding_model, chunk_size, overlap, created_at, chunk_type, has_captions, caption_text, caption_line, context_lines, chapter_title, chapter_path, section_type, content_relevance")
        print("DOCUMENT_CHAPTERS: id, document_id, chapter_title, chapter_path, content, embedding, embedding_model, word_count, section_type, parent_chapter_id, level, created_at")


def remove_unused_document_columns():
    """Remove unused columns from documents table."""
    print("\n🗑️  Removing unused columns from documents table...")

    # Columns to remove from documents table
    columns_to_drop = [
        'status',           # Not needed - all docs are processed
        'custom_fields',    # Replaced by structured content columns
        'reading_time',     # Not essential for retrieval
        'author',           # Not core to retrieval functionality
        'content_type'      # Not essential for chapter-aware retrieval
    ]

    db = SessionLocal()
    try:
        for column in columns_to_drop:
            print(f"  🗑️  Dropping column: {column}")
            db.execute(text(f"ALTER TABLE documents DROP COLUMN IF EXISTS {column};"))

        db.commit()
        print("✅ Documents table columns removed successfully!")

    except Exception as e:
        print(f"❌ Error removing document columns: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def remove_unused_chunk_columns():
    """Remove unused columns from document_chunks table."""
    print("\n🗑️  Removing unused columns from document_chunks table...")

    # Columns to remove from document_chunks table
    columns_to_drop = [
        'chunk_type',       # Old caption-centric processing
        'has_captions',     # Old caption-centric processing
        'caption_text',     # Old caption-centric processing
        'caption_line',     # Old caption-centric processing
        'context_lines',    # Old caption-centric processing
        'chunk_size',       # Not essential for retrieval
        'overlap',          # Not essential for retrieval
        'created_at'        # Timestamp not needed for retrieval
    ]

    db = SessionLocal()
    try:
        for column in columns_to_drop:
            print(f"  🗑️  Dropping column: {column}")
            db.execute(text(f"ALTER TABLE document_chunks DROP COLUMN IF EXISTS {column};"))

        db.commit()
        print("✅ Document chunks table columns removed successfully!")

    except Exception as e:
        print(f"❌ Error removing chunk columns: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def show_final_columns():
    """Show final table structures after cleanup."""
    print("\n✨ Final Optimized Table Structures:")
    print("=" * 50)

    # Use psql command to show structure
    import subprocess
    try:
        result = subprocess.run([
            'psql', '-h', 'localhost', '-U', 'christianhein', '-d', 'rag_system',
            '-c', '\\d documents; \\d document_chunks; \\d document_chapters;'
        ], capture_output=True, text=True, env={'PGPASSWORD': ''})

        print(result.stdout)
    except Exception as e:
        print(f"❌ Error checking final columns: {e}")
        print("Expected final structure:")
        print("\nDOCUMENTS: id, filename, filepath, file_hash, upload_date, last_modified, detected_language, full_content, chapter_content, toc_content, content_structure")
        print("DOCUMENT_CHUNKS: id, document_id, chunk_index, content, embedding_model, chapter_title, chapter_path, section_type, content_relevance")
        print("DOCUMENT_CHAPTERS: id, document_id, chapter_title, chapter_path, content, embedding, embedding_model, word_count, section_type, parent_chapter_id, level, created_at")


def update_models_file():
    """Update the SQLAlchemy models to reflect removed columns."""
    print("\n📝 Updating SQLAlchemy models...")

    models_file = "src/database/models.py"

    # Read current models file
    with open(models_file, 'r') as f:
        content = f.read()

    # Remove unused fields from Document model
    replacements = [
        # Remove status field
        ('    status: Mapped[str] = mapped_column(String(50), default=\'processed\')\n', ''),

        # Remove custom_fields
        ('    custom_fields: Mapped[Optional[dict]] = mapped_column(JSON)  # flexible metadata storage\n', ''),

        # Remove reading_time
        ('    reading_time: Mapped[Optional[int]] = mapped_column(Integer)  # estimated reading time in minutes\n', ''),

        # Remove author
        ('    author: Mapped[Optional[str]] = mapped_column(String(255))\n', ''),

        # Remove content_type
        ('    content_type: Mapped[Optional[str]] = mapped_column(String(100))\n', ''),
    ]

    for old, new in replacements:
        content = content.replace(old, new)

    # Remove unused fields from DocumentChunk model
    chunk_replacements = [
        ('    chunk_type: Mapped[Optional[str]] = mapped_column(String(50))  # \'caption_centric\', \'fallback\', etc.\n', ''),
        ('    has_captions: Mapped[Optional[bool]] = mapped_column(Boolean, default=False)\n', ''),
        ('    caption_text: Mapped[Optional[str]] = mapped_column(Text)  # Caption text if this chunk is caption-centric\n', ''),
        ('    caption_line: Mapped[Optional[int]] = mapped_column(Integer)  # Line number where caption was found\n', ''),
        ('    context_lines: Mapped[Optional[str]] = mapped_column(String(50))  # Range of lines included (e.g., "10-25")\n', ''),
        ('    chunk_size: Mapped[Optional[int]] = mapped_column(Integer)\n', ''),
        ('    overlap: Mapped[Optional[int]] = mapped_column(Integer)\n', ''),
        ('    created_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())\n', ''),
    ]

    for old, new in chunk_replacements:
        content = content.replace(old, new)

    # Write updated models file
    with open(models_file, 'w') as f:
        f.write(content)

    print("✅ SQLAlchemy models updated successfully!")


if __name__ == "__main__":
    print("🗑️  Database Column Cleanup Script")
    print("=" * 50)
    print("This script will remove unused columns to optimize storage and performance:")
    print("Documents table: status, custom_fields, reading_time, author, content_type")
    print("Chunks table: chunk_type, has_captions, caption_text, caption_line, context_lines, chunk_size, overlap, created_at")

    try:
        # Show current state
        show_current_columns()

        # Confirm cleanup
        response = input("\n⚠️  This will permanently delete columns and data. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Column removal cancelled by user")
            sys.exit(0)

        # Remove unused columns
        remove_unused_document_columns()
        remove_unused_chunk_columns()

        # Update models
        update_models_file()

        # Show final state
        show_final_columns()

        print("\n🎉 Column cleanup completed!")
        print("💡 Your database now has only essential columns for chapter-aware retrieval:")
        print("   - Documents: core metadata + structured content")
        print("   - Chunks: content + chapter references only")
        print("   - Chapters: full hierarchical structure")

    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        import traceback
        traceback.print_exc()