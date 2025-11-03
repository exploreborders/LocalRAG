#!/usr/bin/env python3
"""
Update existing chunks with chapter metadata for better LLM context.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.database.models import SessionLocal, Document, DocumentChunk, DocumentChapter
    from sentence_transformers import SentenceTransformer
    from sqlalchemy import text
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def update_chunk_chapter_metadata():
    """Update chunks with chapter metadata from document structure."""
    print("🔄 Updating chunk chapter metadata...")

    db = SessionLocal()

    try:
        # Get documents that have chapters but chunks without chapter metadata
        documents_with_chapters = db.query(Document).filter(
            Document.chapter_content.isnot(None)
        ).all()

        print(f"📊 Found {len(documents_with_chapters)} documents with chapter content")

        total_chunks_updated = 0

        for doc in documents_with_chapters:
            print(f"🔄 Processing {doc.filename[:50]}...")

            # Get chunks for this document
            chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == doc.id
            ).order_by(DocumentChunk.chunk_index).all()

            if not chunks:
                print(f"  ⚠️  No chunks found for document")
                continue

            # Get chapter structure
            chapter_content = doc.chapter_content
            if not isinstance(chapter_content, dict):
                print(f"  ⚠️  Invalid chapter content format")
                continue

            # Create a mapping of chapter titles to chapter data
            chapter_map = {}
            for chapter_path, chapter_data in chapter_content.items():
                if isinstance(chapter_data, dict):
                    # Handle nested chapters
                    for sub_title, sub_content in chapter_data.items():
                        chapter_map[sub_title] = {
                            'path': f"{chapter_path}.{sub_title}",
                            'content': sub_content if isinstance(sub_content, str) else str(sub_content),
                            'type': 'subsection'
                        }
                else:
                    chapter_map[chapter_path] = {
                        'path': chapter_path,
                        'content': chapter_data if isinstance(chapter_data, str) else str(chapter_data),
                        'type': 'chapter'
                    }

            # Update chunks with chapter information
            chunks_updated = 0

            for chunk in chunks:
                # Find which chapter this chunk belongs to
                chunk_text = chunk.content.lower()
                best_match = None
                best_score = 0
                best_chapter_title = None

                for current_chapter_title, chapter_info in chapter_map.items():
                    # Simple text matching - check if chapter title appears in chunk
                    if current_chapter_title.lower() in chunk_text:
                        # Calculate relevance score based on chapter content similarity
                        chapter_words = set(chapter_info['content'].lower().split())
                        chunk_words = set(chunk_text.split())
                        overlap = len(chapter_words.intersection(chunk_words))
                        score = overlap / len(chapter_words) if chapter_words else 0

                        if score > best_score:
                            best_score = score
                            best_match = chapter_info
                            best_chapter_title = current_chapter_title

                if best_match and best_chapter_title:
                    chunk.chapter_title = best_chapter_title[:250]  # Truncate if too long
                    chunk.chapter_path = best_match['path'][:500]
                    chunk.section_type = best_match['type']
                    chunk.content_relevance = min(best_score, 1.0)  # Cap at 1.0
                    chunks_updated += 1

            db.commit()
            print(f"  ✅ Updated {chunks_updated}/{len(chunks)} chunks with chapter metadata")
            total_chunks_updated += chunks_updated

        print(f"\n📈 Total chunks updated: {total_chunks_updated}")

    except Exception as e:
        print(f"❌ Error updating chunk metadata: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def verify_chunk_updates():
    """Verify that chunks now have chapter metadata."""
    print("\n🔍 Verifying chunk updates...")

    db = SessionLocal()

    try:
        # Check total chunks
        total_chunks = db.execute(text("SELECT COUNT(*) FROM document_chunks")).scalar()
        print(f"📊 Total chunks: {total_chunks}")

        # Check chunks with chapter metadata
        chapter_chunks = db.execute(text("SELECT COUNT(*) FROM document_chunks WHERE chapter_title IS NOT NULL")).scalar()
        print(f"📊 Chunks with chapter metadata: {chapter_chunks} ({(chapter_chunks or 0)/(total_chunks or 1)*100:.1f}%)")

        # Show sample chunks with chapter info
        sample_chunks = db.execute(text("""
            SELECT chunk_index, chapter_title, section_type, content_relevance
            FROM document_chunks
            WHERE chapter_title IS NOT NULL
            LIMIT 5
        """)).fetchall()

        if sample_chunks:
            print("📋 Sample chunks with chapter metadata:")
            for chunk in sample_chunks:
                print(f"  Chunk {chunk.chunk_index}: '{chunk.chapter_title}' ({chunk.section_type}, relevance: {chunk.content_relevance:.2f})")

        print("✅ Chunk verification completed!")

    except Exception as e:
        print(f"❌ Error verifying chunks: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    print("📝 Chunk Chapter Metadata Update Script")
    print("=" * 50)
    print("This script will update existing chunks with chapter information:")
    print("- chapter_title: Which chapter/section the chunk belongs to")
    print("- chapter_path: Hierarchical path in document structure")
    print("- section_type: chapter/section/subsection classification")
    print("- content_relevance: How relevant the chunk is to its chapter")

    try:
        update_chunk_chapter_metadata()
        verify_chunk_updates()

        print("\n🎉 Chunk metadata update completed!")
        print("💡 Chunks now have chapter-aware metadata for better LLM context!")

    except Exception as e:
        print(f"❌ Update failed: {e}")
        import traceback
        traceback.print_exc()