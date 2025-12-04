#!/usr/bin/env python3
"""
Final verification script to check that all database columns are populated.
"""

import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from sqlalchemy import text

    from src.database.models import SessionLocal
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def check_database_population():
    """Check that all columns are properly populated."""
    print("🔍 Final Database Population Check")
    print("=" * 50)

    db = SessionLocal()

    try:
        # Check documents
        print("\n📄 DOCUMENTS:")
        docs = db.execute(
            text(
                """
            SELECT
                id,
                filename,
                length(coalesce(full_content, '')) as content_len,
                CASE WHEN chapter_content IS NOT NULL THEN 'populated' ELSE 'empty' END as chapter_status,
                CASE WHEN toc_content IS NOT NULL THEN 'populated' ELSE 'empty' END as toc_status,
                CASE WHEN content_structure IS NOT NULL THEN 'populated' ELSE 'empty' END as structure_status,
                detected_language
            FROM documents
        """
            )
        ).fetchall()

        for doc in docs:
            print(f"  ID {doc.id}: {doc.filename[:40]}...")
            print(f"    Content: {doc.content_len} chars")
            print(f"    Chapters: {doc.chapter_status}")
            print(f"    TOC: {doc.toc_status}")
            print(f"    Structure: {doc.structure_status}")
            print(f"    Language: {doc.detected_language}")

        # Check chapters
        print("\n📖 CHAPTERS:")
        chapter_count = db.execute(
            text("SELECT COUNT(*) FROM document_chapters")
        ).scalar()
        print(f"  Total chapters: {chapter_count}")

        if chapter_count > 0:
            chapters = db.execute(
                text(
                    """
                SELECT chapter_title, word_count, section_type, level
                FROM document_chapters
                ORDER BY id
                LIMIT 5
            """
                )
            ).fetchall()

            for chapter in chapters:
                print(
                    f"  '{chapter.chapter_title[:30]}...' ({chapter.word_count} words, {chapter.section_type}, level {chapter.level})"
                )

        # Check chunks
        print("\n📦 CHUNKS:")
        total_chunks = db.execute(text("SELECT COUNT(*) FROM document_chunks")).scalar()
        chapter_chunks = db.execute(
            text("SELECT COUNT(*) FROM document_chunks WHERE chapter_title IS NOT NULL")
        ).scalar()
        path_chunks = db.execute(
            text("SELECT COUNT(*) FROM document_chunks WHERE chapter_path IS NOT NULL")
        ).scalar()
        type_chunks = db.execute(
            text("SELECT COUNT(*) FROM document_chunks WHERE section_type IS NOT NULL")
        ).scalar()
        relevance_chunks = db.execute(
            text(
                "SELECT COUNT(*) FROM document_chunks WHERE content_relevance IS NOT NULL"
            )
        ).scalar()

        print(f"  Total chunks: {total_chunks}")
        print(
            f"  With chapter_title: {chapter_chunks} ({chapter_chunks / total_chunks * 100:.1f}%)"
        )
        print(
            f"  With chapter_path: {path_chunks} ({path_chunks / total_chunks * 100:.1f}%)"
        )
        print(
            f"  With section_type: {type_chunks} ({type_chunks / total_chunks * 100:.1f}%)"
        )
        print(
            f"  With content_relevance: {relevance_chunks} ({relevance_chunks / total_chunks * 100:.1f}%)"
        )

        # Sample chunk metadata
        if chapter_chunks > 0:
            print("\n📋 Sample Chunk Metadata:")
            sample_chunks = db.execute(
                text(
                    """
                SELECT chunk_index, chapter_title, section_type, content_relevance
                FROM document_chunks
                WHERE chapter_title IS NOT NULL
                ORDER BY chunk_index
                LIMIT 3
            """
                )
            ).fetchall()

            for chunk in sample_chunks:
                print(
                    f"  Chunk {chunk.chunk_index}: '{chunk.chapter_title[:25]}...' ({chunk.section_type}, rel: {chunk.content_relevance:.2f})"
                )

        # Overall assessment
        print("\n🎯 OVERALL ASSESSMENT:")

        all_populated = (
            docs[0].content_len > 0
            and docs[0].chapter_status == "populated"
            and docs[0].toc_status == "populated"
            and docs[0].structure_status == "populated"
            and chapter_count > 0
            and chapter_chunks == total_chunks
            and path_chunks == total_chunks
            and type_chunks == total_chunks
            and relevance_chunks == total_chunks
        )

        if all_populated:
            print("  ✅ ALL COLUMNS FULLY POPULATED!")
            print("  🎉 Chapter-aware database structure is complete and ready!")
        else:
            print("  ⚠️  Some columns still need population:")
            if docs[0].content_len == 0:
                print("    - Document full_content is empty")
            if docs[0].chapter_status != "populated":
                print("    - Document chapter_content is empty")
            if docs[0].toc_status != "populated":
                print("    - Document toc_content is empty")
            if docs[0].structure_status != "populated":
                print("    - Document content_structure is empty")
            if chapter_count == 0:
                print("    - No chapters in document_chapters table")
            if chapter_chunks < total_chunks:
                print(
                    f"    - {total_chunks - chapter_chunks} chunks missing chapter_title"
                )
            if path_chunks < total_chunks:
                print(f"    - {total_chunks - path_chunks} chunks missing chapter_path")
            if type_chunks < total_chunks:
                print(f"    - {total_chunks - type_chunks} chunks missing section_type")
            if relevance_chunks < total_chunks:
                print(
                    f"    - {total_chunks - relevance_chunks} chunks missing content_relevance"
                )

    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback

        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    check_database_population()
