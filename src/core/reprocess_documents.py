#!/usr/bin/env python3
"""
Reprocess existing documents with enhanced chunking and chapter metadata.
"""

import sys
from pathlib import Path

# Add parent src directory to path if running as standalone script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from core.processing.upload_processor import UploadProcessor
from src.database.models import Document, SessionLocal


def reprocess_documents():
    """Reprocess existing documents with enhanced chunking."""
    print("🔄 Reprocessing existing documents with enhanced chunking...")
    print("=" * 60)

    db = SessionLocal()
    processor = UploadProcessor(embedding_model="embeddinggemma:latest")

    try:
        # Get all documents
        documents = db.query(Document).all()
        print(f"Found {len(documents)} documents to reprocess")

        for i, doc in enumerate(documents, 1):
            print(
                f"\n📄 Reprocessing document {i}/{len(documents)}: {doc.filename[:50]}..."
            )

            try:
                # Check if we have stored content to reprocess with
                if not doc.full_content:
                    print(f"  ⚠️  No stored content for document: {doc.filename}")
                    continue

                # Simple reprocessing: just re-chunk the existing content
                from core.processing.document_processor import DocumentProcessor

                doc_processor = DocumentProcessor(db)

                # Detect chapters from the stored content
                all_chapters = doc_processor._detect_all_chapters(doc.full_content)
                print(f"    Detected {len(all_chapters)} chapters")

                # Create chunks from the content and chapters
                chunks = doc_processor._create_chunks(
                    doc.full_content, doc.id, all_chapters
                )
                print(f"    Created {len(chunks)} chunks")

                if chunks:
                    # Clear existing chunks and chapters
                    from database.models import (
                        DocumentChunk,
                        DocumentChapter,
                        DocumentEmbedding,
                    )

                    # Delete existing data
                    db.query(DocumentEmbedding).filter(
                        DocumentEmbedding.chunk_id.in_(
                            db.query(DocumentChunk.id).filter(
                                DocumentChunk.document_id == doc.id
                            )
                        )
                    ).delete(synchronize_session=False)

                    db.query(DocumentChunk).filter(
                        DocumentChunk.document_id == doc.id
                    ).delete()
                    db.query(DocumentChapter).filter(
                        DocumentChapter.document_id == doc.id
                    ).delete()

                    # Add new chapters
                    for chapter in all_chapters:
                        chapter_record = DocumentChapter(
                            document_id=doc.id,
                            chapter_title=chapter["title"][:255],  # Limit title length
                            chapter_path=chapter["path"],
                            content=chapter["content"],
                            word_count=len(chapter["content"].split()),
                            section_type=chapter.get("level", 1),
                            level=chapter.get("level", 1),
                        )
                        db.add(chapter_record)

                    # Add new chunks and collect them for embedding creation
                    chunk_objects = []
                    for i, chunk_data in enumerate(chunks):
                        chunk = DocumentChunk(
                            document_id=doc.id,
                            chunk_index=i,
                            content=chunk_data["content"],
                            chapter_title=chunk_data.get("metadata", {}).get(
                                "chapter_title"
                            ),
                            chapter_path=chunk_data.get("metadata", {}).get(
                                "chapter_path"
                            ),
                            embedding_model="embeddinggemma:latest",  # Updated model
                        )
                        db.add(chunk)
                        chunk_objects.append(chunk)

                    db.commit()  # Commit to get chunk IDs

                    # Generate and save embeddings
                    try:
                        chunk_texts = [chunk.content for chunk in chunk_objects]
                        from core.embeddings import create_embeddings

                        embeddings_array, _ = create_embeddings(
                            chunk_texts,
                            model_name="embeddinggemma:latest",
                            backend="ollama",
                        )

                        if embeddings_array is not None and len(embeddings_array) > 0:
                            # Save embeddings to database
                            for chunk_db, embedding in zip(
                                chunk_objects, embeddings_array
                            ):
                                from database.models import DocumentEmbedding

                                embedding_record = DocumentEmbedding(
                                    chunk_id=chunk_db.id,
                                    embedding=embedding.tolist(),
                                    embedding_model="embeddinggemma:latest",
                                )
                                db.add(embedding_record)

                            # Index the document with chunks and embeddings in Elasticsearch
                            from core.processing.document_processor import (
                                DocumentProcessor,
                            )

                            doc_processor = DocumentProcessor(db)
                            doc_processor._index_document(
                                doc, chunks, embeddings_array.tolist()
                            )
                            print(
                                f"    ✅ Saved {len(chunks)} embeddings to database and indexed in search"
                            )
                        else:
                            print(f"    ⚠️ Failed to generate embeddings")

                    except Exception as e:
                        print(f"    ⚠️ Failed to create embeddings: {e}")

                    db.commit()
                    print(
                        f"  ✅ Successfully reprocessed: {len(chunks)} chunks, {len(all_chapters)} chapters"
                    )
                else:
                    print(f"  ⚠️  No chunks created for document")

                # Create processing result for reprocessing
                processing_result = {
                    "extracted_content": doc.full_content,
                    "chapters": all_chapters,
                    "chunks": chunks,
                    "chapters_detected": len(all_chapters),
                }

                # Reprocess with the new chunking
                result = processor.reprocess_existing_document(
                    doc, processing_result, doc.filepath
                )

                # Reprocess the document using stored content
                # Create a temporary document processor to process the content
                from core.processing.document_processor import DocumentProcessor

                doc_processor = DocumentProcessor(db)

                # Process the document with stored content
                processing_result = doc_processor._process_document_standard(
                    doc.filepath, doc.filename, content=doc.full_content
                )

                if processing_result.get("success"):
                    # Now reprocess with the full results
                    result = processor.reprocess_existing_document(
                        doc, processing_result, doc.filepath
                    )
                else:
                    print(
                        f"  ❌ Failed to reprocess content: {processing_result.get('error', 'Unknown error')}"
                    )
                    continue

                if result["success"]:
                    print(
                        f"  ✅ Successfully reprocessed: {result.get('chunks_created', 0)} chunks"
                    )
                else:
                    print(
                        f"  ❌ Failed to reprocess: {result.get('error', 'Unknown error')}"
                    )

            except Exception as e:
                print(f"  ❌ Error reprocessing: {e}")

        print("\n" + "=" * 60)
        print("🎉 Document reprocessing completed!")

    except Exception as e:
        print(f"❌ Error during reprocessing: {e}")
        import traceback

        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    reprocess_documents()
