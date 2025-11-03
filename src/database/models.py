"""
SQLAlchemy models for the RAG system database.
Defines tables for documents, chunks, and processing jobs.
"""

from sqlalchemy import create_engine, Integer, String, Text, TIMESTAMP, ForeignKey, func, JSON, Table, Column, Boolean, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Mapped, mapped_column
from typing import Optional, List
from datetime import datetime

Base = declarative_base()

class Document(Base):
    """
    Document metadata table.

    Stores information about uploaded documents including file details,
    processing status, and relationships to chunks and jobs.
    """
    __tablename__ = 'documents'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(500), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    detected_language: Mapped[Optional[str]] = mapped_column(String(10))  # ISO 639-1 language code (e.g., 'en', 'de')
    content_type: Mapped[Optional[str]] = mapped_column(String(50))  # File type/extension (e.g., 'pdf', 'txt')
    status: Mapped[str] = mapped_column(String(20), default='uploaded')  # Processing status
    upload_date: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())
    last_modified: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())

    # Content columns for structured document content
    full_content: Mapped[Optional[str]] = mapped_column(Text)  # Complete document content
    chapter_content: Mapped[Optional[dict]] = mapped_column(JSONB)  # Content organized by chapters/subchapters
    toc_content: Mapped[Optional[list]] = mapped_column(JSONB)  # Table of contents structure
    content_structure: Mapped[Optional[dict]] = mapped_column(JSONB)  # Document structure metadata

    chunks = relationship("DocumentChunk", back_populates="document")
    chapters = relationship("DocumentChapter", backref="parent_document")

class DocumentChunk(Base):
    """
    Document chunk table.

    Stores individual text chunks from documents along with their
    embedding information and metadata.
    """
    __tablename__ = 'document_chunks'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey('documents.id'), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)

    # Chapter-aware chunk metadata for better LLM context
    chapter_title: Mapped[Optional[str]] = mapped_column(String(255))  # Chapter/section this chunk belongs to
    chapter_path: Mapped[Optional[str]] = mapped_column(String(500))  # Hierarchical path (e.g., "1.2.3")
    section_type: Mapped[Optional[str]] = mapped_column(String(50))  # 'chapter', 'section', 'subsection', 'paragraph'
    content_relevance: Mapped[Optional[float]] = mapped_column(Float)  # 0-1 score for content density


    document = relationship("Document", backref="document_chunks")


class DocumentChapter(Base):
    """
    Document chapters table for hierarchical retrieval.

    Stores chapter-level content and embeddings for fast high-level retrieval.
    """
    __tablename__ = 'document_chapters'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey('documents.id'), nullable=False)
    chapter_title: Mapped[str] = mapped_column(String(255), nullable=False)
    chapter_path: Mapped[str] = mapped_column(String(100), nullable=False)  # Hierarchical path (e.g., "1.2")
    content: Mapped[str] = mapped_column(Text, nullable=False)  # Full chapter content
    embedding: Mapped[Optional[list]] = mapped_column(JSONB)  # Pre-computed chapter embedding
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100))
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    section_type: Mapped[str] = mapped_column(String(50), default='chapter')  # 'chapter', 'section', 'subsection'
    parent_chapter_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey('document_chapters.id'))
    level: Mapped[int] = mapped_column(Integer, default=1)  # Hierarchy level (1=chapter, 2=section, etc.)

    document = relationship("Document", backref="document_chapters")
    parent = relationship("DocumentChapter", remote_side=[id], backref="subchapters")



# Database connection
import os
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{os.getenv('POSTGRES_USER', 'christianhein')}:{os.getenv('POSTGRES_PASSWORD', '')}@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'rag_system')}"
)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    Dependency function to get database session.

    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()