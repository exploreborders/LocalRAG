"""
SQLAlchemy models for the RAG system database.
Defines tables for documents, chunks, and processing jobs.
"""

from sqlalchemy import create_engine, Integer, String, Text, TIMESTAMP, ForeignKey, func, JSON, Table, Column
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
    content_type: Mapped[Optional[str]] = mapped_column(String(100))
    detected_language: Mapped[Optional[str]] = mapped_column(String(10))  # ISO 639-1 language code (e.g., 'en', 'de')
    upload_date: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())
    last_modified: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())
    status: Mapped[str] = mapped_column(String(50), default='processed')
    author: Mapped[Optional[str]] = mapped_column(String(255))
    reading_time: Mapped[Optional[int]] = mapped_column(Integer)  # estimated reading time in minutes
    custom_fields: Mapped[Optional[dict]] = mapped_column(JSON)  # flexible metadata storage

    chunks = relationship("DocumentChunk", back_populates="document")
    jobs = relationship("ProcessingJob", back_populates="document")
    tags = relationship("DocumentTag", secondary="document_tags_association", back_populates="documents")
    categories = relationship("DocumentCategory", secondary="document_categories_association", back_populates="documents")

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
    chunk_size: Mapped[Optional[int]] = mapped_column(Integer)
    overlap: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())

    document = relationship("Document", back_populates="chunks")

class ProcessingJob(Base):
    """
    Processing job table.

    Tracks document processing jobs including status, timing,
    and error information.
    """
    __tablename__ = 'processing_jobs'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey('documents.id'), nullable=False)
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default='pending')
    started_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP)
    completed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())

    document = relationship("Document", back_populates="jobs")

class DocumentTag(Base):
    """
    Document tag table.

    Stores tags that can be assigned to documents for organization.
    """
    __tablename__ = 'document_tags'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    color: Mapped[Optional[str]] = mapped_column(String(7))  # hex color code like #FF5733
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())

    documents = relationship("Document", secondary="document_tags_association", back_populates="tags")

class DocumentCategory(Base):
    """
    Document category table.

    Stores hierarchical categories for document organization.
    """
    __tablename__ = 'document_categories'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    parent_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey('document_categories.id'))
    created_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())

    documents = relationship("Document", secondary="document_categories_association", back_populates="categories")
    parent = relationship("DocumentCategory", remote_side=[id])
    children = relationship("DocumentCategory", back_populates="parent")

# Association tables for many-to-many relationships
# Association tables for many-to-many relationships
class DocumentTagsAssociation(Base):
    """
    Association table for document-tag many-to-many relationship.
    """
    __tablename__ = 'document_tags_association'

    document_id: Mapped[int] = mapped_column(Integer, ForeignKey('documents.id'), primary_key=True)
    tag_id: Mapped[int] = mapped_column(Integer, ForeignKey('document_tags.id'), primary_key=True)

class DocumentCategoriesAssociation(Base):
    """
    Association table for document-category many-to-many relationship.
    """
    __tablename__ = 'document_categories_association'

    document_id: Mapped[int] = mapped_column(Integer, ForeignKey('documents.id'), primary_key=True)
    category_id: Mapped[int] = mapped_column(Integer, ForeignKey('document_categories.id'), primary_key=True)

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