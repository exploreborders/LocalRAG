"""
SQLAlchemy models for the RAG system database.
Defines tables for documents, chunks, and processing jobs.
"""

from sqlalchemy import create_engine, Integer, String, Text, TIMESTAMP, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Mapped, mapped_column
from typing import Optional
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
    upload_date: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())
    last_modified: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())
    status: Mapped[str] = mapped_column(String(50), default='processed')

    chunks = relationship("DocumentChunk", back_populates="document")
    jobs = relationship("ProcessingJob", back_populates="document")

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