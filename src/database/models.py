"""
Enhanced SQLAlchemy models for AI-powered RAG system with hierarchical structure and topic classification.
"""

from sqlalchemy import create_engine, Integer, String, Text, TIMESTAMP, ForeignKey, func, JSON, Table, Column, Boolean, Float, Date
from datetime import date
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Mapped, mapped_column
from typing import Optional, List
from datetime import datetime

Base = declarative_base()

class Document(Base):
    """
    Enhanced document metadata table with AI-enriched content and structure.
    """
    __tablename__ = 'documents'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(500), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    detected_language: Mapped[Optional[str]] = mapped_column(String(10))  # ISO 639-1 language code
    content_type: Mapped[Optional[str]] = mapped_column(String(50))  # File type/extension
    status: Mapped[str] = mapped_column(String(20), default='uploaded')  # Processing status
    upload_date: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())
    last_modified: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())

    # Content columns for structured document content
    full_content: Mapped[Optional[str]] = mapped_column(Text)  # Complete document content
    chapter_content: Mapped[Optional[dict]] = mapped_column(JSONB)  # Content organized by chapters
    toc_content: Mapped[Optional[list]] = mapped_column(JSONB)  # Table of contents structure
    content_structure: Mapped[Optional[dict]] = mapped_column(JSONB)  # Document structure metadata

    # AI-enriched metadata
    document_summary: Mapped[Optional[str]] = mapped_column(Text)  # AI-generated summary
    key_topics: Mapped[Optional[list]] = mapped_column(JSONB)  # Extracted key topics
    reading_time_minutes: Mapped[Optional[int]] = mapped_column(Integer)  # Estimated reading time
    author: Mapped[Optional[str]] = mapped_column(String(255))  # Document author
    publication_date: Mapped[Optional[date]] = mapped_column(Date)  # Publication date
    custom_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)  # Flexible metadata

    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document")
    chapters = relationship("DocumentChapter", back_populates="document")
    topics = relationship("DocumentTopic", back_populates="document")
    tags = relationship("DocumentTagAssignment", back_populates="document")
    categories = relationship("DocumentCategoryAssignment", back_populates="document")

class DocumentChapter(Base):
    """
    Document chapters table for hierarchical retrieval and navigation.
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
    level: Mapped[int] = mapped_column(Integer, default=1)  # Hierarchy level
    created_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())

    # Relationships
    document = relationship("Document", back_populates="chapters")
    parent = relationship("DocumentChapter", remote_side=[id], backref="subchapters")

class DocumentChunk(Base):
    """
    Enhanced document chunks with chapter awareness and topic classification.
    """
    __tablename__ = 'document_chunks'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey('documents.id'), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)

    # Chapter-aware chunk metadata
    chapter_title: Mapped[Optional[str]] = mapped_column(String(255))  # Chapter/section this chunk belongs to
    chapter_path: Mapped[Optional[str]] = mapped_column(String(500))  # Hierarchical path (e.g., "1.2.3")
    section_type: Mapped[Optional[str]] = mapped_column(String(50))  # 'chapter', 'section', 'subsection', 'paragraph'
    content_relevance: Mapped[Optional[float]] = mapped_column(Float)  # 0-1 score for content density

    # Topic classification
    primary_topic: Mapped[Optional[str]] = mapped_column(String(100))  # Main topic this chunk relates to
    topic_relevance: Mapped[Optional[float]] = mapped_column(Float)  # 0-1 relevance score to primary topic

    created_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())

    # Relationships
    document = relationship("Document", back_populates="chunks")
    embedding = relationship("DocumentEmbedding", back_populates="chunk", uselist=False)

class DocumentEmbedding(Base):
    """
    Vector embeddings storage optimized for pgvector operations.
    """
    __tablename__ = 'document_embeddings'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[int] = mapped_column(Integer, ForeignKey('document_chunks.id'), nullable=False)
    embedding: Mapped[Optional[list]] = mapped_column(JSONB)  # Vector embedding (will be converted to pgvector)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())

    # Relationships
    chunk = relationship("DocumentChunk", back_populates="embedding")

class Topic(Base):
    """
    Topic classification system for cross-document relationships.
    """
    __tablename__ = 'topics'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[Optional[str]] = mapped_column(String(50))  # 'academic', 'technical', 'business', etc.
    parent_topic_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey('topics.id'))
    created_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())

    # Relationships
    parent = relationship("Topic", remote_side=[id], backref="subtopics")
    documents = relationship("DocumentTopic", back_populates="topic")

class DocumentTopic(Base):
    """
    Document-topic relationships for cross-document analysis.
    """
    __tablename__ = 'document_topics'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey('documents.id'), nullable=False)
    topic_id: Mapped[int] = mapped_column(Integer, ForeignKey('topics.id'), nullable=False)
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)  # 0-1 relevance
    confidence: Mapped[Optional[float]] = mapped_column(Float)  # AI confidence in classification
    assigned_by: Mapped[str] = mapped_column(String(50), default='auto')  # 'auto', 'manual', 'user'
    created_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())

    # Relationships
    document = relationship("Document", back_populates="topics")
    topic = relationship("Topic", back_populates="documents")

    __table_args__ = ({'schema': None},)

class DocumentTag(Base):
    """
    Document tags for flexible categorization.
    """
    __tablename__ = 'document_tags'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    color: Mapped[Optional[str]] = mapped_column(String(7))  # Hex color code
    description: Mapped[Optional[str]] = mapped_column(Text)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())

    # Relationships
    assignments = relationship("DocumentTagAssignment", back_populates="tag")

class DocumentTagAssignment(Base):
    """
    Document-tag assignments (many-to-many).
    """
    __tablename__ = 'document_tag_assignments'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey('documents.id'), nullable=False)
    tag_id: Mapped[int] = mapped_column(Integer, ForeignKey('document_tags.id'), nullable=False)
    assigned_by: Mapped[str] = mapped_column(String(50), default='auto')
    created_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())

    # Relationships
    document = relationship("Document", back_populates="tags")
    tag = relationship("DocumentTag", back_populates="assignments")

class DocumentCategory(Base):
    """
    Document categories for hierarchical organization.
    """
    __tablename__ = 'document_categories'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    parent_category_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey('document_categories.id'))
    level: Mapped[int] = mapped_column(Integer, default=1)
    color: Mapped[Optional[str]] = mapped_column(String(7))
    created_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())

    # Relationships
    parent = relationship("DocumentCategory", remote_side=[id], backref="subcategories")
    assignments = relationship("DocumentCategoryAssignment", back_populates="category")

class DocumentCategoryAssignment(Base):
    """
    Document-category assignments.
    """
    __tablename__ = 'document_category_assignments'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey('documents.id'), nullable=False)
    category_id: Mapped[int] = mapped_column(Integer, ForeignKey('document_categories.id'), nullable=False)
    assigned_by: Mapped[str] = mapped_column(String(50), default='auto')
    created_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())

    # Relationships
    document = relationship("Document", back_populates="categories")
    category = relationship("DocumentCategory", back_populates="assignments")

class ProcessingJob(Base):
    """
    Processing jobs table with enhanced tracking for AI pipeline.
    """
    __tablename__ = 'processing_jobs'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey('documents.id'), nullable=False)
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'embedding', 'topic_classification', 'structure_extraction'
    status: Mapped[str] = mapped_column(String(50), default='pending')  # 'pending', 'running', 'completed', 'failed'
    started_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP)
    completed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    progress: Mapped[float] = mapped_column(Float, default=0)  # 0-1 progress indicator
    job_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)  # Additional job-specific data
    created_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, default=func.current_timestamp())

    # Relationships
    document = relationship("Document", backref="processing_jobs")



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