"""
SQLAlchemy models for the RAG system database.
Defines tables for documents, chunks, and processing jobs.
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()

class Document(Base):
    """
    Document metadata table.

    Stores information about uploaded documents including file details,
    processing status, and relationships to chunks and jobs.
    """
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(500), nullable=False)
    file_hash = Column(String(64), nullable=False, unique=True)
    content_type = Column(String(100))
    upload_date = Column(TIMESTAMP, default=func.current_timestamp())
    last_modified = Column(TIMESTAMP, default=func.current_timestamp())
    status = Column(String(50), default='processed')

    chunks = relationship("DocumentChunk", back_populates="document")
    jobs = relationship("ProcessingJob", back_populates="document")

class DocumentChunk(Base):
    """
    Document chunk table.

    Stores individual text chunks from documents along with their
    embedding information and metadata.
    """
    __tablename__ = 'document_chunks'

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding_model = Column(String(100), nullable=False)
    chunk_size = Column(Integer)
    overlap = Column(Integer)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())

    document = relationship("Document", back_populates="chunks")

class ProcessingJob(Base):
    """
    Processing job table.

    Tracks document processing jobs including status, timing,
    and error information.
    """
    __tablename__ = 'processing_jobs'

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    job_type = Column(String(50), nullable=False)
    status = Column(String(50), default='pending')
    started_at = Column(TIMESTAMP)
    completed_at = Column(TIMESTAMP)
    error_message = Column(Text)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())

    document = relationship("Document", back_populates="jobs")

# Database connection
DATABASE_URL = "postgresql://christianhein@localhost:5432/rag_system"
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