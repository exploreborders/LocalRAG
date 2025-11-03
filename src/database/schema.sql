-- Enhanced PostgreSQL schema for AI-powered RAG system with hierarchical structure and topic classification

-- Enable pgvector extension for vector embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table with enhanced metadata
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    filepath VARCHAR(500) NOT NULL,
    file_hash VARCHAR(64) NOT NULL UNIQUE,
    detected_language VARCHAR(10), -- ISO 639-1 language code (e.g., 'en', 'de')
    content_type VARCHAR(50), -- File type/extension (e.g., 'pdf', 'txt')
    status VARCHAR(20) DEFAULT 'uploaded', -- Processing status
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Content columns for structured document content
    full_content TEXT, -- Complete document content
    chapter_content JSONB, -- Content organized by chapters/subchapters
    toc_content JSONB, -- Table of contents structure
    content_structure JSONB, -- Document structure metadata

    -- AI-enriched metadata
    document_summary TEXT, -- AI-generated document summary
    key_topics JSONB, -- Extracted key topics and concepts
    reading_time_minutes INTEGER, -- Estimated reading time
    author VARCHAR(255), -- Document author if available
    publication_date DATE, -- Publication date if available
    custom_metadata JSONB -- Flexible metadata storage
);

-- Document chapters table for hierarchical retrieval
CREATE TABLE IF NOT EXISTS document_chapters (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chapter_title VARCHAR(255) NOT NULL,
    chapter_path VARCHAR(100) NOT NULL, -- Hierarchical path (e.g., "1.2")
    content TEXT NOT NULL, -- Full chapter content
    embedding JSONB, -- Pre-computed chapter embedding
    embedding_model VARCHAR(100),
    word_count INTEGER DEFAULT 0,
    section_type VARCHAR(50) DEFAULT 'chapter', -- 'chapter', 'section', 'subsection'
    parent_chapter_id INTEGER REFERENCES document_chapters(id),
    level INTEGER DEFAULT 1, -- Hierarchy level (1=chapter, 2=section, etc.)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document chunks table with chapter-aware metadata
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding_model VARCHAR(100) NOT NULL,

    -- Chapter-aware chunk metadata for better LLM context
    chapter_title VARCHAR(255), -- Chapter/section this chunk belongs to
    chapter_path VARCHAR(500), -- Hierarchical path (e.g., "1.2.3")
    section_type VARCHAR(50), -- 'chapter', 'section', 'subsection', 'paragraph'
    content_relevance REAL, -- 0-1 score for content density

    -- Topic classification
    primary_topic VARCHAR(100), -- Main topic this chunk relates to
    topic_relevance REAL, -- 0-1 relevance score to primary topic

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Topic classification system
CREATE TABLE IF NOT EXISTS topics (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    category VARCHAR(50), -- 'academic', 'technical', 'business', etc.
    parent_topic_id INTEGER REFERENCES topics(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document-topic relationships for cross-document analysis
CREATE TABLE IF NOT EXISTS document_topics (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    topic_id INTEGER REFERENCES topics(id) ON DELETE CASCADE,
    relevance_score REAL NOT NULL, -- 0-1 relevance to this topic
    confidence REAL, -- AI confidence in classification
    assigned_by VARCHAR(50) DEFAULT 'auto', -- 'auto', 'manual', 'user'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, topic_id)
);

-- Document tags for flexible categorization
CREATE TABLE IF NOT EXISTS document_tags (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    color VARCHAR(7), -- Hex color code (e.g., '#FF5733')
    description TEXT,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document-tag assignments (many-to-many)
CREATE TABLE IF NOT EXISTS document_tag_assignments (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    tag_id INTEGER REFERENCES document_tags(id) ON DELETE CASCADE,
    assigned_by VARCHAR(50) DEFAULT 'auto',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, tag_id)
);

-- Document categories for hierarchical organization
CREATE TABLE IF NOT EXISTS document_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    parent_category_id INTEGER REFERENCES document_categories(id),
    level INTEGER DEFAULT 1,
    color VARCHAR(7),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document-category assignments
CREATE TABLE IF NOT EXISTS document_category_assignments (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES document_categories(id) ON DELETE CASCADE,
    assigned_by VARCHAR(50) DEFAULT 'auto',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, category_id)
);

-- Processing jobs table with enhanced tracking
CREATE TABLE IF NOT EXISTS processing_jobs (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    job_type VARCHAR(50) NOT NULL, -- 'embedding', 'indexing', 'topic_classification', 'structure_extraction'
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed'
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    progress REAL DEFAULT 0, -- 0-1 progress indicator
    metadata JSONB, -- Additional job-specific data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vector embeddings storage (optimized for pgvector)
CREATE TABLE IF NOT EXISTS document_embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES document_chunks(id) ON DELETE CASCADE,
    embedding vector(768), -- nomic-embed-text-v1.5 dimension
    embedding_model VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_modified ON documents(last_modified);
CREATE INDEX IF NOT EXISTS idx_documents_language ON documents(detected_language);
CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(id); -- For category joins

-- Chapter hierarchy indexes
CREATE INDEX IF NOT EXISTS idx_chapters_document_id ON document_chapters(document_id);
CREATE INDEX IF NOT EXISTS idx_chapters_path ON document_chapters(chapter_path);
CREATE INDEX IF NOT EXISTS idx_chapters_parent ON document_chapters(parent_chapter_id);
CREATE INDEX IF NOT EXISTS idx_chapters_level ON document_chapters(level);

-- Chunk indexes with chapter awareness
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id_index ON document_chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_model ON document_chunks(embedding_model);
CREATE INDEX IF NOT EXISTS idx_chunks_chapter_path ON document_chunks(chapter_path);
CREATE INDEX IF NOT EXISTS idx_chunks_topic ON document_chunks(primary_topic);
CREATE INDEX IF NOT EXISTS idx_chunks_relevance ON document_chunks(content_relevance);

-- Topic classification indexes
CREATE INDEX IF NOT EXISTS idx_topics_name ON topics(name);
CREATE INDEX IF NOT EXISTS idx_topics_category ON topics(category);
CREATE INDEX IF NOT EXISTS idx_topics_parent ON topics(parent_topic_id);
CREATE INDEX IF NOT EXISTS idx_document_topics_doc ON document_topics(document_id);
CREATE INDEX IF NOT EXISTS idx_document_topics_topic ON document_topics(topic_id);
CREATE INDEX IF NOT EXISTS idx_document_topics_score ON document_topics(relevance_score);

-- Tag and category indexes
CREATE INDEX IF NOT EXISTS idx_tags_name ON document_tags(name);
CREATE INDEX IF NOT EXISTS idx_tag_assignments_doc ON document_tag_assignments(document_id);
CREATE INDEX IF NOT EXISTS idx_tag_assignments_tag ON document_tag_assignments(tag_id);
CREATE INDEX IF NOT EXISTS idx_categories_name ON document_categories(name);
CREATE INDEX IF NOT EXISTS idx_categories_parent ON document_categories(parent_category_id);
CREATE INDEX IF NOT EXISTS idx_category_assignments_doc ON document_category_assignments(document_id);
CREATE INDEX IF NOT EXISTS idx_category_assignments_cat ON document_category_assignments(category_id);

-- Processing and embedding indexes
CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_document_id ON processing_jobs(document_id);
CREATE INDEX IF NOT EXISTS idx_jobs_type ON processing_jobs(job_type);
CREATE INDEX IF NOT EXISTS idx_jobs_recent ON processing_jobs(created_at) WHERE created_at > NOW() - INTERVAL '7 days';
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON document_embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON document_embeddings(embedding_model);

-- Vector similarity indexes (requires pgvector)
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON document_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Partial indexes for common queries
CREATE INDEX IF NOT EXISTS idx_documents_processed ON documents(status) WHERE status = 'processed';
CREATE INDEX IF NOT EXISTS idx_documents_uploaded_recent ON documents(upload_date) WHERE upload_date > NOW() - INTERVAL '30 days';
CREATE INDEX IF NOT EXISTS idx_jobs_running ON processing_jobs(status) WHERE status = 'running';