#!/usr/bin/env python3
"""
Enhanced database schema migration script for AI-powered RAG system.

This script safely migrates from the basic schema to the enhanced schema
with topic classification, hierarchical structure, and AI-enriched metadata.
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Text, TIMESTAMP, ForeignKey, func
from sqlalchemy.orm import sessionmaker

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database.models import Base

def get_database_url():
    """Get database URL from environment variables."""
    return os.getenv(
        "DATABASE_URL",
        f"postgresql://{os.getenv('POSTGRES_USER', 'christianhein')}:{os.getenv('POSTGRES_PASSWORD', '')}@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'rag_system')}"
    )

def create_enhanced_tables(engine):
    """Create all new tables for the enhanced schema."""
    print("Creating enhanced database tables...")

    with engine.connect() as conn:
        # Try to enable pgvector (skip if not available)
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
            print("✅ pgvector extension enabled")
        except Exception as e:
            print(f"⚠️  pgvector not available (continuing without vectors): {e}")

        # Create all tables from the enhanced schema
        Base.metadata.create_all(bind=engine)

        print("✅ Enhanced tables created successfully")

def migrate_existing_data(engine):
    """Migrate existing data to new schema structure."""
    print("Migrating existing data to enhanced schema...")

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Update existing documents status if needed
        db.execute(text("""
            UPDATE documents
            SET status = 'processed'
            WHERE status IS NULL OR status = '';
        """))

        # Migrate existing chunks to new structure (if any new columns needed)
        # The existing chunks should be compatible, but we can add default values

        db.commit()
        print("✅ Existing data migrated successfully")

    except Exception as e:
        db.rollback()
        print(f"❌ Error migrating data: {e}")
        raise
    finally:
        db.close()

def create_indexes(engine):
    """Create performance indexes for the enhanced schema."""
    print("Creating performance indexes...")

    with engine.connect() as conn:
        indexes = [
            # Document indexes
            "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);",
            "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);",
            "CREATE INDEX IF NOT EXISTS idx_documents_modified ON documents(last_modified);",
            "CREATE INDEX IF NOT EXISTS idx_documents_language ON documents(detected_language);",

            # Chapter hierarchy indexes
            "CREATE INDEX IF NOT EXISTS idx_chapters_document_id ON document_chapters(document_id);",
            "CREATE INDEX IF NOT EXISTS idx_chapters_path ON document_chapters(chapter_path);",
            "CREATE INDEX IF NOT EXISTS idx_chapters_parent ON document_chapters(parent_chapter_id);",
            "CREATE INDEX IF NOT EXISTS idx_chapters_level ON document_chapters(level);",

            # Chunk indexes with chapter awareness
            "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);",
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id_index ON document_chunks(document_id, chunk_index);",
            "CREATE INDEX IF NOT EXISTS idx_chunks_model ON document_chunks(embedding_model);",
            "CREATE INDEX IF NOT EXISTS idx_chunks_chapter_path ON document_chunks(chapter_path);",
            "CREATE INDEX IF NOT EXISTS idx_chunks_topic ON document_chunks(primary_topic);",
            "CREATE INDEX IF NOT EXISTS idx_chunks_relevance ON document_chunks(content_relevance);",

            # Topic classification indexes
            "CREATE INDEX IF NOT EXISTS idx_topics_name ON topics(name);",
            "CREATE INDEX IF NOT EXISTS idx_topics_category ON topics(category);",
            "CREATE INDEX IF NOT EXISTS idx_topics_parent ON topics(parent_topic_id);",
            "CREATE INDEX IF NOT EXISTS idx_document_topics_doc ON document_topics(document_id);",
            "CREATE INDEX IF NOT EXISTS idx_document_topics_topic ON document_topics(topic_id);",
            "CREATE INDEX IF NOT EXISTS idx_document_topics_score ON document_topics(relevance_score);",

            # Tag and category indexes
            "CREATE INDEX IF NOT EXISTS idx_tags_name ON document_tags(name);",
            "CREATE INDEX IF NOT EXISTS idx_tag_assignments_doc ON document_tag_assignments(document_id);",
            "CREATE INDEX IF NOT EXISTS idx_tag_assignments_tag ON document_tag_assignments(tag_id);",
            "CREATE INDEX IF NOT EXISTS idx_categories_name ON document_categories(name);",
            "CREATE INDEX IF NOT EXISTS idx_categories_parent ON document_categories(parent_category_id);",
            "CREATE INDEX IF NOT EXISTS idx_category_assignments_doc ON document_category_assignments(document_id);",
            "CREATE INDEX IF NOT EXISTS idx_category_assignments_cat ON document_category_assignments(category_id);",

            # Processing and embedding indexes
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status);",
            "CREATE INDEX IF NOT EXISTS idx_jobs_document_id ON processing_jobs(document_id);",
            "CREATE INDEX IF NOT EXISTS idx_jobs_type ON processing_jobs(job_type);",
            "CREATE INDEX IF NOT EXISTS idx_jobs_recent ON processing_jobs(created_at) WHERE created_at > NOW() - INTERVAL '7 days';",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON document_embeddings(chunk_id);",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_model ON document_embeddings(embedding_model);",

            # Vector similarity index (requires pgvector)
            "CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON document_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);",

            # Partial indexes for common queries
            "CREATE INDEX IF NOT EXISTS idx_documents_processed ON documents(status) WHERE status = 'processed';",
            "CREATE INDEX IF NOT EXISTS idx_documents_uploaded_recent ON documents(upload_date) WHERE upload_date > NOW() - INTERVAL '30 days';",
            "CREATE INDEX IF NOT EXISTS idx_jobs_running ON processing_jobs(status) WHERE status = 'running';"
        ]

        for index_sql in indexes:
            try:
                conn.execute(text(index_sql))
            except Exception as e:
                print(f"Warning: Could not create index: {e}")

        conn.commit()
        print("✅ Performance indexes created")

def seed_initial_data(engine):
    """Seed initial topics, tags, and categories for the system."""
    print("Seeding initial classification data...")

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Seed initial topics
        initial_topics = [
            ("Machine Learning", "artificial intelligence", None),
            ("Natural Language Processing", "artificial intelligence", None),
            ("Computer Vision", "artificial intelligence", None),
            ("Data Science", "technology", None),
            ("Research Methodology", "academic", None),
            ("Academic Writing", "academic", None),
            ("Technical Documentation", "technical", None),
            ("Business Intelligence", "business", None),
        ]

        for name, category, parent_id in initial_topics:
            # Check if topic exists
            result = db.execute(text("SELECT id FROM topics WHERE name = :name"), {"name": name}).fetchone()
            if not result:
                db.execute(text("""
                    INSERT INTO topics (name, category, parent_topic_id)
                    VALUES (:name, :category, :parent_id)
                """), {"name": name, "category": category, "parent_id": parent_id})

        # Seed initial tags
        initial_tags = [
            ("important", "#FF5733", "High priority content"),
            ("review", "#33FF57", "Needs review or verification"),
            ("draft", "#3357FF", "Draft or preliminary content"),
            ("reference", "#F033FF", "Reference material"),
            ("tutorial", "#33FFF0", "Educational content"),
            ("research", "#F0FF33", "Research-related content"),
        ]

        for name, color, description in initial_tags:
            # Check if tag exists
            result = db.execute(text("SELECT id FROM document_tags WHERE name = :name"), {"name": name}).fetchone()
            if not result:
                db.execute(text("""
                    INSERT INTO document_tags (name, color, description)
                    VALUES (:name, :color, :description)
                """), {"name": name, "color": color, "description": description})

        # Seed initial categories
        initial_categories = [
            ("Academic Research", "Academic and research documents", None, 1),
            ("Technical Documentation", "Technical and software documentation", None, 1),
            ("Business Documents", "Business and corporate documents", None, 1),
            ("Educational Materials", "Tutorials and learning materials", None, 1),
        ]

        for name, description, parent_id, level in initial_categories:
            # Check if category exists
            result = db.execute(text("SELECT id FROM document_categories WHERE name = :name"), {"name": name}).fetchone()
            if not result:
                db.execute(text("""
                    INSERT INTO document_categories (name, description, parent_category_id, level)
                    VALUES (:name, :description, :parent_id, :level)
                """), {"name": name, "description": description, "parent_id": parent_id, "level": level})

        db.commit()
        print("✅ Initial classification data seeded")

    except Exception as e:
        db.rollback()
        print(f"❌ Error seeding data: {e}")
        raise
    finally:
        db.close()

def verify_migration(engine):
    """Verify that the migration was successful."""
    print("Verifying migration...")

    with engine.connect() as conn:
        # Check that all tables exist
        tables_to_check = [
            'documents', 'document_chapters', 'document_chunks', 'document_embeddings',
            'topics', 'document_topics', 'document_tags', 'document_tag_assignments',
            'document_categories', 'document_category_assignments', 'processing_jobs'
        ]

        for table in tables_to_check:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = :table_name
                );
            """), {"table_name": table}).fetchone()

            if result and result[0]:
                print(f"✅ Table '{table}' exists")
            else:
                print(f"❌ Table '{table}' missing")

        # Check pgvector extension
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            );
        """)).fetchone()

        if result and result[0]:
            print("✅ pgvector extension enabled")
        else:
            print("❌ pgvector extension not found")

        print("✅ Migration verification complete")

def main():
    """Main migration function."""
    print("🚀 Starting enhanced schema migration...")
    print("=" * 50)

    try:
        # Get database connection
        database_url = get_database_url()
        print(f"Connecting to database: {database_url.replace(':***@', ':***@')}")

        engine = create_engine(database_url)

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection successful")

        # Perform migration steps
        create_enhanced_tables(engine)
        migrate_existing_data(engine)
        create_indexes(engine)
        seed_initial_data(engine)
        verify_migration(engine)

        print("=" * 50)
        print("🎉 Enhanced schema migration completed successfully!")
        print("\nNext steps:")
        print("1. Update your application code to use the new models")
        print("2. Test document processing with the enhanced pipeline")
        print("3. Verify topic classification and hierarchical features")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()