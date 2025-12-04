#!/usr/bin/env python3
"""
Database schema enhancements for Knowledge Graph functionality.

Adds tag relationships table and enhances category metadata for AI-powered
categorization and relationship mapping.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sqlalchemy import create_engine, text  # noqa: E402

# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://{}:{}@{}:{}/{}".format(
        os.getenv("POSTGRES_USER", "christianhein"),
        os.getenv("POSTGRES_PASSWORD", ""),
        os.getenv("POSTGRES_HOST", "localhost"),
        os.getenv("POSTGRES_PORT", "5432"),
        os.getenv("POSTGRES_DB", "rag_system"),
    ),
)


def run_schema_enhancements():
    """Apply knowledge graph schema enhancements."""

    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        print("🔄 Applying Knowledge Graph schema enhancements...")

        # Add AI confidence and alternative categories to document_categories
        try:
            conn.execute(
                text(
                    """
                ALTER TABLE document_categories
                ADD COLUMN IF NOT EXISTS ai_confidence FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS ai_suggested BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS alternative_categories JSONB;
            """
                )
            )
            print("✅ Enhanced document_categories table")
        except Exception as e:
            print(f"⚠️ Failed to enhance document_categories: {e}")

        # Create tag_relationships table
        try:
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS tag_relationships (
                    id SERIAL PRIMARY KEY,
                    parent_tag_id INTEGER REFERENCES document_tags(id) ON DELETE CASCADE,
                    child_tag_id INTEGER REFERENCES document_tags(id) ON DELETE CASCADE,
                    relationship_type VARCHAR(50) DEFAULT 'parent_child',
                    strength FLOAT DEFAULT 1.0,
                    confidence FLOAT DEFAULT 0.5,
                    created_by VARCHAR(50) DEFAULT 'auto',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(parent_tag_id, child_tag_id, relationship_type)
                );

                CREATE INDEX IF NOT EXISTS idx_tag_relationships_parent ON tag_relationships(parent_tag_id);
                CREATE INDEX IF NOT EXISTS idx_tag_relationships_child ON tag_relationships(child_tag_id);
                CREATE INDEX IF NOT EXISTS idx_tag_relationships_type ON tag_relationships(relationship_type);
            """
                )
            )
            print("✅ Created tag_relationships table")
        except Exception as e:
            print(f"⚠️ Failed to create tag_relationships table: {e}")

        # Create tag_category_relationships table
        try:
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS tag_category_relationships (
                    id SERIAL PRIMARY KEY,
                    tag_id INTEGER REFERENCES document_tags(id) ON DELETE CASCADE,
                    category_id INTEGER REFERENCES document_categories(id) ON DELETE CASCADE,
                    relationship_type VARCHAR(50) DEFAULT 'belongs_to',
                    strength FLOAT DEFAULT 1.0,
                    evidence_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(tag_id, category_id, relationship_type)
                );

                CREATE INDEX IF NOT EXISTS idx_tag_cat_relationships_tag ON tag_category_relationships(tag_id);
                CREATE INDEX IF NOT EXISTS idx_tag_cat_relationships_cat ON tag_category_relationships(category_id);
            """
                )
            )
            print("✅ Created tag_category_relationships table")
        except Exception as e:
            print(f"⚠️ Failed to create tag_category_relationships table: {e}")

        # Add knowledge graph metadata columns to documents
        try:
            conn.execute(
                text(
                    """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS kg_context_expansion JSONB,
                ADD COLUMN IF NOT EXISTS kg_relationship_score FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS kg_last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
            """
                )
            )
            print("✅ Enhanced documents table with KG metadata")
        except Exception as e:
            print(f"⚠️ Failed to enhance documents table: {e}")

        # Create indexes for performance
        try:
            conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_documents_kg_score ON documents(kg_relationship_score);
                CREATE INDEX IF NOT EXISTS idx_categories_ai_confidence ON document_categories(ai_confidence);
                CREATE INDEX IF NOT EXISTS idx_tags_usage_count ON document_tags(usage_count);
            """
                )
            )
            print("✅ Created performance indexes")
        except Exception as e:
            print(f"⚠️ Failed to create indexes: {e}")

        conn.commit()
        print("🎉 Knowledge Graph schema enhancements completed!")


def populate_initial_relationships():
    """Populate initial tag relationships based on existing data."""

    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        print("🔄 Populating initial tag relationships...")

        try:
            # Insert co-occurrence based relationships
            conn.execute(
                text(
                    """
                INSERT INTO tag_relationships (parent_tag_id, child_tag_id, relationship_type, strength, confidence, created_by)
                SELECT DISTINCT
                    t1.id as parent_tag_id,
                    t2.id as child_tag_id,
                    'co_occurs_with' as relationship_type,
                    LEAST(1.0, COUNT(*) / 5.0) as strength,
                    0.6 as confidence,
                    'auto' as created_by
                FROM document_tag_assignments ta1
                JOIN document_tag_assignments ta2 ON ta1.document_id = ta2.document_id
                JOIN document_tags t1 ON ta1.tag_id = t1.id
                JOIN document_tags t2 ON ta2.tag_id = t2.id
                WHERE t1.id < t2.id  -- Avoid duplicates
                  AND t1.id != t2.id
                GROUP BY t1.id, t2.id, t1.name, t2.name
                HAVING COUNT(*) >= 2  -- At least 2 co-occurrences
                ON CONFLICT (parent_tag_id, child_tag_id, relationship_type) DO NOTHING;
            """
                )
            )
            print("✅ Populated co-occurrence relationships")
        except Exception as e:
            print(f"⚠️ Failed to populate co-occurrence relationships: {e}")

        try:
            # Insert tag-category relationships
            conn.execute(
                text(
                    """
                INSERT INTO tag_category_relationships (tag_id, category_id, relationship_type, strength, evidence_count)
                SELECT
                    dt.id as tag_id,
                    dc.id as category_id,
                    'belongs_to' as relationship_type,
                    LEAST(1.0, COUNT(*) / 3.0) as strength,
                    COUNT(*) as evidence_count
                FROM document_tag_assignments dta
                JOIN document_category_assignments dca ON dta.document_id = dca.document_id
                JOIN document_tags dt ON dta.tag_id = dt.id
                JOIN document_categories dc ON dca.category_id = dc.id
                GROUP BY dt.id, dc.id, dt.name, dc.name
                HAVING COUNT(*) >= 1
                ON CONFLICT (tag_id, category_id, relationship_type) DO NOTHING;
            """
                )
            )
            print("✅ Populated tag-category relationships")
        except Exception as e:
            print(f"⚠️ Failed to populate tag-category relationships: {e}")

        conn.commit()
        print("🎉 Initial relationship population completed!")


if __name__ == "__main__":
    print("🤖 Knowledge Graph Schema Enhancement Tool")
    print("=" * 50)

    if len(sys.argv) > 1 and sys.argv[1] == "--populate":
        populate_initial_relationships()
    else:
        run_schema_enhancements()
        print("\n💡 Tip: Run with --populate to also populate initial relationships")
