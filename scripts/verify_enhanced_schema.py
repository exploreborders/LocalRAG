#!/usr/bin/env python3
"""
Verify that the enhanced knowledge graph schema is properly set up.
"""

import os
import sys

def check_enhanced_schema():
    """Check if enhanced schema tables exist."""
    print("🔍 Checking Enhanced Knowledge Graph Schema")
    print("=" * 50)

    # Try to import psycopg2 for direct database access
    try:
        import psycopg2
    except ImportError:
        print("❌ psycopg2 not available - cannot verify database schema")
        print("💡 To verify manually, check if these tables exist:")
        print("   - tag_relationships")
        print("   - tag_category_relationships")
        print("   - Enhanced document_categories with ai_confidence column")
        print("   - Enhanced documents with kg_context_expansion column")
        return False

    # Database connection
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", 5432),
            dbname=os.getenv("POSTGRES_DB", "rag_system"),
            user=os.getenv("POSTGRES_USER", "christianhein"),
            password=os.getenv("POSTGRES_PASSWORD", "")
        )
        cursor = conn.cursor()
    except Exception as e:
        print(f"❌ Cannot connect to database: {e}")
        print("💡 Make sure PostgreSQL is running and credentials are correct")
        return False

    try:
        # Check for enhanced tables
        required_tables = [
            'tag_relationships',
            'tag_category_relationships'
        ]

        existing_tables = []
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        all_tables = [row[0] for row in cursor.fetchall()]

        print("📋 Checking for enhanced schema tables:")
        for table in required_tables:
            if table in all_tables:
                existing_tables.append(table)
                print(f"  ✅ {table} - EXISTS")
            else:
                print(f"  ❌ {table} - MISSING")

        # Check for enhanced columns
        enhanced_columns = {
            'document_categories': ['ai_confidence', 'ai_suggested', 'alternative_categories'],
            'documents': ['kg_context_expansion', 'kg_relationship_score', 'kg_last_updated']
        }

        print("\n📋 Checking for enhanced columns:")
        for table, columns in enhanced_columns.items():
            if table in all_tables:
                cursor.execute(f"""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = '{table}' AND table_schema = 'public'
                """)
                existing_cols = [row[0] for row in cursor.fetchall()]

                for col in columns:
                    if col in existing_cols:
                        print(f"  ✅ {table}.{col} - EXISTS")
                    else:
                        print(f"  ❌ {table}.{col} - MISSING")
            else:
                print(f"  ⚠️  Table {table} doesn't exist - cannot check columns")

        # Check for indexes
        required_indexes = [
            'idx_tag_relationships_parent',
            'idx_tag_relationships_child',
            'idx_tag_cat_relationships_tag',
            'idx_tag_cat_relationships_cat',
            'idx_documents_kg_score',
            'idx_categories_ai_confidence'
        ]

        print("\n📋 Checking for performance indexes:")
        cursor.execute("""
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
        """)
        existing_indexes = [row[0] for row in cursor.fetchall()]

        for idx in required_indexes:
            if idx in existing_indexes:
                print(f"  ✅ {idx} - EXISTS")
            else:
                print(f"  ❌ {idx} - MISSING")

        # Overall assessment
        print("\n🎯 SCHEMA READINESS ASSESSMENT:")

        tables_ready = len(existing_tables) == len(required_tables)
        if tables_ready:
            print("  ✅ All required tables exist")
        else:
            print(f"  ❌ {len(required_tables) - len(existing_tables)} required tables missing")

        # Check if we have the basic schema at least
        basic_tables_exist = all(table in all_tables for table in [
            'documents', 'document_tags', 'document_categories',
            'document_tag_assignments', 'document_category_assignments'
        ])

        if basic_tables_exist:
            print("  ✅ Basic tagging and categorization schema exists")
        else:
            print("  ❌ Basic schema missing - run migrate_database_schema.py first")

        # Final verdict
        if tables_ready and basic_tables_exist:
            print("\n🎉 ENHANCED KNOWLEDGE GRAPH SCHEMA IS READY!")
            print("✅ Database is fully prepared for the Enhanced Architecture")
            return True
        else:
            print("\n⚠️  ENHANCED SCHEMA NEEDS SETUP")
            print("💡 Run: python3 scripts/enhance_knowledge_graph_schema.py")
            return False

    except Exception as e:
        print(f"❌ Error during schema verification: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    success = check_enhanced_schema()
    sys.exit(0 if success else 1)