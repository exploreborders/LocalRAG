#!/usr/bin/env python3
"""
Run the database schema setup.
"""

import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def run_schema():
    """Run the schema.sql file."""
    # Database connection
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", 5432),
        dbname=os.getenv("POSTGRES_DB", "rag_system"),
        user=os.getenv("POSTGRES_USER", "christianhein"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )

    cursor = conn.cursor()

    # Read and execute schema.sql
    with open("src/database/schema.sql", "r") as f:
        schema_sql = f.read()

    try:
        cursor.execute(schema_sql)
        conn.commit()
        print("✅ Database schema created successfully")
    except Exception as e:
        print(f"❌ Failed to create schema: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    run_schema()
