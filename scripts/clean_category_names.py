#!/usr/bin/env python3
"""
Clean up existing category names that contain AI-generated prefixes.

This script cleans category names that were created with full AI responses
instead of clean category names.
"""

import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.ai_enrichment import AIEnrichmentService
from src.database.models import DocumentCategory, SessionLocal


def clean_existing_categories():
    """Clean existing category names in the database."""
    # Create a dummy AI service just to use the cleaning method
    ai_service = AIEnrichmentService.__new__(AIEnrichmentService)
    ai_service.llm_client = None

    db = SessionLocal()

    try:
        categories = db.query(DocumentCategory).all()
        cleaned_count = 0

        # Manual mapping for known messy categories
        manual_cleaning = {
            "Based On The Tags": "Document Analysis",
            "Two Relevant Subcategories For General Could Be Document Analysis": "Document Analysis",
            "Based On The Provided Information": "Content Analysis",
            "I Would Suggest The Following Relevant Subcategories For The General Category Document Analysis": "Document Analysis",
            "Based On The Provided Document": "Content Analysis",
        }

        for category in categories:
            original_name = category.name

            # First try manual mapping
            cleaned_name = manual_cleaning.get(original_name)

            # If no manual mapping, try AI cleaning
            if not cleaned_name:
                cleaned_name = ai_service._clean_category_name(original_name)

            # Only update if the name actually changed and is not empty
            if cleaned_name and cleaned_name != original_name and len(cleaned_name) > 1:
                print(f'Cleaning: "{original_name}" -> "{cleaned_name}"')
                category.name = cleaned_name
                cleaned_count += 1
            elif not cleaned_name and len(original_name.split()) > 3:  # Very long names
                # For very long names that couldn't be cleaned, use a generic name
                generic_name = (
                    "Document Analysis"
                    if "document" in original_name.lower()
                    else "Content Analysis"
                )
                print(f'Replacing long name: "{original_name}" -> "{generic_name}"')
                category.name = generic_name
                cleaned_count += 1

        if cleaned_count > 0:
            db.commit()
            print(f"\n✅ Cleaned {cleaned_count} category names")
        else:
            print("\nℹ️ No categories needed cleaning")

    except Exception as e:
        print(f"❌ Error cleaning categories: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    print("🧹 Cleaning existing category names...")
    clean_existing_categories()
    print("✅ Category cleanup complete!")
