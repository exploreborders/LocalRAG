#!/usr/bin/env python3
"""
Script to populate document metadata using Docling extraction (fast alternative to LLM).

This script extracts metadata directly from document structure and properties using Docling,
providing much faster metadata population than LLM-based approaches.

Extracts:
- Title, Author, Creation/Modification dates
- Page count, word count, reading time estimates
- Language detection, document type
- Table and image counts
- Document structure analysis
"""

import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.document_processor import DocumentProcessor
    from src.database.models import SessionLocal, Document
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc.document import DoclingDocument
    from langdetect import LangDetectException
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


class DoclingMetadataExtractor:
    """
    Extract metadata from documents using Docling (fast alternative to LLM).
    """

    def __init__(self):
        self.doc_converter = DocumentConverter()

    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from a document using Docling.

        Args:
            filepath: Path to the document file

        Returns:
            Dict containing extracted metadata
        """
        try:
            # Convert document with Docling
            result = self.doc_converter.convert(filepath)
            doc = result.document

            metadata = {
                'title': self._extract_title(doc, filepath),
                'author': self._extract_author_from_docling(doc),
                'creation_date': self._extract_creation_date(doc),
                'modification_date': self._extract_modification_date(doc),
                'page_count': doc.num_pages() if hasattr(doc, 'num_pages') and callable(getattr(doc, 'num_pages')) else None,
                'word_count': self._calculate_word_count(doc),
                'reading_time': None,  # Will be calculated from word_count
                'language': self._detect_language(doc),
                'document_type': self._get_document_type(filepath),
                'table_count': len(doc.tables) if hasattr(doc, 'tables') else 0,
                'image_count': len(doc.pictures) if hasattr(doc, 'pictures') else 0,
                'has_tables': len(doc.tables) > 0 if hasattr(doc, 'tables') else False,
                'has_images': len(doc.pictures) > 0 if hasattr(doc, 'pictures') else False,
                'structure_info': self._analyze_document_structure(doc),
                'extracted_at': datetime.now().isoformat(),
                'extraction_method': 'docling'
            }

            # Calculate reading time (200 words per minute)
            if metadata['word_count']:
                metadata['reading_time'] = max(1, round(metadata['word_count'] / 200))

            return metadata

        except Exception as e:
            print(f"❌ Error extracting metadata from {filepath}: {e}")
            return {
                'error': str(e),
                'extraction_method': 'failed'
            }

    def _extract_title(self, doc: DoclingDocument, filepath: str) -> Optional[str]:
        """
        Extract document title from various sources.
        """
        # Try document origin metadata
        if hasattr(doc, 'origin') and doc.origin:
            origin_dict = doc.origin.model_dump()
            if 'title' in origin_dict and origin_dict['title']:
                return origin_dict['title'].strip()

        # Try to find title from document structure
        if hasattr(doc, 'texts'):
            # Look for title items
            for text_item in doc.texts:
                if hasattr(text_item, 'label') and text_item.label == 'title':
                    if hasattr(text_item, 'text') and text_item.text:
                        return text_item.text.strip()

            # Look for first section header as fallback
            for text_item in doc.texts:
                if hasattr(text_item, 'label') and text_item.label == 'section_header':
                    if hasattr(text_item, 'text') and text_item.text:
                        return text_item.text.strip()

        # Fallback to filename without extension
        filename = Path(filepath).stem
        return filename.replace('_', ' ').replace('-', ' ').title()

    def _extract_author_from_docling(self, doc: DoclingDocument) -> Optional[str]:
        """
        Extract author information from Docling document.
        """
        # Try document origin metadata
        if hasattr(doc, 'origin') and doc.origin:
            origin_dict = doc.origin.model_dump()
            if 'author' in origin_dict and origin_dict['author']:
                return origin_dict['author'].strip()
            if 'creator' in origin_dict and origin_dict['creator']:
                return origin_dict['creator'].strip()

        # Try to find author in document text
        if hasattr(doc, 'texts'):
            full_text = ' '.join([item.text for item in doc.texts if hasattr(item, 'text')])
            return self._extract_author_from_text(full_text)

        return None

    def _extract_author_from_text(self, text: str) -> Optional[str]:
        """
        Extract author from text using pattern matching.
        """
        author_patterns = [
            r'Author[:\-]?\s*([^\n\r]{1,100})',
            r'By[:\-]?\s*([^\n\r]{1,100})',
            r'Written by[:\-]?\s*([^\n\r]{1,100})',
            r'Created by[:\-]?\s*([^\n\r]{1,100})',
        ]

        lines = text.split('\n')[:20]  # Check first 20 lines

        for line in lines:
            line = line.strip()
            if not line:
                continue

            for pattern in author_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    author_candidate = match.group(1).strip()
                    if self._is_valid_author(author_candidate):
                        return author_candidate

        return None

    def _is_valid_author(self, author_candidate: str) -> bool:
        """
        Validate if a string looks like a valid author name.
        """
        if not author_candidate or len(author_candidate) < 2 or len(author_candidate) > 100:
            return False

        exclude_patterns = [
            r'^\d+$',  # Just numbers
            r'^[^\w\s]+$',  # Just punctuation
            r'(?i)(table|figure|page|chapter|section|volume)',  # Document structure words
            r'(?i)(http|www|\.com|\.org|\.edu)',  # URLs
            r'^.{1,3}$',  # Very short strings
        ]

        for pattern in exclude_patterns:
            if re.search(pattern, author_candidate):
                return False

        has_letters = bool(re.search(r'[a-zA-Z]', author_candidate))
        number_ratio = len(re.findall(r'\d', author_candidate)) / len(author_candidate)

        return has_letters and number_ratio < 0.5

    def _extract_creation_date(self, doc: DoclingDocument) -> Optional[str]:
        """
        Extract document creation date.
        """
        if hasattr(doc, 'origin') and doc.origin:
            origin_dict = doc.origin.model_dump()
            if 'creation_date' in origin_dict and origin_dict['creation_date']:
                return str(origin_dict['creation_date'])
            if 'created' in origin_dict and origin_dict['created']:
                return str(origin_dict['created'])

        return None

    def _extract_modification_date(self, doc: DoclingDocument) -> Optional[str]:
        """
        Extract document modification date.
        """
        if hasattr(doc, 'origin') and doc.origin:
            origin_dict = doc.origin.model_dump()
            if 'modification_date' in origin_dict and origin_dict['modification_date']:
                return str(origin_dict['modification_date'])
            if 'modified' in origin_dict and origin_dict['modified']:
                return str(origin_dict['modified'])

        return None

    def _calculate_word_count(self, doc: DoclingDocument) -> Optional[int]:
        """
        Calculate total word count in the document.
        """
        if hasattr(doc, 'texts'):
            full_text = ' '.join([item.text for item in doc.texts if hasattr(item, 'text')])
            words = re.findall(r'\w+', full_text)
            return len(words)

        return None

    def _detect_language(self, doc: DoclingDocument) -> Optional[str]:
        """
        Detect document language.
        """
        try:
            from langdetect import detect, LangDetectException

            if hasattr(doc, 'texts') and doc.texts:
                # Use first substantial text block for language detection
                sample_text = ''
                for item in doc.texts:
                    if hasattr(item, 'text') and item.text:
                        sample_text += item.text + ' '
                        if len(sample_text) > 500:  # Use first 500 chars
                            break

                if sample_text.strip():
                    return detect(sample_text[:1000])  # Limit to 1000 chars for speed

        except Exception:
            pass

        return None

    def _get_document_type(self, filepath: str) -> str:
        """
        Determine document type from file extension.
        """
        ext = Path(filepath).suffix.lower()
        type_map = {
            '.pdf': 'PDF Document',
            '.docx': 'Word Document',
            '.doc': 'Word Document',
            '.pptx': 'PowerPoint Presentation',
            '.ppt': 'PowerPoint Presentation',
            '.xlsx': 'Excel Spreadsheet',
            '.xls': 'Excel Spreadsheet',
            '.txt': 'Text Document',
            '.md': 'Markdown Document',
            '.html': 'HTML Document',
        }
        return type_map.get(ext, 'Unknown Document Type')

    def _analyze_document_structure(self, doc: DoclingDocument) -> Dict[str, Any]:
        """
        Analyze document structure and provide summary.
        """
        structure = {
            'total_text_items': len(doc.texts) if hasattr(doc, 'texts') else 0,
            'total_tables': len(doc.tables) if hasattr(doc, 'tables') else 0,
            'total_images': len(doc.pictures) if hasattr(doc, 'pictures') else 0,
            'has_headers': False,
            'has_lists': False,
            'has_tables': len(doc.tables) > 0 if hasattr(doc, 'tables') else False,
            'has_images': len(doc.pictures) > 0 if hasattr(doc, 'pictures') else False,
        }

        # Analyze text items for structure
        if hasattr(doc, 'texts'):
            for item in doc.texts:
                if hasattr(item, 'label'):
                    if item.label == 'section_header':
                        structure['has_headers'] = True
                    elif item.label in ['list_item', 'ordered_list', 'unordered_list']:
                        structure['has_lists'] = True

        return structure


def populate_docling_metadata():
    """
    Populate metadata for all existing documents using Docling extraction.
    """
    print("🔍 Starting Docling metadata extraction...")

    extractor = DoclingMetadataExtractor()
    db = SessionLocal()

    try:
        # Get all documents
        documents = db.query(Document).all()
        total_docs = len(documents)

        print(f"📊 Found {total_docs} documents to process")
        print(f"📄 First document: {documents[0].filename if documents else 'None'}")

        successful = 0
        failed = 0

        for i, doc in enumerate(documents, 1):  # Process all documents
            if not os.path.exists(doc.filepath):
                print(f"⚠️ [{i}/{total_docs}] File not found: {doc.filepath}")
                failed += 1
                continue

            # Skip very large files for now (>5MB)
            file_size = os.path.getsize(doc.filepath)
            if file_size > 5 * 1024 * 1024:  # 5MB
                print(f"⚠️ [{i}/{total_docs}] Skipping large file: {doc.filename} ({file_size//1024//1024}MB)")
                failed += 1
                continue

            print(f"🔄 [{i}/{total_docs}] Processing {doc.filename[:50]}...")

            try:
                # Extract metadata using Docling
                metadata = extractor.extract_metadata(doc.filepath)
                print(f"  📊 Extracted metadata for {doc.filename}")

                if 'error' not in metadata:
                    print(f"  📊 Metadata keys: {list(metadata.keys())}")
                    print(f"  📄 Title: {metadata.get('title')}, Existing title: {(doc.custom_fields or {}).get('title')}")
                    # Update document with extracted metadata
                    if metadata.get('title') and not (doc.custom_fields or {}).get('title'):
                        # Store title in custom_fields if not already set
                        custom_fields = doc.custom_fields or {}
                        custom_fields['title'] = metadata['title']
                        doc.custom_fields = custom_fields

                    if metadata.get('author') and not doc.author:
                        doc.author = metadata['author']

                    if metadata.get('reading_time') and not doc.reading_time:
                        doc.reading_time = metadata['reading_time']

                    # Store comprehensive metadata in custom_fields
                    current_custom_fields = doc.custom_fields or {}
                    new_custom_fields = dict(current_custom_fields)  # Create a new dict
                    new_custom_fields.update({
                        'docling_metadata': metadata,
                        'word_count': metadata.get('word_count'),
                        'page_count': metadata.get('page_count'),
                        'table_count': metadata.get('table_count'),
                        'image_count': metadata.get('image_count'),
                        'document_type': metadata.get('document_type'),
                        'language_detected': metadata.get('language'),
                        'structure_info': metadata.get('structure_info'),
                        'extraction_method': 'docling',
                        'extracted_at': metadata.get('extracted_at')
                    })
                    doc.custom_fields = new_custom_fields
                    print(f"  📝 Updated custom_fields keys: {list((doc.custom_fields or {}).keys())}")

                    successful += 1
                    print(f"  ✅ Extracted: {metadata.get('word_count', 0)} words, "
                          f"{metadata.get('page_count', 0)} pages, "
                          f"{metadata.get('table_count', 0)} tables")

                    # Commit after each successful extraction
                    print(f"  🔄 Attempting to commit for {doc.filename}")
                    print(f"  📊 Before commit: custom_fields keys = {list((doc.custom_fields or {}).keys())}")
                    try:
                        db.flush()
                        print(f"  🔄 Flush successful")
                        db.commit()
                        print(f"  💾 Commit successful for {doc.filename}")
                        # Verify the commit worked
                        db.refresh(doc)
                        custom_fields = doc.custom_fields or {}
                        print(f"  📊 After refresh: custom_fields keys = {list(custom_fields.keys())}")
                        if 'extraction_method' in custom_fields:
                            print(f"  ✅ Verified: extraction_method = {custom_fields['extraction_method']}")
                        else:
                            print(f"  ❌ Verification failed: extraction_method not found in custom_fields")
                    except Exception as commit_error:
                        print(f"  ❌ Commit failed: {commit_error}")
                        import traceback
                        traceback.print_exc()
                        db.rollback()

                else:
                     failed += 1
                     print(f"  ❌ Failed: {metadata['error']}")

            except Exception as e:
                 print(f"  ❌ Exception: {e}")
                 failed += 1
                 db.rollback()

        # Commit all changes
        db.commit()
        print("\n💾 Changes committed to database")
        print("\n📈 Docling Metadata Extraction Summary:")
        print(f"  ✅ Successfully processed: {successful}")
        print(f"  ❌ Failed: {failed}")
        print(f"  📊 Total processed: {successful + failed}")

        if successful > 0:
            print("\n🎉 Documents now have rich metadata extracted using Docling!")
            print("   - Word counts, page counts, reading time estimates")
            print("   - Table and image counts")
            print("   - Document structure analysis")
            print("   - Language detection")
            print("   - Author and title extraction")

    except Exception as e:
        print(f"❌ Error during metadata extraction: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def show_sample_docling_metadata():
    """
    Show a sample of documents with Docling-extracted metadata.
    """
    print("\n📋 Sample of Docling-Enriched Documents:")
    print("-" * 60)

    db = SessionLocal()
    try:
        # Get a few documents with Docling metadata
        docs_with_metadata = db.query(Document).filter(
            Document.custom_fields.op('->>')('extraction_method') == 'docling'
        ).limit(3).all()

        if not docs_with_metadata:
            print("No documents with Docling metadata found yet.")
            return

        for doc in docs_with_metadata:
            print(f"📄 {doc.filename}")
            print(f"   Author: {doc.author or 'Not extracted'}")
            print(f"   Reading Time: {doc.reading_time or 'Not estimated'} minutes")

            if doc.custom_fields:
                docling_meta = doc.custom_fields.get('docling_metadata', {})
                if docling_meta:
                    print(f"   Title: {docling_meta.get('title', 'N/A')}")
                    print(f"   Pages: {docling_meta.get('page_count', 'N/A')}")
                    print(f"   Words: {docling_meta.get('word_count', 'N/A')}")
                    print(f"   Tables: {docling_meta.get('table_count', 0)}")
                    print(f"   Images: {docling_meta.get('image_count', 0)}")
                    print(f"   Language: {docling_meta.get('language', 'N/A')}")
                    print(f"   Type: {docling_meta.get('document_type', 'N/A')}")
                else:
                    print(f"   Word Count: {doc.custom_fields.get('word_count', 'N/A')}")
                    print(f"   Page Count: {doc.custom_fields.get('page_count', 'N/A')}")
            print()

    except Exception as e:
        print(f"❌ Error showing samples: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    print("🚀 Docling Metadata Population Script")
    print("=" * 50)
    print("⚡ Fast metadata extraction using Docling (no LLM required)")

    try:
        populate_docling_metadata()
        show_sample_docling_metadata()

        print("\n✨ Script completed!")
        print("💡 Docling metadata extraction is much faster than LLM-based approaches!")
        print("   Use this for bulk metadata population, then use AI enrichment for summaries.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()