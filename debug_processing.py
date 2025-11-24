#!/usr/bin/env python3
"""
Debug script for chapter detection and language detection issues.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def debug_document_processing():
    """Debug document processing for the German PDF."""
    pdf_path = "data/velpTEC_K4.0053_2.0_AI-Development_Exkurs_PyTorch Geometric und Open3D_01.E.01.pdf"

    print("🔍 Debugging document processing...")
    print(f"PDF path: {pdf_path}")

    # Step 1: Extract content
    print("\n1. Extracting content with split_documents...")
    try:
        from data.loader import split_documents

        content = split_documents([pdf_path])
        print(f"✅ Content extracted, length: {len(content)} characters")
        print(f"First 500 chars: {content[:500]}")
        print(f"Last 500 chars: {content[-500:]}")
    except Exception as e:
        print(f"❌ Failed to extract content: {e}")
        return

    # Step 2: Language detection
    print("\n2. Testing language detection...")
    try:
        from langdetect import detect, LangDetectException

        # Test with different samples
        samples = [
            content[:1000],
            content[1000:3000],
            content[len(content) // 2 : len(content) // 2 + 2000],
            content[-2000:],
        ]

        for i, sample in enumerate(samples, 1):
            try:
                lang = detect(sample)
                print(f"Sample {i} ({len(sample)} chars): Detected language = {lang}")
                print(f"Sample {i} preview: {sample[:100]}...")
            except LangDetectException as e:
                print(f"Sample {i}: Language detection failed: {e}")

    except Exception as e:
        print(f"❌ Language detection test failed: {e}")

    # Step 3: Chapter detection
    print("\n3. Testing chapter detection...")

    # Create a processor instance
    from core.document_manager import DocumentProcessor

    processor = DocumentProcessor()

    # Test the _detect_chapter_info method on different parts
    test_chunks = [
        content[:2000],  # First chunk
        content[len(content) // 2 : len(content) // 2 + 2000],  # Middle chunk
        content[-2000:],  # Last chunk
    ]

    for i, chunk in enumerate(test_chunks, 1):
        print(f"\nTesting chunk {i} ({len(chunk)} chars):")
        print(f"Chunk preview: {chunk[:200]}...")

        try:
            title, path = processor._detect_chapter_info(chunk)
            if title:
                print(f"✅ Detected chapter: '{title}' (path: {path})")
            else:
                print("❌ No chapter detected in this chunk")
        except Exception as e:
            print(f"❌ Chapter detection failed: {e}")

    # Step 4: Check for ## headers in content
    print("\n4. Checking for markdown headers in content...")
    lines = content.split("\n")
    headers_found = []

    for i, line in enumerate(lines):
        if line.strip().startswith("##"):
            headers_found.append((i, line.strip()[:100]))

    if headers_found:
        print(f"✅ Found {len(headers_found)} ## headers:")
        for line_num, header in headers_found[:5]:  # Show first 5
            print(f"  Line {line_num}: {header}")
    else:
        print("❌ No ## headers found in content")

    # Step 5: Check for table format
    print("\n5. Checking for table format in content...")
    import re

    table_matches = re.findall(r"\|\s*(\d+(?:\.\d+)*)\s*\|\s*([^\|]+?)\s*\|", content)
    if table_matches:
        print(f"✅ Found {len(table_matches)} table entries:")
        for number, title in table_matches[:5]:  # Show first 5
            print(f"  {number}: {title.strip()}")
    else:
        print("❌ No table format found")


if __name__ == "__main__":
    debug_document_processing()
