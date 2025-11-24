#!/usr/bin/env python3
"""
Simple test to check PDF content extraction.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from data.loader import split_documents
    from langdetect import detect, LangDetectException

    pdf_path = "data/velpTEC_K4.0053_2.0_AI-Development_Exkurs_PyTorch Geometric und Open3D_01.E.01.pdf"

    print("Testing PDF content extraction...")
    content = split_documents([pdf_path])
    print(f"Content length: {len(content)}")
    print(f"First 1000 chars:\n{content[:1000]}")
    print("\n" + "=" * 50 + "\n")
    print(f"Last 1000 chars:\n{content[-1000:]}")

    # Test language detection
    print("\nTesting language detection...")
    try:
        sample = content[1000:3000] if len(content) > 3000 else content[:1000]
        lang = detect(sample)
        print(f"Detected language: {lang}")
        print(f"Sample text: {sample[:200]}...")
    except Exception as e:
        print(f"Language detection failed: {e}")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
