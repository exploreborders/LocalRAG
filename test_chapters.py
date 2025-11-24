#!/usr/bin/env python3
"""
Test chapter detection with the German PDF.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from core.document_manager import DocumentProcessor

    pdf_path = "data/velpTEC_K4.0053_2.0_AI-Development_Exkurs_PyTorch Geometric und Open3D_01.E.01.pdf"

    print("🧪 Testing chapter detection...")

    # Create processor
    processor = DocumentProcessor()

    # Test chapter detection
    from data.loader import split_documents

    content = split_documents([pdf_path])

    print(f"Content length: {len(content)} characters")

    # Detect chapters
    chapters = processor._detect_all_chapters(content)
    print(f"✅ Detected {len(chapters)} chapters:")

    for i, chapter in enumerate(chapters, 1):
        print(f"  {i}. {chapter['title']} (path: {chapter['path']})")

    print("\n🎉 Chapter detection test completed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
