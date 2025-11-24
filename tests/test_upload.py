#!/usr/bin/env python3
"""
Basic test for upload functionality.
"""

import sys
import os
from pathlib import Path
from io import BytesIO

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.core.document_manager import UploadProcessor


class MockUploadedFile:
    """Mock Streamlit uploaded file for testing."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getbuffer(self):
        return self._content


def test_upload_files():
    """Test the upload_files method with mock files."""
    print("🧪 Testing upload_files functionality...")

    # Create mock uploaded files
    mock_files = [
        MockUploadedFile(
            "test_doc1.txt", b"This is a test document for upload testing."
        ),
        MockUploadedFile(
            "test_doc2.txt", b"Another test document with different content."
        ),
    ]

    # Initialize upload processor
    processor = UploadProcessor()

    # Test upload
    try:
        result = processor.upload_files(
            uploaded_files=mock_files,
            data_dir="test_data",
            use_parallel=False,  # Sequential for testing
        )

        print("✅ Upload test completed!")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Successful uploads: {result.get('successful_uploads', 0)}")
        print(f"   Failed uploads: {result.get('failed_uploads', 0)}")
        print(f"   Total chunks: {result.get('total_chunks', 0)}")
        print(f"   Total chapters: {result.get('total_chapters', 0)}")

        if result.get("success"):
            print("🎉 Upload functionality is working!")
            return True
        else:
            print(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False


if __name__ == "__main__":
    success = test_upload_files()
    sys.exit(0 if success else 1)
