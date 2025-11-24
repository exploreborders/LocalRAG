#!/usr/bin/env python3
"""
Test the full upload process with advanced processing
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.document_manager import UploadProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_upload_with_advanced_processing():
    """Test uploading a PDF with automatic advanced processing detection"""
    pdf_path = "data/velpTEC_K4.0016_Deep Learning_Theorieskript.pdf"

    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        return

    logger.info("Testing upload process with automatic advanced processing...")

    try:
        processor = UploadProcessor()

        # Test the process_single_file method which should auto-detect scanned PDFs
        logger.info("Calling process_single_file (should auto-detect scanned PDF)...")
        result = processor.process_single_file(
            pdf_path, filename="test_deep_learning.pdf"
        )

        logger.info(f"Processing result: success={result.get('success', False)}")
        logger.info(
            f"Advanced processing used: {result.get('advanced_processing', 'unknown')}"
        )
        logger.info(f"Chapters detected: {result.get('chapters_detected', 0)}")
        logger.info(f"Chunks created: {result.get('chunks_created', 0)}")
        logger.info(f"Topics identified: {result.get('topics_identified', 0)}")

        if result.get("success"):
            logger.info("✅ Upload test successful!")
            return True
        else:
            logger.error(
                f"❌ Upload test failed: {result.get('error', 'Unknown error')}"
            )
            return False

    except Exception as e:
        logger.error(f"Upload test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_upload_with_advanced_processing()
    sys.exit(0 if success else 1)
