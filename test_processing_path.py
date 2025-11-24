#!/usr/bin/env python3
"""
Quick test to check which processing path is used
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.document_manager import DocumentProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_processing_path():
    """Test which processing path is used"""
    pdf_path = "data/velpTEC_K4.0016_Deep Learning_Theorieskript.pdf"

    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        return

    logger.info("Testing processing path selection...")

    # Test with use_advanced_processing=True
    logger.info("Testing with use_advanced_processing=True")
    processor = DocumentProcessor()

    # This should trigger advanced processing
    try:
        result = processor.process_document(pdf_path, use_advanced_processing=True)
        logger.info(
            f"Advanced processing result: success={result.get('success', False)}"
        )
        if result.get("success"):
            logger.info("✅ Advanced processing works!")
        else:
            logger.error(
                f"❌ Advanced processing failed: {result.get('error', 'Unknown error')}"
            )
    except Exception as e:
        logger.error(f"Advanced processing exception: {e}")

    # Test with use_advanced_processing=False
    logger.info("Testing with use_advanced_processing=False")
    try:
        result = processor.process_document(pdf_path, use_advanced_processing=False)
        logger.info(
            f"Standard processing result: success={result.get('success', False)}"
        )
        if result.get("success"):
            logger.info("✅ Standard processing works!")
        else:
            logger.error(
                f"❌ Standard processing failed: {result.get('error', 'Unknown error')}"
            )
    except Exception as e:
        logger.error(f"Standard processing exception: {e}")


if __name__ == "__main__":
    test_processing_path()
