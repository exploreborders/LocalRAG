"""
Base processor class for document processing operations.
Contains shared functionality to avoid code duplication.
"""

import logging
import os
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from src.database.models import Document

logger = logging.getLogger(__name__)


class BaseProcessor:
    """Base class for document processors with shared functionality."""

    def __init__(self, db: Session):
        self.db = db

    def process_existing_documents(self) -> Dict[str, Any]:
        """
        Process documents that exist in the database but haven't been processed yet.

        This method finds documents with status 'uploaded' or 'pending' and processes them
        to create chunks and embeddings.
        """
        try:
            # Find documents that need processing
            pending_docs = (
                self.db.query(Document).filter(Document.status.in_(["uploaded", "pending"])).all()
            )

            if not pending_docs:
                logger.info("No pending documents found to process")
                return {
                    "success": True,
                    "processed": 0,
                    "message": "No pending documents",
                }

            logger.info(f"Processing {len(pending_docs)} existing documents...")

            processed = 0
            failed = 0

            for doc in pending_docs:
                try:
                    logger.info(f"Processing document: {doc.filename}")

                    # Check if file still exists
                    if not os.path.exists(doc.filepath):
                        logger.warning(f"File not found: {doc.filepath}")
                        doc.status = "error"
                        self.db.commit()
                        failed += 1
                        continue

                    # Process the document - to be implemented by subclasses
                    result = self._process_single_document(doc.filepath, doc.filename)

                    if result.get("success", False):
                        processed += 1
                        logger.info(f"Successfully processed: {doc.filename}")
                    else:
                        failed += 1
                        logger.error(
                            f"Failed to process {doc.filename}: {result.get('error', 'Unknown error')}"
                        )

                except Exception as e:
                    logger.error(f"Error processing {doc.filename}: {e}")
                    failed += 1

            self.db.commit()

            return {
                "success": True,
                "processed": processed,
                "failed": failed,
                "message": f"Processed {processed} documents, {failed} failed",
            }

        except Exception as e:
            logger.error(f"Error in process_existing_documents: {e}")
            self.db.rollback()
            return {"success": False, "error": str(e)}

    def _process_single_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Process a single document. To be implemented by subclasses.

        Args:
            file_path: Path to the document file
            filename: Name of the document

        Returns:
            Dictionary with processing results
        """
        raise NotImplementedError("Subclasses must implement _process_single_document")

    def _validate_file_path(self, file_path: str) -> bool:
        """
        Validate file path to prevent directory traversal attacks.

        Args:
            file_path: Path to validate

        Returns:
            True if path is safe, False otherwise
        """
        try:
            # Normalize the path
            normalized_path = os.path.normpath(file_path)

            # Check for directory traversal attempts
            if ".." in normalized_path:
                logger.warning(f"Directory traversal attempt detected: {file_path}")
                return False

            # Check if path is absolute (should be relative to data directory)
            if os.path.isabs(normalized_path):
                logger.warning(f"Absolute path not allowed: {file_path}")
                return False

            # Check file exists and is a regular file
            if not os.path.exists(normalized_path) or not os.path.isfile(normalized_path):
                logger.warning(f"File does not exist or is not a regular file: {file_path}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating file path {file_path}: {e}")
            return False
