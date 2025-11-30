"""
Caption-aware document processor for enhanced chunking.

This module provides caption-centric chunking that preserves document structure
by keeping captions and their associated content together during text splitting.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from docling_core.types.doc.document import DoclingDocument
from langchain_core.documents import Document as LangchainDocument


class CaptionAwareProcessor:
    """
    Processor for caption-aware document chunking.

    Extracts captions and their associated content from Docling documents,
    then creates chunks that preserve the relationship between captions
    and their explanatory content.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the caption-aware processor.

        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_document_structure(self, docling_document: DoclingDocument) -> Dict[str, Any]:
        """
        Extract structured content from a Docling document.

        Args:
            docling_document: Docling document object

        Returns:
            Dict containing structured document elements
        """
        structure = {"full_text": "", "captions": [], "sections": []}

        # Get the full markdown content
        structure["full_text"] = docling_document.export_to_markdown()

        # Identify captions from the markdown text
        structure["captions"] = self._identify_captions_from_markdown(structure["full_text"])

        return structure

    def _identify_captions_from_markdown(self, markdown_text: str) -> List[Dict[str, Any]]:
        """
        Identify captions from markdown text content.

        Args:
            markdown_text: Full document content in markdown format

        Returns:
            List of identified captions with their positions
        """
        captions = []

        # Split into lines for processing
        lines = markdown_text.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()

            # Common caption patterns in markdown
            caption_patterns = [
                r"^(?:Table|Figure|Chart|Diagram|Fig\.)\s+\d+\.?\s*[:\-]?\s*(.+)$",  # Table 1: Description
                r"^(?:Source|Note|Caption)[:\-]?\s*(.+)$",  # Source: Description
                r"^\*\s*(.+?)\*$",  # *Caption text*
                r"^_\s*(.+?)_$",  # _Caption text_
                r"^!\[([^\]]*)\]",  # ![alt text] image captions
            ]

            for pattern in caption_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Store the full line as text for complete caption information
                    caption_text = line.strip()
                    # Also store the extracted content for processing
                    extracted_text = match.group(1).strip() if match.groups() else line.strip()
                    # Clean up markdown formatting from extracted text
                    extracted_text = re.sub(r"[\*_`]", "", extracted_text)

                    captions.append(
                        {
                            "text": caption_text,
                            "extracted_text": extracted_text,
                            "line_number": i,
                            "type": "caption",
                            "pattern": pattern,
                            "context": self._get_caption_context(lines, i),
                        }
                    )
                    break

        return captions

    def _get_caption_context(
        self, lines: List[str], caption_line: int, context_window: int = 3
    ) -> str:
        """
        Get surrounding context for a caption.

        Args:
            lines: All lines of the document
            caption_line: Line number of the caption
            context_window: Number of lines to include before/after

        Returns:
            Context text around the caption
        """
        start = max(0, caption_line - context_window)
        end = min(len(lines), caption_line + context_window + 1)

        context_lines = []
        for i in range(start, end):
            if i != caption_line:  # Exclude the caption line itself
                line = lines[i].strip()
                if line:  # Only include non-empty lines
                    context_lines.append(line)

        return " ".join(context_lines)

    def create_caption_centric_chunks(
        self,
        docling_document: DoclingDocument,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[LangchainDocument]:
        """
        Create caption-centric chunks from a Docling document.

        Args:
            docling_document: Docling document object
            metadata: Additional metadata to include in chunks

        Returns:
            List of LangchainDocument objects with caption-aware chunking
        """
        # Extract document structure
        structure = self.extract_document_structure(docling_document)

        # Get full markdown content
        full_content = structure["full_text"]

        # Create caption-centric chunks
        chunks = self._create_chunks_with_captions(structure, metadata or {})

        # If no captions found or chunking failed, fall back to regular chunking
        if not chunks:
            chunks = self._fallback_chunking(full_content, metadata or {})

        return chunks

    def _create_chunks_with_captions(
        self, structure: Dict[str, Any], base_metadata: Dict[str, Any]
    ) -> List[LangchainDocument]:
        """
        Create chunks that preserve caption-content relationships.

        Args:
            structure: Document structure from extract_document_structure
            base_metadata: Base metadata for all chunks

        Returns:
            List of chunks with caption awareness
        """
        captions = structure.get("captions", [])
        full_text = structure.get("full_text", "")

        if not captions:
            return []

        chunks = []
        text_lines = full_text.split("\n")

        # Process each caption and create contextual chunks
        for caption in captions:
            line_number = caption.get("line_number", 0)
            caption_text = caption.get("text", "")
            context = caption.get("context", "")

            # Create a chunk centered around this caption
            # Include some lines before and after the caption
            start_line = max(0, line_number - 5)  # 5 lines before
            end_line = min(len(text_lines), line_number + 15)  # 15 lines after

            chunk_lines = text_lines[start_line:end_line]
            chunk_content = "\n".join(chunk_lines).strip()

            # Skip if chunk is too short
            if len(chunk_content) < 100:
                continue

            # Create metadata for this chunk
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_type": "caption_centric",
                    "has_captions": True,
                    "caption_text": caption_text,
                    "caption_line": line_number,
                    "context_lines": f"{start_line}-{end_line}",
                }
            )

            chunks.append(LangchainDocument(page_content=chunk_content, metadata=chunk_metadata))

        return chunks

    def _fallback_chunking(self, content: str, metadata: Dict[str, Any]) -> List[LangchainDocument]:
        """
        Fallback to regular character-based chunking.

        Args:
            content: Full document content
            metadata: Metadata for chunks

        Returns:
            List of chunks using regular splitting
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            length_function=len,
        )

        chunks = text_splitter.split_text(content)

        return [
            LangchainDocument(
                page_content=chunk,
                metadata={**metadata, "chunk_type": "fallback", "has_captions": False},
            )
            for chunk in chunks
        ]
