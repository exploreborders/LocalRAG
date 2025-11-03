"""
Vision fallback processor for complex document parsing using qwen2.5vl.

This module provides OCR and layout-aware text extraction for documents that
Docling cannot process effectively, such as scanned PDFs or complex layouts.
"""

import os
import logging
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
import requests
from PIL import Image
import io

logger = logging.getLogger(__name__)

class VisionFallbackProcessor:
    """
    Vision-based document processing using qwen2.5vl for complex documents.

    Provides fallback OCR and layout understanding when standard text extraction fails.
    """

    def __init__(self, model_name: str = "qwen2.5vl:7b", base_url: str = "http://localhost:11434"):
        """
        Initialize the vision processor.

        Args:
            model_name: Ollama model name for vision processing
            base_url: Ollama API base URL
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/generate"

    def is_available(self) -> bool:
        """Check if the vision model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(model['name'] == self.model_name for model in models)
            return False
        except Exception as e:
            logger.warning(f"Vision model availability check failed: {e}")
            return False

    def process_document(self, file_path: Path, quality_check_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a document using vision analysis.

        Args:
            file_path: Path to the document file
            quality_check_result: Result from quality assessment

        Returns:
            Dict containing extracted text, structure, and metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        logger.info(f"Processing document with vision fallback: {file_path.name}")

        # Convert document to images
        images = self._convert_to_images(file_path)

        if not images:
            return {
                'text': '',
                'structure': {},
                'confidence': 0.0,
                'method': 'vision_fallback',
                'error': 'No images could be extracted from document'
            }

        # Process images with vision model
        extracted_content = []
        total_confidence = 0.0

        for i, image_data in enumerate(images):
            try:
                page_content = self._process_image(image_data, page_num=i+1)
                if page_content:
                    extracted_content.append(page_content)
                    total_confidence += page_content.get('confidence', 0.5)
            except Exception as e:
                logger.warning(f"Failed to process page {i+1}: {e}")
                continue

        # Combine results
        combined_text = '\n\n'.join([page.get('text', '') for page in extracted_content])
        avg_confidence = total_confidence / len(images) if images else 0.0

        # Extract structure information
        structure = self._extract_structure(extracted_content)

        return {
            'text': combined_text,
            'structure': structure,
            'confidence': avg_confidence,
            'method': 'vision_fallback',
            'pages_processed': len(extracted_content),
            'total_pages': len(images)
        }

    def _convert_to_images(self, file_path: Path) -> List[bytes]:
        """
        Convert document pages to images.

        Args:
            file_path: Path to document

        Returns:
            List of image data as bytes
        """
        try:
            # For PDF files, we would use pdf2image or similar
            # For now, return empty list as this requires additional dependencies
            logger.warning("PDF to image conversion not implemented - requires pdf2image")
            return []

        except Exception as e:
            logger.error(f"Failed to convert document to images: {e}")
            return []

    def _process_image(self, image_data: bytes, page_num: int) -> Optional[Dict[str, Any]]:
        """
        Process a single image using the vision model.

        Args:
            image_data: Image data as bytes
            page_num: Page number for context

        Returns:
            Dict with extracted text and metadata
        """
        try:
            # Convert image to base64
            import base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')

            # Prepare prompt for vision analysis
            prompt = f"""
            Analyze this document page and extract all visible text and structure.

            Instructions:
            1. Extract all readable text from the image
            2. Preserve the original layout and formatting as much as possible
            3. Identify headings, paragraphs, lists, and other structural elements
            4. If you see tables, describe their structure and content
            5. Note any special formatting like bold, italic, or different fonts
            6. Provide confidence score (0-1) for text extraction accuracy

            Return the result as a structured JSON object with:
            - text: The extracted text
            - structure: Description of layout elements
            - confidence: Your confidence in the extraction (0-1)
            - elements: List of identified structural elements
            """

            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent extraction
                    "num_predict": 1024  # Allow longer responses
                }
            }

            # Make API request
            response = requests.post(self.api_url, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')

                # Parse the JSON response from the model
                try:
                    # The model should return JSON, but we need to extract it
                    import json
                    # Look for JSON in the response
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        parsed_result = json.loads(json_str)
                        return parsed_result
                    else:
                        # Fallback: treat entire response as text
                        return {
                            'text': response_text,
                            'structure': {'type': 'unknown'},
                            'confidence': 0.7,
                            'elements': []
                        }
                except json.JSONDecodeError:
                    # If JSON parsing fails, return basic structure
                    return {
                        'text': response_text,
                        'structure': {'type': 'text_block'},
                        'confidence': 0.6,
                        'elements': [{'type': 'paragraph', 'content': response_text}]
                    }
            else:
                logger.error(f"Vision API request failed: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Vision processing failed for page {page_num}: {e}")
            return None

    def _extract_structure(self, page_contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract overall document structure from page contents.

        Args:
            page_contents: List of page processing results

        Returns:
            Document structure information
        """
        structure = {
            'pages': len(page_contents),
            'has_tables': False,
            'has_headers': False,
            'has_lists': False,
            'estimated_reading_time': 0
        }

        total_words = 0
        for page in page_contents:
            text = page.get('text', '')
            elements = page.get('elements', [])

            total_words += len(text.split())

            # Check for structural elements
            for element in elements:
                elem_type = element.get('type', '')
                if 'table' in elem_type.lower():
                    structure['has_tables'] = True
                elif 'header' in elem_type.lower() or 'heading' in elem_type.lower():
                    structure['has_headers'] = True
                elif 'list' in elem_type.lower():
                    structure['has_lists'] = True

        # Estimate reading time (200 words per minute)
        structure['estimated_reading_time'] = max(1, total_words // 200)

        return structure

    def assess_quality(self, docling_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess if vision fallback is needed based on Docling results.

        Args:
            docling_result: Results from Docling processing

        Returns:
            Quality assessment with recommendation
        """
        # Simple quality assessment
        text_length = len(docling_result.get('text', ''))
        has_structure = bool(docling_result.get('structure'))

        # Recommend vision fallback for very short extractions or missing structure
        needs_vision = text_length < 100 or not has_structure

        return {
            'needs_vision_fallback': needs_vision,
            'text_length': text_length,
            'has_structure': has_structure,
            'confidence_score': 0.8 if not needs_vision else 0.3
        }