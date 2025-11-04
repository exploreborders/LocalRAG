"""
Enhanced AI-powered document processor for the Local RAG system.

This module orchestrates the complete AI processing pipeline:
1. Quality assessment and vision fallback
2. Structure extraction and hierarchy analysis
3. Topic classification
4. Hierarchical chunking
5. Relevance scoring
6. Database storage with embeddings
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import time

from pipeline.vision_fallback import VisionFallbackProcessor
from pipeline.structure_extractor import StructureExtractor
from pipeline.hierarchical_chunker import HierarchicalChunker
from pipeline.relevance_scorer import RelevanceScorer

logger = logging.getLogger(__name__)

class EnhancedDocumentProcessor:
    """
    Complete AI-powered document processing pipeline.

    Orchestrates all AI components for comprehensive document intelligence.
    """

    def __init__(self,
                 vision_model: str = "qwen2.5vl:latest",
                 structure_model: str = "llama3.2:latest",
                 ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the enhanced processor.

        Args:
            vision_model: Ollama model for vision processing
            structure_model: Ollama model for structure analysis
            ollama_base_url: Ollama API base URL
        """
        self.vision_processor = VisionFallbackProcessor(
            model_name=vision_model,
            base_url=ollama_base_url
        )

        self.structure_extractor = StructureExtractor(
            model_name=structure_model,
            base_url=ollama_base_url
        )

        self.chunker = HierarchicalChunker(
            chunk_size=1000,
            overlap=200
        )

        self.relevance_scorer = RelevanceScorer()

        logger.info("Enhanced document processor initialized")

    def process_document(self,
                        file_path: str,
                        docling_result: Optional[Dict[str, Any]] = None,
                        force_vision: bool = False) -> Dict[str, Any]:
        """
        Process a document through the complete AI pipeline.

        Args:
            file_path: Path to the document file
            docling_result: Pre-computed Docling results (optional)
            force_vision: Force vision processing even if Docling succeeds

        Returns:
            Complete processing results with chunks and metadata
        """
        start_time = time.time()
        file_path = Path(file_path)

        logger.info(f"Starting enhanced processing for: {file_path.name}")

        # Step 1: Initial processing with Docling (if not provided)
        if docling_result is None:
            docling_result = self._run_docling_processing(file_path)

        # Step 2: Quality assessment
        quality_assessment = self._assess_quality(docling_result)

        # Step 3: Vision fallback if needed
        final_text = docling_result.get('text', '')
        processing_method = 'docling'

        if force_vision or quality_assessment.get('needs_vision_fallback', False):
            logger.info("Applying vision fallback processing")
            vision_result = self.vision_processor.process_document(file_path, quality_assessment)

            if vision_result.get('confidence', 0) > quality_assessment.get('confidence_score', 0):
                final_text = vision_result.get('text', final_text)
                processing_method = 'vision_fallback'
                logger.info("Vision processing provided better results")

        # Step 4: Structure extraction
        logger.info("Extracting document structure")
        structure_analysis = self.structure_extractor.extract_structure(
            final_text, file_path.name
        )

        # Step 5: Hierarchical chunking
        logger.info("Creating hierarchical chunks")
        raw_chunks = self.chunker.chunk_document(
            final_text, structure_analysis, file_path.name
        )

        # Step 6: Relevance scoring
        document_topics = structure_analysis.get('secondary_topics', []) + \
                         [structure_analysis.get('primary_topic', '')]

        logger.info("Scoring chunks for relevance")
        scored_chunks = self.relevance_scorer.score_chunks(
            raw_chunks, document_topics, structure_analysis
        )

        # Step 7: Compile final results
        processing_time = time.time() - start_time

        result = {
            'filename': file_path.name,
            'filepath': str(file_path),
            'processing_method': processing_method,
            'processing_time_seconds': round(processing_time, 2),

            # Content and structure
            'full_text': final_text,
            'structure_analysis': structure_analysis,
            'chunks': scored_chunks,

            # Quality metrics
            'quality_assessment': quality_assessment,
            'chunk_count': len(scored_chunks),
            'avg_chunk_relevance': sum(c.get('relevance_score', 0) for c in scored_chunks) / len(scored_chunks) if scored_chunks else 0,

            # Metadata
            'document_type': structure_analysis.get('document_type', 'unknown'),
            'primary_topic': structure_analysis.get('primary_topic', ''),
            'secondary_topics': structure_analysis.get('secondary_topics', []),
            'technical_level': structure_analysis.get('technical_level', 'intermediate'),
            'estimated_reading_time': structure_analysis.get('reading_time_minutes', 0),

            # Processing status
            'status': 'completed',
            'success': True
        }

        logger.info(f"Enhanced processing completed in {processing_time:.2f}s, "
                   f"generated {len(scored_chunks)} chunks")

        return result

    def _run_docling_processing(self, file_path: Path) -> Dict[str, Any]:
        """
        Run initial Docling processing.

        This is a placeholder - in the real implementation, this would
        integrate with the existing Docling processor.
        """
        try:
            # Import the existing docling processor
            from document_processor import DocumentProcessor

            processor = DocumentProcessor()
            result = processor.process_file(str(file_path))

            return {
                'text': result.get('content', ''),
                'structure': result.get('structure', {}),
                'confidence': 0.8,  # Assume good quality from Docling
                'method': 'docling'
            }

        except Exception as e:
            logger.warning(f"Docling processing failed: {e}")
            return {
                'text': '',
                'structure': {},
                'confidence': 0.0,
                'error': str(e)
            }

    def _assess_quality(self, docling_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the quality of initial processing results.
        """
        text = docling_result.get('text', '')
        text_length = len(text)

        # Basic quality checks
        has_content = text_length > 100
        has_structure = bool(docling_result.get('structure'))

        # Determine if vision fallback is needed
        needs_vision = (
            not has_content or  # No substantial content
            text_length < 500 or  # Very short content
            not has_structure  # No structural information
        )

        return {
            'needs_vision_fallback': needs_vision,
            'text_length': text_length,
            'has_structure': has_structure,
            'confidence_score': docling_result.get('confidence', 0.5)
        }

    def get_processing_status(self) -> Dict[str, Any]:
        """
        Get the status of AI processing components.
        """
        return {
            'vision_processor_available': self.vision_processor.is_available(),
            'structure_extractor_available': self.structure_extractor.is_available(),
            'chunker_ready': True,
            'relevance_scorer_ready': True
        }

    def batch_process_documents(self,
                              file_paths: List[str],
                              max_workers: int = 4) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch mode.

        Args:
            file_paths: List of file paths to process
            max_workers: Maximum number of parallel workers

        Returns:
            List of processing results
        """
        import concurrent.futures

        logger.info(f"Starting batch processing of {len(file_paths)} documents")

        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_document, path): path
                for path in file_paths
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed processing: {Path(path).name}")
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
                    results.append({
                        'filename': Path(path).name,
                        'filepath': path,
                        'status': 'failed',
                        'error': str(e),
                        'success': False
                    })

        logger.info(f"Batch processing completed: {len(results)} results")
        return results

    def optimize_processing(self, document_type: str) -> Dict[str, Any]:
        """
        Get processing optimizations based on document type.

        Args:
            document_type: Type of document being processed

        Returns:
            Optimization settings
        """
        # Default optimizations
        optimizations = {
            'chunk_size': 1000,
            'overlap': 200,
            'vision_priority': 0.5,
            'structure_depth': 3
        }

        # Document-type specific optimizations
        if document_type == 'research_paper':
            optimizations.update({
                'chunk_size': 800,  # Smaller chunks for dense content
                'vision_priority': 0.7,  # Higher vision priority for complex layouts
                'structure_depth': 4  # Deeper structure analysis
            })
        elif document_type == 'manual':
            optimizations.update({
                'chunk_size': 1200,  # Larger chunks for procedural content
                'overlap': 300,  # More overlap for continuity
                'structure_depth': 3
            })
        elif document_type == 'book':
            optimizations.update({
                'chunk_size': 1500,  # Larger chunks for narrative content
                'structure_depth': 5  # Books often have deep hierarchies
            })

        return optimizations