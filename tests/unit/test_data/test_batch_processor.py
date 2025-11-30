"""
Unit tests for BatchEmbeddingService class.

Tests batch processing, device detection, statistics, and async operations.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.data.batch_processor import BatchEmbeddingService, QueryRequest


class TestBatchEmbeddingService:
    """Test the BatchEmbeddingService class functionality."""

    @patch("src.data.batch_processor.TORCH_AVAILABLE", True)
    @patch("src.data.batch_processor.torch")
    def test_init_cpu_fallback(self, mock_torch):
        """Test initialization with CPU fallback."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        service = BatchEmbeddingService()

        assert service.device == "cpu"
        assert service.max_batch_size == 16  # Default

    @patch("src.data.batch_processor.TORCH_AVAILABLE", True)
    @patch("src.data.batch_processor.torch")
    @patch("src.data.batch_processor.platform")
    def test_init_mps_detection(self, mock_platform, mock_torch):
        """Test initialization with Apple Silicon MPS detection."""
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "arm64"
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = False

        service = BatchEmbeddingService(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            backend="sentence-transformers",
        )

        assert service.device == "mps"
        assert service.max_batch_size == 8  # Reduced for unified memory

    @patch("src.data.batch_processor.TORCH_AVAILABLE", True)
    @patch("src.data.batch_processor.torch")
    def test_init_cuda_detection(self, mock_torch):
        """Test initialization with CUDA detection."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = False

        service = BatchEmbeddingService(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            max_batch_size=32,
            backend="sentence-transformers",
        )

        assert service.device == "cuda"
        assert service.max_batch_size == 16  # Capped for GPU

    @patch("src.data.batch_processor.TORCH_AVAILABLE", False)
    @patch("src.data.batch_processor.get_embedding_model")
    def test_init_no_torch(self, mock_get_embedding_model):
        """Test initialization when torch is not available."""
        mock_get_embedding_model.return_value = "ollama:embeddinggemma:latest"
        service = BatchEmbeddingService()
        assert service.device == "cpu"
        assert service.backend == "ollama"

    @patch("src.data.batch_processor.TORCH_AVAILABLE", True)
    @patch("src.data.batch_processor.get_embedding_model")
    @patch("src.data.batch_processor.torch")
    def test_start_stop_processing(self, mock_torch, mock_get_embedding_model):
        """Test starting and stopping the processing service."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_get_embedding_model.return_value = MagicMock()

        service = BatchEmbeddingService()

        # Test starting
        assert not service.is_running
        asyncio.run(service.start_processing())
        assert service.is_running
        assert service.processor_task is not None

        # Test stopping
        asyncio.run(service.stop_processing())
        assert not service.is_running
        assert service.processor_task is None

    @patch("src.data.batch_processor.TORCH_AVAILABLE", True)
    @patch("src.data.batch_processor.create_embeddings")
    @patch("src.data.batch_processor.get_embedding_model")
    @patch("src.data.batch_processor.torch")
    def test_embed_query_sync_with_running_loop(
        self, mock_torch, mock_get_embedding_model, mock_create_embeddings
    ):
        """Test sync embedding when event loop is already running."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        mock_get_embedding_model.return_value = MagicMock()
        mock_create_embeddings.return_value = (
            np.array([[0.1, 0.2, 0.3]]),
            "ollama:embeddinggemma:latest",
        )

        service = BatchEmbeddingService()

        # Mock that loop is running
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.is_running.return_value = True
            mock_get_loop.return_value = mock_loop

            result = service.embed_query_sync("test query")

            # Should fall back to direct embedding
            assert isinstance(result, np.ndarray)
            mock_create_embeddings.assert_called_once()

    @patch("src.data.batch_processor.TORCH_AVAILABLE", True)
    @patch("src.data.batch_processor.get_embedding_model")
    @patch("src.data.batch_processor.torch")
    @pytest.mark.asyncio
    async def test_embed_query_async(self, mock_torch, mock_get_embedding_model):
        """Test asynchronous embedding."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        mock_get_embedding_model.return_value = MagicMock()

        service = BatchEmbeddingService()

        # Mock the queue
        mock_queue = AsyncMock()
        service.queue = mock_queue

        # Create a real future and set result
        future = asyncio.Future()
        future.set_result(np.array([0.1, 0.2, 0.3]))

        with patch("asyncio.Future") as mock_future_class:
            mock_future_class.return_value = future

            result = await service.embed_query_async("test query")

            assert isinstance(result, np.ndarray)
            assert result.shape == (3,)
            mock_queue.put.assert_called_once()

    def test_query_request_creation(self):
        """Test QueryRequest dataclass creation."""
        import time

        start_time = time.time()

        request = QueryRequest(id="test-123", query="test query", timestamp=start_time)

        assert request.id == "test-123"
        assert request.query == "test query"
        assert request.timestamp == start_time
        assert request.callback is None

    @patch("src.data.batch_processor.TORCH_AVAILABLE", True)
    @patch("src.data.batch_processor.get_embedding_model")
    @patch("src.data.batch_processor.torch")
    def test_get_stats(self, mock_torch, mock_get_embedding_model):
        """Test getting service statistics."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_get_embedding_model.return_value = MagicMock()

        service = BatchEmbeddingService()

        stats = service.get_stats()

        expected_keys = [
            "total_queries",
            "batch_count",
            "avg_batch_size",
            "avg_processing_time",
            "gpu_utilization",
            "device",
            "max_batch_size",
            "queue_size",
            "is_running",
        ]

        for key in expected_keys:
            assert key in stats

        assert stats["device"] == "cpu"
        assert stats["max_batch_size"] == 16
        assert stats["is_running"] is False

    @patch("src.data.batch_processor.TORCH_AVAILABLE", True)
    @patch("src.data.batch_processor.get_embedding_model")
    @patch("src.data.batch_processor.torch")
    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_torch, mock_get_embedding_model):
        """Test health check with successful embedding."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        mock_get_embedding_model.return_value = MagicMock()

        service = BatchEmbeddingService()

        # Mock successful embedding
        with patch.object(service, "embed_query_async", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = np.array([0.1] * 768)  # 768-dim embedding

            result = await service.health_check()

            assert result is True
            mock_embed.assert_called_once_with("test")

    @patch("src.data.batch_processor.TORCH_AVAILABLE", True)
    @patch("src.data.batch_processor.get_embedding_model")
    @patch("src.data.batch_processor.torch")
    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_torch, mock_get_embedding_model):
        """Test health check with failed embedding."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        mock_get_embedding_model.return_value = MagicMock()

        service = BatchEmbeddingService()

        # Mock failed embedding
        with patch.object(service, "embed_query_async", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = Exception("Embedding failed")

            result = await service.health_check()

            assert result is False

    @patch("src.data.batch_processor.TORCH_AVAILABLE", True)
    @patch("src.data.batch_processor.get_embedding_model")
    @patch("src.data.batch_processor.torch")
    def test_statistics_update(self, mock_torch, mock_get_embedding_model):
        """Test statistics update functionality."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_get_embedding_model.return_value = MagicMock()

        service = BatchEmbeddingService()

        # Simulate some processing
        service._update_stats(batch_size=5, processing_time=0.1)
        service._update_stats(batch_size=3, processing_time=0.05)

        stats = service.get_stats()

        assert stats["total_queries"] == 8  # 5 + 3
        assert stats["batch_count"] == 2
        assert abs(stats["avg_batch_size"] - 4.0) < 0.1  # (5+3)/2
        assert stats["avg_processing_time"] > 0

    @patch("src.data.batch_processor.TORCH_AVAILABLE", True)
    @patch("src.data.batch_processor.get_embedding_model")
    @patch("src.data.batch_processor.torch")
    def test_device_specific_optimizations(self, mock_torch, mock_get_embedding_model):
        """Test device-specific optimizations."""
        mock_model = MagicMock()
        mock_get_embedding_model.return_value = mock_model

        # Test MPS optimizations
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = False

        with patch("src.data.batch_processor.platform") as mock_platform:
            mock_platform.system.return_value = "Darwin"
            mock_platform.machine.return_value = "arm64"

            service = BatchEmbeddingService(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                backend="sentence-transformers",
            )
            assert service.device == "mps"
            assert service.max_batch_size == 8

        # Test CUDA optimizations
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True

        service = BatchEmbeddingService(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            backend="sentence-transformers",
        )
        assert service.device == "cuda"
        assert service.max_batch_size == 16
