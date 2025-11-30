"""
Unit tests for RedisCache class.

Tests caching functionality, connection handling, and error scenarios.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.cache.redis_cache import RedisCache


class TestRedisCache:
    """Test the RedisCache class functionality."""

    @patch("src.cache.redis_cache.redis")
    def test_init_successful_connection(self, mock_redis):
        """Test successful Redis connection initialization."""
        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        cache = RedisCache(host="localhost", port=6379, password="test", db=1, ttl_hours=48)

        assert cache.redis == mock_redis_instance
        assert cache.ttl_seconds == 48 * 3600  # 48 hours in seconds

        # Verify Redis was initialized with correct parameters
        mock_redis.Redis.assert_called_once_with(
            host="localhost",
            port=6379,
            password="test",
            db=1,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            max_connections=20,
        )

        mock_redis_instance.ping.assert_called_once()

    @patch("src.cache.redis_cache.redis")
    def test_init_connection_failure(self, mock_redis):
        """Test Redis connection failure during initialization."""
        from redis import ConnectionError

        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance
        mock_redis_instance.ping.side_effect = ConnectionError("Connection failed")

        with pytest.raises(ConnectionError):
            RedisCache()

    @patch("src.cache.redis_cache.redis")
    def test_get_cache_hit(self, mock_redis):
        """Test successful cache retrieval."""
        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance

        # Mock cached data
        cached_data = {"response": "test", "timestamp": "2024-01-01"}
        mock_redis_instance.get.return_value = json.dumps(cached_data)

        cache = RedisCache()
        result = cache.get("test_key")

        assert result == cached_data
        mock_redis_instance.get.assert_called_once_with("test_key")

    @patch("src.cache.redis_cache.redis")
    def test_get_cache_miss(self, mock_redis):
        """Test cache miss (key not found)."""
        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance
        mock_redis_instance.get.return_value = None

        cache = RedisCache()
        result = cache.get("nonexistent_key")

        assert result is None
        mock_redis_instance.get.assert_called_once_with("nonexistent_key")

    @patch("src.cache.redis_cache.redis")
    def test_get_cache_error(self, mock_redis):
        """Test cache retrieval error handling."""
        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance
        mock_redis_instance.get.side_effect = Exception("Redis error")

        cache = RedisCache()
        result = cache.get("error_key")

        assert result is None
        mock_redis_instance.get.assert_called_once_with("error_key")

    @patch("src.cache.redis_cache.redis")
    def test_set_cache_success(self, mock_redis):
        """Test successful cache storage."""
        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance

        cache = RedisCache()
        test_data = {"response": "test", "metadata": {"model": "gpt-4"}}

        result = cache.set("test_key", test_data)

        assert result is True
        mock_redis_instance.setex.assert_called_once()
        # Verify TTL was used
        call_args = mock_redis_instance.setex.call_args
        assert call_args[0][0] == "test_key"  # key
        assert call_args[0][1] == cache.ttl_seconds  # ttl
        # Verify data was JSON serialized
        import json

        expected_data = json.dumps(test_data)
        assert call_args[0][2] == expected_data  # value

    @patch("src.cache.redis_cache.redis")
    def test_set_cache_error(self, mock_redis):
        """Test cache storage error handling."""
        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance
        mock_redis_instance.setex.side_effect = Exception("Redis set error")

        cache = RedisCache()
        result = cache.set("error_key", {"data": "test"})

        assert result is False
        mock_redis_instance.setex.assert_called_once()

    @patch("src.cache.redis_cache.redis")
    def test_delete_cache_success(self, mock_redis):
        """Test successful cache deletion."""
        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance
        mock_redis_instance.delete.return_value = 1

        cache = RedisCache()
        result = cache.delete("test_key")

        assert result is True
        mock_redis_instance.delete.assert_called_once_with("test_key")

    @patch("src.cache.redis_cache.redis")
    def test_delete_cache_not_found(self, mock_redis):
        """Test cache deletion when key doesn't exist."""
        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance
        mock_redis_instance.delete.return_value = 0

        cache = RedisCache()
        result = cache.delete("nonexistent_key")

        assert result is True  # Redis delete returns number of keys deleted
        mock_redis_instance.delete.assert_called_once_with("nonexistent_key")

    @patch("src.cache.redis_cache.redis")
    def test_delete_cache_error(self, mock_redis):
        """Test cache deletion error handling."""
        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance
        mock_redis_instance.delete.side_effect = Exception("Redis delete error")

        cache = RedisCache()
        result = cache.delete("error_key")

        assert result is False
        mock_redis_instance.delete.assert_called_once_with("error_key")

    @patch("src.cache.redis_cache.redis")
    def test_clear_pattern_success(self, mock_redis):
        """Test successful pattern-based cache clearing."""
        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance
        mock_redis_instance.keys.return_value = [b"key1", b"key2"]
        mock_redis_instance.delete.return_value = 2

        cache = RedisCache()
        result = cache.clear_pattern("test:*")

        assert result == 2
        mock_redis_instance.keys.assert_called_once_with("test:*")
        mock_redis_instance.delete.assert_called_once_with(b"key1", b"key2")

    @patch("src.cache.redis_cache.redis")
    def test_clear_pattern_no_matches(self, mock_redis):
        """Test pattern clearing when no keys match."""
        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance
        mock_redis_instance.keys.return_value = []

        cache = RedisCache()
        result = cache.clear_pattern("nonexistent:*")

        assert result == 0
        mock_redis_instance.keys.assert_called_once_with("nonexistent:*")
        mock_redis_instance.delete.assert_not_called()

    @patch("src.cache.redis_cache.redis")
    def test_get_stats_success(self, mock_redis):
        """Test successful statistics retrieval."""
        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance

        # Mock Redis info command
        mock_redis_instance.info.return_value = {
            "used_memory_human": "1.00K",
            "total_connections_received": 100,
            "connected_clients": 5,
        }

        cache = RedisCache()
        stats = cache.get_stats()

        assert "used_memory" in stats
        assert "connected_clients" in stats
        assert stats["used_memory"] == "1.00K"
        mock_redis_instance.info.assert_called_once()

    @patch("src.cache.redis_cache.redis")
    def test_get_stats_error(self, mock_redis):
        """Test statistics retrieval error handling."""
        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance
        mock_redis_instance.info.side_effect = Exception("Redis info error")

        cache = RedisCache()
        stats = cache.get_stats()

        assert stats == {}
        mock_redis_instance.info.assert_called_once()

    @patch("src.cache.redis_cache.redis")
    def test_health_check_success(self, mock_redis):
        """Test successful health check."""
        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        cache = RedisCache()
        result = cache.health_check()

        assert result is True
        assert mock_redis_instance.ping.call_count == 2  # Once in __init__, once in health_check

    @patch("src.cache.redis_cache.redis")
    def test_health_check_failure(self, mock_redis):
        """Test health check failure."""
        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance

        # Ping succeeds in __init__, fails in health_check
        call_count = 0

        def ping_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return True  # Success for __init__
            else:
                raise Exception("Connection failed")  # Fail for health_check

        mock_redis_instance.ping.side_effect = ping_side_effect

        cache = RedisCache()
        result = cache.health_check()

        assert result is False
        assert call_count == 2  # Called once in __init__, once in health_check

    @patch("src.cache.redis_cache.redis")
    def test_set_document_metadata_success(self, mock_redis):
        """Test successful document metadata storage."""
        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance

        cache = RedisCache()
        metadata = {"title": "Test Doc", "author": "Test Author"}

        result = cache.set_document_metadata(123, metadata)

        assert result is True
        # Verify the metadata was stored with correct key
        expected_key = f"doc_metadata:{123}"
        mock_redis_instance.setex.assert_called_once()
        call_args = mock_redis_instance.setex.call_args
        assert call_args[0][0] == expected_key
        assert call_args[0][1] == cache.ttl_seconds

    @patch("src.cache.redis_cache.redis")
    def test_get_document_metadata_success(self, mock_redis):
        """Test successful document metadata retrieval."""
        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance

        metadata = {"title": "Test Doc", "author": "Test Author"}
        mock_pipeline = MagicMock()
        mock_redis_instance.pipeline.return_value = mock_pipeline
        mock_pipeline.get.return_value = None  # Not used
        mock_pipeline.execute.return_value = [json.dumps(metadata)]

        cache = RedisCache()
        result = cache.get_document_metadata([123])

        assert result is not None
        assert 123 in result
        assert result[123] == metadata
