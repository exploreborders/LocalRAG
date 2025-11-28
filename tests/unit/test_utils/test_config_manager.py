"""
Unit tests for configuration management utilities.
"""

import pytest
from unittest.mock import patch

from src.utils.config_manager import (
    ConfigManager,
    DatabaseConfig,
    OpenSearchConfig,
    RedisConfig,
    OllamaConfig,
    CacheConfig,
)


class TestConfigManager:
    """Test the ConfigManager class."""

    @patch.dict("os.environ", {"POSTGRES_HOST": "test-host"})
    def test_database_config_from_env(self):
        """Test database config reads from environment variables."""
        config = ConfigManager()
        db_config = config.database

        assert db_config.host == "test-host"
        assert db_config.port == 5432  # default value
        assert db_config.database == "rag_system"  # default value

    @patch.dict(
        "os.environ",
        {
            "POSTGRES_HOST": "prod-db",
            "POSTGRES_PORT": "9999",
            "POSTGRES_DB": "prod_rag",
            "POSTGRES_USER": "prod_user",
            "POSTGRES_PASSWORD": "prod_pass",
        },
    )
    def test_database_config_complete_env(self):
        """Test database config with all environment variables set."""
        config = ConfigManager()
        db_config = config.database

        assert db_config.host == "prod-db"
        assert db_config.port == 9999
        assert db_config.database == "prod_rag"
        assert db_config.user == "prod_user"
        assert db_config.password == "prod_pass"

    def test_database_config_defaults(self):
        """Test database config uses correct defaults."""
        config = ConfigManager()
        db_config = config.database

        assert db_config.host == "localhost"
        assert db_config.port == 5432
        assert db_config.database == "rag_system"
        assert db_config.user == "christianhein"  # This is hardcoded - should be fixed
        assert db_config.password == ""

    @patch.dict("os.environ", {"OPENSEARCH_HOST": "es-host"})
    def test_opensearch_config_from_env(self):
        """Test OpenSearch config reads from environment variables."""
        config = ConfigManager()
        es_config = config.opensearch

        assert es_config.host == "es-host"
        assert es_config.port == 9200  # default value

    @patch.dict(
        "os.environ",
        {
            "OPENSEARCH_HOST": "prod-es",
            "OPENSEARCH_PORT": "9999",
            "OPENSEARCH_USER": "prod_user",
            "OPENSEARCH_PASSWORD": "prod_pass",
        },
    )
    def test_opensearch_config_complete_env(self):
        """Test OpenSearch config with all environment variables set."""
        config = ConfigManager()
        es_config = config.opensearch

        assert es_config.host == "prod-es"
        assert es_config.port == 9999
        assert es_config.user == "prod_user"
        assert es_config.password == "prod_pass"

    def test_opensearch_config_defaults(self):
        """Test OpenSearch config uses correct defaults."""
        config = ConfigManager()
        es_config = config.opensearch

        assert es_config.host == "localhost"
        assert es_config.port == 9200
        assert es_config.user == "elastic"  # This is hardcoded - should be fixed
        assert es_config.password == "changeme"  # This is hardcoded - should be fixed

    @patch.dict("os.environ", {"REDIS_HOST": "redis-host"})
    def test_redis_config_from_env(self):
        """Test Redis config reads from environment variables."""
        config = ConfigManager()
        redis_config = config.redis

        assert redis_config.host == "redis-host"
        assert redis_config.port == 6379  # default value
        assert redis_config.db == 0  # default value

    @patch.dict(
        "os.environ",
        {
            "REDIS_HOST": "prod-redis",
            "REDIS_PORT": "9999",
            "REDIS_DB": "5",
            "REDIS_PASSWORD": "redis_pass",
        },
    )
    def test_redis_config_complete_env(self):
        """Test Redis config with all environment variables set."""
        config = ConfigManager()
        redis_config = config.redis

        assert redis_config.host == "prod-redis"
        assert redis_config.port == 9999
        assert redis_config.db == 5
        assert redis_config.password == "redis_pass"

    def test_redis_config_defaults(self):
        """Test Redis config uses correct defaults."""
        config = ConfigManager()
        redis_config = config.redis

        assert redis_config.host == "localhost"
        assert redis_config.port == 6379
        assert redis_config.db == 0
        assert redis_config.password == ""

    @patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://custom-ollama:8080"})
    def test_ollama_config_from_env(self):
        """Test Ollama config reads from environment variables."""
        config = ConfigManager()
        ollama_config = config.ollama

        assert ollama_config.base_url == "http://custom-ollama:8080"

    def test_ollama_config_defaults(self):
        """Test Ollama config uses correct defaults."""
        config = ConfigManager()
        ollama_config = config.ollama

        assert ollama_config.base_url == "http://localhost:11434"

    @patch.dict(
        "os.environ",
        {"CACHE_ENABLED": "true", "CACHE_TYPE": "redis", "CACHE_TTL_HOURS": "48"},
    )
    def test_cache_config_from_env(self):
        """Test cache config reads from environment variables."""
        config = ConfigManager()
        cache_config = config.cache

        assert cache_config.enabled == True
        assert cache_config.cache_type == "redis"
        assert cache_config.ttl_hours == 48

    def test_cache_config_defaults(self):
        """Test cache config uses correct defaults."""
        config = ConfigManager()
        cache_config = config.cache

        assert cache_config.enabled == True  # default is True
        assert cache_config.cache_type == "redis"  # default
        assert cache_config.ttl_hours == 24  # default

    @patch.dict("os.environ", {"EMBEDDING_MODEL": "custom-embedding-model"})
    def test_embedding_model_from_env(self):
        """Test embedding model reads from environment."""
        config = ConfigManager()
        model = config.get_embedding_model()
        assert model == "custom-embedding-model"

    def test_embedding_model_default(self):
        """Test embedding model uses correct default."""
        config = ConfigManager()
        model = config.get_embedding_model()
        assert model == "nomic-ai/nomic-embed-text-v1.5"

    @patch.dict("os.environ", {"LLM_MODEL": "custom-llm-model"})
    def test_llm_model_from_env(self):
        """Test LLM model reads from environment."""
        config = ConfigManager()
        model = config.get_llm_model()
        assert model == "custom-llm-model"

    def test_llm_model_default(self):
        """Test LLM model uses correct default."""
        config = ConfigManager()
        model = config.get_llm_model()
        assert model == "llama3.2:latest"


class TestConfigClasses:
    """Test the configuration data classes."""

    def test_database_config_creation(self):
        """Test DatabaseConfig creation."""
        config = DatabaseConfig(
            host="test-host",
            port=5432,
            database="test-db",
            user="test-user",
            password="test-pass",
        )
        assert config.host == "test-host"
        assert config.port == 5432
        assert config.database == "test-db"
        assert config.user == "test-user"
        assert config.password == "test-pass"

    def test_opensearch_config_creation(self):
        """Test OpenSearchConfig creation."""
        config = OpenSearchConfig(
            host="es-host", port=9200, user="es-user", password="es-pass"
        )
        assert config.host == "es-host"
        assert config.port == 9200
        assert config.user == "es-user"
        assert config.password == "es-pass"

    def test_redis_config_creation(self):
        """Test RedisConfig creation."""
        config = RedisConfig(host="redis-host", port=6379, password="redis-pass", db=1)
        assert config.host == "redis-host"
        assert config.port == 6379
        assert config.password == "redis-pass"
        assert config.db == 1

    def test_ollama_config_creation(self):
        """Test OllamaConfig creation."""
        config = OllamaConfig(base_url="http://ollama:8080")
        assert config.base_url == "http://ollama:8080"

    def test_cache_config_creation(self):
        """Test CacheConfig creation."""
        config = CacheConfig(enabled=True, cache_type="redis", ttl_hours=24)
        assert config.enabled == True
        assert config.cache_type == "redis"
        assert config.ttl_hours == 24
