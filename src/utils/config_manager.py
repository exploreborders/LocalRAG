"""
Configuration manager for LocalRAG system.
Centralizes all model names, URLs, and configurable parameters.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ModelConfig:
    """Configuration for AI models."""

    # Embedding models
    embedding_model: str = "embeddinggemma:latest"
    embedding_backend: str = "ollama"  # Fixed to Ollama

    # LLM models
    llm_model: str = "llama3.2:latest"
    llm_small_model: str = "llama3.2:latest"

    # Vision models
    vision_model: str = "deepseek-ocr:latest"  # Upgraded from qwen2.5vl
    vision_small_model: str = "microsoft/trocr-base-printed"  # Fallback

    # Structure extraction models
    structure_model: str = "llama3.2:latest"  # Upgraded from phi3.5

    # Topic classification models
    topic_model: str = "llama3.2:latest"  # Upgraded from phi3.5


@dataclass
class DatabaseConfig:
    """Database configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "rag_system"
    user: str = "christianhein"
    password: str = ""

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create database config from environment variables."""
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "rag_system"),
            user=os.getenv("POSTGRES_USER", "christianhein"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
        )


@dataclass
class OpenSearchConfig:
    """OpenSearch/Elasticsearch configuration."""

    host: str = "localhost"
    port: int = 9200
    user: str = "elastic"
    password: str = "changeme"

    @classmethod
    def from_env(cls) -> "OpenSearchConfig":
        """Create OpenSearch config from environment variables."""
        return cls(
            host=os.getenv("OPENSEARCH_HOST", "localhost"),
            port=int(os.getenv("OPENSEARCH_PORT", "9200")),
            user=os.getenv("OPENSEARCH_USER", "elastic"),
            password=os.getenv("OPENSEARCH_PASSWORD", "changeme"),
        )


@dataclass
class RedisConfig:
    """Redis configuration."""

    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0

    @classmethod
    def from_env(cls) -> "RedisConfig":
        """Create Redis config from environment variables."""
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD", ""),
            db=int(os.getenv("REDIS_DB", "0")),
        )


@dataclass
class OllamaConfig:
    """Ollama configuration."""

    base_url: str = "http://localhost:11434"

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        """Create Ollama config from environment variables."""
        return cls(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )


@dataclass
class CacheConfig:
    """Cache configuration."""

    enabled: bool = True
    cache_type: str = "redis"
    ttl_hours: int = 24

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Create cache config from environment variables."""
        return cls(
            enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            cache_type=os.getenv("CACHE_TYPE", "redis"),
            ttl_hours=int(os.getenv("CACHE_TTL_HOURS", "24")),
        )


class ConfigManager:
    """Centralized configuration manager for LocalRAG."""

    def __init__(self):
        self.models = ModelConfig()
        self.database = DatabaseConfig.from_env()
        self.opensearch = OpenSearchConfig.from_env()
        self.redis = RedisConfig.from_env()
        self.ollama = OllamaConfig.from_env()
        self.cache = CacheConfig.from_env()

        # Load custom model configurations from environment
        self._load_model_overrides()

    def _load_model_overrides(self) -> None:
        """Load model name overrides from environment variables."""
        # Embedding model override
        embedding_model = os.getenv("EMBEDDING_MODEL")
        if embedding_model:
            self.models.embedding_model = embedding_model

        # LLM model overrides
        llm_model = os.getenv("LLM_MODEL")
        if llm_model:
            self.models.llm_model = llm_model
        llm_small_model = os.getenv("LLM_SMALL_MODEL")
        if llm_small_model:
            self.models.llm_small_model = llm_small_model

        # Vision model overrides
        vision_model = os.getenv("VISION_MODEL")
        if vision_model:
            self.models.vision_model = vision_model
        vision_small_model = os.getenv("VISION_SMALL_MODEL")
        if vision_small_model:
            self.models.vision_small_model = vision_small_model

        # Structure model override
        structure_model = os.getenv("STRUCTURE_MODEL")
        if structure_model:
            self.models.structure_model = structure_model

        # Topic model override
        topic_model = os.getenv("TOPIC_MODEL")
        if topic_model:
            self.models.topic_model = topic_model

    def get_embedding_model(self) -> str:
        """Get the embedding model name."""
        return self.models.embedding_model

    def get_llm_model(self, model_type: str = "default") -> str:
        """Get LLM model name by type."""
        if model_type == "small":
            return self.models.llm_small_model
        return self.models.llm_model

    def get_vision_model(self, model_type: str = "default") -> str:
        """Get vision model name by type."""
        if model_type == "small":
            return self.models.vision_small_model
        return self.models.vision_model

    def get_structure_model(self) -> str:
        """Get structure extraction model name."""
        return self.models.structure_model

    def get_topic_model(self) -> str:
        """Get topic classification model name."""
        return self.models.topic_model

    def get_database_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return (
            f"postgresql://{self.database.user}:{self.database.password}"
            f"@{self.database.host}:{self.database.port}/{self.database.database}"
        )

    def get_opensearch_url(self) -> str:
        """Get OpenSearch connection URL."""
        return f"http://{self.opensearch.host}:{self.opensearch.port}"

    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        auth_part = f":{self.redis.password}@" if self.redis.password else ""
        return f"redis://{auth_part}{self.redis.host}:{self.redis.port}/{self.redis.db}"

    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return any issues."""
        issues = []

        # Check database configuration
        if (
            not self.database.password
            and os.getenv("REQUIRE_SECURE_PASSWORDS", "false").lower() == "true"
        ):
            issues.append(
                {
                    "severity": "high",
                    "category": "database",
                    "message": "Database password is required in production mode",
                }
            )

        # Check OpenSearch configuration
        if self.opensearch.password == "changeme":
            issues.append(
                {
                    "severity": "critical",
                    "category": "opensearch",
                    "message": "Using default OpenSearch password",
                }
            )

        # Check model availability (basic check)
        required_models = [
            self.models.embedding_model,
            self.models.llm_model,
            self.models.vision_model,
        ]

        for model in required_models:
            if not model:
                issues.append(
                    {
                        "severity": "high",
                        "category": "models",
                        "message": f"Required model is not configured: {model}",
                    }
                )

        return {"valid": len(issues) == 0, "issues": issues}

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (for debugging)."""
        return {
            "models": {
                "embedding_model": self.models.embedding_model,
                "llm_model": self.models.llm_model,
                "llm_small_model": self.models.llm_small_model,
                "vision_model": self.models.vision_model,
                "vision_small_model": self.models.vision_small_model,
                "structure_model": self.models.structure_model,
                "topic_model": self.models.topic_model,
            },
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "user": self.database.user,
                "password": "***" if self.database.password else "",
            },
            "opensearch": {
                "host": self.opensearch.host,
                "port": self.opensearch.port,
                "user": self.opensearch.user,
                "password": "***" if self.opensearch.password else "",
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db,
                "password": "***" if self.redis.password else "",
            },
            "ollama": {
                "base_url": self.ollama.base_url,
            },
            "cache": {
                "enabled": self.cache.enabled,
                "cache_type": self.cache.cache_type,
                "ttl_hours": self.cache.ttl_hours,
            },
        }


# Global configuration instance
config = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global configuration instance."""
    return config
