import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import redis

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        ttl_hours: int = 24,
    ):
        self.ttl_seconds = int(timedelta(hours=ttl_hours).total_seconds())

        try:
            self.redis = redis.Redis(
                host=host,
                port=port,
                password=password,
                db=db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=20,
            )
            self.redis.ping()  # Test connection
            logger.info("Redis cache connected successfully")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if valid"""
        try:
            data = self.redis.get(key)
            if data:
                cached = json.loads(data)
                # Check if entry is still valid (TTL handled by Redis)
                return cached
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
        return None

    def set(self, key: str, response: Dict[str, Any]) -> bool:
        """Store response with TTL"""
        try:
            # Add metadata
            response["cached_at"] = datetime.now().isoformat()
            response["cache_hits"] = response.get("cache_hits", 0)

            data = json.dumps(response)
            return bool(self.redis.setex(key, self.ttl_seconds, data))
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete specific cache entry"""
        try:
            self.redis.delete(key)
            return True  # Redis delete succeeds even if key doesn't exist
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern"""
        try:
            keys = self.redis.keys(pattern)
            if keys:
                return self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear pattern error for {pattern}: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            info = self.redis.info()
            keys = len(self.redis.keys("llm:*"))  # Assuming llm: prefix

            return {
                "total_keys": keys,
                "used_memory": info.get("used_memory_human", "unknown"),
                "hit_rate": info.get("keyspace_hits", 0)
                / max(info.get("keyspace_misses", 0) + info.get("keyspace_hits", 0), 1),
                "connected_clients": info.get("connected_clients", 0),
                "uptime_days": info.get("uptime_in_days", 0),
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {}

    def get_document_metadata(self, doc_ids: list) -> Optional[Dict[int, Dict[str, Any]]]:
        """Retrieve cached document metadata for multiple documents"""
        if not doc_ids:
            return {}

        try:
            # Create pipeline for batch get
            pipeline = self.redis.pipeline()
            keys = [f"doc_metadata:{doc_id}" for doc_id in doc_ids]
            for key in keys:
                pipeline.get(key)
            results = pipeline.execute()

            metadata = {}
            for doc_id, data in zip(doc_ids, results):
                if data:
                    metadata[doc_id] = json.loads(data)
            return metadata if metadata else None
        except Exception as e:
            logger.warning(f"Document metadata cache get error: {e}")
        return None

    def set_document_metadata(self, doc_id: int, metadata: Dict[str, Any]) -> bool:
        """Cache document metadata"""
        try:
            key = f"doc_metadata:{doc_id}"
            data = json.dumps(metadata)
            return bool(self.redis.setex(key, self.ttl_seconds, data))
        except Exception as e:
            logger.error(f"Document metadata cache set error for doc {doc_id}: {e}")
            return False

    def health_check(self) -> bool:
        """Check if Redis is healthy"""
        try:
            return self.redis.ping()
        except Exception:
            return False
