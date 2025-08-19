from __future__ import annotations

import logging
import os
from typing import Optional

from redis.asyncio import Redis
from redis.asyncio.client import PubSub

logger = logging.getLogger(__name__)

# Singleton async Redis client (lazy)
_redis_singleton: Optional[Redis] = None


def _redis_url() -> str:
    """Resolve Redis URL from environment or default."""
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")


async def get_redis() -> Redis:
    """
    Get/create a singleton async Redis client configured from env.

    Environment:
      - REDIS_URL=redis://localhost:6379/0 (default)
    """
    global _redis_singleton
    if _redis_singleton is None:
        url = _redis_url()
        logger.info("Initializing Redis client: %s", url)
        # decode_responses=True so we work with str payloads everywhere
        _redis_singleton = Redis.from_url(
            url,
            decode_responses=True,
            health_check_interval=30,
            retry_on_timeout=True,
        )
        # Validate connection (fail-fast so callers can handle unavailability)
        try:
            await _redis_singleton.ping()
        except Exception as exc:
            # Ensure we don't retain a broken singleton
            await close_redis()
            logger.error("Redis unavailable at %s: %s", url, exc)
            raise
    return _redis_singleton


async def get_pubsub() -> PubSub:
    """
    Return a new PubSub object bound to the singleton client.

    Note: Caller is responsible for closing the PubSub via pubsub.close().
    """
    client = await get_redis()
    # ignore_subscribe_messages=True filters out subscribe/unsubscribe ack messages
    return client.pubsub(ignore_subscribe_messages=True)


async def close_redis() -> None:
    """Close the singleton Redis client (used on graceful shutdown)."""
    global _redis_singleton
    if _redis_singleton is not None:
        try:
            await _redis_singleton.close()
        except Exception as exc:
            logger.warning("Error closing Redis client: %s", exc)
        finally:
            _redis_singleton = None