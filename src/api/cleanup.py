"""Background task: delete expired sessions every hour."""
from __future__ import annotations

import asyncio
import logging

from src.config import SESSION_TTL_HOURS

logger = logging.getLogger("visualmind.api.cleanup")


async def cleanup_loop(interval_minutes: int = 60):
    """Every interval_minutes, delete sessions older than SESSION_TTL_HOURS."""
    from src.api.session_store import cleanup_expired

    while True:
        await asyncio.sleep(interval_minutes * 60)
        try:
            n = cleanup_expired(max_age_hours=SESSION_TTL_HOURS)
            if n:
                logger.info(f"Cleaned up {n} expired sessions")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")