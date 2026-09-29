"""Security + resilience primitives: admin auth and per-IP rate limiting."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from app.config import get_settings


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def require_admin(request: Request) -> None:
    """Guard admin endpoints with ``ADMIN_API_KEY`` (fail closed)."""
    settings = get_settings()
    if not settings.admin_api_key:
        raise HTTPException(status_code=503, detail="admin endpoints are not configured")
    provided = request.headers.get("x-admin-key") or ""
    if not provided or provided != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="unauthorized")


class _SlidingWindowLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, max_requests: int, window_seconds: int) -> bool:
        if max_requests <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            dq = self._hits[key]
            while dq and now - dq[0] > window_seconds:
                dq.popleft()
            if len(dq) >= max_requests:
                return False
            dq.append(now)
            return True


_limiter = _SlidingWindowLimiter()


def rate_limit(request: Request) -> None:
    """Per-IP sliding-window rate limit (applies to the ask/retrieval routes)."""
    settings = get_settings()
    if settings.rate_limit_max <= 0:
        return
    if not _limiter.allow(client_ip(request), settings.rate_limit_max, settings.rate_limit_window):
        raise HTTPException(
            status_code=429,
            detail=f"rate limit exceeded ({settings.rate_limit_max} per "
                   f"{settings.rate_limit_window}s)",
        )
