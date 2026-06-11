"""Rate limiting and audit logging middleware."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

RATE_LIMIT = int(__import__("os").getenv("RATE_LIMIT", "60"))
WINDOW_SECONDS = 60

_audit_log: list[dict] = []
_request_counts: dict[str, list[float]] = defaultdict(list)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiter per client IP."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        timestamps = _request_counts[client_ip]
        timestamps[:] = [t for t in timestamps if now - t < WINDOW_SECONDS]

        if len(timestamps) >= RATE_LIMIT:
            return Response(
                content='{"detail":"Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
            )

        timestamps.append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT)
        response.headers["X-RateLimit-Remaining"] = str(RATE_LIMIT - len(timestamps))
        return response


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Log all API requests for enterprise audit trail."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start) * 1000

        entry = {
            "timestamp": time.time(),
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "client": request.client.host if request.client else "unknown",
        }
        _audit_log.append(entry)
        if len(_audit_log) > 10000:
            _audit_log.pop(0)

        return response


def get_audit_log(limit: int = 100) -> list[dict]:
    return _audit_log[-limit:]
