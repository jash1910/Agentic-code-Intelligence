"""Observability and health check routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from backend.app.auth.jwt_auth import TokenPayload, require_permission
from backend.app.config import settings

router = APIRouter(tags=["system"])


@router.get("/health")
async def health_check(request: Request):
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "mode": settings.RUN_MODE,
        "model": settings.MODEL_ID,
        "rag_enabled": settings.RAG_ENABLED,
        "agents": [
            "planner", "retriever", "knowledge_graph",
            "code_generator", "critic", "execution", "formatter",
        ],
    }


@router.get("/metrics")
async def metrics(
    request: Request,
    user: TokenPayload = Depends(require_permission("admin")),
):
    tracer = getattr(request.app.state, "tracer", None)
    if tracer:
        return tracer.get_metrics()
    return {"message": "Observability not configured", "metrics": {}}

# ArchonAI platform module - architecture reference
