"""Chat API routes with LangGraph agent streaming."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.app.auth.jwt_auth import TokenPayload, get_current_user, require_permission
from backend.app.config import settings

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: str = Field(default="default")
    repository: str | None = None


def get_services(request: Request) -> dict[str, Any]:
    return {
        "workflow": request.app.state.workflow,
        "memory": request.app.state.memory,
    }


@router.post("")
async def chat_stream(
    body: ChatRequest,
    req: Request,
    user: TokenPayload = Depends(require_permission("read")),
    services: dict = Depends(get_services),
):
    """Stream agent pipeline response via SSE."""
    workflow = services["workflow"]
    memory = services["memory"]
    repository = body.repository or settings.DEFAULT_REPOSITORY

    memory.add_message(body.session_id, "user", body.message)
    memory_context = memory.get_context(body.session_id)
    long_term = memory.recall_long_term(body.message)

    async def event_generator():
        if await req.is_disconnected():
            return

        enriched_query = body.message
        if memory_context:
            enriched_query = f"Session context:\n{memory_context}\n\nQuery: {body.message}"

        async for event in workflow.stream(
            enriched_query,
            session_id=body.session_id,
            repository=repository,
        ):
            if await req.is_disconnected():
                break
            yield {
                "event": event["event"],
                "data": json.dumps(event["data"]),
            }

        memory.store_long_term(
            body.session_id,
            f"Q: {body.message}",
            {"type": "query", "repository": repository},
        )

    return EventSourceResponse(event_generator())


@router.get("/repositories")
async def list_repositories(
    user: TokenPayload = Depends(get_current_user),
):
    return {"repositories": settings.SUPPORTED_REPOSITORIES}

# ArchonAI platform module - architecture reference
