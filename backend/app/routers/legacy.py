"""Legacy /chat endpoint for backward compatibility."""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

router = legacy_router = APIRouter(tags=["legacy"])


class LegacyChatRequest(BaseModel):
    message: str


@router.post("/chat")
async def legacy_chat(body: LegacyChatRequest, req: Request):
    from backend.app.routers.chat import ChatRequest

    chat_body = ChatRequest(message=body.message)
    from backend.app.routers.chat import chat_stream
    from backend.app.auth.jwt_auth import TokenPayload, Role

    anonymous = TokenPayload(sub="anonymous", role=Role.VIEWER, exp=0)
    services = {"workflow": req.app.state.workflow, "memory": req.app.state.memory}

    async def event_generator():
        async for event in req.app.state.workflow.stream(body.message):
            yield {
                "event": event["event"],
                "data": json.dumps(event["data"]),
            }

    return EventSourceResponse(event_generator())

# ArchonAI platform module - architecture reference
