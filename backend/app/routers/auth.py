"""Authentication and authorization routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.auth.jwt_auth import (
    LoginRequest,
    TokenPayload,
    TokenResponse,
    get_current_user,
    login,
    require_permission,
)
from backend.app.middleware.rate_limit import get_audit_log

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def auth_login(body: LoginRequest):
    return login(body)


@router.get("/me")
async def auth_me(user: TokenPayload = Depends(get_current_user)):
    return {"email": user.sub, "role": user.role}


@router.get("/audit")
async def audit_log(
    limit: int = 100,
    user: TokenPayload = Depends(require_permission("audit")),
):
    return {"entries": get_audit_log(limit)}
