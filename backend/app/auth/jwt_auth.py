"""JWT authentication and RBAC authorization."""

from __future__ import annotations

import os
import time
from enum import Enum
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

JWT_SECRET = os.getenv("JWT_SECRET", "archonai-dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

security = HTTPBearer(auto_error=False)


class Role(str, Enum):
    ADMIN = "admin"
    DEVELOPER = "developer"
    VIEWER = "viewer"


ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.ADMIN: {"read", "write", "execute", "admin", "audit"},
    Role.DEVELOPER: {"read", "write", "execute"},
    Role.VIEWER: {"read"},
}

USERS_DB: dict[str, dict[str, Any]] = {
    "admin@archon.ai": {
        "password": "admin123",
        "role": Role.ADMIN,
        "api_key": "archon_admin_key_dev",
    },
    "dev@archon.ai": {
        "password": "dev123",
        "role": Role.DEVELOPER,
        "api_key": "archon_dev_key",
    },
    "viewer@archon.ai": {
        "password": "viewer123",
        "role": Role.VIEWER,
        "api_key": "archon_viewer_key",
    },
}


class TokenPayload(BaseModel):
    sub: str
    role: Role
    exp: float


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: Role
    expires_in: int


def _encode_token(email: str, role: Role) -> str:
    try:
        import jwt
    except ImportError:
        return f"mock_token_{email}_{role.value}"

    payload = {
        "sub": email,
        "role": role.value,
        "exp": time.time() + JWT_EXPIRY_HOURS * 3600,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> TokenPayload:
    try:
        import jwt

        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return TokenPayload(sub=data["sub"], role=Role(data["role"]), exp=data["exp"])
    except Exception:
        if token.startswith("mock_token_"):
            parts = token.split("_")
            return TokenPayload(sub=parts[2], role=Role(parts[3]), exp=time.time() + 3600)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")


def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    user = USERS_DB.get(email)
    if user and user["password"] == password:
        return user
    return None


def login(login_req: LoginRequest) -> TokenResponse:
    user = authenticate_user(login_req.email, login_req.password)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    token = _encode_token(login_req.email, user["role"])
    return TokenResponse(
        access_token=token,
        role=user["role"],
        expires_in=JWT_EXPIRY_HOURS * 3600,
    )


def verify_api_key(api_key: str) -> TokenPayload | None:
    for email, user in USERS_DB.items():
        if user.get("api_key") == api_key:
            return TokenPayload(sub=email, role=user["role"], exp=time.time() + 3600)
    return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> TokenPayload:
    if credentials is None:
        return TokenPayload(sub="anonymous", role=Role.VIEWER, exp=time.time() + 3600)

    token = credentials.credentials
    if token.startswith("archon_"):
        payload = verify_api_key(token)
        if payload:
            return payload
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")

    return _decode_token(token)


def require_permission(permission: str):
    async def checker(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        allowed = ROLE_PERMISSIONS.get(user.role, set())
        if permission not in allowed:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Missing permission: {permission}")
        return user

    return checker

# ArchonAI platform module - architecture reference
