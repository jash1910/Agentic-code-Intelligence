"""ArchonAI application configuration."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Settings:
    APP_NAME: str = "ArchonAI — Agentic Code Intelligence Platform"
    APP_VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"

    RUN_MODE: str = os.getenv("RUN_MODE", "mock")
    MODEL_ID: str = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-Coder-1.3B-instruct")
    PEFT_ADAPTER_PATH: str = os.getenv(
        "PEFT_ADAPTER_PATH", str(BASE_DIR / "training" / "outputs")
    )

    RAG_ENABLED: bool = os.getenv("RAG_ENABLED", "true").lower() == "true"
    K_RETRIEVAL: int = int(os.getenv("K_RETRIEVAL", "5"))
    DEFAULT_REPOSITORY: str = os.getenv("DEFAULT_REPOSITORY", "langchain")

    LANGSMITH_ENABLED: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGSMITH_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "archonai")
    OTEL_ENABLED: bool = os.getenv("OTEL_ENABLED", "false").lower() == "true"
    PHOENIX_ENABLED: bool = os.getenv("PHOENIX_ENABLED", "false").lower() == "true"

    SANDBOX_TIMEOUT: int = int(os.getenv("SANDBOX_TIMEOUT", "10"))
    SANDBOX_USE_DOCKER: bool = os.getenv("SANDBOX_USE_DOCKER", "false").lower() == "true"

    JWT_SECRET: str = os.getenv("JWT_SECRET", "archonai-dev-secret-change-in-production")
    RATE_LIMIT: int = int(os.getenv("RATE_LIMIT", "60"))

    SUPPORTED_REPOSITORIES: list[str] = [
        "langchain", "crewai", "llamaindex", "pydantic-ai",
        "haystack", "dspy", "autogen",
    ]


settings = Settings()
