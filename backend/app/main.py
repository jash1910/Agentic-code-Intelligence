"""ArchonAI FastAPI application entry point."""

from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.workflow import create_workflow
from backend.app.config import settings
from backend.app.middleware.rate_limit import AuditLogMiddleware, RateLimitMiddleware
from backend.app.model import LLMManager
from backend.app.routers import auth, chat, health
from backend.app.services.memory import MemoryManager
from backend.app.services.sandbox import CodeSandbox
from knowledge_graph.neo4j_store import Neo4jGraphStore
from observability.tracer import ArchonTracer
from rag.hybrid_retriever import HybridRetriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[ArchonAI] Initializing agentic platform...")
    llm_manager = LLMManager()
    hybrid_retriever = HybridRetriever() if settings.RAG_ENABLED else None
    graph_query = Neo4jGraphStore()
    sandbox = CodeSandbox(
        timeout=settings.SANDBOX_TIMEOUT,
        use_docker=settings.SANDBOX_USE_DOCKER,
    )
    tracer = ArchonTracer()

    app.state.llm_manager = llm_manager
    app.state.hybrid_retriever = hybrid_retriever
    app.state.graph_query = graph_query
    app.state.sandbox = sandbox
    app.state.memory = MemoryManager()
    app.state.tracer = tracer
    app.state.workflow = create_workflow(
        llm_manager=llm_manager,
        hybrid_retriever=hybrid_retriever,
        graph_query=graph_query,
        sandbox=sandbox,
    )
    print("[ArchonAI] All systems online.")
    yield
    graph_query.close()
    print("[ArchonAI] Shutdown complete.")


app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade agentic code intelligence platform",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuditLogMiddleware)

app.include_router(health.router)
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(chat.router, prefix=settings.API_V1_STR)

# Legacy endpoint for backward compatibility
from backend.app.routers.legacy import legacy_router
app.include_router(legacy_router)

frontend_dist = os.path.join(BASE_DIR, "frontend", "dist")
if os.path.exists(frontend_dist):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")
