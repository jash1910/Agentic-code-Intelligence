"""Planner Agent — understands intent and creates execution plans."""

from __future__ import annotations

import json
import time
from typing import Any

from agents.state import AgentStep, ArchonState

SUPPORTED_REPOSITORIES = [
    "langchain",
    "crewai",
    "llamaindex",
    "pydantic-ai",
    "haystack",
    "dspy",
    "autogen",
]

REPO_KEYWORDS: dict[str, list[str]] = {
    "langchain": ["langchain", "lcel", "llmchain", "runnable", "agentexecutor"],
    "crewai": ["crewai", "crew", "agent crew"],
    "llamaindex": ["llamaindex", "llama index", "query engine"],
    "pydantic-ai": ["pydantic-ai", "pydantic ai"],
    "haystack": ["haystack", "deepset"],
    "dspy": ["dspy", "teleprompter"],
    "autogen": ["autogen", "autogen agent"],
}


def _detect_repository(query: str, default: str = "langchain") -> str:
    query_lower = query.lower()
    for repo, keywords in REPO_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return repo
    return default


def _classify_intent(query: str) -> str:
    query_lower = query.lower()
    if any(w in query_lower for w in ["debug", "fix", "error", "traceback", "bug"]):
        return "debug"
    if any(w in query_lower for w in ["refactor", "rewrite", "improve", "optimize"]):
        return "refactor"
    if any(w in query_lower for w in ["architecture", "design", "diagram", "structure"]):
        return "architecture"
    if any(w in query_lower for w in ["write", "generate", "implement", "create", "code"]):
        return "code_generation"
    if any(w in query_lower for w in ["explain", "what is", "how does", "describe"]):
        return "explanation"
    return "general"


def _build_plan(query: str, intent: str, repository: str) -> tuple[str, list[str]]:
    steps = [
        f"Identify relevant {repository} source files and documentation",
        "Retrieve hybrid RAG context (vector + BM25)",
        "Query knowledge graph for dependencies and inheritance",
    ]

    if intent == "code_generation":
        steps.extend([
            "Generate production-ready code with citations",
            "Run critic review for hallucinations",
            "Execute code in sandbox and self-correct if needed",
        ])
    elif intent == "debug":
        steps.extend([
            "Analyze error patterns in retrieved context",
            "Generate fix with root-cause explanation",
            "Validate fix in execution sandbox",
        ])
    elif intent == "architecture":
        steps.extend([
            "Map component relationships via knowledge graph",
            "Generate architecture diagram (Mermaid)",
            "Provide cited architectural explanation",
        ])
    elif intent == "refactor":
        steps.extend([
            "Identify refactoring targets from graph traversal",
            "Generate refactored code preserving behavior",
            "Critic review and sandbox validation",
        ])
    else:
        steps.extend([
            "Synthesize answer from retrieved sources",
            "Cross-reference knowledge graph relationships",
            "Format response with citations and confidence score",
        ])

    plan = (
        f"Intent: {intent} | Repository: {repository}\n"
        f"Strategy: Multi-agent pipeline with hybrid retrieval, "
        f"knowledge graph enrichment, and verified code generation."
    )
    return plan, steps


def planner_node(state: ArchonState, llm_manager: Any | None = None) -> dict[str, Any]:
    """LangGraph node: analyze query and produce execution plan."""
    start = time.perf_counter()
    query = state.get("user_query", "")
    repository = state.get("repository") or _detect_repository(query)
    intent = _classify_intent(query)
    plan, plan_steps = _build_plan(query, intent, repository)

    step: AgentStep = {
        "agent": "planner",
        "status": "completed",
        "output": plan,
        "duration_ms": (time.perf_counter() - start) * 1000,
        "metadata": {"intent": intent, "repository": repository, "steps": plan_steps},
    }

    trace = list(state.get("agent_trace", []))
    trace.append(step)

    return {
        "repository": repository,
        "plan": plan,
        "plan_steps": plan_steps,
        "agent_trace": trace,
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", 3),
    }

# ArchonAI platform module - architecture reference
