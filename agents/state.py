"""Shared LangGraph state definitions for the ArchonAI agent pipeline."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages


class Citation(TypedDict, total=False):
    source: str
    type: str
    content: str
    score: float
    rank: int


class AgentStep(TypedDict, total=False):
    agent: str
    status: Literal["pending", "running", "completed", "failed"]
    output: str
    duration_ms: float
    metadata: dict[str, Any]


class GraphNode(TypedDict, total=False):
    name: str
    type: str
    relationships: list[str]


class ExecutionResult(TypedDict, total=False):
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    attempts: int


class ArchonState(TypedDict, total=False):
    """State passed between all agents in the LangGraph workflow."""

    messages: Annotated[list, add_messages]
    user_query: str
    session_id: str
    repository: str
    plan: str
    plan_steps: list[str]
    retrieved_docs: list[dict[str, Any]]
    citations: list[Citation]
    graph_context: list[GraphNode]
    generated_code: str
    generated_explanation: str
    architecture_diagram: str
    critic_feedback: str
    critic_score: float
    hallucination_flags: list[str]
    execution_result: ExecutionResult
    final_response: str
    confidence_score: float
    agent_trace: list[AgentStep]
    retry_count: int
    max_retries: int
    error: str | None
