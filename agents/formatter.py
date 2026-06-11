"""Response Formatter Agent — assembles final streamed response with citations."""

from __future__ import annotations

import time
from typing import Any

from agents.state import AgentStep, ArchonState


def _compute_confidence(state: ArchonState) -> float:
    critic = state.get("critic_score", 0.5)
    citation_count = len(state.get("citations", []))
    exec_ok = state.get("execution_result", {}).get("success", True)
    hallucinations = len(state.get("hallucination_flags", []))

    score = critic * 0.5
    score += min(citation_count * 0.08, 0.25)
    score += 0.15 if exec_ok else -0.1
    score -= hallucinations * 0.05
    return max(0.0, min(1.0, score))


def formatter_node(state: ArchonState) -> dict[str, Any]:
    """LangGraph node: format final response with metadata."""
    start = time.perf_counter()
    explanation = state.get("generated_explanation", "")
    code = state.get("generated_code", "")
    diagram = state.get("architecture_diagram", "")
    feedback = state.get("critic_feedback", "")
    exec_result = state.get("execution_result", {})
    confidence = _compute_confidence(state)

    sections = [explanation]

    if diagram:
        sections.append(f"\n\n### Architecture\n\n```mermaid\n{diagram}\n```")

    if code and code not in explanation:
        sections.append(f"\n\n### Generated Code\n\n```python\n{code}\n```")

    if feedback:
        sections.append(f"\n\n---\n**Critic Review** (confidence: {confidence:.0%})\n{feedback}")

    if exec_result.get("stdout"):
        sections.append(f"\n\n**Execution Output:**\n```\n{exec_result['stdout']}\n```")
    if exec_result.get("stderr") and not exec_result.get("success"):
        sections.append(f"\n\n**Execution Error:**\n```\n{exec_result['stderr']}\n```")

    final_response = "".join(sections)

    step: AgentStep = {
        "agent": "formatter",
        "status": "completed",
        "output": f"Formatted response ({len(final_response)} chars, confidence={confidence:.2f})",
        "duration_ms": (time.perf_counter() - start) * 1000,
        "metadata": {"confidence": confidence},
    }

    trace = list(state.get("agent_trace", []))
    trace.append(step)

    return {
        "final_response": final_response,
        "confidence_score": confidence,
        "agent_trace": trace,
    }
