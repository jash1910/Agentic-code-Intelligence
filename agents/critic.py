"""Critic Agent — reviews generated code for quality and hallucinations."""

from __future__ import annotations

import re
import time
from typing import Any

from agents.state import AgentStep, ArchonState

HALLUCINATION_PATTERNS = [
    r"ChatPromptTemplate\s*\(\s*system_message\s*=",
    r"LLMChain\s*\(\s*prompt\s*=.*output_key\s*=.*deprecated_param",
    r"from langchain\.nonexistent",
    r"import\s+FakeLangChain",
]

QUALITY_CHECKS = [
    ("has_imports", r"^(from|import)\s+\w+", re.MULTILINE),
    ("has_docstring_or_comment", r'"""|# ', 0),
    ("uses_modern_api", r"ChatPromptTemplate|Runnable|@tool|create_.*_agent", 0),
]


def critic_node(state: ArchonState, llm_manager: Any | None = None) -> dict[str, Any]:
    """LangGraph node: review output and flag potential hallucinations."""
    start = time.perf_counter()
    code = state.get("generated_code", "")
    explanation = state.get("generated_explanation", "")
    combined = f"{code}\n{explanation}"

    hallucination_flags: list[str] = []
    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            hallucination_flags.append(f"Suspicious pattern: {pattern}")

    passed_checks = 0
    for name, pattern, flags in QUALITY_CHECKS:
        flag_val = flags if isinstance(flags, int) and flags == 0 else flags
        if re.search(pattern, combined, flag_val):
            passed_checks += 1

    citation_count = len(state.get("citations", []))
    base_score = 0.5 + (passed_checks / len(QUALITY_CHECKS)) * 0.3
    citation_bonus = min(citation_count * 0.05, 0.15)
    hallucination_penalty = len(hallucination_flags) * 0.1
    critic_score = max(0.0, min(1.0, base_score + citation_bonus - hallucination_penalty))

    feedback_parts = []
    if hallucination_flags:
        feedback_parts.append("Hallucination risks detected:")
        feedback_parts.extend(f"  - {f}" for f in hallucination_flags)
    if passed_checks < 2:
        feedback_parts.append("Consider adding imports, docstrings, and modern LangChain APIs.")
    if critic_score >= 0.7:
        feedback_parts.append("Quality check passed. Code appears consistent with retrieved sources.")
    else:
        feedback_parts.append("Recommend revision before execution.")

    critic_feedback = "\n".join(feedback_parts) or "No issues detected."

    step: AgentStep = {
        "agent": "critic",
        "status": "completed",
        "output": f"Score: {critic_score:.2f} | Flags: {len(hallucination_flags)}",
        "duration_ms": (time.perf_counter() - start) * 1000,
        "metadata": {"score": critic_score, "flags": hallucination_flags},
    }

    trace = list(state.get("agent_trace", []))
    trace.append(step)

    return {
        "critic_feedback": critic_feedback,
        "critic_score": critic_score,
        "hallucination_flags": hallucination_flags,
        "agent_trace": trace,
    }
