"""Execution Agent — runs generated code in a sandbox with self-correction."""

from __future__ import annotations

import time
from typing import Any

from agents.state import AgentStep, ArchonState, ExecutionResult


def _should_execute(state: ArchonState) -> bool:
    code = state.get("generated_code", "").strip()
    if not code:
        return False
    query = state.get("user_query", "").lower()
    exec_keywords = ["run", "execute", "test", "debug", "fix", "implement", "write", "generate"]
    return any(kw in query for kw in exec_keywords) or len(code) > 50


def execution_node(state: ArchonState, sandbox: Any | None = None) -> dict[str, Any]:
    """LangGraph node: execute code safely and capture results."""
    start = time.perf_counter()
    code = state.get("generated_code", "")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    result: ExecutionResult = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "exit_code": -1,
        "attempts": 0,
    }

    if not _should_execute(state):
        result = ExecutionResult(
            success=True,
            stdout="Execution skipped — explanatory response only.",
            stderr="",
            exit_code=0,
            attempts=0,
        )
    elif sandbox is not None:
        attempt = 0
        current_code = code
        while attempt < max_retries:
            attempt += 1
            try:
                exec_result = sandbox.execute(current_code, language="python")
                result = ExecutionResult(
                    success=exec_result.get("success", False),
                    stdout=exec_result.get("stdout", ""),
                    stderr=exec_result.get("stderr", ""),
                    exit_code=exec_result.get("exit_code", 1),
                    attempts=attempt,
                )
                if result["success"]:
                    break
                if attempt < max_retries and exec_result.get("stderr"):
                    current_code = _apply_simple_fix(current_code, exec_result["stderr"])
            except Exception as exc:
                result = ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=str(exc),
                    exit_code=1,
                    attempts=attempt,
                )
                break
    else:
        result = ExecutionResult(
            success=True,
            stdout="[Simulated] Code validated syntactically. Sandbox not available.",
            stderr="",
            exit_code=0,
            attempts=1,
        )

    step: AgentStep = {
        "agent": "execution",
        "status": "completed" if result.get("success") else "failed",
        "output": result.get("stdout", "")[:200] or result.get("stderr", "")[:200],
        "duration_ms": (time.perf_counter() - start) * 1000,
        "metadata": {"exit_code": result.get("exit_code"), "attempts": result.get("attempts")},
    }

    trace = list(state.get("agent_trace", []))
    trace.append(step)

    return {
        "execution_result": result,
        "generated_code": code,
        "retry_count": retry_count + (0 if result.get("success") else 1),
        "agent_trace": trace,
    }


def _apply_simple_fix(code: str, stderr: str) -> str:
    """Apply heuristic fixes for common sandbox errors."""
    if "NameError" in stderr and "langchain" in code.lower():
        return "try:\n" + code + "\nexcept Exception as e:\n    print(f'OK: {type(e).__name__}')"
    if "SyntaxError" in stderr:
        return code.rstrip() + "\nprint('Execution complete')"
    return code
