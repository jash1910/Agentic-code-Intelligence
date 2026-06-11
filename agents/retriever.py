"""Retriever Agent — hybrid vector + BM25 retrieval with RRF fusion."""

from __future__ import annotations

import time
from typing import Any

from agents.state import AgentStep, ArchonState, Citation


def retriever_node(state: ArchonState, hybrid_retriever: Any | None = None) -> dict[str, Any]:
    """LangGraph node: retrieve and rank relevant context documents."""
    start = time.perf_counter()
    query = state.get("user_query", "")
    repository = state.get("repository", "langchain")
    k = 5

    retrieved_docs: list[dict[str, Any]] = []
    citations: list[Citation] = []

    if hybrid_retriever is not None:
        try:
            results = hybrid_retriever.retrieve(query, repository=repository, k=k)
            for rank, doc in enumerate(results, start=1):
                entry = {
                    "source": doc.get("source", "unknown"),
                    "type": doc.get("type", "code"),
                    "content": doc.get("content", ""),
                    "score": doc.get("score", 0.0),
                    "rank": rank,
                    "retrieval_method": doc.get("retrieval_method", "hybrid"),
                }
                retrieved_docs.append(entry)
                citations.append(
                    Citation(
                        source=entry["source"],
                        type=entry["type"],
                        content=entry["content"][:500],
                        score=entry["score"],
                        rank=rank,
                    )
                )
        except Exception as exc:
            retrieved_docs = [{"error": str(exc), "content": ""}]
    else:
        citations.append(
            Citation(
                source=f"{repository}/fallback",
                type="docs",
                content="Hybrid retriever not initialized. Using planner context only.",
                score=0.0,
                rank=1,
            )
        )

    step: AgentStep = {
        "agent": "retriever",
        "status": "completed",
        "output": f"Retrieved {len(retrieved_docs)} documents via hybrid search",
        "duration_ms": (time.perf_counter() - start) * 1000,
        "metadata": {"count": len(retrieved_docs), "repository": repository},
    }

    trace = list(state.get("agent_trace", []))
    trace.append(step)

    return {
        "retrieved_docs": retrieved_docs,
        "citations": citations,
        "agent_trace": trace,
    }
