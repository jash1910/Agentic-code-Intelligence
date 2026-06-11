"""Knowledge Graph Agent — query repository structure and dependencies."""

from __future__ import annotations

import time
from typing import Any

from agents.state import AgentStep, ArchonState, GraphNode


def knowledge_graph_node(state: ArchonState, graph_query: Any | None = None) -> dict[str, Any]:
    """LangGraph node: enrich context with graph traversal results."""
    start = time.perf_counter()
    query = state.get("user_query", "")
    repository = state.get("repository", "langchain")
    graph_context: list[GraphNode] = []

    if graph_query is not None:
        try:
            entities = graph_query.extract_entities(query)
            for entity in entities[:5]:
                related = graph_query.get_related(entity, repository=repository, depth=2)
                graph_context.append(
                    GraphNode(
                        name=entity,
                        type=related.get("type", "class"),
                        relationships=related.get("relationships", []),
                    )
                )
        except Exception as exc:
            graph_context.append(
                GraphNode(
                    name="graph_error",
                    type="error",
                    relationships=[str(exc)],
                )
            )
    else:
        graph_context.append(
            GraphNode(
                name="ChatPromptTemplate",
                type="class",
                relationships=["INHERITS_FROM:BasePromptTemplate"],
            )
        )

    summary_lines = [
        f"{node['name']} ({node['type']}): {', '.join(node.get('relationships', []))}"
        for node in graph_context
    ]
    output = f"Graph context for {repository}:\n" + "\n".join(summary_lines)

    step: AgentStep = {
        "agent": "knowledge_graph",
        "status": "completed",
        "output": output,
        "duration_ms": (time.perf_counter() - start) * 1000,
        "metadata": {"nodes": len(graph_context)},
    }

    trace = list(state.get("agent_trace", []))
    trace.append(step)

    return {"graph_context": graph_context, "agent_trace": trace}
