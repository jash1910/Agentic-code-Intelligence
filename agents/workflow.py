"""LangGraph workflow orchestrating the full ArchonAI agent pipeline."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

from langgraph.graph import END, StateGraph

from agents.code_generator import code_generator_node
from agents.critic import critic_node
from agents.execution import execution_node
from agents.formatter import formatter_node
from agents.knowledge_graph_agent import knowledge_graph_node
from agents.planner import planner_node
from agents.retriever import retriever_node
from agents.state import ArchonState


class ArchonWorkflow:
    """Production LangGraph agent workflow for ArchonAI."""

    def __init__(
        self,
        llm_manager: Any | None = None,
        hybrid_retriever: Any | None = None,
        graph_query: Any | None = None,
        sandbox: Any | None = None,
    ) -> None:
        self.llm_manager = llm_manager
        self.hybrid_retriever = hybrid_retriever
        self.graph_query = graph_query
        self.sandbox = sandbox
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(ArchonState)

        workflow.add_node("planner", lambda s: planner_node(s, self.llm_manager))
        workflow.add_node("retriever", lambda s: retriever_node(s, self.hybrid_retriever))
        workflow.add_node("knowledge_graph", lambda s: knowledge_graph_node(s, self.graph_query))
        workflow.add_node("code_generator", lambda s: code_generator_node(s, self.llm_manager))
        workflow.add_node("critic", lambda s: critic_node(s, self.llm_manager))
        workflow.add_node("execution", lambda s: execution_node(s, self.sandbox))
        workflow.add_node("formatter", formatter_node)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "retriever")
        workflow.add_edge("retriever", "knowledge_graph")
        workflow.add_edge("knowledge_graph", "code_generator")
        workflow.add_edge("code_generator", "critic")
        workflow.add_edge("critic", "execution")
        workflow.add_edge("execution", "formatter")
        workflow.add_edge("formatter", END)

        return workflow.compile()

    def invoke(
        self,
        query: str,
        session_id: str = "default",
        repository: str | None = None,
    ) -> ArchonState:
        """Run the full agent pipeline synchronously."""
        initial: ArchonState = {
            "user_query": query,
            "session_id": session_id,
            "repository": repository or "",
            "messages": [],
            "agent_trace": [],
            "retry_count": 0,
            "max_retries": 3,
        }
        return self.graph.invoke(initial)

    async def stream(
        self,
        query: str,
        session_id: str = "default",
        repository: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream agent events for SSE consumption."""
        yield {"event": "agent_start", "data": {"query": query, "session_id": session_id}}

        loop = asyncio.get_event_loop()
        start = time.perf_counter()

        def run_pipeline() -> ArchonState:
            return self.invoke(query, session_id, repository)

        result = await loop.run_in_executor(None, run_pipeline)

        for step in result.get("agent_trace", []):
            yield {"event": "agent_step", "data": step}
            await asyncio.sleep(0.01)

        yield {
            "event": "citations",
            "data": {"citations": result.get("citations", [])},
        }

        response = result.get("final_response", "")
        words = response.split(" ")
        chunk_size = 4
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size]) + " "
            yield {"event": "token", "data": {"text": chunk}}
            await asyncio.sleep(0.02)

        yield {
            "event": "metrics",
            "data": {
                "confidence_score": result.get("confidence_score", 0.0),
                "critic_score": result.get("critic_score", 0.0),
                "latency_ms": (time.perf_counter() - start) * 1000,
                "agent_trace": result.get("agent_trace", []),
                "architecture_diagram": result.get("architecture_diagram", ""),
            },
        }
        yield {"event": "done", "data": {"status": "completed"}}


def create_workflow(
    llm_manager: Any | None = None,
    hybrid_retriever: Any | None = None,
    graph_query: Any | None = None,
    sandbox: Any | None = None,
) -> ArchonWorkflow:
    """Factory function for ArchonWorkflow."""
    return ArchonWorkflow(
        llm_manager=llm_manager,
        hybrid_retriever=hybrid_retriever,
        graph_query=graph_query,
        sandbox=sandbox,
    )
