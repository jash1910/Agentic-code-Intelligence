"""NetworkX fallback graph store when Neo4j is unavailable."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import networkx as nx

from knowledge_graph.builder import CodeGraphBuilder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_CACHE = os.path.join(BASE_DIR, "knowledge_graph", "cache", "graph.json")


class NetworkXGraphStore:
    """In-memory graph store with persistence for development."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._load_or_build()

    def _load_or_build(self) -> None:
        if os.path.exists(GRAPH_CACHE):
            try:
                with open(GRAPH_CACHE, encoding="utf-8") as f:
                    data = json.load(f)
                for entity_id, meta in data.get("entities", {}).items():
                    self.graph.add_node(entity_id, **meta)
                for rel in data.get("relationships", []):
                    self.graph.add_edge(
                        rel["source"], rel["target"], type=rel["type"]
                    )
                return
            except (json.JSONDecodeError, OSError):
                pass
        self._build_default_graph()

    def _build_default_graph(self) -> None:
        defaults = [
            ("ChatPromptTemplate", "class", "INHERITS_FROM", "BasePromptTemplate"),
            ("AgentExecutor", "class", "DEPENDS_ON", "BaseAgent"),
            ("AgentExecutor", "class", "USES", "BaseTool"),
            ("LLMChain", "class", "INHERITS_FROM", "Chain"),
            ("LLMChain", "class", "USES", "BasePromptTemplate"),
            ("RunnableSequence", "class", "IMPLEMENTS", "Runnable"),
            ("create_tool_calling_agent", "function", "RETURNS", "RunnableAgent"),
        ]
        for name, etype, rel, target in defaults:
            self.graph.add_node(name, name=name, type=etype)
            self.graph.add_node(target, name=target, type="class")
            self.graph.add_edge(name, target, type=rel)

        os.makedirs(os.path.dirname(GRAPH_CACHE), exist_ok=True)
        self.save()

    def save(self) -> None:
        entities = {n: dict(self.graph.nodes[n]) for n in self.graph.nodes}
        relationships = [
            {"source": u, "target": v, "type": d.get("type", "RELATED")}
            for u, v, d in self.graph.edges(data=True)
        ]
        with open(GRAPH_CACHE, "w", encoding="utf-8") as f:
            json.dump({"entities": entities, "relationships": relationships}, f, indent=2)

    def ingest_builder(self, builder: CodeGraphBuilder) -> int:
        data = builder.to_dict()
        for eid, meta in data["entities"].items():
            self.graph.add_node(eid, **meta)
        for rel in data["relationships"]:
            self.graph.add_edge(rel["source"], rel["target"], type=rel["type"])
        self.save()
        return len(data["entities"])

    def get_neighbors(self, entity: str, depth: int = 2) -> list[dict[str, Any]]:
        matches = [
            n for n in self.graph.nodes
            if entity.lower() in n.lower() or entity.lower() in self.graph.nodes[n].get("name", "").lower()
        ]
        if not matches:
            return []

        start = matches[0]
        results: list[dict[str, Any]] = []
        visited = {start}

        frontier = [(start, 0)]
        while frontier:
            node, d = frontier.pop(0)
            if d >= depth:
                continue
            for _, neighbor, data in self.graph.out_edges(node, data=True):
                if neighbor not in visited:
                    visited.add(neighbor)
                    results.append({
                        "entity": neighbor,
                        "relationship": data.get("type", "RELATED"),
                        "depth": d + 1,
                    })
                    frontier.append((neighbor, d + 1))
            for predecessor, _, data in self.graph.in_edges(node, data=True):
                if predecessor not in visited:
                    visited.add(predecessor)
                    results.append({
                        "entity": predecessor,
                        "relationship": f"INCOMING_{data.get('type', 'RELATED')}",
                        "depth": d + 1,
                    })
                    frontier.append((predecessor, d + 1))

        return results
