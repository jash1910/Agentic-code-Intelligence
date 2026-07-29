"""Neo4j graph store with NetworkX fallback."""

from __future__ import annotations

import os
import re
from typing import Any

from knowledge_graph.networkx_store import NetworkXGraphStore

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "archonai")


class Neo4jGraphStore:
    """Neo4j-backed knowledge graph with automatic fallback."""

    def __init__(self) -> None:
        self.driver = None
        self.fallback = NetworkXGraphStore()
        self._connect()

    def _connect(self) -> None:
        try:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("[KnowledgeGraph] Connected to Neo4j")
        except Exception as exc:
            print(f"[KnowledgeGraph] Neo4j unavailable ({exc}), using NetworkX fallback")
            self.driver = None

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def get_related(self, entity: str, repository: str = "langchain", depth: int = 2) -> dict[str, Any]:
        if self.driver:
            try:
                with self.driver.session() as session:
                    result = session.run(
                        """
                        MATCH (n)-[r*1..$depth]-(m)
                        WHERE toLower(n.name) CONTAINS toLower($entity)
                        RETURN n.name AS name, labels(n) AS labels,
                               type(r) AS rel, m.name AS related
                        LIMIT 20
                        """,
                        entity=entity,
                        depth=depth,
                    )
                    records = list(result)
                    relationships = [
                        f"{rec['rel']}:{rec['related']}"
                        for rec in records
                        if rec.get("related")
                    ]
                    if records:
                        return {
                            "type": records[0].get("labels", ["class"])[0],
                            "relationships": relationships,
                        }
            except Exception:
                pass

        neighbors = self.fallback.get_neighbors(entity, depth)
        return {
            "type": "class",
            "relationships": [
                f"{n['relationship']}:{n['entity']}" for n in neighbors
            ],
        }

    def extract_entities(self, query: str) -> list[str]:
        """Extract likely code entities from natural language query."""
        patterns = [
            r"`([A-Za-z_][A-Za-z0-9_]*)`",
            r"\b([A-Z][a-zA-Z0-9]*(?:Template|Chain|Agent|Executor|Retriever|Parser|Tool|Memory|Loader))\b",
            r"\b(create_\w+|Base[A-Z]\w+)\b",
        ]
        entities: list[str] = []
        for pattern in patterns:
            entities.extend(re.findall(pattern, query))
        return list(dict.fromkeys(entities))[:5] or ["ChatPromptTemplate"]

# ArchonAI platform module - architecture reference
