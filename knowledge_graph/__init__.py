"""Knowledge graph query interface."""

from knowledge_graph.neo4j_store import Neo4jGraphStore

__all__ = ["Neo4jGraphStore", "GraphQuery"]

GraphQuery = Neo4jGraphStore
