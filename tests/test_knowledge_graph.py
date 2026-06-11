"""Tests for knowledge graph."""

import pytest

from knowledge_graph.builder import CodeGraphBuilder
from knowledge_graph.networkx_store import NetworkXGraphStore


SAMPLE_CODE = '''
class ChatPromptTemplate(BasePromptTemplate):
    """Chat prompt template."""
    def format(self):
        pass

def helper():
    return 1
'''


class TestGraphBuilder:
    def test_parse_extracts_classes(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text(SAMPLE_CODE)
        builder = CodeGraphBuilder()
        builder.parse_file(str(f))
        assert len(builder.entities) >= 2
        assert any("ChatPromptTemplate" in e.name for e in builder.entities.values())

    def test_inheritance_relationship(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text(SAMPLE_CODE)
        builder = CodeGraphBuilder()
        builder.parse_file(str(f))
        inherits = [r for r in builder.relationships if r.rel_type == "INHERITS_FROM"]
        assert len(inherits) >= 1


class TestNetworkXStore:
    def test_get_neighbors(self):
        store = NetworkXGraphStore()
        neighbors = store.get_neighbors("ChatPromptTemplate")
        assert isinstance(neighbors, list)

    def test_extract_entities(self):
        from knowledge_graph.neo4j_store import Neo4jGraphStore
        store = Neo4jGraphStore()
        entities = store.extract_entities("How does ChatPromptTemplate work with AgentExecutor?")
        assert "ChatPromptTemplate" in entities
