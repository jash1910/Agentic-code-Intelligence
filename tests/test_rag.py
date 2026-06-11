"""Tests for hybrid RAG retrieval."""

import pytest

from rag.hybrid_retriever import BM25Index, HybridRetriever, reciprocal_rank_fusion


class TestBM25:
    def test_search_returns_results(self):
        bm25 = BM25Index()
        docs = [
            {"content": "ChatPromptTemplate is a prompt template for chat models"},
            {"content": "AgentExecutor runs agents with tools"},
        ]
        bm25.add_documents(docs)
        results = bm25.search("ChatPromptTemplate", k=2)
        assert len(results) > 0
        assert results[0][1] > 0


class TestRRF:
    def test_fusion_merges_lists(self):
        list1 = [("doc_a", 0.9), ("doc_b", 0.5)]
        list2 = [("doc_b", 0.8), ("doc_c", 0.3)]
        fused = reciprocal_rank_fusion([list1, list2])
        ids = [d[0] for d in fused]
        assert "doc_b" in ids
        assert fused[0][1] >= fused[-1][1]


class TestHybridRetriever:
    def test_retrieve_returns_docs(self):
        retriever = HybridRetriever()
        results = retriever.retrieve("ChatPromptTemplate", repository="langchain", k=3)
        assert len(results) > 0
        assert "source" in results[0]
        assert "score" in results[0]
