"""Hybrid RAG: ChromaDB vector search + BM25 keyword search with Reciprocal Rank Fusion."""

from __future__ import annotations

import math
import os
import re
import sys
from collections import defaultdict
from typing import Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

SUPPORTED_REPOSITORIES = [
    "langchain",
    "crewai",
    "llamaindex",
    "pydantic-ai",
    "haystack",
    "dspy",
    "autogen",
]

REPO_PATHS: dict[str, str] = {
    "langchain": os.path.join(BASE_DIR, "data_collection", "langchain_repo"),
}


class BM25Index:
    """Lightweight BM25 implementation without external dependencies."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[dict[str, Any]] = []
        self.doc_lengths: list[int] = []
        self.avgdl: float = 0.0
        self.df: dict[str, int] = defaultdict(int)
        self.doc_freqs: list[dict[str, int]] = []
        self.n_docs: int = 0

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.doc_freqs = []
        self.df = defaultdict(int)
        total_len = 0

        for doc in documents:
            tokens = self._tokenize(doc.get("content", ""))
            freq: dict[str, int] = defaultdict(int)
            for token in tokens:
                freq[token] += 1
            self.doc_freqs.append(dict(freq))
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)
            for token in freq:
                self.df[token] += 1

        self.n_docs = len(documents)
        self.avgdl = total_len / max(self.n_docs, 1)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        scores: list[tuple[int, float]] = []

        for idx, doc_freq in enumerate(self.doc_freqs):
            score = 0.0
            dl = self.doc_lengths[idx]
            for token in query_tokens:
                if token not in doc_freq:
                    continue
                tf = doc_freq[token]
                df = self.df[token]
                idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                score += idf * (tf * (self.k1 + 1)) / max(denom, 1e-9)
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = defaultdict(float)

    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """Production hybrid retriever combining ChromaDB and BM25 with RRF."""

    def __init__(self, chroma_dir: str | None = None) -> None:
        self.chroma_dir = chroma_dir or os.path.join(BASE_DIR, "rag", "chroma_db")
        self.embeddings = None
        self.vector_store = None
        self.bm25_indices: dict[str, BM25Index] = {}
        self.corpus: dict[str, list[dict[str, Any]]] = {}
        self._initialized = False
        self._initialize()

    def _initialize(self) -> None:
        try:
            from langchain_chroma import Chroma
            from langchain_community.embeddings import HuggingFaceEmbeddings

            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            if os.path.exists(self.chroma_dir):
                self.vector_store = Chroma(
                    persist_directory=self.chroma_dir,
                    embedding_function=self.embeddings,
                )
            self._load_corpus()
            self._initialized = True
        except Exception as exc:
            print(f"[HybridRetriever] Init warning: {exc}")
            self._load_fallback_corpus()

    def _load_fallback_corpus(self) -> None:
        fallback = [
            {
                "id": "lc-1",
                "source": "langchain/prompts/chat.py",
                "type": "code",
                "content": (
                    "class ChatPromptTemplate(BasePromptTemplate):\n"
                    "    \"\"\"Prompt template for chat models.\"\"\"\n"
                    "    messages: List[BaseMessagePromptTemplate]"
                ),
                "repository": "langchain",
            },
            {
                "id": "lc-2",
                "source": "langchain/agents/agent.py",
                "type": "code",
                "content": (
                    "class AgentExecutor(Chain):\n"
                    "    \"\"\"Agent that uses tools.\"\"\"\n"
                    "    agent: BaseSingleActionAgent\n"
                    "    tools: List[BaseTool]"
                ),
                "repository": "langchain",
            },
            {
                "id": "lc-3",
                "source": "docs/lcel.md",
                "type": "docs",
                "content": (
                    "LangChain Expression Language (LCEL) allows composing runnables "
                    "with the pipe operator: chain = prompt | llm | output_parser"
                ),
                "repository": "langchain",
            },
        ]
        for repo in SUPPORTED_REPOSITORIES:
            docs = [d for d in fallback if d["repository"] == repo] or fallback
            self.corpus[repo] = docs
            bm25 = BM25Index()
            bm25.add_documents(docs)
            self.bm25_indices[repo] = bm25

    def _load_corpus(self) -> None:
        """Build BM25 indices from ChromaDB or repository files."""
        all_docs: list[dict[str, Any]] = []

        if self.vector_store is not None:
            try:
                collection = self.vector_store._collection
                data = collection.get(include=["documents", "metadatas"])
                for i, content in enumerate(data.get("documents", []) or []):
                    meta = (data.get("metadatas") or [{}])[i] or {}
                    all_docs.append({
                        "id": f"vec-{i}",
                        "source": meta.get("source", "unknown"),
                        "type": meta.get("type", "code"),
                        "content": content or "",
                        "repository": meta.get("repository", "langchain"),
                    })
            except Exception:
                pass

        if not all_docs:
            self._load_fallback_corpus()
            return

        for repo in SUPPORTED_REPOSITORIES:
            repo_docs = [d for d in all_docs if d.get("repository") == repo]
            if not repo_docs:
                repo_docs = all_docs[:20]
            self.corpus[repo] = repo_docs
            bm25 = BM25Index()
            bm25.add_documents(repo_docs)
            self.bm25_indices[repo] = bm25

    def _vector_search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self.vector_store is None:
            return []

        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            ranked: list[tuple[str, float]] = []
            for doc, score in results:
                doc_id = doc.metadata.get("source", doc.page_content[:50])
                ranked.append((doc_id, 1.0 / (1.0 + score)))
            return ranked
        except Exception:
            return []

    def _bm25_search(self, query: str, repository: str, k: int) -> list[tuple[str, float]]:
        bm25 = self.bm25_indices.get(repository)
        docs = self.corpus.get(repository, [])
        if not bm25 or not docs:
            return []

        results = bm25.search(query, k=k)
        return [(docs[idx]["id"], score) for idx, score in results if score > 0]

    def retrieve(self, query: str, repository: str = "langchain", k: int = 5) -> list[dict[str, Any]]:
        """Hybrid retrieval with RRF fusion."""
        if repository not in self.corpus:
            repository = "langchain"

        vector_ranked = self._vector_search(query, k=k * 2)
        bm25_ranked = self._bm25_search(query, repository, k=k * 2)
        fused = reciprocal_rank_fusion([vector_ranked, bm25_ranked])

        doc_lookup: dict[str, dict[str, Any]] = {}
        for doc in self.corpus.get(repository, []):
            doc_lookup[doc["id"]] = doc
            doc_lookup[doc.get("source", "")] = doc

        if self.vector_store:
            try:
                for doc, _ in self.vector_store.similarity_search_with_score(query, k=k):
                    key = doc.metadata.get("source", "")
                    doc_lookup[key] = {
                        "id": key,
                        "source": key,
                        "type": doc.metadata.get("type", "code"),
                        "content": doc.page_content,
                        "repository": repository,
                    }
            except Exception:
                pass

        results: list[dict[str, Any]] = []
        seen: set[str] = set()

        for doc_key, score in fused[:k]:
            doc = doc_lookup.get(doc_key)
            if not doc or doc_key in seen:
                continue
            seen.add(doc_key)
            results.append({
                "source": doc.get("source", doc_key),
                "type": doc.get("type", "code"),
                "content": doc.get("content", ""),
                "score": round(score, 4),
                "retrieval_method": "hybrid_rrf",
            })

        if not results:
            for doc in self.corpus.get(repository, [])[:k]:
                results.append({
                    "source": doc["source"],
                    "type": doc["type"],
                    "content": doc["content"],
                    "score": 0.5,
                    "retrieval_method": "fallback",
                })

        return results

# ArchonAI platform module - architecture reference
