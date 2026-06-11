"""Memory system — short-term session + long-term ChromaDB store."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionMemory:
    session_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class MemoryManager:
    """Manages short-term session memory and long-term vector memory."""

    def __init__(self, chroma_dir: str | None = None) -> None:
        self.sessions: dict[str, SessionMemory] = {}
        self.chroma_dir = chroma_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "rag", "memory_db",
        )
        self.long_term_store = None
        self._init_long_term()

    def _init_long_term(self) -> None:
        try:
            from langchain_chroma import Chroma
            from langchain_community.embeddings import HuggingFaceEmbeddings

            os.makedirs(self.chroma_dir, exist_ok=True)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.long_term_store = Chroma(
                persist_directory=self.chroma_dir,
                embedding_function=embeddings,
                collection_name="archon_memory",
            )
        except Exception as exc:
            print(f"[Memory] Long-term store unavailable: {exc}")

    def get_session(self, session_id: str) -> SessionMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionMemory(session_id=session_id)
        return self.sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        session = self.get_session(session_id)
        session.messages.append({"role": role, "content": content})
        session.updated_at = time.time()

    def set_preference(self, session_id: str, key: str, value: Any) -> None:
        session = self.get_session(session_id)
        session.preferences[key] = value
        session.updated_at = time.time()

    def get_context(self, session_id: str, max_messages: int = 10) -> str:
        session = self.get_session(session_id)
        recent = session.messages[-max_messages:]
        return "\n".join(f"{m['role']}: {m['content']}" for m in recent)

    def store_long_term(self, session_id: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        if not self.long_term_store:
            return
        meta = metadata or {}
        meta["session_id"] = session_id
        meta["timestamp"] = time.time()
        try:
            self.long_term_store.add_texts([content], metadatas=[meta])
        except Exception:
            pass

    def recall_long_term(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        if not self.long_term_store:
            return []
        try:
            docs = self.long_term_store.similarity_search(query, k=k)
            return [
                {"content": d.page_content, "metadata": d.metadata}
                for d in docs
            ]
        except Exception:
            return []
