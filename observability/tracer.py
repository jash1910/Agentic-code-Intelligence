"""Unified observability: LangSmith, OpenTelemetry, and Phoenix."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any


class ArchonTracer:
    """Production observability tracer with multi-backend support."""

    def __init__(self) -> None:
        self.metrics: dict[str, Any] = defaultdict(list)
        self.langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        self.otel_enabled = os.getenv("OTEL_ENABLED", "false").lower() == "true"
        self.phoenix_enabled = os.getenv("PHOENIX_ENABLED", "false").lower() == "true"
        self._setup_backends()

    def _setup_backends(self) -> None:
        if self.langsmith_enabled:
            os.environ.setdefault("LANGCHAIN_PROJECT", "archonai")
            print("[Observability] LangSmith tracing enabled")

        if self.phoenix_enabled:
            try:
                import phoenix as px
                px.launch_app()
                print("[Observability] Arize Phoenix launched")
            except Exception as exc:
                print(f"[Observability] Phoenix unavailable: {exc}")

        if self.otel_enabled:
            try:
                from opentelemetry import trace
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.resources import Resource

                provider = TracerProvider(resource=Resource.create({"service.name": "archonai"}))
                trace.set_tracer_provider(provider)
                self.tracer = trace.get_tracer("archonai")
                print("[Observability] OpenTelemetry initialized")
            except Exception as exc:
                print(f"[Observability] OTEL unavailable: {exc}")
                self.tracer = None
        else:
            self.tracer = None

    def trace_agent(self, agent_name: str, duration_ms: float, metadata: dict | None = None) -> None:
        entry = {
            "agent": agent_name,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self.metrics["agent_traces"].append(entry)

        if self.tracer:
            with self.tracer.start_as_current_span(f"agent.{agent_name}") as span:
                span.set_attribute("duration_ms", duration_ms)
                for k, v in (metadata or {}).items():
                    span.set_attribute(k, str(v))

    def trace_query(
        self,
        query: str,
        latency_ms: float,
        tokens: int = 0,
        confidence: float = 0.0,
        retrieval_quality: float = 0.0,
    ) -> None:
        self.metrics["queries"].append({
            "query": query[:200],
            "latency_ms": latency_ms,
            "tokens": tokens,
            "confidence": confidence,
            "retrieval_quality": retrieval_quality,
            "timestamp": time.time(),
        })

    def get_metrics(self) -> dict[str, Any]:
        queries = self.metrics.get("queries", [])
        traces = self.metrics.get("agent_traces", [])
        avg_latency = (
            sum(q["latency_ms"] for q in queries) / len(queries) if queries else 0
        )
        return {
            "total_queries": len(queries),
            "total_agent_steps": len(traces),
            "avg_latency_ms": round(avg_latency, 2),
            "backends": {
                "langsmith": self.langsmith_enabled,
                "opentelemetry": self.otel_enabled,
                "phoenix": self.phoenix_enabled,
            },
            "recent_queries": queries[-10:],
            "recent_traces": traces[-20:],
        }
