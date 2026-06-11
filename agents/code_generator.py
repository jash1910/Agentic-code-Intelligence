"""Code Generator Agent — produces code, explanations, and architecture diagrams."""

from __future__ import annotations

import re
import time
from typing import Any

from agents.state import AgentStep, ArchonState


def _build_context_prompt(state: ArchonState) -> str:
    parts = [
        f"Repository: {state.get('repository', 'langchain')}",
        f"Plan: {state.get('plan', '')}",
        "",
        "### Retrieved Context:",
    ]
    for doc in state.get("retrieved_docs", [])[:5]:
        parts.append(f"- [{doc.get('source', '?')}] {doc.get('content', '')[:400]}")

    parts.append("\n### Knowledge Graph:")
    for node in state.get("graph_context", [])[:5]:
        rels = ", ".join(node.get("relationships", []))
        parts.append(f"- {node.get('name')}: {rels}")

    parts.append(f"\n### User Query:\n{state.get('user_query', '')}")
    return "\n".join(parts)


def _extract_code_blocks(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\n([\s\S]*?)```", text)
    return "\n\n".join(blocks) if blocks else ""


def _generate_mermaid(state: ArchonState) -> str:
    repo = state.get("repository", "langchain")
    nodes = state.get("graph_context", [])
    lines = ["graph TD"]
    for i, node in enumerate(nodes[:4]):
        nid = f"N{i}"
        lines.append(f'    {nid}["{node.get("name", "Node")}"]')
        for rel in node.get("relationships", [])[:2]:
            if ":" in rel:
                rel_type, target = rel.split(":", 1)
                lines.append(f'    {nid} -->|{rel_type}| T{i}_{target.replace(" ", "")}["{target}"]')
    if len(lines) == 1:
        lines.extend([
            '    A["User Query"] --> B["Planner"]',
            '    B --> C["Retriever"]',
            '    C --> D["Code Generator"]',
            f'    D --> E["{repo.title()} Response"]',
        ])
    return "\n".join(lines)


def code_generator_node(state: ArchonState, llm_manager: Any | None = None) -> dict[str, Any]:
    """LangGraph node: generate code and explanations from enriched context."""
    start = time.perf_counter()
    prompt = _build_context_prompt(state)
    generated_explanation = ""
    generated_code = ""

    if llm_manager is not None:
        try:
            tokens: list[str] = []
            for token in llm_manager.stream_generate(
                f"You are ArchonAI, an expert code assistant.\n{prompt}\n\nProvide a detailed, "
                f"cited explanation with Python code examples where appropriate.\n\n### Response:\n"
            ):
                tokens.append(token)
            generated_explanation = "".join(tokens)
            generated_code = _extract_code_blocks(generated_explanation)
        except Exception as exc:
            generated_explanation = f"Generation error: {exc}"
    else:
        query = state.get("user_query", "").lower()
        repo = state.get("repository", "langchain")
        generated_explanation = (
            f"### ArchonAI Analysis ({repo})\n\n"
            f"Based on hybrid retrieval and knowledge graph context, here is the answer "
            f"to your question about `{query[:80]}...`:\n\n"
            f"```python\n"
            f"from langchain.prompts import ChatPromptTemplate\n\n"
            f"prompt = ChatPromptTemplate.from_messages([\n"
            f"    ('system', 'You are a helpful assistant.'),\n"
            f"    ('human', '{{input}}'),\n"
            f"])\n"
            f"```\n\n"
            f"See retrieved citations for source verification."
        )
        generated_code = (
            "from langchain.prompts import ChatPromptTemplate\n\n"
            "prompt = ChatPromptTemplate.from_messages([\n"
            "    ('system', 'You are a helpful assistant.'),\n"
            "    ('human', '{input}'),\n"
            "])"
        )

    architecture_diagram = _generate_mermaid(state)

    step: AgentStep = {
        "agent": "code_generator",
        "status": "completed",
        "output": f"Generated {len(generated_code)} chars of code",
        "duration_ms": (time.perf_counter() - start) * 1000,
        "metadata": {"has_code": bool(generated_code)},
    }

    trace = list(state.get("agent_trace", []))
    trace.append(step)

    return {
        "generated_code": generated_code,
        "generated_explanation": generated_explanation,
        "architecture_diagram": architecture_diagram,
        "agent_trace": trace,
    }
