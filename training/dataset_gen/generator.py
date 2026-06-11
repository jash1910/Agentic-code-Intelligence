"""Synthetic training dataset generation from repository AST parsing."""

from __future__ import annotations

import ast
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterator


class SyntheticDatasetGenerator:
    """Generate instruction-tuning datasets from code repositories."""

    TASK_TEMPLATES = {
        "qa": "Explain what `{name}` does in {repo}.",
        "completion": "Complete the implementation of `{name}`:",
        "debug": "Fix the bug in this `{name}` implementation:\n{code}",
        "architecture": "Describe the architecture and dependencies of `{name}`.",
        "refactor": "Refactor `{name}` to use modern {repo} patterns.",
    }

    def __init__(self, repo_path: str, repository: str = "langchain") -> None:
        self.repo_path = Path(repo_path)
        self.repository = repository
        self.samples: list[dict[str, str]] = []

    def _extract_entities(self, file_path: Path) -> Iterator[dict[str, Any]]:
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except (SyntaxError, OSError):
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [
                    n.name for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                docstring = ast.get_docstring(node) or ""
                yield {
                    "name": node.name,
                    "type": "class",
                    "methods": methods,
                    "docstring": docstring,
                    "file": str(file_path.relative_to(self.repo_path)),
                    "line": node.lineno,
                }
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                docstring = ast.get_docstring(node) or ""
                yield {
                    "name": node.name,
                    "type": "function",
                    "docstring": docstring,
                    "file": str(file_path.relative_to(self.repo_path)),
                    "line": node.lineno,
                }

    def generate(self, max_samples: int = 500) -> list[dict[str, str]]:
        if not self.repo_path.exists():
            return self._fallback_samples()

        for py_file in self.repo_path.rglob("*.py"):
            if "test" in str(py_file).lower() or "__pycache__" in str(py_file):
                continue
            for entity in self._extract_entities(py_file):
                for task_type, template in self.TASK_TEMPLATES.items():
                    instruction = template.format(
                        name=entity["name"],
                        repo=self.repository,
                        code=entity.get("docstring", "")[:200],
                    )
                    response = self._generate_response(entity, task_type)
                    if self._quality_filter(instruction, response):
                        self.samples.append({
                            "instruction": instruction,
                            "response": response,
                            "task_type": task_type,
                            "source_file": entity["file"],
                            "repository": self.repository,
                        })
                    if len(self.samples) >= max_samples:
                        return self.samples

        return self.samples or self._fallback_samples()

    def _generate_response(self, entity: dict[str, Any], task_type: str) -> str:
        name = entity["name"]
        doc = entity.get("docstring", "")
        methods = entity.get("methods", [])

        if task_type == "qa":
            return f"`{name}` is a {entity['type']} defined in `{entity['file']}`. {doc or 'See source for details.'}"
        if task_type == "architecture":
            deps = ", ".join(methods[:5]) if methods else "standalone"
            return f"`{name}` architecture: type={entity['type']}, methods=[{deps}], file={entity['file']}"
        if task_type == "completion":
            return f"```python\nclass {name}:\n    \"\"\"{doc or 'Auto-generated.'}\"\"\"\n    pass\n```"
        if task_type == "debug":
            return f"Fixed `{name}` by correcting imports and adding error handling in `{entity['file']}`."
        return f"Refactored `{name}` to use LCEL patterns with Runnable composition."

    def _quality_filter(self, instruction: str, response: str) -> bool:
        if len(instruction) < 20 or len(response) < 30:
            return False
        if len(response) > 5000:
            return False
        return True

    def _fallback_samples(self) -> list[dict[str, str]]:
        return [
            {
                "instruction": "How do you use ChatPromptTemplate?",
                "response": "Use ChatPromptTemplate.from_messages with role tuples.",
                "task_type": "qa",
                "source_file": "prompts/chat.py",
                "repository": self.repository,
            },
            {
                "instruction": "Explain AgentExecutor dependencies.",
                "response": "AgentExecutor depends on BaseAgent and BaseTool.",
                "task_type": "architecture",
                "source_file": "agents/agent.py",
                "repository": self.repository,
            },
        ]

    def export_jsonl(self, output_path: str, split_ratio: float = 0.9) -> tuple[str, str]:
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * split_ratio)
        train = self.samples[:split_idx]
        val = self.samples[split_idx:]

        train_path = output_path.replace(".jsonl", "_train.jsonl") if split_ratio < 1 else output_path
        val_path = output_path.replace(".jsonl", "_val.jsonl")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(train_path, "w") as f:
            for s in train:
                f.write(json.dumps(s) + "\n")
        with open(val_path, "w") as f:
            for s in val:
                f.write(json.dumps(s) + "\n")

        return train_path, val_path
