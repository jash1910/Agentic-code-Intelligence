"""Knowledge graph builder — extract code structure via AST parsing."""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphEntity:
    name: str
    entity_type: str
    file_path: str
    line: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRelationship:
    source: str
    target: str
    rel_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class CodeGraphBuilder:
    """Build knowledge graph from Python source files using AST."""

    def __init__(self) -> None:
        self.entities: dict[str, GraphEntity] = {}
        self.relationships: list[GraphRelationship] = []

    def parse_file(self, file_path: str, repository: str = "langchain") -> None:
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                source = f.read()
            tree = ast.parse(source)
        except (SyntaxError, OSError):
            return

        module_name = os.path.splitext(os.path.basename(file_path))[0]
        module_id = f"{repository}:{module_name}"

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_id = f"{module_id}.{node.name}"
                bases = [self._resolve_base(b) for b in node.bases]
                self.entities[class_id] = GraphEntity(
                    name=node.name,
                    entity_type="class",
                    file_path=file_path,
                    line=node.lineno,
                    metadata={"bases": bases},
                )
                for base in bases:
                    if base:
                        self.relationships.append(
                            GraphRelationship(class_id, base, "INHERITS_FROM")
                        )
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_id = f"{class_id}.{item.name}"
                        self.entities[method_id] = GraphEntity(
                            name=item.name,
                            entity_type="method",
                            file_path=file_path,
                            line=item.lineno,
                        )
                        self.relationships.append(
                            GraphRelationship(method_id, class_id, "BELONGS_TO")
                        )

            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                func_id = f"{module_id}.{node.name}"
                self.entities[func_id] = GraphEntity(
                    name=node.name,
                    entity_type="function",
                    file_path=file_path,
                    line=node.lineno,
                )

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.relationships.append(
                        GraphRelationship(module_id, alias.name, "IMPORTS")
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                self.relationships.append(
                    GraphRelationship(module_id, node.module, "IMPORTS_FROM")
                )

    def _resolve_base(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def parse_directory(self, directory: str, repository: str = "langchain") -> None:
        if not os.path.isdir(directory):
            return
        for root, _, files in os.walk(directory):
            for fname in files:
                if fname.endswith(".py"):
                    self.parse_file(os.path.join(root, fname), repository)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": {
                k: {
                    "name": v.name,
                    "type": v.entity_type,
                    "file": v.file_path,
                    "line": v.line,
                    "metadata": v.metadata,
                }
                for k, v in self.entities.items()
            },
            "relationships": [
                {"source": r.source, "target": r.target, "type": r.rel_type}
                for r in self.relationships
            ],
        }
