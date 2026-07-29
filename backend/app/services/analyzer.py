"""Static Code Analyzer service using AST-based metrics calculation.

Computes code volume, cyclomatic complexity, maintainability index, and Halstead metrics.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Dict, List, Set, Tuple


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor to calculate Cyclomatic Complexity of functions and classes."""

    def __init__(self) -> None:
        self.complexity: int = 1
        self.nodes_visited: int = 0

    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_And(self, node: ast.And) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_Or(self, node: ast.Or) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.complexity += 1
        self.generic_visit(node)


class HalsteadVisitor(ast.NodeVisitor):
    """AST visitor to count distinct and total operators and operands."""

    def __init__(self) -> None:
        self.operators: List[str] = []
        self.operands: List[str] = []

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.operators.append(node.op.__class__.__name__)
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        self.operators.append(node.op.__class__.__name__)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            self.operators.append(op.__class__.__name__)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.operators.append("Assign")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.operators.append(node.op.__class__.__name__ + "Assign")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Load, ast.Store)):
            self.operands.append(node.id)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        self.operands.append(str(node.value))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.operators.append("Call")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.operands.append(node.attr)
        self.generic_visit(node)


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to extract all import dependencies from a file."""

    def __init__(self) -> None:
        self.imports: Set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.add(node.module)
        self.generic_visit(node)


class StaticCodeAnalyzer:
    """Core analysis engine for extracting Python source code metrics."""

    @staticmethod
    def parse_loc(code: str) -> Dict[str, int]:
        """Calculates physical, logical, blank, and comment lines of code."""
        lines = code.splitlines()
        total = len(lines)
        blanks = 0
        comments = 0
        sloc = 0  # source lines of code (non-blank, non-comment)

        in_multiline_comment = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                blanks += 1
                continue

            # Handle single-line docstrings/comments
            if (stripped.startswith('"""') and stripped.endswith('"""') and len(stripped) >= 6) or \
               (stripped.startswith("'''") and stripped.endswith("'''") and len(stripped) >= 6):
                comments += 1
                continue

            # Handle multiline string comments/docstrings
            if '"""' in stripped or "'''" in stripped:
                if stripped.count('"""') % 2 != 0 or stripped.count("'''") % 2 != 0:
                    in_multiline_comment = not in_multiline_comment
                    comments += 1
                    continue

            if in_multiline_comment:
                comments += 1
                continue

            if stripped.startswith("#"):
                comments += 1
            else:
                sloc += 1

        return {
            "total_lines": total,
            "blank_lines": blanks,
            "comment_lines": comments,
            "source_lines": sloc,
        }

    @staticmethod
    def calculate_cyclomatic_complexity(tree: ast.AST) -> Tuple[int, List[Dict[str, Any]]]:
        """Calculates the cyclomatic complexity of functions, classes, and module-level code."""
        module_visitor = ComplexityVisitor()
        module_visitor.visit(tree)
        module_complexity = module_visitor.complexity

        details = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_visitor = ComplexityVisitor()
                for child in node.body:
                    fn_visitor.visit(child)
                details.append({
                    "type": "function",
                    "name": node.name,
                    "lineno": node.lineno,
                    "complexity": fn_visitor.complexity,
                })
            elif isinstance(node, ast.ClassDef):
                class_visitor = ComplexityVisitor()
                for child in node.body:
                    class_visitor.visit(child)
                details.append({
                    "type": "class",
                    "name": node.name,
                    "lineno": node.lineno,
                    "complexity": class_visitor.complexity,
                })

        return module_complexity, details

    @staticmethod
    def calculate_halstead_metrics(tree: ast.AST) -> Dict[str, Any]:
        """Calculates Halstead complexity metrics for the target AST."""
        visitor = HalsteadVisitor()
        visitor.visit(tree)

        n1 = len(set(visitor.operators))  # distinct operators
        n2 = len(set(visitor.operands))   # distinct operands
        N1 = len(visitor.operators)      # total operators
        N2 = len(visitor.operands)       # total operands

        vocabulary = n1 + n2
        length = N1 + N2

        if vocabulary > 0:
            volume = length * math.log2(vocabulary)
        else:
            volume = 0.0

        if n2 > 0:
            difficulty = (n1 / 2) * (N2 / n2)
        else:
            difficulty = 0.0

        effort = difficulty * volume
        time_seconds = effort / 18.0 if effort > 0 else 0.0
        bugs = volume / 3000.0 if volume > 0 else 0.0

        return {
            "distinct_operators": n1,
            "distinct_operands": n2,
            "total_operators": N1,
            "total_operands": N2,
            "vocabulary": vocabulary,
            "length": length,
            "volume": round(volume, 2),
            "difficulty": round(difficulty, 2),
            "effort": round(effort, 2),
            "time_seconds": round(time_seconds, 2),
            "bugs": round(bugs, 4),
        }

    @staticmethod
    def calculate_maintainability_index(
        volume: float,
        complexity: int,
        source_lines: int
    ) -> float:
        """Calculates the Maintainability Index (MI) score."""
        if volume <= 0 or source_lines <= 0:
            return 100.0

        # Maintainability Index standard formula
        mi = 171.0 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(source_lines)
        # Normalize to 0-100 range
        mi_normalized = max(0.0, min(100.0, (mi * 100.0 / 171.0)))
        return round(mi_normalized, 2)

    def analyze_source_code(self, code: str) -> Dict[str, Any]:
        """Runs the full suite of static analyses on the provided source string."""
        loc_metrics = self.parse_loc(code)
        
        try:
            tree = ast.parse(code)
            syntax_valid = True
            error_message = None
        except SyntaxError as e:
            tree = None
            syntax_valid = False
            error_message = str(e)

        if not syntax_valid or tree is None:
            return {
                "syntax_valid": False,
                "error": error_message,
                "loc": loc_metrics,
                "complexity": {"module": 1, "details": []},
                "halstead": {},
                "maintainability_index": 0.0,
                "dependencies": []
            }

        # Calculate metrics
        module_complexity, details = self.calculate_cyclomatic_complexity(tree)
        halstead = self.calculate_halstead_metrics(tree)
        
        mi = self.calculate_maintainability_index(
            volume=halstead["volume"],
            complexity=module_complexity,
            source_lines=loc_metrics["source_lines"]
        )

        import_visitor = ImportVisitor()
        import_visitor.visit(tree)
        dependencies = sorted(list(import_visitor.imports))

        return {
            "syntax_valid": True,
            "loc": loc_metrics,
            "complexity": {
                "module_complexity": module_complexity,
                "elements": details,
            },
            "halstead": halstead,
            "maintainability_index": mi,
            "dependencies": dependencies,
        }

    def generate_report(self, code: str, filepath: str = "unknown") -> str:
        """Generates a formatted text report from source analysis results."""
        res = self.analyze_source_code(code)
        if not res["syntax_valid"]:
            return f"Analysis Report for {filepath}\nSyntax Error: {res['error']}\n"

        loc = res["loc"]
        comp = res["complexity"]
        h = res["halstead"]
        mi = res["maintainability_index"]
        deps = res["dependencies"]

        status = "Good" if mi >= 65 else ("Moderate" if mi >= 40 else "Poor")

        lines = [
            f"==================================================",
            f"ARCHONAI CODE METRICS REPORT FOR: {filepath}",
            f"==================================================",
            f"Lines of Code (LOC):",
            f"  - Total Physical Lines:  {loc['total_lines']}",
            f"  - Source Code Lines:     {loc['source_lines']}",
            f"  - Comment Lines:         {loc['comment_lines']}",
            f"  - Blank Lines:           {loc['blank_lines']}",
            f"--------------------------------------------------",
            f"Maintainability Profile:",
            f"  - Maintainability Index: {mi}/100 ({status})",
            f"  - Module Complexity:     {comp['module_complexity']}",
            f"--------------------------------------------------",
            f"Halstead Complexity Metrics:",
            f"  - Vocabulary:            {h['vocabulary']}",
            f"  - Program Length:        {h['length']}",
            f"  - Volume (bits):         {h['volume']}",
            f"  - Difficulty:            {h['difficulty']}",
            f"  - Effort:                {h['effort']}",
            f"  - Estimated Dev Time:    {h['time_seconds']} seconds",
            f"  - Estimated Bug Count:   {h['bugs']}",
            f"--------------------------------------------------",
            f"Imports & Dependencies ({len(deps)} total):",
        ]
        for d in deps:
            lines.append(f"  - {d}")

        if comp["elements"]:
            lines.append("--------------------------------------------------")
            lines.append("Complexity breakdown for sub-elements:")
            for item in comp["elements"]:
                lines.append(f"  - [{item['type'].upper()}] {item['name']} (Line {item['lineno']}): CC={item['complexity']}")

        lines.append("==================================================")
        return "\n".join(lines)
