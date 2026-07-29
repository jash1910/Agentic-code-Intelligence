"""Refactoring Agent - Analyzes code architecture, design smells, and suggests refactoring plans."""

from __future__ import annotations

import ast
import time
from typing import Any, Dict, List

from agents.state import AgentStep, ArchonState
from backend.app.services.analyzer import StaticCodeAnalyzer


class RefactorRuleEngine:
    """Rules engine to match design smells and recommend code changes."""

    @staticmethod
    def inspect_ast_smells(code: str) -> List[Dict[str, Any]]:
        """Parses the code to match standard architectural smells."""
        smells = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return [{"smell": "Syntax Error", "severity": "HIGH", "recommendation": "Fix syntactic errors before analyzing design smells."}]

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # 1. Long Function Smell
                lines = len(node.body)
                if lines > 25:
                    smells.append({
                        "smell": "Long Function",
                        "target": node.name,
                        "line": node.lineno,
                        "severity": "MEDIUM",
                        "recommendation": f"Function '{node.name}' has {lines} statements. Extract sub-logic into separate utility functions."
                    })
                
                # 2. Too many arguments
                arg_count = len(node.args.args)
                if arg_count > 5:
                    smells.append({
                        "smell": "Too Many Parameters",
                        "target": node.name,
                        "line": node.lineno,
                        "severity": "MEDIUM",
                        "recommendation": f"Function '{node.name}' accepts {arg_count} parameters. Group parameters into a structured BaseModel or configuration dataclass."
                    })

                # 3. Missing docstring
                doc = ast.get_docstring(node)
                if not doc or len(doc.strip()) < 5:
                    smells.append({
                        "smell": "Undocumented Block",
                        "target": node.name,
                        "line": node.lineno,
                        "severity": "LOW",
                        "recommendation": f"Function '{node.name}' is missing a docstring. Document inputs, outputs, and behaviour."
                    })

            elif isinstance(node, ast.ClassDef):
                # 4. Large class smell
                methods = sum(1 for item in node.body if isinstance(item, ast.FunctionDef))
                if methods > 12:
                    smells.append({
                        "smell": "God Class (Large Class)",
                        "target": node.name,
                        "line": node.lineno,
                        "severity": "HIGH",
                        "recommendation": f"Class '{node.name}' contains {methods} methods. Split responsibilities using composition or the Single Responsibility Principle."
                    })

            # 5. Nested loops check
            elif isinstance(node, (ast.For, ast.While)):
                nested_depth = 0
                current = node
                while True:
                    found_nested = False
                    for child in ast.iter_child_nodes(current):
                        if isinstance(child, (ast.For, ast.While)):
                            nested_depth += 1
                            current = child
                            found_nested = True
                            break
                    if not found_nested:
                        break
                if nested_depth >= 2:
                    smells.append({
                        "smell": "Deeply Nested Loops",
                        "target": "loop block",
                        "line": node.lineno,
                        "severity": "MEDIUM",
                        "recommendation": f"Loops are nested {nested_depth + 1} levels deep. Extract inner loop structures or flatten using list comprehensions / itertools."
                    })

        return smells


def refactoring_agent_node(state: ArchonState) -> dict[str, Any]:
    """LangGraph node: reviews generated code metrics and compile design improvements."""
    start = time.perf_counter()
    code = state.get("generated_code", "")

    # Run static metrics analyzer
    analyzer = StaticCodeAnalyzer()
    metrics = analyzer.analyze_source_code(code)

    # Run design smell engine
    smells = RefactorRuleEngine.inspect_ast_smells(code)

    # Compile score
    base_refactor_score = 1.0
    for s in smells:
        if s["severity"] == "HIGH":
            base_refactor_score -= 0.15
        elif s["severity"] == "MEDIUM":
            base_refactor_score -= 0.08
        else:
            base_refactor_score -= 0.03
            
    refactor_score = max(0.0, round(base_refactor_score, 2))

    # Format refactoring proposals
    proposal_lines = []
    if smells:
        proposal_lines.append(f"Refactoring Proposals: Identified {len(smells)} design issue(s).")
        for s in smells:
            line_str = f" (Line {s['line']})" if "line" in s else ""
            target_str = f" in '{s['target']}'" if "target" in s else ""
            proposal_lines.append(
                f"- [{s['smell']}]{target_str}{line_str}:\n"
                f"  Recommendation: {s['recommendation']}"
            )
    else:
        proposal_lines.append("Code Quality Profile: Excellent. No refactoring improvements required.")

    refactoring_feedback = "\n\n".join(proposal_lines)

    step: AgentStep = {
        "agent": "refactoring_agent",
        "status": "completed",
        "output": f"Refactoring score: {refactor_score:.2f} | Smells: {len(smells)}",
        "duration_ms": (time.perf_counter() - start) * 1000,
        "metadata": {
            "refactor_score": refactor_score,
            "smells_count": len(smells),
            "maintainability_index": metrics.get("maintainability_index", 100.0),
            "cyclomatic_complexity": metrics.get("complexity", {}).get("module_complexity", 1)
        }
    }

    trace = list(state.get("agent_trace", []))
    trace.append(step)

    return {
        "refactor_score": refactor_score,
        "refactor_findings": smells,
        "refactoring_report": refactoring_feedback,
        "agent_trace": trace
    }
