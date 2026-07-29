"""FastAPI router endpoints for static code analysis, security auditing, and refactoring."""

from __future__ import annotations

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.auth.jwt_auth import TokenPayload, get_current_user, require_permission
from backend.app.services.analyzer import StaticCodeAnalyzer
from agents.security_agent import SecurityScanner
from agents.refactoring_agent import RefactorRuleEngine

router = APIRouter(prefix="/analytics", tags=["analytics"])


class CodePayload(BaseModel):
    """Payload representing a code block to be statically analyzed."""
    code: str = Field(..., min_length=1, description="Source code text to analyze.")
    filepath: str | None = Field(default="snippet.py", description="Optional file name context.")


class ComplexityResponse(BaseModel):
    """Response payload containing syntax validation, LOC, and AST complexity metrics."""
    syntax_valid: bool
    loc: Dict[str, int]
    complexity: Dict[str, Any]
    halstead: Dict[str, Any]
    maintainability_index: float
    dependencies: List[str]
    report: str


class FindingResponse(BaseModel):
    """Response detail for a scan finding (vulnerability or design smell)."""
    rule_id: str | None = None
    name: str = Field(..., alias="vulnerability")
    severity: str
    description: str
    line: int
    snippet: str

    class Config:
        populate_by_name = True


class SecurityResponse(BaseModel):
    """Response payload containing security scanner outputs and ratings."""
    security_score: float
    findings: List[Dict[str, Any]]
    report: str


class RefactoringResponse(BaseModel):
    """Response payload containing design code smells and refactoring targets."""
    refactor_score: float
    findings: List[Dict[str, Any]]
    report: str


@router.post(
    "/complexity",
    response_model=ComplexityResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute code complexity, loc, and maintainability metrics."
)
async def analyze_complexity(
    payload: CodePayload,
    user: TokenPayload = Depends(require_permission("read"))
) -> Dict[str, Any]:
    """Parses code using AST and returns Halstead, Cyclomatic, and Maintainability profiles."""
    analyzer = StaticCodeAnalyzer()
    res = analyzer.analyze_source_code(payload.code)
    report = analyzer.generate_report(payload.code, filepath=payload.filepath or "unknown")
    
    return {
        "syntax_valid": res["syntax_valid"],
        "loc": res["loc"],
        "complexity": res["complexity"],
        "halstead": res["halstead"],
        "maintainability_index": res["maintainability_index"],
        "dependencies": res["dependencies"],
        "report": report
    }


@router.post(
    "/security",
    response_model=SecurityResponse,
    status_code=status.HTTP_200_OK,
    summary="Scans code for common vulnerability signatures."
)
async def analyze_security(
    payload: CodePayload,
    user: TokenPayload = Depends(require_permission("read"))
) -> Dict[str, Any]:
    """Runs patterns scan to check for credential leakage, injections, and insecure practices."""
    findings = SecurityScanner.scan_code(payload.code)
    
    # Calculate score
    score = 1.0
    for f in findings:
        severity = f["severity"]
        if severity == "CRITICAL":
            score -= 0.35
        elif severity == "HIGH":
            score -= 0.20
        elif severity == "MEDIUM":
            score -= 0.10
        elif severity == "LOW":
            score -= 0.05
    score = max(0.0, round(score, 2))
    
    # Generate report
    report_lines = []
    if findings:
        report_lines.append(f"Security Scan Report for {payload.filepath}:")
        for f in findings:
            report_lines.append(f"  - [{f['severity']}] {f['vulnerability']} at line {f['line']}: {f['description']}")
    else:
        report_lines.append(f"Security Scan Report for {payload.filepath}: No issues found.")
        
    return {
        "security_score": score,
        "findings": findings,
        "report": "\n".join(report_lines)
    }


@router.post(
    "/refactor",
    response_model=RefactoringResponse,
    status_code=status.HTTP_200_OK,
    summary="Scans code for structural design smells and clean code improvements."
)
async def analyze_refactoring(
    payload: CodePayload,
    user: TokenPayload = Depends(require_permission("read"))
) -> Dict[str, Any]:
    """Inspects class size, function statements length, parameters length, and nesting depth."""
    smells = RefactorRuleEngine.inspect_ast_smells(payload.code)
    
    # Calculate score
    score = 1.0
    for s in smells:
        severity = s.get("severity", "LOW")
        if severity == "HIGH":
            score -= 0.15
        elif severity == "MEDIUM":
            score -= 0.08
        else:
            score -= 0.03
    score = max(0.0, round(score, 2))
    
    # Generate report
    report_lines = []
    if smells:
        report_lines.append(f"Refactoring Recommendations for {payload.filepath}:")
        for s in smells:
            target_str = f" in '{s['target']}'" if "target" in s else ""
            line_str = f" on line {s['line']}" if "line" in s else ""
            report_lines.append(f"  - {s['smell']}{target_str}{line_str}: {s['recommendation']}")
    else:
        report_lines.append(f"Refactoring recommendations for {payload.filepath}: Structure is optimal.")
        
    return {
        "refactor_score": score,
        "findings": smells,
        "report": "\n".join(report_lines)
    }
