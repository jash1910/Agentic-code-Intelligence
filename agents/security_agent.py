"""Security Agent - Static security scanning for generated code snippets."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from agents.state import AgentStep, ArchonState

# Common vulnerability pattern matching
VULN_PATTERNS = [
    {
        "id": "SEC-001",
        "name": "OS Command Injection Risk",
        "severity": "CRITICAL",
        "pattern": r"(os\.system|os\.popen|subprocess\.(Popen|run|call|check_output|check_call)\s*\([^)]*shell\s*=\s*True)",
        "description": "Executing shell commands with shell=True or using os.system is highly vulnerable to command injection.",
    },
    {
        "id": "SEC-002",
        "name": "Dynamic Code Execution Risk",
        "severity": "CRITICAL",
        "pattern": r"\b(eval|exec|__import__)\b\s*\(",
        "description": "Dynamic execution using eval, exec, or __import__ can run arbitrary code from untrusted input.",
    },
    {
        "id": "SEC-003",
        "name": "SQL Injection Risk",
        "severity": "HIGH",
        "pattern": r"(SELECT|INSERT|UPDATE|DELETE).*\+.*\b(request|query|params|user|input|args)\b",
        "description": "Constructing SQL queries using string formatting or concatenation exposes the system to SQL injection.",
    },
    {
        "id": "SEC-004",
        "name": "Hardcoded Secrets / Credentials",
        "severity": "HIGH",
        "pattern": r"\b\w*(api_key|password|secret|token|passwd|private_key|auth_header)\w*\b\s*=\s*['\"][a-zA-Z0-9_\-]{8,}['\"]",
        "description": "Hardcoding authentication keys, private keys, passwords, or tokens in source code is insecure.",
    },
    {
        "id": "SEC-005",
        "name": "Path Traversal Risk",
        "severity": "MEDIUM",
        "pattern": r"(open|read|write|delete)\s*\(\s*.*['\"].*\.\.[/\\]",
        "description": "Directory traversal using '..' enables attackers to read or write sensitive files outside of target directories.",
    },
    {
        "id": "SEC-006",
        "name": "Insecure XML Parsing",
        "severity": "MEDIUM",
        "pattern": r"\b(xml\.etree\.ElementTree|xml\.dom\.minidom)\b",
        "description": "Standard XML parsers in Python are vulnerable to XML Entity Expansion (billion laughs) and XXE attacks.",
    },
    {
        "id": "SEC-007",
        "name": "Weak Cryptographic Hashing",
        "severity": "LOW",
        "pattern": r"\b(hashlib\.(md5|sha1))\b\s*\(",
        "description": "Using weak cryptographic algorithms (MD5, SHA1) is deprecated for signature validation or passwords.",
    },
]

class SecurityScanner:
    """Utility scanner to search code snippets for security vulnerabilities."""

    @staticmethod
    def scan_code(code: str) -> List[Dict[str, Any]]:
        """Scans code using standard vulnerability patterns."""
        findings = []
        if not code:
            return findings

        # Run regex checks
        for rule in VULN_PATTERNS:
            matches = list(re.finditer(rule["pattern"], code, re.IGNORECASE | re.MULTILINE))
            for match in matches:
                # Find line number
                line_no = code[:match.start()].count("\n") + 1
                snippet = code[max(0, match.start() - 30):min(len(code), match.end() + 30)].strip().replace("\n", " ")
                findings.append({
                    "rule_id": rule["id"],
                    "vulnerability": rule["name"],
                    "severity": rule["severity"],
                    "description": rule["description"],
                    "line": line_no,
                    "snippet": f"... {snippet} ..."
                })
        return findings


def security_agent_node(state: ArchonState) -> dict[str, Any]:
    """LangGraph node: audits the generated code for security vulnerabilities."""
    start = time.perf_counter()
    code = state.get("generated_code", "")
    
    findings = SecurityScanner.scan_code(code)
    
    # Calculate security score starting from 1.0 (perfect)
    score = 1.0
    critical_count = 0
    high_count = 0
    med_count = 0
    low_count = 0
    
    for f in findings:
        severity = f["severity"]
        if severity == "CRITICAL":
            score -= 0.35
            critical_count += 1
        elif severity == "HIGH":
            score -= 0.20
            high_count += 1
        elif severity == "MEDIUM":
            score -= 0.10
            med_count += 1
        elif severity == "LOW":
            score -= 0.05
            low_count += 1
            
    score = max(0.0, round(score, 2))
    
    # Format a detailed report
    report_lines = []
    if findings:
        report_lines.append(f"Security Scan Completed: Found {len(findings)} vulnerability indicator(s).")
        for f in findings:
            report_lines.append(
                f"[{f['severity']}] {f['vulnerability']} (Line {f['line']}):\n"
                f"  Description: {f['description']}\n"
                f"  Snippet: {f['snippet']}"
            )
    else:
        report_lines.append("Security Scan Completed: No vulnerabilities detected. Code appears safe for execution.")
        
    security_feedback = "\n\n".join(report_lines)
    
    # Store results in metadata step
    step: AgentStep = {
        "agent": "security_agent",
        "status": "completed",
        "output": f"Score: {score:.2f} | Criticals: {critical_count} | Highs: {high_count}",
        "duration_ms": (time.perf_counter() - start) * 1000,
        "metadata": {
            "security_score": score,
            "findings_count": len(findings),
            "breakdown": {
                "critical": critical_count,
                "high": high_count,
                "medium": med_count,
                "low": low_count
            }
        }
    }
    
    trace = list(state.get("agent_trace", []))
    trace.append(step)
    
    return {
        "security_score": score,
        "security_findings": findings,
        "security_report": security_feedback,
        "agent_trace": trace
    }
