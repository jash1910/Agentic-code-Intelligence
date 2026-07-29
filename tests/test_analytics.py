"""Unit and integration tests for Static Code Analyzer, Security Agent, and Refactoring Agent."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.services.analyzer import StaticCodeAnalyzer
from agents.security_agent import SecurityScanner
from agents.refactoring_agent import RefactorRuleEngine


SAMPLE_PYTHON_CODE = """
def calculate_factorial(n):
    \"\"\"Calculate factorials recursively.\"\"\"
    if n < 0:
        raise ValueError("Must be non-negative")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

class MathHelper:
    \"\"\"Helper class for basic operations.\"\"\"
    def add(self, a, b):
        return a + b
        
    def subtract(self, a, b):
        return a - b
"""

VULNERABLE_CODE = """
import os
import sqlite3

def login(user, password):
    # Insecure query composition (SQL Injection)
    query = "SELECT * FROM users WHERE username = '" + user + "' AND password = '" + password + "'"
    conn = sqlite3.connect("db.sqlite")
    cursor = conn.cursor()
    cursor.execute(query)
    
    # Insecure command execution (Command Injection)
    os.system("echo User logged in: " + user)
    
    # Hardcoded secrets
    auth_token = "s3cr3t_t0k3n_12345"
    return True
"""

COMPLEX_CODE = """
def nested_loops_example(matrix):
    total = 0
    # Deep nesting smell
    for row in matrix:
        for val in row:
            for item in val:
                if item > 10:
                    total += item
    return total

def long_and_verbose_function(a, b, c, d, e, f):
    # Long parameter list smell
    x = a + b
    y = c + d
    z = e + f
    # Long body list smell
    print(x)
    print(y)
    print(z)
    print(x)
    print(y)
    print(z)
    print(x)
    print(y)
    print(z)
    print(x)
    print(y)
    print(z)
    print(x)
    print(y)
    print(z)
    print(x)
    print(y)
    print(z)
    print(x)
    print(y)
    print(z)
    print(x)
    print(y)
    print(z)
    return x + y + z
"""


class TestStaticAnalyzer:
    def test_parse_loc(self) -> None:
        loc = StaticCodeAnalyzer.parse_loc(SAMPLE_PYTHON_CODE)
        assert loc["total_lines"] >= 16
        assert loc["source_lines"] >= 10
        assert loc["comment_lines"] >= 2

    def test_complexity_calculation(self) -> None:
        analyzer = StaticCodeAnalyzer()
        res = analyzer.analyze_source_code(SAMPLE_PYTHON_CODE)
        assert res["syntax_valid"] is True
        
        complexity = res["complexity"]
        assert complexity["module_complexity"] >= 4
        
        elements = complexity["elements"]
        factorial_fn = next(item for item in elements if item["name"] == "calculate_factorial")
        assert factorial_fn["complexity"] >= 3

    def test_halstead_metrics(self) -> None:
        analyzer = StaticCodeAnalyzer()
        res = analyzer.analyze_source_code(SAMPLE_PYTHON_CODE)
        h = res["halstead"]
        assert h["vocabulary"] > 0
        assert h["volume"] > 0.0
        assert h["difficulty"] > 0.0

    def test_maintainability_index(self) -> None:
        analyzer = StaticCodeAnalyzer()
        res = analyzer.analyze_source_code(SAMPLE_PYTHON_CODE)
        mi = res["maintainability_index"]
        assert 0.0 <= mi <= 100.0
        assert mi > 50.0  # clean logic should score high

    def test_syntax_error(self) -> None:
        analyzer = StaticCodeAnalyzer()
        res = analyzer.analyze_source_code("class Invalid python syntax code :::")
        assert res["syntax_valid"] is False
        assert "error" in res


class TestSecurityScanner:
    def test_scan_vulnerabilities(self) -> None:
        findings = SecurityScanner.scan_code(VULNERABLE_CODE)
        assert len(findings) >= 3
        
        rule_ids = [f["rule_id"] for f in findings]
        assert "SEC-001" in rule_ids  # OS Command Injection
        assert "SEC-003" in rule_ids  # SQL Injection
        assert "SEC-004" in rule_ids  # Hardcoded secrets


class TestRefactoringEngine:
    def test_detect_design_smells(self) -> None:
        smells = RefactorRuleEngine.inspect_ast_smells(COMPLEX_CODE)
        assert len(smells) >= 3
        
        smell_types = [s["smell"] for s in smells]
        assert "Deeply Nested Loops" in smell_types
        assert "Too Many Parameters" in smell_types
        assert "Long Function" in smell_types


class TestAnalyticsAPI:
    @pytest.fixture
    def client(self) -> TestClient:
        from backend.app.main import app
        return TestClient(app)

    @pytest.fixture
    def headers(self, client: TestClient) -> dict[str, str]:
        # Login to retrieve token
        response = client.post("/api/v1/auth/login", json={
            "email": "admin@archon.ai",
            "password": "admin123"
        })
        assert response.status_code == 200
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_complexity_endpoint(self, client: TestClient, headers: dict[str, str]) -> None:
        res = client.post(
            "/api/v1/analytics/complexity",
            json={"code": SAMPLE_PYTHON_CODE, "filepath": "factorial.py"},
            headers=headers
        )
        assert res.status_code == 200
        data = res.json()
        assert data["syntax_valid"] is True
        assert "loc" in data
        assert "maintainability_index" in data
        assert "report" in data

    def test_security_endpoint(self, client: TestClient, headers: dict[str, str]) -> None:
        res = client.post(
            "/api/v1/analytics/security",
            json={"code": VULNERABLE_CODE, "filepath": "insecure.py"},
            headers=headers
        )
        assert res.status_code == 200
        data = res.json()
        assert "security_score" in data
        assert len(data["findings"]) >= 3
        assert "report" in data

    def test_refactor_endpoint(self, client: TestClient, headers: dict[str, str]) -> None:
        res = client.post(
            "/api/v1/analytics/refactor",
            json={"code": COMPLEX_CODE, "filepath": "nested.py"},
            headers=headers
        )
        assert res.status_code == 200
        data = res.json()
        assert "refactor_score" in data
        assert len(data["findings"]) >= 3
        assert "report" in data
