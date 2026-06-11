"""Secure code execution sandbox with Docker and subprocess fallback."""

from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
from typing import Any

DEFAULT_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "10"))
DEFAULT_MEMORY = os.getenv("SANDBOX_MEMORY", "256m")
USE_DOCKER = os.getenv("SANDBOX_USE_DOCKER", "false").lower() == "true"


class CodeSandbox:
    """Execute Python code in an isolated environment."""

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        memory_limit: str = DEFAULT_MEMORY,
        use_docker: bool = USE_DOCKER,
    ) -> None:
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.use_docker = use_docker

    def execute(self, code: str, language: str = "python") -> dict[str, Any]:
        if language != "python":
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Unsupported language: {language}",
                "exit_code": 1,
            }

        wrapped = textwrap.dedent(f"""
            import sys
            try:
        {textwrap.indent(code, '        ')}
            except Exception as e:
                print(f"RuntimeError: {{e}}", file=sys.stderr)
                sys.exit(1)
        """)

        if self.use_docker:
            return self._execute_docker(wrapped)
        return self._execute_subprocess(wrapped)

    def _execute_subprocess(self, code: str) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {self.timeout}s",
                "exit_code": 124,
            }
        finally:
            os.unlink(tmp_path)

    def _execute_docker(self, code: str) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    f"--memory={self.memory_limit}",
                    f"--network=none",
                    "-v", f"{tmp_path}:/code/script.py:ro",
                    "python:3.11-slim",
                    "python", "/code/script.py",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout + 5,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Docker execution timed out after {self.timeout}s",
                "exit_code": 124,
            }
        except FileNotFoundError:
            return self._execute_subprocess(code)
        finally:
            os.unlink(tmp_path)
