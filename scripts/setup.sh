#!/usr/bin/env python3
"""One-command ArchonAI setup script."""

import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*60}\n{desc}\n{'='*60}")
    subprocess.check_call(cmd)


def main() -> None:
    venv_python = os.path.join(BASE, "venv", "bin", "python")
    if not os.path.exists(venv_python):
        run([sys.executable, "-m", "venv", "venv"], "Creating virtual environment")

    pip = os.path.join(BASE, "venv", "bin", "pip")
    run([pip, "install", "-r", "backend/requirements.txt"], "Installing Python dependencies")

    ingest = os.path.join(BASE, "rag", "ingest.py")
    if os.path.exists(ingest):
        run([venv_python, ingest], "Building vector index")

    kg_cache = os.path.join(BASE, "knowledge_graph", "cache")
    os.makedirs(kg_cache, exist_ok=True)

    print("\n✅ ArchonAI setup complete!")
    print("\nStart the platform:")
    print("  venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000")
    print("\nOr with Docker:")
    print("  docker compose -f docker/docker-compose.yml up --build")


if __name__ == "__main__":
    main()
