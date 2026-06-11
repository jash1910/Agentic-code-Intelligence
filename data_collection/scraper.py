import os
import subprocess
import sys

REPO_URL = "https://github.com/langchain-ai/langchain.git"
CLONE_DIR = os.path.join(os.path.dirname(__file__), "langchain_repo")

def clone_langchain():
    """
    Clones the LangChain GitHub repository to a local directory using a shallow clone.
    """
    if os.path.exists(CLONE_DIR):
        print(f"[Scraper] LangChain repository already exists at {CLONE_DIR}. Skipping clone.")
        return True

    print(f"[Scraper] Clones LangChain repository from {REPO_URL} (shallow clone, depth=1)...")
    try:
        # Run shallow clone command
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, CLONE_DIR],
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        print("[Scraper] Repository successfully cloned.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Scraper] Error during repository cloning: {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("[Scraper] Error: Git is not installed or not in PATH.", file=sys.stderr)
        return False

if __name__ == "__main__":
    success = clone_langchain()
    if not success:
        sys.exit(1)
