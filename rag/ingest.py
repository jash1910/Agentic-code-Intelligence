import os
import sys

# Add root folder to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

REPO_DIR = os.path.join(BASE_DIR, "data_collection", "langchain_repo")
CHROMA_DIR = os.path.join(BASE_DIR, "rag", "chroma_db")

# Target folders to index
CODE_TARGET_PATHS = [
    os.path.join("libs", "core", "langchain_core"),
    os.path.join("libs", "langchain", "langchain"),
]

DOC_TARGET_PATHS = [
    os.path.join("docs", "docs"),
]

def load_documents():
    """
    Scans the target directories and loads Python and Markdown files.
    """
    documents = []
    
    # 1. Load Python files
    py_count = 0
    for target in CODE_TARGET_PATHS:
        target_path = os.path.join(REPO_DIR, target)
        if not os.path.exists(target_path):
            continue
        for root, _, files in os.walk(target_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        if len(content.strip()) > 50:
                            rel_path = os.path.relpath(file_path, REPO_DIR)
                            documents.append(Document(
                                page_content=content,
                                metadata={"source": rel_path, "type": "code"}
                            ))
                            py_count += 1
                    except Exception as e:
                        print(f"[Ingest] Error reading {file_path}: {e}")
                        
    print(f"[Ingest] Loaded {py_count} Python files.")

    # 2. Load Markdown files
    md_count = 0
    for target in DOC_TARGET_PATHS:
        target_path = os.path.join(REPO_DIR, target)
        if not os.path.exists(target_path):
            continue
        for root, _, files in os.walk(target_path):
            for file in files:
                if file.endswith('.md') or file.endswith('.mdx'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        if len(content.strip()) > 50:
                            rel_path = os.path.relpath(file_path, REPO_DIR)
                            documents.append(Document(
                                page_content=content,
                                metadata={"source": rel_path, "type": "docs"}
                            ))
                            md_count += 1
                    except Exception as e:
                        print(f"[Ingest] Error reading {file_path}: {e}")
                        
    print(f"[Ingest] Loaded {md_count} Markdown files.")
    return documents

def ingest():
    if not os.path.exists(REPO_DIR):
        print(f"[Ingest] Error: Repository directory {REPO_DIR} does not exist. Please run scraper.py first.")
        return

    # Load
    raw_docs = load_documents()
    if not raw_docs:
        print("[Ingest] No documents found to ingest. Creating fallback test document.")
        raw_docs = [
            Document(
                page_content="class LLMChain(BaseChain):\n    \"\"\"Chain to run queries against LLMs.\"\"\"\n    prompt: BasePromptTemplate\n    llm: BaseLanguageModel\n    def _call(self, inputs): return self.llm.generate(self.prompt.format(**inputs))",
                metadata={"source": "libs/langchain/langchain/chains/llm.py", "type": "code"}
            ),
            Document(
                page_content="LangChain has three core pillars: Prompts, LLMs, and Chains. Chains combine components to execute custom tasks.",
                metadata={"source": "docs/docs/index.md", "type": "docs"}
            )
        ]

    # Split
    print("[Ingest] Splitting documents into chunks...")
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=800,
        chunk_overlap=150
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    
    split_docs = []
    for doc in raw_docs:
        if doc.metadata["type"] == "code":
            chunks = python_splitter.split_documents([doc])
        else:
            chunks = text_splitter.split_documents([doc])
        split_docs.extend(chunks)
        
    print(f"[Ingest] Generated {len(split_docs)} chunks.")

    # Embed & Index
    print("[Ingest] Initializing HuggingFace Embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"[Ingest] Storing vectors in ChromaDB at {CHROMA_DIR}...")
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    print("[Ingest] Vector store successfully persisted.")
    return vector_store

if __name__ == "__main__":
    ingest()

# ArchonAI platform module - architecture reference
