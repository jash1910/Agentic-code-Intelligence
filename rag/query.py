import os
import sys

# Add root folder to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = os.path.join(BASE_DIR, "rag", "chroma_db")

class LangChainRAG:
    def __init__(self):
        print("[RAG] Initializing RAG Retrieval Engine...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Check if vector db exists
        if not os.path.exists(CHROMA_DIR):
            print(f"[RAG] Warning: ChromaDB directory {CHROMA_DIR} not found. Running ingest first may be required.")
            self.db = None
        else:
            self.db = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=self.embeddings
            )
            print("[RAG] Vector store loaded successfully.")

    def retrieve(self, query: str, k: int = 4):
        """
        Retrieves the top-k relevant chunks from ChromaDB for a given query.
        """
        if not self.db:
            print("[RAG] Vector database is not initialized. Returning empty citations.")
            return []
            
        try:
            docs = self.db.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"[RAG] Error during similarity search: {e}")
            return []

    def build_prompt(self, query: str, docs) -> str:
        """
        Combines the user query with the retrieved documents context to build the prompt.
        """
        context_str = ""
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown Source")
            doc_type = doc.metadata.get("type", "unknown")
            context_str += f"--- Source [{i+1}]: {source} ({doc_type}) ---\n"
            context_str += f"{doc.page_content}\n\n"

        if not context_str:
            context_str = "No specific documentation context was found for this query. Answer using your general training knowledge."

        prompt = (
            f"You are a helpful and expert AI assistant specialized in LangChain.\n"
            f"Answer the user's question about LangChain internals, code, or documentation using the following retrieved context chunks.\n"
            f"Be detailed, precise, and refer directly to the source code if applicable.\n\n"
            f"### Context:\n{context_str}"
            f"### Question:\n{query}\n\n"
            f"### Response:\n"
        )
        return prompt

if __name__ == "__main__":
    # Test script run
    rag = LangChainRAG()
    test_query = "How do you use LLMChain?"
    results = rag.retrieve(test_query, k=2)
    print(f"\n[Test Search] Query: '{test_query}'")
    print(f"[Test Search] Retrieved {len(results)} chunks:")
    for doc in results:
        print(f"- File: {doc.metadata.get('source')} | Preview: {doc.page_content[:100]}...")
