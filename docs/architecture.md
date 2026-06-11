# ArchonAI Architecture

## System Overview

ArchonAI is a production-grade agentic code intelligence platform combining multi-agent orchestration, hybrid retrieval, knowledge graphs, fine-tuned LLMs, and enterprise observability.

```mermaid
graph TB
    User([User]) --> Frontend[React + Tailwind Frontend]
    VSCode[VS Code Extension] --> Backend
    Frontend --> Backend[FastAPI Backend]
    Backend --> LangGraph[LangGraph Agent System]
    
    LangGraph --> Planner[Planner Agent]
    Planner --> Retriever[Retriever Agent]
    Retriever --> KG[Knowledge Graph Agent]
    KG --> CodeGen[Code Generator Agent]
    CodeGen --> Critic[Critic Agent]
    Critic --> Execution[Execution Agent]
    Execution --> Formatter[Response Formatter]
    
    Retriever --> HybridRAG[Hybrid RAG]
    HybridRAG --> ChromaDB[(ChromaDB)]
    HybridRAG --> BM25[BM25 Index]
    
    KG --> Neo4j[(Neo4j)]
    KG --> NetworkX[NetworkX Fallback]
    
    Execution --> Sandbox[Docker Sandbox]
    Backend --> Memory[Memory System]
    Memory --> SessionMem[Session Memory]
    Memory --> LongTerm[(ChromaDB Memory)]
    
    Backend --> Observability[Observability]
    Observability --> LangSmith[LangSmith]
    Observability --> Phoenix[Arize Phoenix]
    Observability --> OTEL[OpenTelemetry]
```

## Agent Pipeline

| Agent | Responsibility |
|-------|---------------|
| Planner | Intent classification, execution planning, repository selection |
| Retriever | Hybrid vector + BM25 search with RRF fusion |
| Knowledge Graph | Dependency discovery, inheritance analysis |
| Code Generator | Code generation, explanations, Mermaid diagrams |
| Critic | Hallucination detection, quality scoring |
| Execution | Sandbox execution with self-correction |
| Formatter | Citation attachment, confidence scoring, streaming |

## Deployment Targets

- **Local**: `uvicorn backend.app.main:app`
- **Docker**: `docker compose -f docker/docker-compose.yml up --build`
- **Railway**: Deploy backend container with env vars
- **Hugging Face Spaces**: Gradio/Streamlit evaluation dashboard
- **Vercel**: Frontend static deployment
- **AWS EC2**: Full stack with Docker Compose
