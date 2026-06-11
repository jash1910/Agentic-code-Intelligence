# ArchonAI Deployment Guide

## Docker Compose (Recommended)

```bash
docker compose -f docker/docker-compose.yml up --build
```

Services:
- **Backend**: http://localhost:8000
- **Frontend**: http://localhost:5173
- **Neo4j Browser**: http://localhost:7474
- **ChromaDB**: http://localhost:8001
- **Phoenix** (optional): http://localhost:6006

## Railway

1. Connect GitHub repository
2. Set Dockerfile path: `docker/Dockerfile`
3. Configure environment variables:
   ```
   RUN_MODE=mock
   JWT_SECRET=<your-secret>
   NEO4J_URI=bolt://<neo4j-host>:7687
   ```

## Hugging Face Spaces

Deploy evaluation dashboard:
```bash
streamlit run evaluation/dashboard.py
```

## Vercel (Frontend)

```bash
cd frontend
npm run build
vercel deploy --prod
```

Set `VITE_API_BASE` to your backend URL.

## AWS EC2

```bash
# Install Docker
sudo yum install docker -y
sudo systemctl start docker

# Clone and deploy
git clone <repo>
cd archon_ai
docker compose -f docker/docker-compose.yml up -d --build
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| RUN_MODE | mock | mock / local / gpu |
| MODEL_ID | deepseek-ai/DeepSeek-Coder-1.3B-instruct | Base model |
| JWT_SECRET | (dev secret) | JWT signing key |
| NEO4J_URI | bolt://localhost:7687 | Neo4j connection |
| LANGCHAIN_TRACING_V2 | false | Enable LangSmith |
| OTEL_ENABLED | false | Enable OpenTelemetry |
| PHOENIX_ENABLED | false | Enable Arize Phoenix |
| SANDBOX_USE_DOCKER | false | Docker code sandbox |
