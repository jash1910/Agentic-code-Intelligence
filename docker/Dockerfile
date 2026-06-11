# --- Stage 1: Build React Frontend ---
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Python Backend ---
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend
COPY agents/ ./agents
COPY rag/ ./rag
COPY knowledge_graph/ ./knowledge_graph
COPY observability/ ./observability
COPY evaluation/ ./evaluation
COPY training/ ./training
COPY data_collection/ ./data_collection
COPY dataset/ ./dataset

COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

ENV RUN_MODE=mock
ENV PORT=8000
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
