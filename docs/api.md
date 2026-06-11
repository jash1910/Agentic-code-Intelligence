# ArchonAI API Reference

Base URL: `http://localhost:8000`

## Health

```
GET /health
```

Returns system status, version, and active agents.

## Authentication

```
POST /api/v1/auth/login
Body: { "email": "admin@archon.ai", "password": "admin123" }
```

Returns JWT token. Use `Authorization: Bearer <token>` or API key header.

Default users:
- `admin@archon.ai` / `admin123` (Admin)
- `dev@archon.ai` / `dev123` (Developer)
- `viewer@archon.ai` / `viewer123` (Viewer)

## Chat (SSE Streaming)

```
POST /api/v1/chat
Body: {
  "message": "How does ChatPromptTemplate work?",
  "session_id": "session-123",
  "repository": "langchain"
}
```

SSE Events:
- `agent_start` — Pipeline initiated
- `agent_step` — Individual agent completion
- `citations` — Retrieved source documents
- `token` — Streamed response tokens
- `metrics` — Confidence score, latency, agent trace
- `done` — Pipeline complete

Legacy endpoint: `POST /chat` (backward compatible)

## Repositories

```
GET /api/v1/chat/repositories
```

## Audit Log (Admin)

```
GET /api/v1/auth/audit?limit=100
```

## Metrics (Admin)

```
GET /metrics
```
