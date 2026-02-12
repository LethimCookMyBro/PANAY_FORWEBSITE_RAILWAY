# PLC Assistant

An AI-powered chatbot for **industrial automation** and **PLCnext** technical support. Uses **RAG (Retrieval-Augmented Generation)** to answer questions from your PLC documentation.

## Key Features

- **RAG-Powered Answers** - Retrieves information from embedded documents for accurate responses
- **Multi-Input Support** - Text, and voice (Whisper)
- **Local LLM** - Runs offline with LLaMA 3.2 via Ollama (no API costs)
- **GPU Accelerated** - NVIDIA GPU support for 5-10x faster responses

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React + Vite + TailwindCSS |
| Backend | FastAPI (Python) |
| Database | PostgreSQL + pgvector |
| LLM | Ollama (LLaMA 3.2) |
| Embeddings | BAAI/bge-m3 |
| Deployment | Docker Compose |

## Quick Start

### Prerequisites

- Docker Desktop (with GPU support for NVIDIA)
- 8GB+ RAM recommended

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/MetasitKaewsritong/Panya.git
cd Panya

# 2. Create environment file
cp .env.example .env

# 3. Start all services
docker compose up -d

# 4. Pull required LLM models
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull phi3:mini
```

### Access

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:5000
- **pgAdmin:** http://localhost:5050 (admin@admin.com / admin)

## Embedding Documents

To add your own documents to the knowledge base:

### 1. Map your folder in `docker-compose.yml`

```yaml
services:
  backend:
    volumes:
      - ./backend:/app
      - "D:/YourDocs:/app/data/custom"  # Add your path
```

### 2. Restart and run embed command

```bash
# Restart to apply volume changes
docker compose restart backend

# Embed all PDFs in folder
docker compose exec backend python embed.py //app/data/custom

# Embed specific file
docker compose exec backend python embed.py //app/data/custom/manual.pdf

# Dry-run (preview only)
docker compose exec backend python embed.py //app/data/custom --dry-run

# Custom options
docker compose exec backend python embed.py //app/data/custom \
  --collection plcnext \
  --chunk-size 1000 \
  --batch-size 500
```

### Embed Options

| Option | Default | Description |
|--------|---------|-------------|
| `--collection` | `plcnext` | Vector store collection name |
| `--chunk-size` | `1000` | Characters per chunk |
| `--chunk-overlap` | `200` | Overlap between chunks |
| `--batch-size` | `1000` | Embeddings per batch |
| `--dry-run` | `false` | Preview without saving |

## Common Commands

```bash
# Start all services
docker compose up -d

# View backend logs
docker compose logs -f backend

# Restart backend after code changes
docker compose restart backend

# Stop all services (keeps data)
docker compose down

# Stop and delete all data (caution!)
docker compose down -v
```

## Environment Variables

Key settings in `.env`:

```env
# LLM
OLLAMA_MODEL=llama3.2
LLM_TEMPERATURE=0.7

# Embeddings
EMBED_MODEL=BAAI/bge-m3

# RAG Settings
RETRIEVE_LIMIT=50
RERANK_TOPN=8
```

## Railway Deploy (Important)

If the browser shows `API returned HTML instead of JSON`, your frontend is hitting a static page instead of backend API.

Recommended layout is **2 Railway services**:

1. `frontend` service
2. `backend` service

### Frontend service

- Root Directory: `frontend`
- Build Command: `npm run build`
- Start Command: `npm start` (runs `server.cjs`)

`server.cjs` serves `dist` and proxies `/api/*` to backend while preserving the full path.

Set:

```env
API_PROXY_TARGET=https://<your-backend-service>.up.railway.app
```

Do not use `npm run preview` or `serve -s dist` as production start commands.
Do not set `VITE_API_URL` on the frontend production service.

### Backend service

- Root Directory: `backend`
- Run with the provided backend Dockerfile / uvicorn startup

Ensure backend is reachable on its Railway domain.

### Verification

```bash
curl -i https://<frontend-service>.up.railway.app/api/auth/me
curl -i -X POST https://<frontend-service>.up.railway.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{}'
```

Expected: backend JSON responses (`401`/`422` are acceptable), not HTML and not frontend `404`.

If `/api/*` returns `431 Request Header Fields Too Large`, the frontend proxy target usually points to the frontend itself, causing a proxy loop. Set `API_PROXY_TARGET` to the backend service domain.

## License

MIT License
