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

The embed pipeline is now **incremental** with a persistent ingest state file.

Behavior:
- Existing file with same checksum: skip
- New file: embed
- Existing file with changed checksum: replace old chunks then embed new
- First run without state file: bootstrap state from DB (no re-embed for existing sources)

### Docker Compose (local)

```yaml
services:
  backend:
    volumes:
      - ./backend:/app
      - ./data:/data
```

```bash
# Optional: create volume-like folders
mkdir -p ./data/Knowledge ./data/models ./data/ingest

# Embed knowledge folder manually
docker compose exec backend python embed.py /data/Knowledge \
  --collection plcnext \
  --knowledge-root /data/Knowledge \
  --state-path /data/ingest/state.json \
  --skip-mode checksum \
  --bootstrap-from-db \
  --replace-updated

# Dry-run (no DB/state write)
docker compose exec backend python embed.py /data/Knowledge --dry-run
```

### Embed Options

| Option | Default | Description |
|--------|---------|-------------|
| `--collection` | `plcnext` | Vector store collection name |
| `--knowledge-root` | `KNOWLEDGE_DIR` | Root path used to build `source_key` |
| `--state-path` | `INGEST_STATE_PATH` | Persistent ingest state JSON |
| `--device` | `auto` | Embedding device (`auto`, `cpu`, `cuda`, `cuda:N`) |
| `--skip-mode` | `checksum` | Skip by `checksum` or `filename` |
| `--bootstrap-from-db` | `true` | Build state from existing DB docs if state missing |
| `--replace-updated` | `true` | Replace old rows when checksum changes |
| `--replace-all` | `false` | Force rebuild all discovered sources |
| `--prune-missing` | `false` | Delete rows/state for files missing from disk |
| `--chunk-size` | `800` | Characters per chunk |
| `--chunk-overlap` | `150` | Overlap between chunks |
| `--batch-size` | `1000` | Embeddings per batch |
| `--max-embed-tokens` | `480` | Hard token cap per chunk before embedding |
| `--embed-token-overlap` | `64` | Token overlap when oversized chunk is split |
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
EMBED_DEVICE=auto
MODEL_CACHE=/data/models
KNOWLEDGE_DIR=/data/Knowledge
INGEST_STATE_PATH=/data/ingest/state.json
EMBED_MAX_TOKENS=480
EMBED_TOKEN_OVERLAP=64

# Production recommendation: use manual ingest command instead of startup auto-embed
AUTO_EMBED_KNOWLEDGE=false
AUTO_EMBED_SYNC_IF_NOT_EMPTY=false
ALLOW_STARTUP_INGEST_IN_PRODUCTION=false

# RAG Settings
RETRIEVE_LIMIT=50
RERANK_TOPN=8
```

## Railway Deploy (Important)

If the browser shows `API returned HTML instead of JSON`, your frontend is hitting a static page instead of backend API.

### Option A: Single Railway service (frontend + backend together)

This repo now includes a root `Dockerfile` that runs both services in one container:

- FastAPI backend on `127.0.0.1:8000`
- Node web server on `$PORT` (Railway public port), serving `frontend/dist`
- Runtime proxy `/api/*` -> `http://127.0.0.1:8000`

Railway setup:

- Service Root Directory: repo root (`Panya/`)
- Deploy with Docker (uses root `Dockerfile`)

Required env (same as backend requirements):

- `APP_ENV=production`
- `JWT_SECRET=<strong-random-secret>`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- database config using one of:
  - `DATABASE_URL=<postgresql://...>`
  - or all fallback vars: `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`
- plus your existing backend env variables
- Railway volume setup:
  - mount path: `/data`
  - set `KNOWLEDGE_DIR=/data/Knowledge`
  - set `MODEL_CACHE=/data/models`
  - set `INGEST_STATE_PATH=/data/ingest/state.json`
  - set `AUTO_EMBED_KNOWLEDGE=false`
  - set `AUTO_EMBED_SYNC_IF_NOT_EMPTY=false`
  - set `ALLOW_STARTUP_INGEST_IN_PRODUCTION=false`

Recommended for single-service mode:

- Do not set `API_PROXY_TARGET` (defaults to internal backend `http://127.0.0.1:8000`)
- Do not set `VITE_API_URL`
- If `API_PROXY_TARGET` is already set to `http://127.0.0.1:$PORT` (for example `5000`), remove it.

If you set `API_PROXY_TARGET`, it must point to backend only. Never set it to your frontend public domain.

Invalid examples (do not use):

```env
DATABASE_URL=${DATABASE_URL}
API_PROXY_TARGET=http://127.0.0.1:$PORT
```

Run ingest manually from Railway shell:

```bash
mkdir -p /data/Knowledge /data/models /data/ingest
python /app/backend/embed.py /data/Knowledge \
  --collection plcnext \
  --knowledge-root /data/Knowledge \
  --state-path /data/ingest/state.json \
  --skip-mode checksum \
  --bootstrap-from-db \
  --replace-updated
```

Or use the helper script:

```bash
# incremental (default)
sh /app/scripts/ingest_knowledge.sh

# full rebuild (delete+re-embed all discovered files)
INGEST_MODE=rebuild sh /app/scripts/ingest_knowledge.sh
```

### Option B: Two Railway services (frontend + backend split)

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
