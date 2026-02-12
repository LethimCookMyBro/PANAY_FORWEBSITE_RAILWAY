FROM node:20-alpine AS frontend-builder
WORKDIR /build/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg tesseract-ocr curl ca-certificates gnupg \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get update \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UVICORN_LOOP=asyncio \
    ANYIO_BACKEND=asyncio \
    BACKEND_HOST=127.0.0.1 \
    BACKEND_PORT=8000 \
    API_PROXY_TARGET=http://127.0.0.1:8000

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --retries 5 --timeout 1000 --no-cache-dir -r /app/backend/requirements.txt

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY backend /app/backend
COPY server.js /app/server.js
COPY start-single-service.sh /app/start-single-service.sh
COPY --from=frontend-builder /build/frontend/dist /app/frontend/dist

RUN chmod +x /app/start-single-service.sh

EXPOSE 3000

CMD ["/app/start-single-service.sh"]
