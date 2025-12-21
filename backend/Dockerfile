# syntax=docker/dockerfile:1.7
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg tesseract-ocr curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UVICORN_LOOP=asyncio \
    ANYIO_BACKEND=asyncio

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 5 --timeout 1000 --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","5000","--loop","asyncio"]
