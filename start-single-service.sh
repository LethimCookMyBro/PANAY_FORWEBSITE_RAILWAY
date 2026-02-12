#!/bin/sh
set -eu

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-5000}"
FRONTEND_PORT="${PORT:-3000}"

if [ -z "${API_PROXY_TARGET:-}" ]; then
  API_PROXY_TARGET="http://${BACKEND_HOST}:${BACKEND_PORT}"
fi

export API_PROXY_TARGET
export PORT="${FRONTEND_PORT}"

echo "[single] backend -> ${BACKEND_HOST}:${BACKEND_PORT}"
echo "[single] frontend -> 0.0.0.0:${FRONTEND_PORT}"
echo "[single] proxy /api -> ${API_PROXY_TARGET}"

python -m uvicorn main:app \
  --app-dir /app/backend \
  --host "${BACKEND_HOST}" \
  --port "${BACKEND_PORT}" \
  --loop asyncio &
BACKEND_PID=$!

node /app/server.js &
FRONTEND_PID=$!

cleanup() {
  echo "[single] stopping services..."
  kill "${FRONTEND_PID}" "${BACKEND_PID}" 2>/dev/null || true
}

trap cleanup INT TERM

while kill -0 "${BACKEND_PID}" 2>/dev/null && kill -0 "${FRONTEND_PID}" 2>/dev/null; do
  sleep 1
done

if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
  echo "[single] backend process exited"
fi
if ! kill -0 "${FRONTEND_PID}" 2>/dev/null; then
  echo "[single] frontend process exited"
fi

cleanup
wait "${BACKEND_PID}" 2>/dev/null || true
wait "${FRONTEND_PID}" 2>/dev/null || true
exit 1
