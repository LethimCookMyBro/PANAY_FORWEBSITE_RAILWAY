#!/bin/sh
set -eu

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${PORT:-3000}"
APP_ENV="$(printf '%s' "${APP_ENV:-development}" | tr '[:upper:]' '[:lower:]')"

is_placeholder() {
  value="$(printf '%s' "${1:-}" | tr -d '[:space:]')"
  [ -z "${value}" ] && return 0
  case "${value}" in
    '${'*'}'|'${{'*'}}'|'{{'*'}}') return 0 ;;
  esac
  return 1
}

case "${APP_ENV}" in
  production|development) ;;
  *)
    echo "[single] unknown APP_ENV '${APP_ENV}', defaulting to development"
    APP_ENV="development"
    ;;
esac
export APP_ENV

if [ "${APP_ENV}" = "production" ]; then
  if is_placeholder "${JWT_SECRET:-}" || [ "${JWT_SECRET:-}" = "dev-secret" ]; then
    echo "[single] invalid JWT_SECRET for production. Set APP_ENV=production with a real JWT_SECRET." >&2
    exit 1
  fi
fi

if is_placeholder "${DATABASE_URL:-}"; then
  missing_pg=""
  for key in PGHOST PGPORT PGUSER PGPASSWORD PGDATABASE; do
    eval "val=\${$key:-}"
    if is_placeholder "${val}"; then
      missing_pg="${missing_pg} ${key}"
    fi
  done
  if [ -n "${missing_pg}" ]; then
    echo "[single] invalid database env: DATABASE_URL is empty/placeholder and PG fallback vars are missing:${missing_pg}" >&2
    echo "[single] set DATABASE_URL to a real DSN or provide PGHOST PGPORT PGUSER PGPASSWORD PGDATABASE" >&2
    exit 1
  fi
fi

if [ "${BACKEND_PORT}" = "${FRONTEND_PORT}" ]; then
  echo "[single] invalid config: BACKEND_PORT (${BACKEND_PORT}) must not equal PORT (${FRONTEND_PORT})" >&2
  exit 1
fi

if [ -z "${API_PROXY_TARGET:-}" ]; then
  API_PROXY_TARGET="http://${BACKEND_HOST}:${BACKEND_PORT}"
fi

case "${API_PROXY_TARGET}" in
  "http://127.0.0.1:${FRONTEND_PORT}"|"http://localhost:${FRONTEND_PORT}"|"http://0.0.0.0:${FRONTEND_PORT}")
    echo "[single] invalid API_PROXY_TARGET (${API_PROXY_TARGET}): it points to frontend port ${FRONTEND_PORT}. Use backend port ${BACKEND_PORT}." >&2
    exit 1
    ;;
esac

export API_PROXY_TARGET
export PORT="${FRONTEND_PORT}"

echo "[single] backend -> ${BACKEND_HOST}:${BACKEND_PORT}"
echo "[single] frontend -> 0.0.0.0:${FRONTEND_PORT}"
echo "[single] proxy /api -> ${API_PROXY_TARGET}"
echo "[single] app_env -> ${APP_ENV}"

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
