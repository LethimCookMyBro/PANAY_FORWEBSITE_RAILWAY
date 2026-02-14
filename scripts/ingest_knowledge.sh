#!/bin/sh
set -eu

# Usage:
#   sh /app/scripts/ingest_knowledge.sh
#   INGEST_MODE=rebuild sh /app/scripts/ingest_knowledge.sh
#
# Modes:
#   incremental (default): checksum-based + bootstrap from DB
#   rebuild: force replace all files (ignore bootstrap)

INGEST_MODE="${INGEST_MODE:-incremental}"  # incremental|rebuild
KNOWLEDGE_DIR="${KNOWLEDGE_DIR:-/data/Knowledge}"
MODEL_CACHE="${MODEL_CACHE:-/data/models}"
STATE_PATH="${INGEST_STATE_PATH:-/data/ingest/state.json}"
COLLECTION="${DEFAULT_COLLECTION:-plcnext}"
EMBED_DEVICE="${EMBED_DEVICE:-auto}"
MAX_TOKENS="${EMBED_MAX_TOKENS:-480}"
TOKEN_OVERLAP="${EMBED_TOKEN_OVERLAP:-64}"
CHUNK_SIZE="${AUTO_EMBED_CHUNK_SIZE:-800}"
CHUNK_OVERLAP="${AUTO_EMBED_CHUNK_OVERLAP:-150}"
BATCH_SIZE="${AUTO_EMBED_BATCH_SIZE:-1000}"

case "${INGEST_MODE}" in
  incremental|rebuild) ;;
  *)
    echo "[ingest] invalid INGEST_MODE='${INGEST_MODE}' (use incremental|rebuild)" >&2
    exit 1
    ;;
esac

mkdir -p "${KNOWLEDGE_DIR}" "${MODEL_CACHE}" "$(dirname "${STATE_PATH}")"

echo "[ingest] mode=${INGEST_MODE}"
echo "[ingest] collection=${COLLECTION}"
echo "[ingest] knowledge_dir=${KNOWLEDGE_DIR}"
echo "[ingest] state_path=${STATE_PATH}"
echo "[ingest] model_cache=${MODEL_CACHE}"
echo "[ingest] embed_device=${EMBED_DEVICE}"
echo "[ingest] max_tokens=${MAX_TOKENS} token_overlap=${TOKEN_OVERLAP}"

set -- \
  python /app/backend/embed.py "${KNOWLEDGE_DIR}" \
  --collection "${COLLECTION}" \
  --knowledge-root "${KNOWLEDGE_DIR}" \
  --state-path "${STATE_PATH}" \
  --skip-mode checksum \
  --replace-updated \
  --chunk-size "${CHUNK_SIZE}" \
  --chunk-overlap "${CHUNK_OVERLAP}" \
  --batch-size "${BATCH_SIZE}" \
  --model-cache "${MODEL_CACHE}" \
  --device "${EMBED_DEVICE}" \
  --max-embed-tokens "${MAX_TOKENS}" \
  --embed-token-overlap "${TOKEN_OVERLAP}"

if [ "${INGEST_MODE}" = "rebuild" ]; then
  set -- "$@" --replace-all --no-bootstrap-from-db
else
  set -- "$@" --bootstrap-from-db
fi

echo "[ingest] exec: $*"
"$@"
