import hashlib
import json
import logging
import os
from typing import Any, Dict, List

from psycopg2.extras import execute_values

from app.embed_logic import get_embedding_instruction

logger = logging.getLogger(__name__)


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _load_seed_records(json_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(json_path):
        logger.warning("Seed file not found: %s", json_path)
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        logger.error("Failed to read seed file %s: %s", json_path, e)
        return []

    if not isinstance(raw, list):
        logger.warning("Seed file format is not a list: %s", json_path)
        return []

    source_name = os.path.basename(json_path)
    rows: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        question = (item.get("question") or item.get("reference_question") or "").strip()
        context = (item.get("context") or "").strip()
        answer = (item.get("answer") or item.get("reference_answer") or "").strip()
        if not (question or context or answer):
            continue

        text_lines: List[str] = []
        if question:
            text_lines.append(f"Question: {question}")
        if context:
            text_lines.append(f"Context: {context}")
        if answer:
            text_lines.append(f"Answer: {answer}")
        text = "\n".join(text_lines).strip()
        if len(text) < 10:
            continue

        metadata = {
            "source": source_name,
            "chunk_type": "golden_qa_seed",
            "category": item.get("category", "seed"),
        }
        rows.append({"text": text, "metadata": metadata})

    return rows


def seed_golden_qa_if_empty(
    db_pool,
    embedder,
    collection: str,
    json_path: str,
) -> Dict[str, Any]:
    """
    Seed vector data from golden_qa.json when target collection has no documents.
    Idempotent by both collection count check and hash unique constraint.
    """
    if embedder is None:
        return {"seeded": 0, "reason": "embedder_unavailable"}

    records = _load_seed_records(json_path)
    if not records:
        return {"seeded": 0, "reason": "no_seed_records"}

    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents WHERE collection = %s", (collection,))
            existing = cur.fetchone()[0]
            if existing > 0:
                return {"seeded": 0, "reason": "collection_not_empty", "existing": existing}

        instruction = get_embedding_instruction("golden_qa")
        texts = [instruction + rec["text"] for rec in records]
        embeddings = embedder.encode(texts, show_progress_bar=False, batch_size=32, normalize_embeddings=True)

        batch_data = []
        for rec, emb in zip(records, embeddings):
            text = rec["text"]
            metadata_json = json.dumps(rec["metadata"], ensure_ascii=False)
            hash_ = hashlib.sha256(text.encode("utf-8")).hexdigest()
            batch_data.append((text, emb.tolist(), collection, hash_, metadata_json))

        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO documents (content, embedding, collection, hash, metadata)
                VALUES %s
                ON CONFLICT (hash) DO NOTHING
                """,
                batch_data,
                template="(%s, %s, %s, %s, %s)",
            )
            inserted = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        conn.commit()
        return {"seeded": inserted, "source": json_path, "collection": collection}
    except Exception as e:
        conn.rollback()
        logger.error("Failed to seed golden QA: %s", e, exc_info=True)
        return {"seeded": 0, "reason": "seed_failed", "error": str(e)}
    finally:
        db_pool.putconn(conn)


def should_auto_seed() -> bool:
    return _as_bool(os.getenv("AUTO_SEED_GOLDEN_QA", "true"), default=True)
