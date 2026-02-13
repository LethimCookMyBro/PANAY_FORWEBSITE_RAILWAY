import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from psycopg2.extras import execute_values
from langchain_core.documents import Document

from app.embed_logic import (
    create_json_qa_chunks,
    create_pdf_chunks,
    get_embedding_instruction,
)

logger = logging.getLogger(__name__)
DEFAULT_KNOWLEDGE_DIR_CANDIDATES = (
    "/app/data/Knowledge",
    "/app/backend/data/Knowledge",
    "data/Knowledge",
    "backend/data/Knowledge",
)


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _safe_int(value: Optional[str], default: int) -> int:
    try:
        return max(1, int(str(value)))
    except Exception:
        return default


def _safe_non_negative_int(value: Optional[str], default: int) -> int:
    try:
        return max(0, int(str(value)))
    except Exception:
        return default


def resolve_knowledge_dir(preferred: str = "") -> str:
    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(DEFAULT_KNOWLEDGE_DIR_CANDIDATES)
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(preferred) if preferred else ""


def _discover_knowledge_files(knowledge_dir: str) -> List[str]:
    root = Path(knowledge_dir)
    files: List[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".json", ".pdf"}:
            files.append(str(path))
    files.sort(key=lambda p: (0 if p.lower().endswith(".json") else 1, p.lower()))
    return files


def _collection_count(conn, collection: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents WHERE collection = %s", (collection,))
        row = cur.fetchone()
    return int(row[0]) if row else 0


def _collection_sources(conn, collection: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT metadata->>'source'
            FROM documents
            WHERE collection = %s
            """,
            (collection,),
        )
        rows = cur.fetchall()
    return {str(row[0]) for row in rows if row and row[0]}


def _load_chunks_from_file(
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    lower = file_path.lower()
    if lower.endswith(".json"):
        return create_json_qa_chunks(file_path)
    if lower.endswith(".pdf"):
        from langchain_docling import DoclingLoader
        from langchain_docling.loader import ExportType

        loader = DoclingLoader(file_path=file_path, export_type=ExportType.DOC_CHUNKS)
        pages = loader.load()
        return create_pdf_chunks(
            pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return []


def _insert_chunk_batch(
    conn,
    embedder,
    chunks: List[Document],
    collection: str,
) -> int:
    if not chunks:
        return 0

    texts: List[str] = []
    for chunk in chunks:
        chunk_type = (chunk.metadata or {}).get("chunk_type", "prose")
        instruction = get_embedding_instruction(chunk_type)
        texts.append(instruction + chunk.page_content)

    embeddings = embedder.encode(
        texts,
        show_progress_bar=False,
        batch_size=32,
        normalize_embeddings=True,
    )

    batch_data = []
    for chunk, emb in zip(chunks, embeddings):
        text = chunk.page_content
        metadata_json = json.dumps(chunk.metadata or {}, ensure_ascii=False)
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
    return inserted


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


def should_auto_embed_knowledge() -> bool:
    return _as_bool(os.getenv("AUTO_EMBED_KNOWLEDGE", "true"), default=True)


def should_auto_embed_force_rescan() -> bool:
    return _as_bool(os.getenv("AUTO_EMBED_FORCE_RESCAN", "false"), default=False)


def auto_embed_knowledge_if_empty(
    db_pool,
    embedder,
    collection: str,
    knowledge_dir: str = "",
    batch_size: int = 1000,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    sync_if_not_empty: bool = True,
    skip_known_sources: bool = True,
) -> Dict[str, Any]:
    """
    Auto-ingest PDF/JSON knowledge files when target collection is empty.
    Intended for first deployment bootstrapping.
    """
    if embedder is None:
        return {"seeded": 0, "reason": "embedder_unavailable"}

    resolved_dir = resolve_knowledge_dir(knowledge_dir)
    if not resolved_dir or not os.path.isdir(resolved_dir):
        return {
            "seeded": 0,
            "reason": "knowledge_dir_not_found",
            "knowledge_dir": knowledge_dir or None,
        }

    files = _discover_knowledge_files(resolved_dir)
    if not files:
        return {"seeded": 0, "reason": "no_knowledge_files", "knowledge_dir": resolved_dir}

    conn = db_pool.getconn()
    try:
        existing_before = _collection_count(conn, collection)
        if existing_before > 0 and not sync_if_not_empty:
            return {
                "seeded": 0,
                "reason": "collection_not_empty",
                "existing": existing_before,
            }

        known_sources = (
            _collection_sources(conn, collection)
            if existing_before > 0 and skip_known_sources
            else set()
        )

        pending: List[Document] = []
        inserted_total = 0
        files_processed = 0
        files_skipped_by_source = 0
        chunks_created = 0
        failed_files: List[str] = []

        for file_path in files:
            source_name = os.path.basename(file_path)
            if source_name in known_sources:
                files_skipped_by_source += 1
                continue

            try:
                chunks = _load_chunks_from_file(
                    file_path=file_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            except Exception as e:
                logger.error("Knowledge ingest failed for %s: %s", file_path, e, exc_info=True)
                failed_files.append(file_path)
                continue

            files_processed += 1
            chunks_created += len(chunks)
            pending.extend(chunks)

            while len(pending) >= batch_size:
                batch = pending[:batch_size]
                pending = pending[batch_size:]
                inserted_total += _insert_chunk_batch(
                    conn=conn,
                    embedder=embedder,
                    chunks=batch,
                    collection=collection,
                )

        if pending:
            inserted_total += _insert_chunk_batch(
                conn=conn,
                embedder=embedder,
                chunks=pending,
                collection=collection,
            )

        return {
            "seeded": inserted_total,
            "collection": collection,
            "knowledge_dir": resolved_dir,
            "existing_before": existing_before,
            "existing_after": _collection_count(conn, collection),
            "files_discovered": len(files),
            "files_processed": files_processed,
            "files_skipped_by_source": files_skipped_by_source,
            "chunks_created": chunks_created,
            "failed_files": failed_files,
            "sync_if_not_empty": sync_if_not_empty,
            "skip_known_sources": skip_known_sources,
        }
    except Exception as e:
        conn.rollback()
        logger.error("Failed to auto-embed knowledge files: %s", e, exc_info=True)
        return {"seeded": 0, "reason": "auto_embed_failed", "error": str(e)}
    finally:
        db_pool.putconn(conn)


def get_default_golden_qa_path(preferred: str = "") -> str:
    if preferred:
        return preferred
    resolved_dir = resolve_knowledge_dir("")
    if resolved_dir:
        return os.path.join(resolved_dir, "golden_qa.json")
    return "/app/data/Knowledge/golden_qa.json"


def get_auto_embed_batch_size() -> int:
    preferred = os.getenv("AUTO_EMBED_BATCH_SIZE")
    fallback = os.getenv("EMBED_BATCH_SIZE")
    return _safe_int(preferred or fallback, 1000)


def get_auto_embed_chunk_size() -> int:
    preferred = os.getenv("AUTO_EMBED_CHUNK_SIZE")
    fallback = os.getenv("CHUNK_SIZE")
    return _safe_int(preferred or fallback, 800)


def get_auto_embed_chunk_overlap() -> int:
    preferred = os.getenv("AUTO_EMBED_CHUNK_OVERLAP")
    fallback = os.getenv("CHUNK_OVERLAP")
    return _safe_non_negative_int(preferred or fallback, 150)


def get_auto_embed_knowledge_dir() -> str:
    return (os.getenv("KNOWLEDGE_DIR", "") or "").strip()
