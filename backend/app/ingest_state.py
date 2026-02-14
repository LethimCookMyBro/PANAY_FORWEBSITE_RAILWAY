import json
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)

STATE_VERSION = 1


try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - windows fallback
    fcntl = None

try:
    import msvcrt  # type: ignore
except Exception:  # pragma: no cover - linux default
    msvcrt = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _empty_state(collection: str, knowledge_root: str) -> Dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "collection": collection,
        "knowledge_root": knowledge_root,
        "files": {},
    }


def build_lock_path(state_path: str) -> str:
    state = Path(state_path)
    if state.suffix:
        return str(state.with_suffix(".lock"))
    return f"{state_path}.lock"


def _backup_corrupt_state(path: Path) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = path.with_suffix(path.suffix + f".corrupt.{timestamp}.bak")
    shutil.move(str(path), str(backup_path))
    return str(backup_path)


def load_state(
    state_path: str,
    collection: str,
    knowledge_root: str,
) -> Tuple[Dict[str, Any], bool]:
    """
    Load ingest state from JSON.

    Returns:
      (state, created_new)
    """
    path = Path(state_path)
    if not path.exists():
        return _empty_state(collection, knowledge_root), True

    try:
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception as e:
        backup_path = _backup_corrupt_state(path)
        logger.error(
            "State file is corrupt; moved to backup and recreating. path=%s backup=%s error=%s",
            state_path,
            backup_path,
            e,
        )
        return _empty_state(collection, knowledge_root), True

    if not isinstance(loaded, dict):
        backup_path = _backup_corrupt_state(path)
        logger.error(
            "State file is invalid (not object); moved to backup and recreating. path=%s backup=%s",
            state_path,
            backup_path,
        )
        return _empty_state(collection, knowledge_root), True

    files = loaded.get("files")
    if not isinstance(files, dict):
        files = {}

    state = {
        "version": int(loaded.get("version") or STATE_VERSION),
        "collection": str(loaded.get("collection") or collection),
        "knowledge_root": str(loaded.get("knowledge_root") or knowledge_root),
        "files": files,
    }

    # Keep state file scoped to one collection + one knowledge root.
    if state["collection"] != collection or state["knowledge_root"] != knowledge_root:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = path.with_suffix(path.suffix + f".mismatch.{timestamp}.bak")
        shutil.move(str(path), str(backup_path))
        logger.warning(
            "State collection/root mismatch; backed up old state and starting new one. old_collection=%s new_collection=%s old_root=%s new_root=%s backup=%s",
            state["collection"],
            collection,
            state["knowledge_root"],
            knowledge_root,
            backup_path,
        )
        return _empty_state(collection, knowledge_root), True

    return state, False


def save_state(state_path: str, state: Dict[str, Any]) -> None:
    """Write state atomically using temp file + rename."""
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized = {
        "version": int(state.get("version") or STATE_VERSION),
        "collection": str(state.get("collection") or ""),
        "knowledge_root": str(state.get("knowledge_root") or ""),
        "files": state.get("files") if isinstance(state.get("files"), dict) else {},
    }

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(normalized, tmp, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_name = tmp.name

    os.replace(temp_name, str(path))


@contextmanager
def with_lock(lock_path: str):
    """Cross-platform advisory lock for ingest state operations."""
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:
            lock_file.seek(0)
            lock_file.write("0")
            lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)

        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def bootstrap_state_from_db(
    conn,
    state: Dict[str, Any],
    collection: str,
    file_records: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Bootstrap missing ingest state from already embedded sources in DB.

    First-run behavior:
      - If file exists in DB (source_key or legacy basename source), create state entry.
      - Mark entry bootstrapped=true to avoid re-embedding old content.
    """
    files_map = state.setdefault("files", {})
    if files_map:
        return {"bootstrapped": 0, "keys": []}

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              COALESCE(NULLIF(metadata->>'source_key', ''), NULLIF(metadata->>'source', '')) AS source_id,
              COUNT(*)
            FROM documents
            WHERE collection = %s
              AND COALESCE(NULLIF(metadata->>'source_key', ''), NULLIF(metadata->>'source', '')) IS NOT NULL
            GROUP BY 1
            """,
            (collection,),
        )
        rows = cur.fetchall()

    if not rows:
        return {"bootstrapped": 0, "keys": []}

    db_counts: Dict[str, int] = {str(row[0]): int(row[1]) for row in rows if row and row[0]}
    db_keys = set(db_counts.keys())

    records = list(file_records)
    basename_counts: Dict[str, int] = {}
    for record in records:
        source_name = str(record.get("source") or "").strip()
        if source_name:
            basename_counts[source_name] = basename_counts.get(source_name, 0) + 1

    bootstrapped_keys: List[str] = []
    for record in records:
        source_key = str(record.get("source_key") or "").strip()
        source_name = str(record.get("source") or "").strip()
        if not source_key:
            continue

        matched_key = None
        if source_key in db_keys:
            matched_key = source_key
        elif source_name and basename_counts.get(source_name, 0) == 1 and source_name in db_keys:
            matched_key = source_name

        if matched_key is None:
            continue

        files_map[source_key] = {
            "sha256": str(record.get("sha256") or ""),
            "size": int(record.get("size") or 0),
            "mtime": float(record.get("mtime") or 0.0),
            "source": source_name or os.path.basename(source_key),
            "chunk_count": int(db_counts.get(matched_key, 0)),
            "last_embedded_at": _utc_now_iso(),
            "bootstrapped": True,
        }
        bootstrapped_keys.append(source_key)

    return {"bootstrapped": len(bootstrapped_keys), "keys": bootstrapped_keys}
