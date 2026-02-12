import os
import re
from typing import List, Optional, Tuple
from urllib.parse import quote, urlsplit


_PLACEHOLDER_PATTERNS = (
    re.compile(r"^\$\{[^}]+\}$"),      # ${VAR}
    re.compile(r"^\$\{\{[^}]+\}\}$"),  # ${{ VAR }}
    re.compile(r"^\{\{[^}]+\}\}$"),    # {{ VAR }}
)

_PG_REQUIRED_KEYS = ("PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE")


def is_placeholder(value: Optional[str]) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    return any(pattern.match(text) for pattern in _PLACEHOLDER_PATTERNS)


def _missing_required_pg_env() -> List[str]:
    missing = []
    for key in _PG_REQUIRED_KEYS:
        if is_placeholder(os.getenv(key)):
            missing.append(key)
    return missing


def build_database_url_from_pg_env() -> Optional[str]:
    missing = _missing_required_pg_env()
    if missing:
        return None

    host = os.getenv("PGHOST", "").strip()
    port = os.getenv("PGPORT", "").strip()
    user = quote(os.getenv("PGUSER", "").strip(), safe="")
    password = quote(os.getenv("PGPASSWORD", "").strip(), safe="")
    database = quote(os.getenv("PGDATABASE", "").strip(), safe="")

    dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    sslmode = os.getenv("PGSSLMODE")
    if sslmode and not is_placeholder(sslmode):
        dsn += f"?sslmode={quote(sslmode.strip(), safe='')}"

    return dsn


def resolve_database_url() -> Tuple[str, str]:
    database_url = os.getenv("DATABASE_URL", "")
    if not is_placeholder(database_url):
        return database_url.strip(), "DATABASE_URL"

    fallback = build_database_url_from_pg_env()
    if fallback:
        return fallback, "PG_ENV"

    missing = _missing_required_pg_env()
    invalid_detail = ""
    if database_url.strip():
        invalid_detail = (
            f"DATABASE_URL is unresolved placeholder '{database_url.strip()}'. "
        )

    raise RuntimeError(
        invalid_detail
        + "Unable to resolve database connection string. "
        + "Set DATABASE_URL to a real DSN or provide all PG vars: "
        + ", ".join(_PG_REQUIRED_KEYS)
        + f". Missing/invalid: {', '.join(missing)}"
    )


def redact_database_url(dsn: str) -> str:
    if not dsn:
        return "<missing>"

    try:
        parsed = urlsplit(dsn)
    except Exception:
        return "<invalid-dsn>"

    if not parsed.scheme or not parsed.netloc:
        return "<redacted-non-url-dsn>"

    host = parsed.hostname or "unknown-host"
    port = f":{parsed.port}" if parsed.port else ""
    db_name = parsed.path.lstrip("/") or "<unknown-db>"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme}://{host}{port}/{db_name}{query}"
