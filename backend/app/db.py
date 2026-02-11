# backend/app/db.py
import os
from psycopg2 import pool

_db_pool: pool.SimpleConnectionPool | None = None


def init_db_pool() -> pool.SimpleConnectionPool:
    """
    Initialize PostgreSQL connection pool (singleton).
    Called once at FastAPI startup.
    """
    global _db_pool

    if _db_pool is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL environment variable is required. "
                "Please set it in your .env file."
            )

        minconn = int(os.getenv("DB_POOL_MIN", "1"))
        maxconn = int(os.getenv("DB_POOL_MAX", "10"))
        _db_pool = pool.SimpleConnectionPool(
            minconn=minconn,
            maxconn=maxconn,
            dsn=database_url,
        )

    return _db_pool


def get_db_pool() -> pool.SimpleConnectionPool:
    """
    Get initialized DB pool.
    """
    if _db_pool is None:
        raise RuntimeError(
            "Database pool is not initialized. "
            "Did you forget to call init_db_pool() on startup?"
        )
    return _db_pool


def ensure_schema(db_pool: pool.SimpleConnectionPool) -> None:
    """
    Apply idempotent schema migrations required by the current backend code.
    Fail-fast: raise on any error so startup does not continue with broken schema.
    """
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            statements = [
                # pgvector + vector store table
                """
                CREATE EXTENSION IF NOT EXISTS vector;
                """,
                """
                CREATE TABLE IF NOT EXISTS public.documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    collection VARCHAR(255) NOT NULL,
                    hash VARCHAR(64) UNIQUE NOT NULL,
                    embedding VECTOR(1024),
                    created_at TIMESTAMPTZ DEFAULT now()
                );
                """,
                """
                ALTER TABLE public.documents
                ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now();
                """,
                """
                ALTER TABLE public.documents
                ALTER COLUMN metadata SET DEFAULT '{}'::jsonb;
                """,
                """
                UPDATE public.documents
                SET metadata = '{}'::jsonb
                WHERE metadata IS NULL;
                """,
                """
                UPDATE public.documents
                SET metadata = jsonb_set(
                    metadata,
                    '{source}',
                    to_jsonb(regexp_replace(metadata->>'source', '^.*/', '')),
                    true
                )
                WHERE metadata ? 'source'
                  AND (metadata->>'source') LIKE '%/%';
                """,
                """
                WITH inferred_pages AS (
                    SELECT
                        id,
                        (regexp_match(content, '(?i)(?:---\\s*page\\s*|\\bpage\\s*)(\\d{1,5})'))[1]::int AS page_num
                    FROM public.documents
                    WHERE CASE
                        WHEN (metadata->>'page') ~ '^\\d+$' THEN (metadata->>'page')::int
                        ELSE 0
                    END = 0
                    AND regexp_match(content, '(?i)(?:---\\s*page\\s*|\\bpage\\s*)(\\d{1,5})') IS NOT NULL
                )
                UPDATE public.documents d
                SET metadata = jsonb_set(
                    d.metadata,
                    '{page}',
                    to_jsonb(inferred_pages.page_num),
                    true
                )
                FROM inferred_pages
                WHERE d.id = inferred_pages.id
                  AND inferred_pages.page_num > 0;
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_collection
                ON public.documents (collection);
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_hnsw_embedding
                ON public.documents USING hnsw (embedding vector_l2_ops);
                """,
                # auth tables
                """
                CREATE TABLE IF NOT EXISTS public.users (
                  id SERIAL PRIMARY KEY,
                  email TEXT NOT NULL UNIQUE,
                  password_hash TEXT NOT NULL,
                  full_name TEXT,
                  is_active BOOLEAN DEFAULT true,
                  role TEXT DEFAULT 'user',
                  created_at TIMESTAMPTZ DEFAULT now()
                );
                """,
                """
                CREATE TABLE IF NOT EXISTS public.refresh_tokens (
                  id SERIAL PRIMARY KEY,
                  user_id INTEGER NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
                  token_hash TEXT NOT NULL,
                  user_agent TEXT,
                  ip TEXT,
                  created_at TIMESTAMPTZ DEFAULT now(),
                  expires_at TIMESTAMPTZ,
                  revoked BOOLEAN DEFAULT FALSE
                );
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user_id
                ON public.refresh_tokens(user_id);
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_refresh_tokens_token_hash
                ON public.refresh_tokens(token_hash);
                """,
                # chat tables
                """
                CREATE TABLE IF NOT EXISTS public.chat_sessions (
                  id SERIAL PRIMARY KEY,
                  user_id INTEGER NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
                  title TEXT,
                  created_at TIMESTAMPTZ DEFAULT now(),
                  updated_at TIMESTAMPTZ DEFAULT now()
                );
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id
                ON public.chat_sessions(user_id);
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated_at
                ON public.chat_sessions(user_id, updated_at DESC);
                """,
                """
                CREATE TABLE IF NOT EXISTS public.chat_messages (
                  id SERIAL PRIMARY KEY,
                  session_id INTEGER NOT NULL REFERENCES public.chat_sessions(id) ON DELETE CASCADE,
                  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                  content TEXT NOT NULL,
                  metadata JSONB DEFAULT '{}'::jsonb,
                  created_at TIMESTAMPTZ DEFAULT now()
                );
                """,
                """
                ALTER TABLE public.chat_messages
                ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;
                """,
                """
                UPDATE public.chat_messages
                SET metadata = '{}'::jsonb
                WHERE metadata IS NULL;
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
                ON public.chat_messages(session_id);
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created_at
                ON public.chat_messages(session_id, created_at DESC);
                """,
            ]

            for statement in statements:
                cur.execute(statement)

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        db_pool.putconn(conn)
