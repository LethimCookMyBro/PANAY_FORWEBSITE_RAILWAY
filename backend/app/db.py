# backend/app/db.py
import os
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

_db_pool: pool.SimpleConnectionPool | None = None


def init_db_pool():
    """
    Initialize PostgreSQL connection pool (singleton)
    Called once at FastAPI startup
    """
    global _db_pool

    if _db_pool is None:
        DATABASE_URL = os.getenv(
            "DATABASE_URL",
            "postgresql://user:password@postgres:5432/plcnextdb"
        )

        _db_pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=DATABASE_URL
        )

    return _db_pool


def get_db_pool() -> pool.SimpleConnectionPool:
    """
    Get initialized DB pool
    """
    if _db_pool is None:
        raise RuntimeError(
            "Database pool is not initialized. "
            "Did you forget to call init_db_pool() on startup?"
        )

    return _db_pool