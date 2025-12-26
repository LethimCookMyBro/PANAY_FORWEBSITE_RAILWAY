from typing import List, Dict, Optional, Any
import json


# =========================
# Chat Session
# =========================

def create_chat_session(db_pool, user_id: int, title: Optional[str] = None) -> int:
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_sessions (user_id, title)
                VALUES (%s, %s)
                RETURNING id
                """,
                (user_id, title),
            )
            session_id = cur.fetchone()[0]
            conn.commit()
            return session_id
    finally:
        db_pool.putconn(conn)


def get_chat_sessions(db_pool, user_id: int) -> List[Dict]:
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM chat_sessions
                WHERE user_id = %s
                ORDER BY updated_at DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "title": r[1],
                "created_at": r[2],
                "updated_at": r[3],
            }
            for r in rows
        ]
    finally:
        db_pool.putconn(conn)


def update_chat_session_title(
    db_pool,
    session_id: int,
    user_id: int,
    title: str,
) -> bool:
    """
    Rename chat session (only owner can rename)
    """
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE chat_sessions
                SET title = %s,
                    updated_at = NOW()
                WHERE id = %s AND user_id = %s
                """,
                (title, session_id, user_id),
            )
            conn.commit()
            return cur.rowcount > 0
    finally:
        db_pool.putconn(conn)


def delete_chat_session(
    db_pool,
    session_id: int,
    user_id: int,
) -> bool:
    """
    Delete chat session + its messages (only owner)
    """
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            # Delete messages first (in case ON DELETE CASCADE is not set)
            cur.execute(
                """
                DELETE FROM chat_messages
                WHERE session_id = %s
                """,
                (session_id,),
            )

            # Delete session
            cur.execute(
                """
                DELETE FROM chat_sessions
                WHERE id = %s AND user_id = %s
                """,
                (session_id, user_id),
            )

            conn.commit()
            return cur.rowcount > 0
    finally:
        db_pool.putconn(conn)


# =========================
# Chat Messages
# =========================

def insert_chat_message(
    db_pool,
    session_id: int,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    role: 'user' | 'assistant'
    metadata: optional dict with processing_time, ragas scores, etc.
    """
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, metadata)
                VALUES (%s, %s, %s, %s)
                """,
                (session_id, role, content, json.dumps(metadata or {})),
            )

            # update session updated_at
            cur.execute(
                """
                UPDATE chat_sessions
                SET updated_at = NOW()
                WHERE id = %s
                """,
                (session_id,),
            )

            conn.commit()
    finally:
        db_pool.putconn(conn)


def get_chat_messages(
    db_pool, 
    session_id: int, 
    user_id: int,
    limit: int = 50,
    offset: int = 0,
) -> Dict:
    """
    Get chat messages with pagination.
    Returns dict with items, total, has_more.
    """
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            # Get total count first
            cur.execute(
                """
                SELECT COUNT(*) FROM chat_messages m
                JOIN chat_sessions s ON m.session_id = s.id
                WHERE m.session_id = %s AND s.user_id = %s
                """,
                (session_id, user_id),
            )
            total = cur.fetchone()[0]
            
            # Get messages with pagination (newest first, then reverse for display)
            cur.execute(
                """
                SELECT m.role, m.content, m.created_at, m.metadata
                FROM chat_messages m
                JOIN chat_sessions s ON m.session_id = s.id
                WHERE m.session_id = %s AND s.user_id = %s
                ORDER BY m.created_at DESC
                LIMIT %s OFFSET %s
                """,
                (session_id, user_id, limit, offset),
            )
            rows = cur.fetchall()

        messages = [
            {
                "role": r[0],
                "content": r[1],
                "created_at": r[2],
                "metadata": r[3] if r[3] else {},
            }
            for r in reversed(rows)  # Reverse to show oldest first
        ]
        
        return {
            "items": messages,
            "total": total,
            "has_more": (offset + limit) < total,
        }
    finally:
        db_pool.putconn(conn)
