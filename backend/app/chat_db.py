from typing import List, Dict, Optional


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
            # ลบ messages ก่อน (เผื่อไม่ได้ตั้ง ON DELETE CASCADE)
            cur.execute(
                """
                DELETE FROM chat_messages
                WHERE session_id = %s
                """,
                (session_id,),
            )

            # ลบ session
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
):
    """
    role: 'user' | 'assistant'
    """
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages (session_id, role, content)
                VALUES (%s, %s, %s)
                """,
                (session_id, role, content),
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


def get_chat_messages(db_pool, session_id: int, user_id: int) -> List[Dict]:
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            # Join กับ chat_sessions เพื่อเช็คว่า session_id นี้เป็นของ user_id นี้จริงๆ
            cur.execute(
                """
                SELECT m.role, m.content, m.created_at
                FROM chat_messages m
                JOIN chat_sessions s ON m.session_id = s.id
                WHERE m.session_id = %s AND s.user_id = %s
                ORDER BY m.created_at ASC
                """,
                (session_id, user_id),
            )
            rows = cur.fetchall()

        return [
            {
                "role": r[0],
                "content": r[1],
                "created_at": r[2],
            }
            for r in rows
        ]
    finally:
        db_pool.putconn(conn)
