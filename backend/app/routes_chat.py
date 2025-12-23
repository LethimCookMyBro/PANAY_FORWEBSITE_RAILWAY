from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.chatbot import answer_question
from app.routes_auth import get_current_user
from app.chat_db import (
    create_chat_session,
    get_chat_sessions,
    insert_chat_message,
    get_chat_messages,
    update_chat_session_title,
    delete_chat_session,
)

from app.db import get_db_pool
from app.embed_logic import get_embedder
from app.retriever import (
    PostgresVectorRetriever,
    EnhancedFlashrankRerankRetriever,
)
from app.utils import get_llm

router = APIRouter(prefix="/api/chat", tags=["chat"])


# =========================
# Schemas
# =========================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[int] = None
    collection: str = "plc_docs"


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None

class UpdateSessionRequest(BaseModel):
    title: str


# =========================
# Chat (Send message)
# =========================

@router.post("")
def chat(
    payload: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    db_pool = get_db_pool()
    llm = get_llm()
    embedder = get_embedder()

    # 1) Create session if not provided
    if payload.session_id is None:
        session_id = create_chat_session(
            db_pool=db_pool,
            user_id=current_user["id"],
            title=payload.message[:50],
        )
        chat_history = []  # New session, no history
    else:
        session_id = payload.session_id
        # Fetch recent messages for context (last 10 messages = 5 exchanges)
        messages = get_chat_messages(db_pool, session_id, current_user["id"]) or []
        chat_history = [{"role": m["role"], "content": m["content"]} for m in messages[-10:]]

    # 2) Save USER message
    insert_chat_message(
        db_pool=db_pool,
        session_id=session_id,
        role="user",
        content=payload.message,
    )

    # 3) Ask LLM with chat history for context
    result = answer_question(
        question=payload.message,
        db_pool=db_pool,
        llm=llm,
        embedder=embedder,
        collection=payload.collection,
        retriever_class=PostgresVectorRetriever,
        reranker_class=EnhancedFlashrankRerankRetriever,
        chat_history=chat_history,  # Pass conversation history
    )

    if "reply" not in result:
        raise HTTPException(status_code=500, detail="LLM did not return a reply")

    # 4) Save ASSISTANT message
    insert_chat_message(
        db_pool=db_pool,
        session_id=session_id,
        role="assistant",
        content=result["reply"],
    )

    return {
        "session_id": session_id,
        "reply": result["reply"],
        "metadata": {
            "processing_time": result.get("processing_time", 0.0),
            "retrieval_time": result.get("retrieval_time", 0.0),
            "context_count": result.get("context_count", 0),
            "max_score": result.get("max_score"),
        },
    }


# =========================
# Chat Sessions
# =========================

@router.get("/sessions")
def list_chat_sessions(
    current_user: dict = Depends(get_current_user),
):
    db_pool = get_db_pool()
    sessions = get_chat_sessions(db_pool, current_user["id"])

    return {
        "count": len(sessions),
        "items": sessions,
    }


@router.post("/sessions")
def create_session(
    payload: CreateSessionRequest,
    current_user: dict = Depends(get_current_user),
):
    db_pool = get_db_pool()
    session_id = create_chat_session(
        db_pool=db_pool,
        user_id=current_user["id"],
        title=payload.title,
    )

    return {
        "session_id": session_id
    }


@router.patch("/sessions/{session_id}")
def rename_chat_session(
    session_id: int,
    payload: UpdateSessionRequest,
    current_user: dict = Depends(get_current_user),
):
    db_pool = get_db_pool()

    updated = update_chat_session_title(
        db_pool=db_pool,
        session_id=session_id,
        user_id=current_user["id"],
        title=payload.title,
    )

    if not updated:
        raise HTTPException(status_code=404, detail="Chat session not found")

    return {
        "session_id": session_id,
        "title": payload.title,
    }


@router.delete("/sessions/{session_id}")
def delete_chat_session_route(
    session_id: int,
    current_user: dict = Depends(get_current_user),
):
    db_pool = get_db_pool()

    deleted = delete_chat_session(
        db_pool=db_pool,
        session_id=session_id,
        user_id=current_user["id"],
    )

    if not deleted:
        raise HTTPException(status_code=404, detail="Chat session not found")

    return {
        "success": True,
        "session_id": session_id,
    }


# =========================
# Chat Messages
# =========================

@router.get("/sessions/{session_id}")
def get_messages(
    session_id: int,
    current_user: dict = Depends(get_current_user),
):
    db_pool = get_db_pool()
    messages = get_chat_messages(
        db_pool,
        session_id,
        current_user["id"],
    )

    if messages is None:
        raise HTTPException(status_code=404, detail="Chat session not found")

    return {
        "session_id": session_id,
        "count": len(messages),
        "items": messages,
    }