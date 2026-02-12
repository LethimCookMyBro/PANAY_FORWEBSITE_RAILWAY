import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Any, Dict, Optional

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
logger = logging.getLogger(__name__)


def _build_llm_unavailable_reply() -> str:
    return (
        "The AI model service is not ready yet, so I can't answer right now.\n\n"
        "Please check deployment settings:\n"
        "1. Ensure OLLAMA_BASE_URL points to a reachable Ollama service\n"
        "2. Ensure OLLAMA_MODEL is pulled on that service\n"
        "3. Retry after the model is ready"
    )


def _build_service_error_reply() -> str:
    return (
        "I hit a backend error while generating the answer.\n\n"
        "Please try again in a few seconds. If it keeps happening, check backend logs."
    )


def _coerce_answer_result(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return {"reply": result}
    if result is None:
        return {}
    return {"reply": str(result)}


def _answer_direct_llm(llm, message: str, chat_history: list) -> str:
    history_lines = []
    for m in (chat_history or [])[-6:]:
        role = "User" if m.get("role") == "user" else "Assistant"
        content = (m.get("content") or "")[:300]
        if content:
            history_lines.append(f"{role}: {content}")
    history_block = "\n".join(history_lines)
    history_section = f"Conversation history:\n{history_block}\n\n" if history_block else ""

    prompt = (
        "You are Panya, an industrial automation assistant.\n"
        "Answer clearly and concisely in English.\n\n"
        + history_section
        + f"User question: {message}\n\n"
        + "Answer:"
    )
    return str(llm.invoke(prompt)).strip()


# =========================
# Schemas
# =========================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[int] = None
    collection: str = "plcnext"


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
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    llm = get_llm()
    embedder = None
    try:
        embedder = get_embedder()
    except Exception as e:
        logger.warning("Embedder unavailable, will use direct LLM fallback: %s", e)

    # 1) Create session if not provided
    if payload.session_id is None:
        session_id = create_chat_session(
            db_pool=db_pool,
            user_id=current_user["id"],
            title=message[:50],
        )
        chat_history = []  # New session, no history
    else:
        session_id = payload.session_id
        # Fetch recent messages for context (last 10 messages = 5 exchanges)
        result = get_chat_messages(db_pool, session_id, current_user["id"])
        if result is None:
            raise HTTPException(status_code=404, detail="Chat session not found")
        messages = result.get("items", []) if isinstance(result, dict) else []
        if not isinstance(messages, list):
            messages = []
        chat_history = [{"role": m["role"], "content": m["content"]} for m in messages[-10:]]

    # 2) Save USER message
    insert_chat_message(
        db_pool=db_pool,
        session_id=session_id,
        role="user",
        content=message,
    )

    # 3) Ask LLM with chat history for context
    if llm is None:
        result = {
            "reply": _build_llm_unavailable_reply(),
            "processing_time": 0.0,
            "ragas": None,
            "retrieval_time": 0.0,
            "context_count": 0,
            "max_score": None,
            "metadata": {"error": "llm_unavailable"},
        }
    else:
        try:
            if embedder is not None:
                result = answer_question(
                    question=message,
                    db_pool=db_pool,
                    llm=llm,
                    embedder=embedder,
                    collection=payload.collection,
                    retriever_class=PostgresVectorRetriever,
                    reranker_class=EnhancedFlashrankRerankRetriever,
                    chat_history=chat_history,  # Pass conversation history
                )
            else:
                reply = _answer_direct_llm(llm, message, chat_history)
                result = {
                    "reply": reply,
                    "processing_time": 0.0,
                    "ragas": None,
                    "retrieval_time": 0.0,
                    "context_count": 0,
                    "max_score": None,
                }
        except Exception as e:
            logger.error("Chat generation failed: %s", e, exc_info=True)
            result = {
                "reply": _build_service_error_reply(),
                "processing_time": 0.0,
                "ragas": None,
                "retrieval_time": 0.0,
                "context_count": 0,
                "max_score": None,
                "metadata": {"error": "generation_failed", "detail": str(e)},
            }

    result = _coerce_answer_result(result)
    if not str(result.get("reply", "")).strip():
        result["reply"] = _build_service_error_reply()

    # 4) Save ASSISTANT message with metrics
    assistant_saved = True
    try:
        insert_chat_message(
            db_pool=db_pool,
            session_id=session_id,
            role="assistant",
            content=result["reply"],
            metadata={
                "processing_time": result.get("processing_time", 0.0),
                "ragas": result.get("ragas"),
                "error": (result.get("metadata") or {}).get("error"),
            },
        )
    except Exception as e:
        assistant_saved = False
        logger.error(
            "Failed to persist assistant message for session %s: %s",
            session_id,
            e,
            exc_info=True,
        )

    return {
        "session_id": session_id,
        "sessionId": session_id,
        "id": session_id,
        "reply": result["reply"],
        "processing_time": result.get("processing_time", 0.0),
        "ragas": result.get("ragas"),
        "metadata": {
            "retrieval_time": result.get("retrieval_time", 0.0),
            "context_count": result.get("context_count", 0),
            "max_score": result.get("max_score"),
            "error": (result.get("metadata") or {}).get("error"),
            "storage_warning": None if assistant_saved else "assistant_message_not_saved",
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
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    db_pool = get_db_pool()
    result = get_chat_messages(
        db_pool,
        session_id,
        current_user["id"],
        limit=limit,
        offset=offset,
    )

    if result is None:
        raise HTTPException(status_code=404, detail="Chat session not found")

    return {
        "session_id": session_id,
        "count": len(result["items"]),
        "total": result["total"],
        "has_more": result["has_more"],
        "items": result["items"],
    }
