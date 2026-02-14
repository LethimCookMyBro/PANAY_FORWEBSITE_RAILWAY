import logging
import os
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from typing import Any, Dict, Optional

from app.chatbot import answer_question, invoke_llm_with_fallback
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
    NoRerankRetriever,
)
from app.utils import get_llm

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)


def _env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(str(value).strip())
    except Exception:
        return default


CHAT_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, _env_int("CHAT_EXECUTOR_WORKERS", 4)),
    thread_name_prefix="chat-worker",
)


def _get_request_embedder(request: Request):
    embedder = getattr(request.app.state, "embedder", None)
    if embedder is not None:
        return embedder

    if not _env_bool("LOAD_EMBEDDER_ON_DEMAND", False):
        return None

    try:
        loaded = get_embedder()
        request.app.state.embedder = loaded
        logger.info("Embedder loaded on-demand from chat route")
        return loaded
    except Exception as e:
        logger.warning("Embedder unavailable, will use direct LLM fallback: %s", e)
        return None


def _build_llm_unavailable_reply() -> str:
    return (
        "The AI model service is not ready yet, so I can't answer right now.\n\n"
        "Please check deployment settings:\n"
        "1. Ensure OLLAMA_BASE_URL points to a reachable Ollama service\n"
        "2. Ensure OLLAMA_MODEL is pulled on that service\n"
        "3. Retry after the model is ready"
    )


def _build_service_error_reply(detail: str = "") -> str:
    msg = "I hit a backend error while generating the answer."
    if detail:
        msg += f"\n\nError details: {detail}"
    msg += "\n\nPlease try again in a few seconds. If it keeps happening, check backend logs."
    return msg

def _build_connection_error_reply() -> str:
    return (
        "I cannot connect to the AI service (Ollama).\n\n"
        "Please check if the Ollama service is running and accessible at the configured URL."
    )

def _build_timeout_error_reply(timeout_s: Optional[float] = None) -> str:
    if timeout_s is None:
        return (
            "The AI service timed out while generating the answer.\n\n"
            "The model might be loading or the query is too complex. Please try again."
        )
    return (
        f"The request exceeded the server response budget ({timeout_s:.1f}s), so I stopped generation to keep the API responsive.\n\n"
        "Please retry, ask a shorter/more specific question, or reduce model load."
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
    return invoke_llm_with_fallback(llm, prompt)


def _run_chat_generation(
    db_pool,
    llm,
    embedder,
    message: str,
    collection: str,
    chat_history: list,
) -> Dict[str, Any]:
    use_rerank = _env_bool("CHAT_USE_RERANK", False)
    reranker_cls = EnhancedFlashrankRerankRetriever if use_rerank else NoRerankRetriever

    if embedder is not None:
        return answer_question(
            question=message,
            db_pool=db_pool,
            llm=llm,
            embedder=embedder,
            collection=collection,
            retriever_class=PostgresVectorRetriever,
            reranker_class=reranker_cls,
            chat_history=chat_history,
        )

    reply = _answer_direct_llm(llm, message, chat_history)
    return {
        "reply": reply,
        "processing_time": 0.0,
        "ragas": None,
        "retrieval_time": 0.0,
        "context_count": 0,
        "max_score": None,
    }


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
    request: Request,
    payload: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    db_pool = get_db_pool()
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    llm = get_llm()
    embedder = _get_request_embedder(request)
    request_budget_seconds = max(5.0, _env_float("CHAT_REQUEST_TIMEOUT_SECONDS", 180.0))

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
        future = None
        try:
            future = CHAT_EXECUTOR.submit(
                _run_chat_generation,
                db_pool,
                llm,
                embedder,
                message,
                payload.collection,
                chat_history,
            )
            result = future.result(timeout=request_budget_seconds)
        except FutureTimeoutError:
            if future is not None:
                future.cancel()
            logger.error(
                "Chat generation exceeded request budget (%.1fs) for session %s",
                request_budget_seconds,
                session_id,
            )
            result = {
                "reply": _build_timeout_error_reply(timeout_s=request_budget_seconds),
                "processing_time": request_budget_seconds,
                "ragas": None,
                "retrieval_time": 0.0,
                "context_count": 0,
                "max_score": None,
                "metadata": {
                    "error": "request_timeout",
                    "timeout_seconds": request_budget_seconds,
                },
            }
        except requests.exceptions.ConnectionError:
            logger.error("Chat generation failed: Connection refused to AI service")
            result = {
                "reply": _build_connection_error_reply(),
                "processing_time": 0.0,
                "ragas": None,
                "retrieval_time": 0.0,
                "context_count": 0,
                "max_score": None,
                "metadata": {"error": "connection_error"},
            }
        except requests.exceptions.Timeout:
            logger.error("Chat generation failed: AI service timed out")
            result = {
                "reply": _build_timeout_error_reply(),
                "processing_time": 0.0,
                "ragas": None,
                "retrieval_time": 0.0,
                "context_count": 0,
                "max_score": None,
                "metadata": {"error": "timeout"},
            }
        except Exception as e:
            detail = str(e)
            logger.error("Chat generation failed: %s", detail, exc_info=True)
            normalized = detail.lower()
            if "405" in normalized and "method not allowed" in normalized:
                result = {
                    "reply": (
                        "The AI model endpoint rejected this request method (405).\n\n"
                        "Please verify model provider settings:\n"
                        "1. OLLAMA_BASE_URL points to an Ollama-compatible endpoint\n"
                        "2. The target service supports chat/generate APIs\n"
                        "3. Retry after updating configuration"
                    ),
                    "processing_time": 0.0,
                    "ragas": None,
                    "retrieval_time": 0.0,
                    "context_count": 0,
                    "max_score": None,
                    "metadata": {
                        "error": "llm_method_not_allowed",
                        "detail": detail,
                    },
                }
            else:
                result = {
                    "reply": _build_service_error_reply(detail),
                    "processing_time": 0.0,
                    "ragas": None,
                    "retrieval_time": 0.0,
                    "context_count": 0,
                    "max_score": None,
                    "metadata": {"error": "generation_failed", "detail": detail},
                }

    result = _coerce_answer_result(result)
    if not str(result.get("reply", "")).strip():
        result["reply"] = _build_service_error_reply("")

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
                "sources": result.get("sources", []),
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
        "sources": result.get("sources", []),
        "metadata": {
            "retrieval_time": result.get("retrieval_time", 0.0),
            "context_count": result.get("context_count", 0),
            "max_score": result.get("max_score"),
            "sources": result.get("sources", []),
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
