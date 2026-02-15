# backend/app/chatbot.py
# VERSION 5.0 - PURE RAG (NO DB SIDE EFFECTS)

from langchain_core.prompts import PromptTemplate
from typing import List, Optional, Dict, Any
import math
import logging
import re
import time
import os
import hashlib
import requests
from urllib.parse import urlparse, urlunparse, urljoin

# Setup logging
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_KEEP = 2
ALPHA = 0.6
HARD_MIN = 0.10
SOFT_MIN = 0.15

DEFAULT_QA_TOPK = 4
DEFAULT_PROCEDURE_TOPK = 8
DEFAULT_QA_RERANK_TOPN = 5
DEFAULT_PROCEDURE_RERANK_TOPN = 10
DEFAULT_QA_MAX_CANDIDATES = 8
DEFAULT_PROCEDURE_MAX_CANDIDATES = 14

# RAGAS quality threshold - if average score below this, respond "I don't know"
RAGAS_MIN_THRESHOLD = 0.40  # 40% minimum average quality
RAGAS_FAITHFULNESS_HARD_FAIL = 0.20
RAGAS_RELEVANCY_HARD_FAIL = 0.20
RAGAS_METRIC_WEIGHTS = {
    "answer_relevancy": 0.35,
    "faithfulness": 0.35,
    "context_precision": 0.15,
    "context_recall": 0.15,
}

# Questions matching these patterns skip RAGAS quality check (identity/greeting questions)
SKIP_RAGAS_PATTERNS = (
    "your name",
    "who are you",
    "what are you",
    "introduce yourself",
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "thank you",
    "thanks",
    "bye",
    "goodbye",
    "see you",
    "how are you",
    "what can you do",
    "help me",
    "what is panya",
)

PROCEDURE_QUERY_HINTS = (
    "how to",
    "steps",
    "step by step",
    "setup",
    "set up",
    "configure",
    "configuration",
    "commission",
    "parameter",
    "wiring",
    "troubleshoot",
    "fix",
    "reset",
    "cc-link",
    "field network",
    "gx works",
    "error code",
)

TROUBLESHOOT_QUERY_HINTS = (
    "error led",
    "err led",
    "alarm",
    "fault",
    "error code",
    "diagnostic",
    "buffer memory",
    "led indication",
    "solid on",
    "blinking",
)

RETRIEVAL_INCLUDE_TERMS = (
    "setting",
    "parameter",
    "buffer memory",
    "led indication",
    "procedure",
    "error code",
    "diagnostic",
)

RETRIEVAL_EXCLUDE_TERMS = (
    "introduction",
    "feature overview",
    "marketing",
    "network concept explanation",
)

_PROMPT_LEAK_PATTERNS = (
    r"(?i)\bcritical\s+rule\s*:\s*make\s+sure\s+to\s+answer\s+using\s+information\s+from\s+the\s+context\s+above\b[^\n.]*[.]?",
    r"(?im)^\s*INTERNAL INSTRUCTIONS.*$",
    r"(?im)^\s*CRITICAL RULES\s*:.*$",
    r"(?im)^\s*FORMATTING\s*:.*$",
    r"(?im)^\s*RESPONSE QUALITY\s*:.*$",
    r"(?im)^\s*CURRENT QUESTION\s*:.*$",
    r"(?im)^\s*CONTEXT\s*:.*$",
    r"(?i)\bnever\s+expose\s+or\s+quote\s+these\s+internal\s+instructions\b[^\n.]*[.]?",
)

PROCEDURE_BUCKET_ORDER = (
    "hardware_mode",
    "network",
    "station",
    "refresh",
    "write",
    "reset",
    "diagnostic",
    "other",
)

PROCEDURE_BUCKET_HINTS = {
    "hardware_mode": ("hardware", "cpu", "module", "mode", "plc mode"),
    "network": ("network", "ethernet", "cc-link", "cc link", "profinet", "ip address"),
    "station": ("station", "node", "slave", "master", "parameter"),
    "refresh": ("refresh", "cyclic", "link refresh", "mapping"),
    "write": ("write", "download", "apply", "save", "write to plc"),
    "reset": ("reset", "power cycle", "reboot", "restart"),
    "diagnostic": ("diagnostic", "monitor", "verify", "led", "buffer memory", "error"),
}

PARAMETER_VALUE_HINTS = (
    "parameter",
    "station",
    "address",
    "refresh",
    "register",
    "setting",
    "timer",
    "value",
    "baud",
    "timeout",
    "retry",
    "scan",
)

CC_LINK_FULL_TERM = "CC-Link IE Field Network Basic (CC-Link IEF)"
CC_LINK_SHORT_TERM = "CC-Link IEF"

MODEL_IDENTIFIER_PATTERN = re.compile(r"\b[A-Z]{2,}\d{2,}[A-Z0-9-]*\b")
IPV4_PATTERN = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
HEX_VALUE_PATTERN = re.compile(r"\b0x[0-9a-fA-F]+\b")
REGISTER_VALUE_PATTERN = re.compile(r"\b(?:D|R|W|M|X|Y|B)\d+\b", flags=re.IGNORECASE)
UNIT_VALUE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds|%|hz|khz|mhz|kbps|mbps)\b",
    flags=re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")


def _looks_broken_reply(text: Any) -> bool:
    """
    Detect obviously broken model outputs (empty/punctuation-only such as '=')
    so the UI does not get unusable replies.
    """
    if text is None:
        return True
    s = str(text).strip()
    if not s:
        return True
    if len(s) <= 3 and all(ch in "=.-_~*" for ch in s):
        return True
    alnum = sum(1 for ch in s if ch.isalnum())
    if len(s) <= 10 and alnum == 0:
        return True
    return False


def _sanitize_prompt_leakage(text: Any) -> str:
    """
    Remove obvious instruction leakage from model output.
    Keeps user-facing content clean when the model accidentally echoes prompt policy text.
    """
    cleaned = str(text or "")
    for pattern in _PROMPT_LEAK_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _is_procedure_query(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    return any(hint in q for hint in PROCEDURE_QUERY_HINTS)


def _is_troubleshoot_query(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    return any(hint in q for hint in TROUBLESHOOT_QUERY_HINTS)


def _question_mode(question: str) -> str:
    if _is_troubleshoot_query(question):
        return "troubleshoot"
    if _is_procedure_query(question):
        return "procedure"
    return "qa"


def _build_task_prompt(question: str, mode: str) -> str:
    if mode == "troubleshoot":
        return (
            "TASK: PLC_TROUBLESHOOT\n\n"
            f"Observed symptom:\n{question}\n\n"
            "Requirements:\n"
            "- Combine related troubleshooting steps across multiple manual sections\n"
            "- Keep only executable inspection/recovery actions\n"
            "- Do not invent numeric parameter values not present in context\n"
            "- If multiple model variants appear, label them clearly"
        )
    if mode == "procedure":
        return (
            "TASK: PLC_CONFIGURATION\n\n"
            f"Question:\n{question}\n\n"
            "Constraints:\n"
            "- Combine related configuration steps across multiple sections into a single executable procedure\n"
            "- Do not invent numeric parameter values not present in context\n"
            "- If model differs, separate sections\n"
            "- Put model-specific branching notes before the numbered steps\n"
            "- For enable/disable options, state explicit selection criteria\n"
            "- Explain why refresh/apply steps are required and when to execute them\n"
            "- No generic explanation"
        )
    return (
        "TASK: PLC_QA\n\n"
        f"Question:\n{question}\n\n"
        "Constraints:\n"
        "- Answer only using manuals\n"
        "- Do not invent numeric parameter values not present in context\n"
        "- No generic explanation"
    )


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(str(raw).strip())
    except Exception:
        return default


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _normalize_intent_text(text: str) -> str:
    lowered = str(text or "").lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    lowered = re.sub(r"\s{2,}", " ", lowered)
    return lowered.strip()


def _should_skip_ragas(question: str) -> bool:
    normalized = _normalize_intent_text(question)
    if not normalized:
        return False
    return any(
        re.search(rf"\b{re.escape(pattern)}\b", normalized) is not None
        for pattern in SKIP_RAGAS_PATTERNS
    )


def _topk_for_mode(mode: str) -> int:
    if mode in {"procedure", "troubleshoot"}:
        value = _env_int("CHAT_TOPK_PROCEDURE", DEFAULT_PROCEDURE_TOPK)
        return _clamp_int(value, 6, 10)
    value = _env_int("CHAT_TOPK_QA", DEFAULT_QA_TOPK)
    return _clamp_int(value, 3, 5)


def _rerank_topn_for_mode(mode: str) -> int:
    if mode in {"procedure", "troubleshoot"}:
        value = _env_int("CHAT_RERANK_TOPN_PROCEDURE", DEFAULT_PROCEDURE_RERANK_TOPN)
        return _clamp_int(value, 6, 12)
    value = _env_int("CHAT_RERANK_TOPN_QA", DEFAULT_QA_RERANK_TOPN)
    return _clamp_int(value, 3, 8)


def _max_candidates_for_mode(mode: str) -> int:
    if mode in {"procedure", "troubleshoot"}:
        value = _env_int("CHAT_MAX_CANDIDATES_PROCEDURE", DEFAULT_PROCEDURE_MAX_CANDIDATES)
        return _clamp_int(value, 8, 20)
    value = _env_int("CHAT_MAX_CANDIDATES_QA", DEFAULT_QA_MAX_CANDIDATES)
    return _clamp_int(value, 5, 12)


def _normalize_technical_terms(text: str) -> str:
    normalized = str(text or "")
    if not normalized:
        return normalized

    normalized = re.sub(
        r"(?i)\bcc\s*ie\s*field\s*configuration\b",
        f"CC IE Field Configuration (used for {CC_LINK_SHORT_TERM} setup)",
        normalized,
    )
    normalized = re.sub(
        r"(?i)\bcc[- ]?link\s*ief\s*basic\b",
        CC_LINK_FULL_TERM,
        normalized,
    )
    normalized = re.sub(
        r"(?i)\bcc[- ]?link\s*ie\s*field\s*network\s*basic\b",
        CC_LINK_FULL_TERM,
        normalized,
    )
    normalized = re.sub(
        r"(?i)\bcc[- ]?link\s*ie\s*field\s*network\s*basic\s*\(cc[- ]?link\s*ief\)\s*\(cc[- ]?link\s*ief\)",
        CC_LINK_FULL_TERM,
        normalized,
    )

    first_idx = normalized.find(CC_LINK_FULL_TERM)
    if first_idx >= 0:
        head = normalized[: first_idx + len(CC_LINK_FULL_TERM)]
        tail = normalized[first_idx + len(CC_LINK_FULL_TERM) :]
        tail = tail.replace(CC_LINK_FULL_TERM, CC_LINK_SHORT_TERM)
        normalized = head + tail

    return normalized


def _normalize_ollama_base_url(raw_url: str) -> str:
    value = (raw_url or "").strip()
    if not value:
        return "http://ollama:11434"

    parsed = urlparse(value)
    if not parsed.scheme:
        value = f"http://{value}"
        parsed = urlparse(value)

    host = (parsed.hostname or "").lower()
    scheme = (parsed.scheme or "").lower()
    if scheme == "http" and host.endswith(".railway.app"):
        return urlunparse(parsed._replace(scheme="https"))
    return value


def _is_method_not_allowed_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return "405" in msg and "method not allowed" in msg


def _extract_llm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(value, dict):
        for key in ("response", "reply", "content", "text"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return str(value)


def _invoke_ollama_chat_fallback(prompt: str) -> str:
    base_url = _normalize_ollama_base_url(
        os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    ).rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    timeout = _env_int("LLM_TIMEOUT", 20)
    num_predict = int(os.getenv("LLM_NUM_PREDICT", "420"))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    top_p = float(os.getenv("LLM_TOP_P", "0.1"))
    frequency_penalty = float(os.getenv("LLM_FREQUENCY_PENALTY", "0.2"))
    repeat_penalty = float(
        os.getenv("LLM_REPEAT_PENALTY", str(1.0 + max(0.0, frequency_penalty)))
    )

    request_payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "num_predict": num_predict,
        },
    }

    target_url = f"{base_url}/api/chat"
    response = requests.post(
        target_url,
        json=request_payload,
        timeout=timeout,
        allow_redirects=False,
    )
    if response.status_code in {301, 302, 307, 308}:
        redirected_url = response.headers.get("Location")
        if redirected_url:
            follow_url = (
                redirected_url
                if redirected_url.startswith("http")
                else urljoin(target_url, redirected_url)
            )
            response = requests.post(
                follow_url,
                json=request_payload,
                timeout=timeout,
                allow_redirects=False,
            )
    response.raise_for_status()
    payload = response.json() if response.content else {}

    message_content = (
        (payload.get("message") or {}).get("content")
        if isinstance(payload, dict)
        else None
    )
    if isinstance(message_content, str) and message_content.strip():
        return message_content.strip()

    choices = payload.get("choices") if isinstance(payload, dict) else None
    if isinstance(choices, list) and choices:
        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        choice_text = (
            ((first_choice.get("message") or {}).get("content"))
            or first_choice.get("text")
            or first_choice.get("response")
        )
        if isinstance(choice_text, str) and choice_text.strip():
            return choice_text.strip()

    fallback = payload.get("response") if isinstance(payload, dict) else None
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()

    raise RuntimeError("LLM fallback response missing content")


def invoke_llm_with_fallback(llm: Any, prompt: str) -> str:
    try:
        raw = llm.invoke(prompt)
        return _extract_llm_text(raw).strip()
    except Exception as exc:
        if not _is_method_not_allowed_error(exc):
            raise
        logger.warning(
            "LLM invoke failed with 405. Falling back to Ollama /api/chat endpoint."
        )
        return _invoke_ollama_chat_fallback(prompt)


def _content_fingerprint(text: str) -> str:
    payload = (text or "").encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()[:10]


def _preview_text(text: str, limit: int) -> str:
    compact = (text or "").replace("\n", " ").strip()
    if limit <= 0:
        return ""
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


# ============================================================
# LOGGING UTILITIES
# ============================================================

def log_chat_request(
    question: str,
    answer: str,
    retrieval_time: float,
    rerank_time: float,
    llm_time: float,
    total_time: float,
    retrieved_docs: List,
    reranked_docs: List,
    selected_docs: List,
    max_score: Optional[float],
    ragas_scores: Optional[Dict[str, Any]] = None,
):
    """
    Log comprehensive information about a chat request.
    Includes: question, timing metrics, reranking process, chunk details, and RAGAS scores.
    """
    separator = "=" * 70
    include_context_preview = _env_bool("LOG_CONTEXT_PREVIEW", False)
    preview_chars = _clamp_int(_env_int("LOG_CONTEXT_PREVIEW_CHARS", 200), 80, 400)
    
    # Build log message
    log_parts = [
        "",
        separator,
        "📩 CHAT REQUEST RECEIVED",
        separator,
        f"❓ Question: {question[:200]}{'...' if len(question) > 200 else ''}",
        "",
        "⏱️  TIMING BREAKDOWN:",
        f"   • Retrieval:   {retrieval_time:.3f}s",
        f"   • Reranking:   {rerank_time:.3f}s", 
        f"   • LLM:         {llm_time:.3f}s",
        f"   • Total:       {total_time:.3f}s",
        "",
    ]
    
    # Reranking section
    log_parts.append("🔄 RERANKING PROCESS:")
    log_parts.append(f"   Retrieved: {len(retrieved_docs)} docs → Reranked: {len(reranked_docs)} docs → Selected: {len(selected_docs)} docs")
    
    if reranked_docs:
        log_parts.append("")
        log_parts.append("   TOP RERANKED DOCS (before selection):")
        for i, doc in enumerate(reranked_docs[:5]):
            metadata = getattr(doc, "metadata", {}) or {}
            score = getattr(doc, "score", None) or metadata.get("score", "N/A")
            if isinstance(score, float):
                score = f"{score:.4f}"
            source = str(metadata.get("source", "unknown"))[:60]
            content = getattr(doc, "page_content", "") or ""
            preview = _preview_text(content, preview_chars)
            fingerprint = _content_fingerprint(content)
            log_parts.append(f"   [{i+1}] Score: {score} | {source}")
            if include_context_preview:
                log_parts.append(f"       Preview: {preview}")
            else:
                log_parts.append(
                    f"       Content: [omitted] chars={len(content)} sha1={fingerprint}"
                )
    
    log_parts.append("")
    
    # Selected chunks section
    log_parts.append("📄 SELECTED CHUNKS (used for context):")
    if selected_docs:
        for i, doc in enumerate(selected_docs):
            metadata = getattr(doc, "metadata", {}) or {}
            score = getattr(doc, "score", None) or metadata.get("score", "N/A")
            if isinstance(score, float):
                score = f"{score:.4f}"
            source = metadata.get("source", "unknown")
            chunk_type = metadata.get("chunk_type", "standard")
            content = getattr(doc, "page_content", "") or ""
            preview = _preview_text(content, preview_chars)
            fingerprint = _content_fingerprint(content)
            log_parts.append(f"   ── Chunk {i+1} ──")
            log_parts.append(f"   Score: {score} | Type: {chunk_type}")
            log_parts.append(f"   Source: {source}")
            if include_context_preview:
                log_parts.append(f"   Content: {preview}")
            else:
                log_parts.append(
                    f"   Content: [omitted] chars={len(content)} sha1={fingerprint}"
                )
            log_parts.append("")
    else:
        log_parts.append("   (No relevant chunks found)")
        log_parts.append("")
    
    # Max score
    if max_score is not None:
        log_parts.append(f"📊 MAX RELEVANCE SCORE: {max_score:.4f}")
    else:
        log_parts.append("📊 MAX RELEVANCE SCORE: N/A")
    
    log_parts.append("")
    
    # RAGAS Scores section
    log_parts.append("📈 RAGAS METRICS (Quality Scores):")
    if ragas_scores and ragas_scores.get("scores"):
        scores = ragas_scores["scores"]
        
        # Answer Relevancy
        ar = scores.get("answer_relevancy")
        ar_str = f"{ar * 100:.1f}%" if ar is not None else "N/A"
        
        # Faithfulness
        faith = scores.get("faithfulness")
        faith_str = f"{faith * 100:.1f}%" if faith is not None else "N/A"
        
        # Context Precision
        cp = scores.get("context_precision")
        cp_str = f"{cp * 100:.1f}%" if cp is not None else "N/A"
        
        # Context Recall
        cr = scores.get("context_recall")
        cr_str = f"{cr * 100:.1f}%" if cr is not None else "N/A"
        
        log_parts.append(f"   • Answer Relevancy:    {ar_str}")
        log_parts.append(f"   • Faithfulness:        {faith_str}")
        log_parts.append(f"   • Context Precision:   {cp_str}")
        log_parts.append(f"   • Context Recall:      {cr_str}")
    else:
        log_parts.append("   (RAGAS evaluation not available)")
    
    log_parts.append(separator)
    log_parts.append("")
    
    # Print the log
    logger.info("\n".join(log_parts))


# ============================================================
# SCORE UTILITIES
# ============================================================

def normalize_score(raw_score: float) -> float:
    if raw_score is None:
        return 0.0
    if 0 <= raw_score <= 1:
        return raw_score
    return 1 / (1 + math.exp(-raw_score))


def get_doc_score(doc) -> Optional[float]:
    score = getattr(doc, "score", None)
    if score is not None:
        return normalize_score(score)
    metadata = getattr(doc, "metadata", {})
    if isinstance(metadata, dict):
        score = metadata.get("score") or metadata.get("relevance_score")
        if score is not None:
            return normalize_score(score)
        # No-rerank fallback: pgvector returns L2 distance in metadata["distance"].
        # Convert distance (lower is better) to a bounded similarity-like score.
        distance = metadata.get("distance")
        if distance is not None:
            try:
                dist_val = float(distance)
                if math.isfinite(dist_val):
                    dist_val = max(0.0, dist_val)
                    return 1.0 / (1.0 + dist_val)
            except Exception:
                pass
    return None


def _coerce_positive_int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if value > 0 else 0
    if isinstance(value, float):
        page = int(value)
        return page if page > 0 else 0
    if isinstance(value, str):
        match = re.search(r"\d+", value.strip())
        if match:
            try:
                page = int(match.group(0))
                return page if page > 0 else 0
            except Exception:
                return 0
    return 0


def _doc_source_page(doc, fallback_index: int = 0) -> Dict[str, Any]:
    metadata = getattr(doc, "metadata", {}) or {}
    source = metadata.get("source") or f"Document {fallback_index + 1}"
    source = os.path.basename(str(source).strip()) or f"Document {fallback_index + 1}"
    page = _coerce_positive_int(metadata.get("page"))
    if page <= 0:
        content = getattr(doc, "page_content", "") or ""
        snippet = content[:1000]
        for pattern in (
            r"---\s*PAGE\s*(\d{1,5})\s*---",
            r"\bpage\s*[:#-]?\s*(\d{1,5})\b",
        ):
            match = re.search(pattern, snippet, flags=re.IGNORECASE)
            if match:
                page = _coerce_positive_int(match.group(1))
                if page > 0:
                    break
    return {"source": source, "page": page}


def build_source_citations(selected_docs: List, max_items: int = 6) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    seen = set()

    for i, doc in enumerate(selected_docs or []):
        item = _doc_source_page(doc, fallback_index=i)
        key = (item["source"], item["page"])
        if key in seen:
            continue
        seen.add(key)
        citations.append(item)
        if len(citations) >= max_items:
            break

    return citations


def build_citations_from_docs(docs):
    citations = []
    for item in build_source_citations(docs, max_items=10):
        source = item["source"]
        page = item["page"]
        citations.append(f"{source} (page {page})" if page > 0 else source)
    return citations


# ============================================================
# CONTEXT SELECTION
# ============================================================

def _doc_source_key(doc) -> str:
    md = getattr(doc, "metadata", {}) or {}
    source = md.get("source") or md.get("source_key") or "unknown"
    return os.path.basename(str(source))


def _doc_page(doc) -> int:
    md = getattr(doc, "metadata", {}) or {}
    return _coerce_positive_int(md.get("page"))


def _doc_dedupe_key(doc) -> tuple:
    source = _doc_source_key(doc)
    page = _doc_page(doc)
    preview = (getattr(doc, "page_content", "") or "")[:120]
    return (source, page, preview)


def _keyword_hits(text: str, keywords: tuple) -> int:
    lower = (text or "").lower()
    return sum(1 for kw in keywords if kw in lower)


def _is_actionable_manual_chunk(doc) -> bool:
    content = getattr(doc, "page_content", "") or ""
    include_hits = _keyword_hits(content, RETRIEVAL_INCLUDE_TERMS)
    if include_hits <= 0:
        return False
    exclude_hits = _keyword_hits(content, RETRIEVAL_EXCLUDE_TERMS)
    if exclude_hits > 0 and include_hits <= 1:
        return False
    return True


def _compact_context_text(text: str, max_chars: int) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    compact = "\n".join(lines)
    compact = re.sub(r"\n{3,}", "\n\n", compact)
    if max_chars > 0 and len(compact) > max_chars:
        return compact[:max_chars].rstrip()
    return compact


def _clean_list_prefix(line: str) -> str:
    cleaned = re.sub(r"^\s*\d+[.)]\s*", "", str(line or ""))
    cleaned = re.sub(r"^\s*[-*•]\s*", "", cleaned)
    return cleaned.strip()


def _normalize_line_key(line: str) -> str:
    return re.sub(r"\W+", " ", str(line or "").lower()).strip()


def _dedupe_lines(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        key = _normalize_line_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item.strip())
    return result


def _normalize_model_note(note: str, total_steps: int) -> str:
    text = str(note or "").strip()
    if not text:
        return text

    low = text.lower()
    if "iq-f" in low and "step" in low and total_steps > 0:
        end_step = min(5, total_steps)
        return (
            f"If MELSEC iQ-F models are used, follow Steps 1-{end_step} only. "
            f"For other models, follow Steps 1-{total_steps}."
        )

    if total_steps > 0:
        text = re.sub(r"(?i)\ball\s+\d+\s+steps\b", f"all {total_steps} steps", text)
        text = re.sub(
            r"(?i)\ball\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+steps\b",
            f"all {total_steps} steps",
            text,
        )
        if total_steps != 7:
            text = re.sub(r"(?i)\bseven steps\b", f"{total_steps} steps", text)

    return text


def _normalize_step_phrasing(step: str) -> str:
    text = str(step or "").strip().rstrip(".")
    if not text:
        return text

    low = text.lower()
    if "enable or disable" in low and "when" not in low:
        text = re.sub(r"(?i)\s*\(default[^)]*\)\s*", " ", text).strip().rstrip(".")
        text += (
            ". Set to Enable when CC-Link IEF communication is required; "
            "set to Disable for Ethernet-only operation"
        )

    if (
        ("configure network configuration" in low or "network configuration setting" in low)
        and "link scan timeout" not in low
    ):
        text += (
            ". Include key settings such as link scan timeout, retry count, "
            "and remote-station detection behavior"
        )

    if "refresh" in low and "update" not in low and "apply" not in low:
        text += (
            ". Use Refresh to apply updated network parameters to the module "
            "after changing network configuration"
        )

    return text


def _step_bucket(step_text: str) -> str:
    low = (step_text or "").lower()
    for bucket in PROCEDURE_BUCKET_ORDER:
        if bucket == "other":
            continue
        hints = PROCEDURE_BUCKET_HINTS.get(bucket, ())
        if any(h in low for h in hints):
            return bucket
    return "other"


def _normalize_engineering_steps(raw: str, mode: str) -> str:
    text = _normalize_technical_terms((raw or "").strip())
    if mode not in {"procedure", "troubleshoot"}:
        return text

    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text

    steps: List[str] = []
    notes: List[str] = []
    troubleshooting: List[str] = []
    in_troubleshooting = False

    for raw_line in lines:
        stripped = raw_line.strip()
        if re.match(r"(?i)^\*{0,2}\s*troubleshooting\s*\*{0,2}\s*:?\s*$", stripped):
            in_troubleshooting = True
            continue

        if re.match(r"(?i)^sources?\s*:", stripped):
            break

        cleaned = _clean_list_prefix(stripped)
        low = cleaned.lower()

        if in_troubleshooting:
            if len(cleaned) >= 6:
                troubleshooting.append(cleaned)
            continue

        if re.match(r"^\d+[.)]\s+", stripped):
            if len(cleaned) >= 8:
                steps.append(cleaned)
            continue

        if re.match(r"(?i)^\*{0,2}\s*(prerequisites?|notes?|steps?)\s*\*{0,2}\s*:?\s*$", stripped):
            continue

        if (
            low.startswith("note:")
            or ("iq-f" in low and "step" in low)
            or ("follow steps" in low and "model" in low)
            or ("prerequisite" in low)
        ):
            notes.append(re.sub(r"(?i)^note:\s*", "", cleaned))
            continue

        if stripped.startswith(("-", "*", "•")):
            if any(
                hint in low
                for hint in (
                    "error",
                    "fault",
                    "alarm",
                    "timeout",
                    "led",
                    "diagnostic",
                    "not detected",
                    "disconnected",
                    "link scan",
                )
            ):
                troubleshooting.append(cleaned)
            else:
                notes.append(cleaned)
            continue

        if len(cleaned) >= 10:
            notes.append(cleaned)

    steps = _dedupe_lines(steps)
    if not steps:
        return text

    buckets: Dict[str, List[str]] = {name: [] for name in PROCEDURE_BUCKET_ORDER}
    for step in steps:
        buckets[_step_bucket(step)].append(_normalize_step_phrasing(step))

    ordered_steps: List[str] = []
    for bucket in PROCEDURE_BUCKET_ORDER:
        ordered_steps.extend(buckets.get(bucket, []))

    ordered_steps = _dedupe_lines(ordered_steps)
    total_steps = len(ordered_steps)
    if total_steps <= 0:
        return text

    notes = _dedupe_lines([_normalize_model_note(n, total_steps) for n in notes])
    troubleshooting = _dedupe_lines(troubleshooting)

    if not notes:
        notes = [
            "Confirm the exact PLC model/module and engineering software version before applying settings.",
        ]

    default_troubleshooting = [
        "Check common failures: station not detected, timeout, or link scan failure.",
        "Verify LEDs for the target module: RUN (expected ON), ERR (expected OFF), LINK (expected ON or blinking by traffic).",
        "Confirm station number, network parameters, and module settings match on both ends.",
        "Capture the exact error code from diagnostics and verify it against the manual section for the selected model.",
    ]
    for item in default_troubleshooting:
        if len(troubleshooting) >= 4:
            break
        troubleshooting.append(item)
    troubleshooting = _dedupe_lines(troubleshooting)

    parts: List[str] = []
    parts.append("**Prerequisites / Notes**")
    parts.extend(f"- {note}" for note in notes)
    parts.append("")
    parts.append("**Steps**")
    parts.extend(f"{idx}. {step}" for idx, step in enumerate(ordered_steps, start=1))
    parts.append("")
    parts.append("**Troubleshooting**")
    parts.extend(f"- {item}" for item in troubleshooting)

    return "\n".join(parts).strip()


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _extract_context_tokens(context_texts: List[str]) -> Dict[str, Any]:
    raw_context = "\n".join(context_texts or []).lower()
    normalized_context = _normalize_for_match(raw_context)
    tokens = set()
    patterns = (
        IPV4_PATTERN,
        HEX_VALUE_PATTERN,
        REGISTER_VALUE_PATTERN,
        UNIT_VALUE_PATTERN,
        NUMBER_PATTERN,
    )
    for pattern in patterns:
        for match in pattern.finditer(raw_context):
            token = match.group(0).strip().lower()
            if token:
                tokens.add(token)
                tokens.add(_normalize_for_match(token))
    return {
        "raw": raw_context,
        "normalized": normalized_context,
        "tokens": tokens,
    }


def _extract_parameter_tokens(line: str) -> List[str]:
    candidates: List[str] = []
    for pattern in (IPV4_PATTERN, HEX_VALUE_PATTERN, REGISTER_VALUE_PATTERN, UNIT_VALUE_PATTERN):
        candidates.extend(m.group(0).strip() for m in pattern.finditer(line))

    lower = (line or "").lower()
    if any(hint in lower for hint in PARAMETER_VALUE_HINTS):
        candidates.extend(m.group(0).strip() for m in NUMBER_PATTERN.finditer(line))

    seen = set()
    deduped: List[str] = []
    for token in candidates:
        key = token.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(token)
    return deduped


def _is_model_identifier(token: str) -> bool:
    upper = str(token or "").upper()
    return bool(MODEL_IDENTIFIER_PATTERN.fullmatch(upper)) and not bool(
        REGISTER_VALUE_PATTERN.fullmatch(upper)
    )


def _token_found_in_context(token: str, context_data: Dict[str, Any]) -> bool:
    token_low = str(token or "").lower().strip()
    if not token_low:
        return True
    if token_low in context_data["raw"]:
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?", token_low):
        return token_low in context_data["tokens"]
    normalized = _normalize_for_match(token_low)
    if not normalized:
        return True
    if normalized in context_data["tokens"]:
        return True
    return normalized in context_data["normalized"]


def _enforce_numeric_guardrail(text: str, context_texts: List[str], mode: str) -> str:
    if mode not in {"procedure", "troubleshoot"}:
        return text
    if not text or not context_texts:
        return text

    context_data = _extract_context_tokens(context_texts)
    masked = False
    safe_lines: List[str] = []

    for raw in text.splitlines():
        line = raw
        content = re.sub(r"^\d+[.)]\s*", "", raw).strip()
        low = content.lower()
        is_parameter_line = any(hint in low for hint in PARAMETER_VALUE_HINTS)
        if not is_parameter_line:
            safe_lines.append(line)
            continue

        tokens = _extract_parameter_tokens(content)
        unknown_tokens: List[str] = []
        for token in tokens:
            if _is_model_identifier(token):
                continue
            if re.fullmatch(r"\d", token.strip()):
                continue
            if _token_found_in_context(token, context_data):
                continue
            unknown_tokens.append(token)

        if unknown_tokens:
            masked = True
            for token in sorted(set(unknown_tokens), key=len, reverse=True)[:12]:
                line = re.sub(
                    re.escape(token),
                    "<verify-in-manual>",
                    line,
                    flags=re.IGNORECASE,
                )
        safe_lines.append(line)

    guarded = "\n".join(safe_lines).strip()
    if masked:
        guarded += (
            "\n\nNote: Some parameter values were masked because they were not found in retrieved manual context."
        )
    return guarded


def _select_default_context_docs(
    retrieved_docs: List,
    *,
    top_k: int,
    max_candidates: int,
) -> List:
    candidates = (retrieved_docs or [])[:max_candidates]
    if not candidates:
        return []

    max_score = get_doc_score(candidates[0])
    if max_score is None or max_score < HARD_MIN:
        return []

    cutoff = max(max_score * ALPHA, SOFT_MIN)
    final_docs = []
    seen = set()
    for i, doc in enumerate(candidates):
        score = get_doc_score(doc) or max_score
        if i >= MIN_KEEP and score < cutoff:
            continue
        key = _doc_dedupe_key(doc)
        if key in seen:
            continue
        seen.add(key)
        final_docs.append(doc)
        if len(final_docs) >= top_k:
            break
    return final_docs


def _select_procedure_context_docs(
    retrieved_docs: List,
    *,
    top_k: int,
    max_candidates: int,
) -> List:
    candidates = (retrieved_docs or [])[:max_candidates]
    if not candidates:
        return []

    max_score = get_doc_score(candidates[0])
    if max_score is None or max_score < HARD_MIN:
        return []

    # Keep a wider band to allow cross-page synthesis.
    cutoff = max(max_score * 0.45, SOFT_MIN * 0.8)
    scored = []
    for i, doc in enumerate(candidates):
        score = get_doc_score(doc) or max_score
        if i >= MIN_KEEP and score < cutoff:
            continue
        scored.append((doc, float(score)))

    if not scored:
        return []

    # Prefer actionable chunks, but do not hard-reject all context when keyword hits are sparse.
    boosted_scored = []
    for doc, score in scored:
        content = getattr(doc, "page_content", "") or ""
        include_hits = _keyword_hits(content, RETRIEVAL_INCLUDE_TERMS)
        exclude_hits = _keyword_hits(content, RETRIEVAL_EXCLUDE_TERMS)
        boost = min(include_hits, 4) * 0.03
        if _is_actionable_manual_chunk(doc):
            boost += 0.08
        if exclude_hits > 0 and include_hits <= 0:
            boost -= 0.05
        adjusted = max(0.0, float(score) + boost)
        boosted_scored.append((doc, float(score), adjusted))

    boosted_scored.sort(key=lambda item: item[2], reverse=True)
    scored = [(doc, adjusted) for doc, _, adjusted in boosted_scored]

    by_source: Dict[str, List[tuple]] = {}
    source_weight: Dict[str, float] = {}
    for doc, score in scored:
        source = _doc_source_key(doc)
        by_source.setdefault(source, []).append((doc, score))
        source_weight[source] = source_weight.get(source, 0.0) + score

    # Favor one dominant manual to prevent mixed-document summaries.
    primary_source = max(
        source_weight.items(),
        key=lambda kv: (kv[1], len(by_source.get(kv[0], []))),
    )[0]

    primary_docs = by_source.get(primary_source, [])
    other_docs = [
        (doc, score)
        for source, pairs in by_source.items()
        if source != primary_source
        for doc, score in pairs
    ]

    # Order primary docs in reading order for cross-page procedure assembly.
    primary_docs.sort(
        key=lambda pair: (
            0 if _doc_page(pair[0]) > 0 else 1,
            _doc_page(pair[0]) if _doc_page(pair[0]) > 0 else 10**9,
            -pair[1],
        )
    )
    other_docs.sort(key=lambda pair: pair[1], reverse=True)

    final_docs = []
    seen = set()
    for doc, _ in primary_docs:
        key = _doc_dedupe_key(doc)
        if key in seen:
            continue
        seen.add(key)
        final_docs.append(doc)
        if len(final_docs) >= top_k:
            return final_docs

    for doc, _ in other_docs:
        key = _doc_dedupe_key(doc)
        if key in seen:
            continue
        seen.add(key)
        final_docs.append(doc)
        if len(final_docs) >= top_k:
            break

    return final_docs


def select_context_docs(
    retrieved_docs: List,
    question: str = "",
    question_mode: str = "qa",
    top_k: Optional[int] = None,
    max_candidates: Optional[int] = None,
) -> List:
    mode = (question_mode or _question_mode(question)).strip().lower()
    top_k = top_k or _topk_for_mode(mode)
    max_candidates = max_candidates or _max_candidates_for_mode(mode)
    if mode in {"procedure", "troubleshoot"}:
        docs = _select_procedure_context_docs(
            retrieved_docs,
            top_k=top_k,
            max_candidates=max_candidates,
        )
        if docs:
            return docs
    return _select_default_context_docs(
        retrieved_docs,
        top_k=top_k,
        max_candidates=max_candidates,
    )


# ============================================================
# QUERY PREPROCESSING
# ============================================================

def fix_markdown_tables(text: str) -> str:
    """
    Fix malformed markdown tables that are on a single line.
    Converts: | A | B | | --- | --- | | 1 | 2 |
    To proper multi-line format.
    """
    if not text or '|' not in text:
        return text
    
    lines = text.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Check if this line looks like an inline table (has separator pattern inline)
        # Pattern: | text | text | | --- | --- | | text | text |
        if re.search(r'\|\s*-{2,}\s*\|.*\|', line) and line.count('|') > 8:
            # This looks like an inline table, try to fix it
            # Split by | and filter empty parts
            parts = [p.strip() for p in line.split('|')]
            parts = [p for p in parts if p]  # Remove empty strings
            
            if len(parts) >= 4:
                # Find separator indices (cells that are just dashes)
                sep_indices = [i for i, p in enumerate(parts) if re.match(r'^-+$', p)]
                
                if sep_indices and len(sep_indices) >= 1:
                    # Number of columns = position of first separator
                    num_cols = sep_indices[0]
                    
                    if num_cols > 0 and num_cols == len(sep_indices):
                        # Build proper table rows
                        result_rows = []
                        for i in range(0, len(parts), num_cols):
                            row_parts = parts[i:i+num_cols]
                            if len(row_parts) == num_cols:
                                result_rows.append('| ' + ' | '.join(row_parts) + ' |')
                        
                        if result_rows:
                            fixed_lines.append('\n'.join(result_rows))
                            continue
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def preprocess_query(query: str) -> str:
    if not query:
        return query

    expansions = {
        "plc": "Programmable Logic Controller",
        "hmi": "Human Machine Interface",
        "profinet": "PROFINET",
        "i/o": "input output",
    }

    processed = str(query).strip()
    for abbr, full in expansions.items():
        token_pattern = rf"(?<![A-Za-z0-9]){re.escape(abbr)}(?![A-Za-z0-9])"
        processed = re.sub(
            rf"(?i){token_pattern}(?!\s*\()",
            lambda m: f"{m.group(0)} ({full})",
            processed,
        )

    processed = re.sub(r"\s{2,}", " ", processed).strip()
    return processed


# ============================================================
# PROMPTS
# ============================================================

def build_enhanced_prompt() -> PromptTemplate:
    template = """You are an industrial PLC field engineer.

Use only MANUAL CONTEXT below.
Do not output retrieval/debug text.
Do not invent numeric parameter values (addresses, station numbers, timers, register values) unless explicitly present in context.
Never merge instructions across incompatible PLC models without labeling model differences.
Write everything in English only.

Core synthesis rule:
Combine related configuration steps across multiple sections into a single executable procedure.
Use consistent terminology:
- First mention: CC-Link IE Field Network Basic (CC-Link IEF)
- After first mention: CC-Link IEF

Output policy:
- If QUESTION_MODE is procedure/troubleshoot: use this structure exactly:
  1) Prerequisites / Notes
  2) Steps
  3) Troubleshooting
- If QUESTION_MODE is qa: answer directly, then add short actionable steps only when useful.
- If key info is missing after using all context, say exactly what is missing and continue with remaining verified steps.
- Keep step numbering internally consistent (do not reference a total step count that does not exist).
- For model-specific branching, write branching notes before steps (not inside the middle of step flow).
- For enable/disable settings, provide explicit decision criteria.
- In troubleshooting, include common failure patterns (for example: station not detected, timeout, link scan fail), relevant LEDs (RUN/ERR/LINK), and likely diagnostic checks.

{history_section}MANUAL CONTEXT:
{context}

USER TASK TEMPLATE:
{task_prompt}

QUESTION_MODE: {question_mode}
CURRENT QUESTION:
{question}

ANSWER:"""
    return PromptTemplate(
        input_variables=["history_section", "context", "task_prompt", "question_mode", "question"],
        template=template,
    )


def build_no_context_prompt() -> PromptTemplate:
    template = """You are an industrial PLC field engineer.

No reliable manual context was retrieved for this question.

Respond with:
1. A brief limitation statement.
2. What exact manual details are missing (model, parameter section, page, or error code table).
3. The best next query the user should ask to retrieve the correct section.

QUESTION:
{question}

ANSWER:"""
    return PromptTemplate(input_variables=["question"], template=template)


def format_chat_history(chat_history: List[dict], max_messages: int = 6) -> str:
    """
    Format chat history for inclusion in the prompt.
    
    Args:
        chat_history: List of {"role": "user"|"assistant", "content": "..."}
        max_messages: Maximum number of recent messages to include
    
    Returns:
        Formatted string for the prompt, or empty string if no history
    """
    if not chat_history:
        return ""
    
    # Take only the last N messages
    recent = chat_history[-max_messages:]
    
    if not recent:
        return ""
    
    # Format as numbered exchanges with clear structure
    formatted_lines = []
    exchange_num = 1
    i = 0
    
    while i < len(recent):
        msg = recent[i]
        if msg.get("role") == "user":
            user_content = msg.get("content", "")[:200]
            # Check if there's a following assistant message
            assistant_content = ""
            if i + 1 < len(recent) and recent[i + 1].get("role") == "assistant":
                assistant_content = recent[i + 1].get("content", "")[:200]
                i += 1
            
            formatted_lines.append(f"[Exchange {exchange_num}]")
            formatted_lines.append(f"  Q: {user_content}")
            if assistant_content:
                formatted_lines.append(f"  A: {assistant_content}")
            exchange_num += 1
        i += 1
    
    if not formatted_lines:
        return ""
    
    return "=== PREVIOUS CONVERSATION ===\n" + "\n".join(formatted_lines) + "\n=== END PREVIOUS CONVERSATION ===\n\n"


# ============================================================
# MAIN RAG FUNCTION (PURE)
# ============================================================

def answer_question(
    question: str,
    db_pool,
    llm,
    embedder,
    collection: str,
    retriever_class,
    reranker_class,
    chat_history: List[dict] = None,
) -> dict:

    processed_msg = preprocess_query((question or "").strip())
    if not processed_msg:
        return {"reply": "Please enter a question."}

    t0 = time.perf_counter()
    question_mode = _question_mode(processed_msg)
    # Keep procedure/troubleshoot prompts lean to reduce latency and leakage.
    history_section = (
        ""
        if question_mode in {"procedure", "troubleshoot"}
        else format_chat_history(chat_history or [], max_messages=5)
    )

    # ============ RETRIEVAL PHASE ============
    t_retrieval_start = time.perf_counter()
    rerank_top_n = _rerank_topn_for_mode(question_mode)
    selected_top_k = _topk_for_mode(question_mode)
    max_candidates = _max_candidates_for_mode(question_mode)
    
    base_retriever = retriever_class(
        connection_pool=db_pool,
        embedder=embedder,
        collection=collection,
    )
    
    # Get raw retrieved docs from base retriever
    raw_retrieved_docs = base_retriever.invoke(processed_msg) or []
    t_retrieval_end = time.perf_counter()
    retrieval_time = t_retrieval_end - t_retrieval_start

    # ============ RERANKING PHASE ============
    t_rerank_start = time.perf_counter()
    
    reranker = None
    for kwargs in (
        {"base_retriever": base_retriever, "prefetched_docs": raw_retrieved_docs, "top_n": rerank_top_n},
        {"base_retriever": base_retriever, "prefetched_docs": raw_retrieved_docs},
        {"base_retriever": base_retriever, "top_n": rerank_top_n},
        {"base_retriever": base_retriever},
    ):
        try:
            reranker = reranker_class(**kwargs)
            break
        except TypeError:
            continue
    if reranker is None:
        raise RuntimeError("Failed to initialize reranker with compatible arguments")

    retrieved_docs = reranker.invoke(processed_msg) or []
    selected_docs = select_context_docs(
        retrieved_docs,
        question=processed_msg,
        question_mode=question_mode,
        top_k=selected_top_k,
        max_candidates=max_candidates,
    )
    
    t_rerank_end = time.perf_counter()
    rerank_time = t_rerank_end - t_rerank_start

    context_chars = (
        _env_int("CHAT_CONTEXT_MAX_CHARS_PROCEDURE", 600)
        if question_mode in {"procedure", "troubleshoot"}
        else _env_int("CHAT_CONTEXT_MAX_CHARS_QA", 420)
    )
    context_texts = [
        _compact_context_text(getattr(d, "page_content", "") or "", context_chars)
        for d in selected_docs
    ]
    citation_items = build_source_citations(selected_docs)
    max_score = get_doc_score(retrieved_docs[0]) if retrieved_docs else None
    task_prompt = _build_task_prompt(processed_msg, question_mode)



    # ============ LLM PHASE ============
    t_llm_start = time.perf_counter()
    reply = None

    if reply is None and context_texts:
        context_headers = []
        for i, doc in enumerate(selected_docs):
            source_page = _doc_source_page(doc, fallback_index=i)
            page = source_page["page"]
            page_label = str(page) if page > 0 else "unknown"
            source_label = str(source_page["source"]).replace("\n", " ").strip()
            context_headers.append(
                f"<<<DOC {i + 1} source={source_label} page={page_label}>>>"
            )

        context_str = "\n\n---\n\n".join(
            f"{header}\n{content}" for header, content in zip(context_headers, context_texts)
        )
        prompt_template = build_enhanced_prompt()
        prompt_inputs = {
            "history_section": history_section,
            "context": context_str,
            "task_prompt": task_prompt,
            "question_mode": question_mode,
            "question": processed_msg,
        }
    elif reply is None:
        prompt_template = build_no_context_prompt()
        prompt_inputs = {
            "question": processed_msg,
        }
    if reply is None:
        rendered_prompt = prompt_template.format(**prompt_inputs)

    # LLM call with retry logic (exponential backoff)
    if reply is None:
        max_retries = max(1, _env_int("LLM_MAX_RETRIES", 1))
        for attempt in range(max_retries):
            try:
                reply = invoke_llm_with_fallback(llm, rendered_prompt)
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                    raise

    reply = fix_markdown_tables(str(reply))  # Fix malformed markdown tables
    reply = _sanitize_prompt_leakage(reply)
    reply = _normalize_engineering_steps(reply, question_mode)
    reply = _enforce_numeric_guardrail(reply, context_texts, question_mode)
    if _looks_broken_reply(reply):
        logger.warning("⚠️ Broken/empty LLM reply detected. Replacing with safe fallback.")
        reply = (
            "I couldn't generate a usable answer right now.\n\n"
            "Please try:\n"
            "1. Rephrase the question in one clear sentence\n"
            "2. Ask a specific topic (for example: model, error code, or protocol)\n"
            "3. Try again in a few seconds"
        )

    # If we already have context-backed content, avoid a contradictory trailing fallback sentence.
    if context_texts:
        reply = re.sub(
            r"\n*\s*I couldn't find specific information about this\.?\s*$",
            "",
            reply,
            flags=re.IGNORECASE,
        ).strip()
    
    t_llm_end = time.perf_counter()
    llm_time = t_llm_end - t_llm_start

    # ============ RAGAS EVALUATION ============
    ragas_scores = None
    enable_chat_ragas = _env_bool("ENABLE_CHAT_RAGAS", False)
    allow_source_citations = True

    if enable_chat_ragas and context_texts:
        skip_ragas_check = _should_skip_ragas(processed_msg)

        try:
            from app.ragas_eval import simple_ragas_eval

            ragas_scores = simple_ragas_eval(
                question=processed_msg,
                answer=reply,
                contexts=context_texts,
            )

            # Check if quality is too low (but skip for identity/greeting questions)
            if not skip_ragas_check and ragas_scores and ragas_scores.get("scores"):
                scores = ragas_scores["scores"]
                threshold = _env_float("RAGAS_MIN_THRESHOLD", RAGAS_MIN_THRESHOLD)
                faith_hard_fail = _env_float(
                    "RAGAS_FAITHFULNESS_HARD_FAIL",
                    RAGAS_FAITHFULNESS_HARD_FAIL,
                )
                relevancy_hard_fail = _env_float(
                    "RAGAS_RELEVANCY_HARD_FAIL",
                    RAGAS_RELEVANCY_HARD_FAIL,
                )

                weighted_sum = 0.0
                weight_total = 0.0
                for metric, weight in RAGAS_METRIC_WEIGHTS.items():
                    val = scores.get(metric)
                    if val is None:
                        continue
                    weighted_sum += float(val) * float(weight)
                    weight_total += float(weight)
                weighted_quality = (weighted_sum / weight_total) if weight_total > 0 else None

                faithfulness = scores.get("faithfulness")
                relevancy = scores.get("answer_relevancy")
                hard_fail = (
                    (faithfulness is not None and float(faithfulness) < faith_hard_fail)
                    or (relevancy is not None and float(relevancy) < relevancy_hard_fail)
                )
                low_quality = (
                    weighted_quality is not None and weighted_quality < threshold
                )

                if hard_fail or low_quality:
                    reply = (
                        "I'm sorry, but I don't have enough reliable information in my documents "
                        "to answer this question accurately. The context I found may not be "
                        "relevant or sufficient.\n\n"
                        "**Please try:**\n"
                        "• Rephrasing your question\n"
                        "• Asking about a more specific topic\n"
                        "• Checking if the topic is covered in the documentation"
                    )
                    logger.warning(
                        "⚠️ RAGAS quality gate triggered "
                        f"(weighted={weighted_quality if weighted_quality is not None else 'N/A'}, "
                        f"faithfulness={faithfulness}, relevancy={relevancy}, "
                        f"threshold={threshold}, hard_fail={hard_fail}). "
                        "Replacing with fallback response."
                    )
                    allow_source_citations = False
        except Exception as e:
            logger.warning(f"RAGAS evaluation failed: {e}")
            ragas_scores = None

    if question_mode == "qa" and allow_source_citations and _env_bool("APPEND_SOURCE_CITATIONS", True):
        citations = build_citations_from_docs(selected_docs)
        if citations:
            reply += "\n\nSources:\n- " + "\n- ".join(citations)

    total_time = time.perf_counter() - t0

    # ============ LOG THE REQUEST ============
    log_chat_request(
        question=question,
        answer=reply,
        retrieval_time=retrieval_time,
        rerank_time=rerank_time,
        llm_time=llm_time,
        total_time=total_time,
        retrieved_docs=raw_retrieved_docs,
        reranked_docs=retrieved_docs,
        selected_docs=selected_docs,
        max_score=max_score,
        ragas_scores=ragas_scores,
    )

    return {
        "reply": reply,
        "processing_time": round(total_time, 2),
        "retrieval_time": round(retrieval_time, 2),
        "rerank_time": round(rerank_time, 2),
        "llm_time": round(llm_time, 2),
        "question_mode": question_mode,
        "context_count": len(context_texts),
        "max_score": max_score,
        "ragas": ragas_scores,
        "sources": citation_items if allow_source_citations else [],
        "contexts_list": context_texts,
    }
