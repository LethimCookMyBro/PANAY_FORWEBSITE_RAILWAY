# backend/app/chatbot.py
# VERSION 5.0 - PURE RAG (NO DB SIDE EFFECTS)

from langchain_core.prompts import PromptTemplate
from typing import List, Optional, Dict, Any
import math
import logging
import re
import time
import os
import requests
from urllib.parse import urlparse, urlunparse, urljoin

# Setup logging
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

FINAL_K = 3
MIN_KEEP = 2
ALPHA = 0.6
HARD_MIN = 0.10
SOFT_MIN = 0.15
MAX_CANDIDATES = 5

# RAGAS quality threshold - if average score below this, respond "I don't know"
RAGAS_MIN_THRESHOLD = 0.40  # 40% minimum average quality

# Questions matching these patterns skip RAGAS quality check (identity/greeting questions)
SKIP_RAGAS_PATTERNS = [
    "your name", "who are you", "what are you", "introduce yourself",
    "hello", "hi ", "hey ", "good morning", "good afternoon", "good evening",
    "thank you", "thanks", "bye", "goodbye", "see you",
    "how are you", "what can you do", "help me", "what is panya",
]


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
    num_predict = int(os.getenv("LLM_NUM_PREDICT", "1024"))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    request_payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "temperature": temperature,
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
            score = getattr(doc, 'score', None) or doc.metadata.get('score', 'N/A')
            if isinstance(score, float):
                score = f"{score:.4f}"
            content_preview = doc.page_content[:80].replace('\n', ' ')
            source = doc.metadata.get('source', 'unknown')[:30]
            log_parts.append(f"   [{i+1}] Score: {score} | {source}")
            log_parts.append(f"       Preview: {content_preview}...")
    
    log_parts.append("")
    
    # Selected chunks section
    log_parts.append("📄 SELECTED CHUNKS (used for context):")
    if selected_docs:
        for i, doc in enumerate(selected_docs):
            score = getattr(doc, 'score', None) or doc.metadata.get('score', 'N/A')
            if isinstance(score, float):
                score = f"{score:.4f}"
            content_preview = doc.page_content[:120].replace('\n', ' ')
            source = doc.metadata.get('source', 'unknown')
            chunk_type = doc.metadata.get('chunk_type', 'standard')
            log_parts.append(f"   ── Chunk {i+1} ──")
            log_parts.append(f"   Score: {score} | Type: {chunk_type}")
            log_parts.append(f"   Source: {source}")
            log_parts.append(f"   Content: {content_preview}...")
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
    seen = set()
    citations = []

    for d in docs:
        md = getattr(d, "metadata", {}) or {}

        source = md.get("source_key") or md.get("source") or "unknown"
        page = md.get("page") or 0

        try:
            page = int(page)
        except Exception:
            page = 0

        key = (source, page)
        if key in seen:
            continue
        seen.add(key)

        if page > 0:
            citations.append(f"{source} (page {page})")
        else:
            citations.append(f"{source}")

    return citations


# ============================================================
# CONTEXT SELECTION
# ============================================================

def select_context_docs(retrieved_docs: List, max_candidates: int = MAX_CANDIDATES) -> List:
    candidates = (retrieved_docs or [])[:max_candidates]

    if not candidates:
        return []

    max_score = get_doc_score(candidates[0])
    if max_score is None or max_score < HARD_MIN:
        return []

    cutoff = max(max_score * ALPHA, SOFT_MIN)

    final_docs = []
    for i, doc in enumerate(candidates):
        score = get_doc_score(doc) or max_score
        if i < MIN_KEEP or score >= cutoff:
            final_docs.append(doc)
        if len(final_docs) >= FINAL_K:
            break

    return final_docs


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

    abbreviations = {
        "plc": "Programmable Logic Controller",
        "hmi": "Human Machine Interface",
        "profinet": "PROFINET",
        "i/o": "input output",
        "gds": "Global Data Space",
        "esm": "Execution and Synchronization Manager",
    }

    processed = query.lower()
    for abbr, full in abbreviations.items():
        processed = re.sub(rf"\b{re.escape(abbr)}\b", full, processed)

    return processed if processed != query.lower() else query


# ============================================================
# PROMPTS
# ============================================================

def build_enhanced_prompt() -> PromptTemplate:
    template = """You are Panya, an Industrial Automation and PLC expert assistant.

{history_section}CONTEXT:
{context}

CRITICAL RULES:
- Answer using information from the context above
- If the answer is NOT found, say: "I couldn't find specific information about this."
- DO NOT make up facts or specifications
- If context tags include [Source: ... | Page: ...], use them as evidence and never invent page numbers
- Respond in English only
- Do not produce contradictory statements in the same answer
- If context is ambiguous/conflicting, explicitly say it is conflicting and present both interpretations with their source/page tags
- Every concrete specification or error-code meaning must be supported by the provided context
- Answer ONLY the CURRENT QUESTION below

FORMATTING:
- For step-by-step procedures or "how to" questions: Use NUMBERED LISTS (1. 2. 3.)
- For feature lists, specifications, or options: Use bullet points (•)
- Use **bold** for important terms, menu items [like this], and specifications
- Structure your response with clear sections if the answer is complex
- Keep responses clear, concise and scannable

RESPONSE QUALITY:
- Prefer exact manual wording over generic guidance
- If evidence is insufficient, say exactly what is missing instead of guessing
- If the question asks for model-specific behavior, do not answer with cross-vendor assumptions

CURRENT QUESTION:
{question}

ANSWER:"""
    return PromptTemplate(input_variables=["history_section", "context", "question"], template=template)


def build_no_context_prompt() -> PromptTemplate:
    template = """You are Panya, an Industrial Automation assistant.

{history_section}IMPORTANT: No relevant documents were found in my knowledge base for this question.

GUIDELINES:
- Clearly state that you don't have specific documentation for this topic
- Do not provide guessed values, specs, or model-specific claims
- Do not provide broad generic troubleshooting lists as a substitute for documentation evidence
- Ask the user for the exact model/manual keyword if needed
- Respond in English only
- Answer ONLY the CURRENT QUESTION below

FORMATTING:
- Use bullet points (•) for lists
- Use **bold** for important terms
- Keep responses clear and concise

CURRENT QUESTION (answer this):
{question}

ANSWER:"""
    return PromptTemplate(input_variables=["history_section", "question"], template=template)


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

    # Format chat history for prompt (last 5 messages)
    history_section = format_chat_history(chat_history or [], max_messages=5)

    # ============ RETRIEVAL PHASE ============
    t_retrieval_start = time.perf_counter()
    
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
    
    try:
        reranker = reranker_class(
            base_retriever=base_retriever,
            prefetched_docs=raw_retrieved_docs,
        )
    except TypeError:
        # Backward compatibility for custom rerankers without prefetched_docs.
        reranker = reranker_class(base_retriever=base_retriever)
    retrieved_docs = reranker.invoke(processed_msg) or []
    selected_docs = select_context_docs(retrieved_docs)
    
    t_rerank_end = time.perf_counter()
    rerank_time = t_rerank_end - t_rerank_start

    context_texts = [d.page_content for d in selected_docs]
    citation_items = build_source_citations(selected_docs)
    max_score = get_doc_score(retrieved_docs[0]) if retrieved_docs else None



    # ============ LLM PHASE ============
    t_llm_start = time.perf_counter()
    
    if context_texts:
        context_headers = []
        for i, doc in enumerate(selected_docs):
            source_page = _doc_source_page(doc, fallback_index=i)
            page = source_page["page"]
            page_label = str(page) if page > 0 else "unknown"
            context_headers.append(
                f"[Source: {source_page['source']} | Page: {page_label}]"
            )

        context_str = "\n\n---\n\n".join(
            f"{header}\n{content}" for header, content in zip(context_headers, context_texts)
        )
        prompt_template = build_enhanced_prompt()
        prompt_inputs = {
            "history_section": history_section,
            "context": context_str,
            "question": processed_msg,
        }
    else:
        prompt_template = build_no_context_prompt()
        prompt_inputs = {
            "history_section": history_section,
            "question": processed_msg,
        }
    rendered_prompt = prompt_template.format(**prompt_inputs)

    # LLM call with retry logic (exponential backoff)
    max_retries = max(1, _env_int("LLM_MAX_RETRIES", 1))
    reply = None
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
        # Check if this is an identity/greeting question that should skip RAGAS check
        question_lower = question.lower()
        skip_ragas_check = any(pattern in question_lower for pattern in SKIP_RAGAS_PATTERNS)

        try:
            from app.ragas_eval import simple_ragas_eval

            ragas_scores = simple_ragas_eval(
                question=question,
                answer=reply,
                contexts=context_texts,
            )

            # Check if quality is too low (but skip for identity/greeting questions)
            if not skip_ragas_check and ragas_scores and ragas_scores.get("scores"):
                scores = ragas_scores["scores"]
                quality_scores = []
                for key in ["faithfulness", "context_precision", "context_recall"]:
                    val = scores.get(key)
                    if val is not None:
                        quality_scores.append(val)

                if quality_scores:
                    avg_quality = sum(quality_scores) / len(quality_scores)
                    if avg_quality < RAGAS_MIN_THRESHOLD:
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
                            f"⚠️ Low quality response detected (avg={avg_quality:.2f} < {RAGAS_MIN_THRESHOLD}). "
                            f"Replacing with 'I don't know' message."
                        )
                        allow_source_citations = False
        except Exception as e:
            logger.warning(f"RAGAS evaluation failed: {e}")
            ragas_scores = None

    if allow_source_citations and _env_bool("APPEND_SOURCE_CITATIONS", True):
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
        "context_count": len(context_texts),
        "max_score": max_score,
        "ragas": ragas_scores,
        "sources": citation_items if allow_source_citations else [],
        "contexts_list": context_texts,
    }
