# backend/app/ragas_eval.py
import os
import re
import time
import math
import logging
import warnings
from typing import Any, Dict, List, Optional, Union

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger(__name__)

_llm_cache = None
_embeddings_cache = None
_fast_embedder_cache = None

# -----------------------------
# Helpers
# -----------------------------
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _extract_ground_truth_from_contexts(question: str, contexts: List[str]) -> Optional[str]:
    """(Optional) Try to extract Ground Truth from Q/A blocks in contexts if present."""
    if not contexts:
        return None
    q = (question or "").strip().lower()
    patterns = [
        r"Question:\s*(.+?)\s*\n\s*Answer:\s*(.+?)(?:\n\n|\nQuestion:|$)",
        r"Q:\s*(.+?)\s*\n\s*A:\s*(.+?)(?:\n\n|$)",
        r"question[:\s]+(.+?)\s*answer[:\s]+(.+?)(?:\n|$)",
    ]
    best, best_overlap = None, 0.0
    for ctx in contexts:
        for pat in patterns:
            for q_text, a_text in re.findall(pat, ctx, flags=re.I | re.S):
                ctx_q = q_text.strip().lower()
                if len(ctx_q) < 5:
                    continue
                qs = set(q.split())
                cs = set(ctx_q.split())
                if not qs:
                    continue
                overlap = len(qs & cs) / max(1, len(qs | cs))
                if overlap > best_overlap and overlap >= 0.55:
                    best_overlap = overlap
                    best = a_text.strip()
    return best


def _choose_contexts(question: str, contexts: List[str], k: int, max_chars: int) -> List[str]:
    """
    Choose K contexts: Score based on Jaccard overlap with question + bonus if Q/A block.
    (Written straightforwardly without walrus operator to avoid syntax issues)
    """
    qset = set((question or "").lower().split())
    scored: List[tuple] = []
    for c in contexts:
        c_low = (c or "").lower()
        c_words = c_low.split()
        cset = set(c_words)
        jacc = len(qset & cset) / max(1, len(qset | cset))
        bonus = 0.2 if ("question:" in c_low and "answer:" in c_low) else 0.0
        scored.append((jacc + bonus, (c or "")[:max_chars]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:max(1, k)]]

# -----------------------------
# Models
# -----------------------------
def _build_langchain_llm():
    """Build LLM for RAGAS (supports ollama/openai) + caching."""
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache

    from ragas.llms import LangchainLLMWrapper
    provider = os.getenv("RAGAS_LLM_PROVIDER", "ollama").strip().lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set but RAGAS_LLM_PROVIDER=openai")
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("RAGAS_LLM_TEMPERATURE", "0"))
        max_tokens = int(os.getenv("RAGAS_LLM_MAX_TOKENS", "512"))
        lc = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens, timeout=60)
        _llm_cache = LangchainLLMWrapper(lc)
        logger.info(f"[RAGAS LLM] OpenAI model={model}")
        return _llm_cache

    # ---- OLLAMA (default) ----
    try:
        from langchain_ollama import ChatOllama  # New package, recommended
        backend = "langchain_ollama"
    except Exception:
        from langchain_community.chat_models import ChatOllama  # Legacy fallback
        backend = "langchain_community"

    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    # Use judge model if available, otherwise fallback to main model
    model = os.getenv("RAGAS_LLM_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2"))

    model_kwargs = {
        "num_ctx": int(os.getenv("RAGAS_NUM_CTX", "256")),
        "num_predict": int(os.getenv("RAGAS_NUM_PREDICT", "16")),
        "temperature": float(os.getenv("RAGAS_LLM_TEMPERATURE", "0.0")),
        "top_k": int(os.getenv("RAGAS_TOP_K", "1")),
        "top_p": float(os.getenv("RAGAS_TOP_P", "0.05")),
        "repeat_penalty": float(os.getenv("RAGAS_REPEAT_PENALTY", "1.05")),
    }

    lc = ChatOllama(model=model, base_url=base_url, model_kwargs=model_kwargs)
    _llm_cache = LangchainLLMWrapper(lc)
    logger.info(f"[RAGAS LLM] provider=ollama backend={backend} model={model} kwargs={model_kwargs}")
    return _llm_cache


def _build_embeddings():
    """Build Embeddings for RAGAS/embedding-only + caching."""
    global _embeddings_cache
    if _embeddings_cache is not None:
        return _embeddings_cache

    # Support both env names for compatibility
    model_name = os.getenv("RAGAS_EMBED_MODEL_EVAL", "") or os.getenv("EVAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    _embeddings_cache = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "cpu"},
    )
    logger.info(f"[RAGAS] Embeddings: {model_name}")
    return _embeddings_cache

# -----------------------------
# Evaluator (LLM-judge core)
# -----------------------------
def _eval_metrics_seq(dataset, metrics, llm, embeddings) -> Dict[str, Optional[float]]:
    """
    Evaluate metrics sequentially + supports RunConfig + total time budget per round.
    """
    from ragas import evaluate
    scores: Dict[str, Optional[float]] = {m.name: None for m in metrics}
    timeout = int(os.getenv("RAGAS_TIMEOUT", "90"))
    budget_s = int(os.getenv("RAGAS_BUDGET_S", "75"))
    t0 = time.time()

    try:
        from ragas.run_config import RunConfig
        run_cfg = RunConfig(timeout=timeout, max_workers=1)
        use_run_cfg = True
    except Exception:
        run_cfg, use_run_cfg = None, False
        logger.warning("[RAGAS] run_config not available; evaluate() will use defaults")

    for m in metrics:
        # Stop if total budget exceeded
        if time.time() - t0 > budget_s:
            logger.warning(f"[RAGAS] budget exceeded before metric {m.name}, skipping.")
            break
        try:
            if use_run_cfg:
                res = evaluate(dataset, metrics=[m], llm=llm, embeddings=embeddings, run_config=run_cfg)
            else:
                res = evaluate(dataset, metrics=[m], llm=llm, embeddings=embeddings)
            df = res.to_pandas() if hasattr(res, "to_pandas") else None
            if df is not None and not df.empty and m.name in df.columns:
                v = df[m.name].iloc[0]
                scores[m.name] = None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        except Exception as e:
            logger.warning(f"[RAGAS] metric {m.name} failed: {e}")
    return scores


def _ragas_eval_llm(question: str, answer: str, contexts: List[str], llm, embeddings) -> Dict:
    """
    Use RAGAS LLM-judge:
    - Select K contexts (try to keep GT block if Question/Answer format exists)
    - Metric order: context_precision -> context_recall -> answer_relevancy -> faithfulness
    - If any metric times out/fails, fill with embed-only estimate to "always have complete scores"
    """
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset

    max_chars = int(os.getenv("RAGAS_CONTEXT_MAX_CHARS", "240"))
    k = int(os.getenv("RAGAS_CONTEXTS_K", "1"))

    # ---- Keep GT block if present (same Question/Answer format) ----
    gt_block = None
    for c in contexts or []:
        if c.lower().startswith("question:") and "\nanswer:" in c.lower():
            if (question.strip().lower() in c.strip().lower()):
                gt_block = c[:max_chars]
                break

    # Select K contexts by proximity + truncate
    picked = _choose_contexts(question, contexts or [], k=k, max_chars=max_chars)
    if gt_block and gt_block not in picked:
        # Force GT to be included by pushing out the last item if exceeds K
        picked = ([gt_block] + picked)[:max(1, k)]

    ctx_used = [(c or "")[:max_chars] for c in picked]

    if len(ctx_used) == 0:
        eo = _ragas_eval_embed_only(question, answer, [])
        eo["scores"]["context_precision"] = 0.0
        eo["scores"]["context_recall"] = 0.0
        return {"status": "completed", "scores": eo["scores"]}

    # ---- Prepare dataset ----
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [ctx_used],
        "ground_truth": [""]  # optional
    }
    ds = Dataset.from_dict(data)

    # ---- Metric order: context metrics first ----
    metrics = [context_precision, context_recall, answer_relevancy, faithfulness]

    # ---- Run within budget ----
    scores = _eval_metrics_seq(ds, metrics, llm=llm, embeddings=embeddings)

    # ---- Fill missing metrics with embed-only estimates ----
    try:
        eo_scores = _ragas_eval_embed_only(question=question, answer=answer, contexts=ctx_used)["scores"]
        for k_ in ("context_precision", "context_recall", "answer_relevancy", "faithfulness"):
            if k_ in scores and (scores[k_] is None):
                scores[k_] = eo_scores.get(k_)
    except Exception:
        pass

    return {"status": "completed", "scores": scores}

# -----------------------------
# Embedding-only evaluator (fast)
# -----------------------------
def _ragas_eval_embed_only(question: str, answer: str, contexts: List[str]) -> Dict:
    """
    Fast version (no LLM):
    - faithfulness (approx): cosine(answer, centroid(contexts))
    - answer_relevancy: cosine(answer, question)
    - context_precision/recall: simple vector proximity metrics
    """
    global _fast_embedder_cache
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # Use cached embedder for performance
    if _fast_embedder_cache is None:
        model_name = os.getenv("RAGAS_EMBED_MODEL_EVAL", "") or os.getenv("EVAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _fast_embedder_cache = SentenceTransformer(model_name)
    
    emb = _fast_embedder_cache

    def vec(x: Union[List[str], str]):
        xs = x if isinstance(x, list) else [x]
        m = emb.encode(xs, normalize_embeddings=True)
        return m if isinstance(x, list) else m[0]

    qv = vec(question)
    av = vec(answer)
    cv = vec(contexts) if contexts else None

    def cos(a, b):
        return float(np.clip(np.dot(a, b), -1.0, 1.0))

    scores: Dict[str, Optional[float]] = {}
    scores["answer_relevancy"] = (cos(av, qv) + 1) / 2

    if cv is not None and len(contexts) > 0:
        centroid = cv.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        scores["faithfulness"] = (cos(av, centroid) + 1) / 2

        q2c = [cos(qv, c) for c in cv]
        a2c = [cos(av, c) for c in cv]
        high = [1 for v in a2c if v > 0.3]
        scores["context_precision"] = sum(high) / max(len(a2c), 1)
        scores["context_recall"] = (sum(q2c) / max(len(q2c), 1) + 1) / 2
    else:
        scores["faithfulness"] = None
        scores["context_precision"] = 0.0
        scores["context_recall"] = 0.0

    return {"status": "completed", "scores": scores}

# -----------------------------
# Public API
# -----------------------------
def local_ragas_eval(question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
    """
    Main mode: If ENABLE_RAGAS_LLM=true, uses LLM-judge; otherwise uses embed-only.
    (Important) Must create llm/embeddings then pass to _ragas_eval_llm().
    """
    try:
        if _env_bool("ENABLE_RAGAS_LLM", True):
            llm = _build_langchain_llm()
            embeddings = _build_embeddings()
            return _ragas_eval_llm(question, answer, contexts, llm=llm, embeddings=embeddings)
        return _ragas_eval_embed_only(question, answer, contexts)
    except Exception as e:
        logger.warning(f"[RAGAS] LLM evaluation failed → fallback: {e}")
        return _ragas_eval_embed_only(question, answer, contexts)


def simple_ragas_eval(question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
    """Fast mode: Always uses embed-only."""
    return _ragas_eval_embed_only(question, answer, contexts)
