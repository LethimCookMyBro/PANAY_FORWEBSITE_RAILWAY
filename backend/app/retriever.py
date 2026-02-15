import os
import json
import logging
import re
from typing import List, Any, Tuple, Optional

from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from pgvector.psycopg2 import register_vector
import psycopg2  # noqa: F401 (used with connection_pool)
import numpy as np  # noqa: F401 (for type checking in some cases)

# flashrank is optional - falls back to base score if not available
try:
    from flashrank import Ranker, RerankRequest  # type: ignore
    _FLASHRANK_AVAILABLE = True
except Exception:
    _FLASHRANK_AVAILABLE = False

# Singleton ranker instance for performance
_ranker_instance = None

def _get_ranker():
    """Get or create singleton Ranker instance."""
    global _ranker_instance
    if _ranker_instance is None and _FLASHRANK_AVAILABLE:
        model = os.getenv("RERANK_MODEL", "ms-marco-MiniLM-L-12-v2")
        cache_dir = os.getenv("MODEL_CACHE", "/data/models")
        _ranker_instance = Ranker(model_name=model, cache_dir=cache_dir)
        logging.info(f"[Ranker] Initialized: {model}")
    return _ranker_instance


def _safe_load_json(val):
    """Always returns a dict, even if metadata is a string or None."""
    if not val:
        return {}
    if isinstance(val, dict):
        return val
    try:
        return json.loads(val)
    except Exception:
        return {"_raw_meta": str(val)}


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


_HYBRID_HINT_TERMS = (
    "station",
    "parameter",
    "parameters",
    "refresh",
    "write",
    "reset",
    "diagnostic",
    "network",
    "mode",
    "buffer memory",
    "cc-link",
    "profinet",
    "gx works",
)

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "have",
    "what",
    "when",
    "where",
    "which",
    "into",
    "your",
    "about",
    "please",
    "need",
    "show",
    "help",
}


def _extract_query_keywords(query: str, max_terms: int = 12) -> List[str]:
    q = (query or "").lower().strip()
    if not q:
        return []

    terms: List[str] = []
    for phrase in _HYBRID_HINT_TERMS:
        if phrase in q:
            terms.append(phrase)

    for token in re.findall(r"[a-z0-9][a-z0-9\-_/]{1,}", q):
        if len(token) < 3 or token in _STOPWORDS:
            continue
        terms.append(token)

    seen = set()
    unique: List[str] = []
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        unique.append(term)
        if len(unique) >= max_terms:
            break
    return unique


def _doc_dedupe_key(content: str, metadata: Any) -> Tuple[str, str, int]:
    meta = _safe_load_json(metadata)
    source_key = str(meta.get("source_key") or meta.get("source") or "")
    try:
        page = int(meta.get("page") or 0)
    except Exception:
        page = 0
    preview = (content or "")[:180]
    return (source_key, preview, page)


# ===============================
# Base Vector Retriever (pgvector)
# ===============================
class PostgresVectorRetriever(BaseRetriever):
    """
    Retrieve documents from the 'documents' table (pgvector).
    - Uses connection_pool (psycopg2.pool) prepared by main.py
    - Calls register_vector(conn) before each query
    - Returns list[Document] with metadata['distance']
    """
    connection_pool: Any = Field(...)
    embedder: Any = Field(...)
    collection: str = Field(default="plcnext")
    limit: int = Field(default_factory=lambda: _env_int("RETRIEVE_LIMIT", 50))

    def _fetch_semantic_rows(self, cur, query_vector) -> List[Tuple[Any, Any, Any]]:
        cur.execute(
            """
            SELECT content, metadata, embedding <-> %s AS distance
            FROM documents
            WHERE collection = %s
            ORDER BY embedding <-> %s
            LIMIT %s
            """,
            (query_vector, self.collection, query_vector, self.limit),
        )
        return cur.fetchall()

    def _fetch_keyword_rows(self, cur, query_vector, query: str) -> List[Tuple[Any, Any, Any]]:
        if not _env_bool("HYBRID_KEYWORD_ENABLED", True):
            return []

        keyword_terms = _extract_query_keywords(query, max_terms=_env_int("HYBRID_KEYWORD_MAX_TERMS", 12))
        if not keyword_terms:
            return []

        keyword_limit = _env_int("HYBRID_KEYWORD_LIMIT", 20)
        likes = [f"%{term}%" for term in keyword_terms]
        match_clause = " OR ".join(["lower(content) LIKE %s"] * len(likes))
        sql = f"""
            SELECT content, metadata, embedding <-> %s AS distance
            FROM documents
            WHERE collection = %s
              AND ({match_clause})
            ORDER BY embedding <-> %s
            LIMIT %s
        """
        params: List[Any] = [query_vector, self.collection]
        params.extend(likes)
        params.extend([query_vector, keyword_limit])
        cur.execute(sql, params)
        return cur.fetchall()

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # SentenceTransformer embedder returns numpy.ndarray
        query_vector = self.embedder.encode(query)

        conn = self.connection_pool.getconn()
        try:
            register_vector(conn)
            with conn.cursor() as cur:
                semantic_rows = self._fetch_semantic_rows(cur, query_vector)
                keyword_rows = self._fetch_keyword_rows(cur, query_vector, query)

            merged_rows: List[Tuple[Any, Any, Any]] = []
            seen = set()
            for row in semantic_rows:
                key = _doc_dedupe_key(row[0], row[1])
                if key in seen:
                    continue
                seen.add(key)
                merged_rows.append(row)
            for row in keyword_rows:
                key = _doc_dedupe_key(row[0], row[1])
                if key in seen:
                    continue
                seen.add(key)
                merged_rows.append(row)

            docs: List[Document] = []
            semantic_seen = {_doc_dedupe_key(r[0], r[1]) for r in semantic_rows}
            for content, metadata, distance in merged_rows:
                meta = _safe_load_json(metadata)
                meta["distance"] = float(distance)
                meta["retrieval_match"] = "semantic" if _doc_dedupe_key(content, metadata) in semantic_seen else "keyword"
                docs.append(Document(page_content=content, metadata=meta))
            return docs

        except Exception as e:
            logging.error("🔥 Error in PostgresVectorRetriever: %s", e, exc_info=True)
            return []
        finally:
            self.connection_pool.putconn(conn)


# ===============================
# Flashrank-based Reranker
# ===============================
class EnhancedFlashrankRerankRetriever(BaseRetriever):
    """
    Rerank stage (lexical+semantic) using Flashrank with domain-boost.
    - base_retriever: first retrieves candidates
    - top_n: number of final results (env: RERANK_TOPN)
    - Limits candidates sent to Flashrank via env: RERANK_CANDIDATES_MAX (for speed/stability)
    """
    base_retriever: BaseRetriever = Field(...)
    top_n: int = Field(default_factory=lambda: _env_int("RERANK_TOPN", 8))
    prefetched_docs: Optional[List[Document]] = Field(default=None)

    # Domain-specific keywords for boosting
    _PLC_TERMS = [
        # Phoenix Contact
        "plcnext", "phoenix contact", "gds", "esm", "profinet", "axc f",
        "axc f 2152", "axc f 3152", "axc f 1152", "plcnext engineer",
        # Mitsubishi
        "melsec", "fx3", "fx3u", "fx3g", "iq-r", "rcpu", "qcpu", "lcpu",
        "cc-link", "cc link", "edgecross", "data collector", "gx works",
        "iq edgecross", "mitsubishi"
    ]
    _PROTO_TERMS = [
        "protocol", "mode", "rs-485", "rs485", "profinet", "ethernet",
        "serial", "communication", "modbus", "tcp", "udp", "interface",
        "opcua", "opc ua"
    ]

    def _rank(self, query: str, docs: List[Document]) -> List[Tuple[float, Document]]:
        # 1) Get scores from Flashrank (if available), otherwise use 1 - distance as similarity
        ranker = _get_ranker()
        if ranker is not None:
            try:
                passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
                req = RerankRequest(query=query, passages=passages)
                result = ranker.rerank(req)
                # Field names may be "id/score" or "index/relevance_score" depending on flashrank version
                pairs: List[Tuple[float, Document]] = []
                for it in result:
                    idx = int(it.get("id", it.get("index")))
                    sc = float(it.get("score", it.get("relevance_score")))
                    pairs.append((sc, docs[idx]))
            except Exception as e:
                logging.warning("⚠️ Flashrank failed, fallback to base scores: %s", e)
                pairs = [(1.0 - float(d.metadata.get("distance", 1.0)), d) for d in docs]
        else:
            pairs = [(1.0 - float(d.metadata.get("distance", 1.0)), d) for d in docs]

        # 2) Apply domain-specific boosts (soft + capped)
        boosted: List[Tuple[float, Document]] = []
        q_tokens = (query or "").lower().split()
        q_keywords = _extract_query_keywords(query, max_terms=12)
        query_upper = (query or "").upper()
        
        # Extract error/event codes from query (pattern: letter + numbers + H, e.g., F800H, 9801H)
        code_pattern = re.compile(r'\b[A-F0-9]{4,5}H\b', re.IGNORECASE)
        query_codes = set(code_pattern.findall(query_upper))
        
        for s, d in pairs:
            text_low = (d.page_content or "").lower()
            text_upper = (d.page_content or "").upper()
            bonus = 0.0
            
            # HIGH PRIORITY: Exact error/event code match (e.g., F800H, F389H)
            if query_codes:
                chunk_codes = set(code_pattern.findall(text_upper))
                matching_codes = query_codes & chunk_codes
                if matching_codes:
                    bonus += 2.0  # Strong boost for exact code match
            
            if any(w in text_low for w in self._PLC_TERMS):
                bonus += 0.10
            if any(tok in text_low for tok in q_tokens if tok and len(tok) > 2):
                bonus += 0.20
            keyword_hits = sum(1 for kw in q_keywords if kw in text_low)
            if keyword_hits > 0:
                bonus += min(0.60, keyword_hits * 0.12)
            proto_hits = sum(1 for t in self._PROTO_TERMS if t in text_low)
            if proto_hits > 0:
                bonus += min(0.30, proto_hits * 0.08)  # cap 0.30
            ctype = (d.metadata or {}).get("chunk_type")
            if ctype == "golden_qa":
                bonus += 10.0
            elif ctype == "spec_pair":
                bonus += 0.15
            boosted.append((s + bonus, d))

        boosted.sort(key=lambda x: x[0], reverse=True)
        return boosted

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Reuse candidates if caller already fetched them to avoid duplicate DB retrieval.
        cand = list(self.prefetched_docs) if self.prefetched_docs is not None else (self.base_retriever.invoke(query) or [])
        if not cand:
            return []
        # Limit candidates sent to Flashrank for speed/stability
        cap = _env_int("RERANK_CANDIDATES_MAX", 32)
        ranked = self._rank(query, cand[:cap])
        
        # Store score in metadata
        results = []
        for score, doc in ranked[:self.top_n]:
            doc.metadata["score"] = score
            results.append(doc)
        return results


# ===============================
# No-op Reranker (for A/B testing)
# ===============================
class NoRerankRetriever(BaseRetriever):
    """
    No reranking - passes through results from base_retriever and truncates to Top-N.
    """
    base_retriever: BaseRetriever = Field(...)
    top_n: int = Field(default=8)
    prefetched_docs: Optional[List[Document]] = Field(default=None)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = list(self.prefetched_docs) if self.prefetched_docs is not None else (self.base_retriever.invoke(query) or [])
        return docs[: self.top_n]
