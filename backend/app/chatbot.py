# backend/app/chatbot.py
# VERSION 5.0 - PURE RAG (NO DB SIDE EFFECTS)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Optional
import math
import logging
import re
import time


# ============================================================
# CONFIGURATION
# ============================================================

FINAL_K = 3
MIN_KEEP = 2
ALPHA = 0.6
HARD_MIN = 0.10
SOFT_MIN = 0.15
MAX_CANDIDATES = 5


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
    return None


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

REFERENCE DOCUMENTS:
{context}

RULES:
- Answer strictly based on the documents above
- Be concise and technical
- If information is missing, say so clearly

QUESTION:
{question}

ANSWER:"""
    return PromptTemplate(input_variables=["context", "question"], template=template)


def build_no_context_prompt() -> PromptTemplate:
    template = """You are Panya, an Industrial Automation assistant.

No relevant documents were found.
Answer using general automation knowledge only.

QUESTION:
{question}

ANSWER:"""
    return PromptTemplate(input_variables=["question"], template=template)


# ============================================================
# MAIN RAG FUNCTION (PURE)
# ============================================================

def answer_question(
    question: str,
    session_id: int,
    db_pool,
    llm,
    embedder,
    collection: str,
    retriever_class,
    reranker_class,
) -> dict:

    processed_msg = preprocess_query((question or "").strip())
    if not processed_msg:
        return {"reply": "Please enter a question."}

    t0 = time.perf_counter()

    base_retriever = retriever_class(
        connection_pool=db_pool,
        embedder=embedder,
        collection=collection,
    )
    reranker = reranker_class(base_retriever=base_retriever)

    retrieved_docs = reranker.invoke(processed_msg) or []
    selected_docs = select_context_docs(retrieved_docs)

    context_texts = [d.page_content for d in selected_docs]
    max_score = get_doc_score(retrieved_docs[0]) if retrieved_docs else None

    if context_texts:
        context_str = "\n\n---\n\n".join(
            f"[Document {i+1}]\n{c}" for i, c in enumerate(context_texts)
        )
        chain = (
            {"context": (lambda _: context_str), "question": RunnablePassthrough()}
            | build_enhanced_prompt()
            | llm
            | StrOutputParser()
        )
    else:
        chain = (
            {"question": RunnablePassthrough()}
            | build_no_context_prompt()
            | llm
            | StrOutputParser()
        )

    reply = chain.invoke(processed_msg)

    return {
        "reply": reply,
        "processing_time": round(time.perf_counter() - t0, 2),
        "context_count": len(context_texts),
        "max_score": max_score,
    }
