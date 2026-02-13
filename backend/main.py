# ============================================================================
# backend/main.py v3.0 - Universal PLC Assistant
# ============================================================================
# CHANGES FROM ORIGINAL:
# 1. ✅ Removed Auto mode entirely - only Fast and Deep modes
# 2. ✅ Generic PLC branding (removed all PLCnext-specific references)
# 3. ✅ Improved code organization with clear sections
# 4. ✅ Better error handling and logging
# 5. ✅ Performance optimizations
# 6. ✅ Cleaner prompt engineering
# 7. ✅ Added comprehensive documentation
#
# MODE EXPLANATION:
# ─────────────────────────────────────────────────────────────────────────────
# FAST MODE (default):
#   - Direct LLM response WITHOUT searching the vector database
#   - Optional web search for current information
#   - Best for: General PLC concepts, quick troubleshooting tips, syntax help
#   - Response time: ~5-15 seconds
#   - Use when: You need quick answers or asking about general topics
#
# DEEP MODE:
#   - Uses RAG (Retrieval-Augmented Generation) pipeline
#   - Searches vector database for relevant documentation chunks
#   - Applies reranking for better context selection
#   - Best for: Specific documentation lookups, detailed specs, accuracy-critical
#   - Response time: ~30-60 seconds  
#   - Use when: You need precise information from your embedded documents
# ─────────────────────────────────────────────────────────────────────────────
# ============================================================================

import os
import logging
import requests
import time
import math
import re
import json
import io
from uuid import uuid4
import mimetypes
import warnings
from urllib.parse import urlparse, urlunparse

from contextlib import asynccontextmanager
from typing import Any, Optional, List, Dict
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
import pytesseract
from PIL import Image

from app.db import init_db_pool, ensure_schema
from app.env_resolver import resolve_database_url, redact_database_url, is_placeholder
from app.routes_auth import router as auth_router
from app.routes_chat import router as chat_router
from app.routes_auth import get_current_user
from app.chat_db import get_user_chat_history
from app.seed import (
    auto_embed_knowledge_if_empty,
    get_auto_embed_batch_size,
    get_auto_embed_chunk_overlap,
    get_auto_embed_chunk_size,
    get_auto_embed_knowledge_dir,
    get_default_golden_qa_path,
    seed_golden_qa_if_empty,
    should_auto_embed_force_rescan,
    should_auto_embed_knowledge,
    should_auto_seed,
)

# Local imports
from app.retriever import (
    PostgresVectorRetriever, 
    EnhancedFlashrankRerankRetriever, 
    NoRerankRetriever
)
from app.chatbot import answer_question
from app.embed_logic import get_embedder
from app.utils import set_llm
from app.errors import (
    ErrorResponse, ErrorCode, AppException,
    create_error_response
)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

def normalize_ollama_base_url(raw_url: str) -> str:
    """
    Normalize Ollama base URL and prevent POST->GET downgrade on HTTP redirects.
    Railway public domains should use HTTPS.
    """
    value = (raw_url or "").strip()
    if not value:
        return "http://ollama:11434"

    parsed = urlparse(value)
    if not parsed.scheme:
        # Keep local container targets on plain HTTP by default.
        value = f"http://{value}"
        parsed = urlparse(value)

    host = (parsed.hostname or "").lower()
    scheme = (parsed.scheme or "").lower()
    should_force_https = (
        scheme == "http"
        and host
        and host.endswith(".railway.app")
    )
    if should_force_https:
        return urlunparse(parsed._replace(scheme="https"))

    return value

class Config:
    """Centralized configuration management"""

    APP_ENV: str = (os.getenv("APP_ENV", "development") or "development").strip().lower()
    
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_BASE_URL: str = normalize_ollama_base_url(
        os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    )
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "180"))
    LLM_NUM_PREDICT: int = int(os.getenv("LLM_NUM_PREDICT", "1024"))  # Max output tokens
    
    # Embeddings
    EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    
    # File processing limits
    FAST_MODE_CHARS: int = int(os.getenv("FAST_MODE_CHARS", "8000"))
    DEEP_MODE_CHARS: int = int(os.getenv("DEEP_MODE_CHARS", "60000"))
    
    # Web search
    WEB_SEARCH_TIMEOUT: int = int(os.getenv("WEB_SEARCH_TIMEOUT", "10"))
    WEB_SEARCH_MAX_RESULTS: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
    
    # Database pool
    DB_POOL_MIN: int = int(os.getenv("DB_POOL_MIN", "1"))
    DB_POOL_MAX: int = int(os.getenv("DB_POOL_MAX", "10"))
    
    # Default collection name for vector store
    DEFAULT_COLLECTION: str = os.getenv("DEFAULT_COLLECTION", "plcnext")


config = Config()

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("PLCAssistant")

logger.info("=" * 60)
logger.info("🤖 PLC Assistant v3.0 - Starting up")
logger.info("=" * 60)

logger.info(f"  App Env: {config.APP_ENV}")
logger.info(f"  Ollama URL: {config.OLLAMA_BASE_URL}")
logger.info(f"  Ollama Model: {config.OLLAMA_MODEL}")
logger.info(f"  Embed Model: {config.EMBED_MODEL_NAME}")
logger.info("=" * 60)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def to_bool(val: Any) -> Optional[bool]:
    """Convert various types to boolean"""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return None


def get_app_env() -> str:
    env = (os.getenv("APP_ENV", config.APP_ENV or "development") or "development").strip().lower()
    if env not in {"development", "production"}:
        logger.warning("Unknown APP_ENV '%s'; defaulting to 'development'", env)
        return "development"
    return env


def is_weak_jwt_secret(secret: Optional[str]) -> bool:
    value = (secret or "").strip()
    return not value or value == "dev-secret" or is_placeholder(value)


def validate_runtime_security_config(app_env: str) -> None:
    jwt_secret = os.getenv("JWT_SECRET", "")
    if app_env == "production" and is_weak_jwt_secret(jwt_secret):
        raise RuntimeError(
            "JWT_SECRET is missing/weak for production. "
            "Set APP_ENV=production with a real JWT_SECRET (not 'dev-secret' and not placeholder)."
        )

    if app_env != "production" and is_weak_jwt_secret(jwt_secret):
        logger.warning(
            "JWT_SECRET is missing/weak while APP_ENV=%s. This is okay for local development but unsafe for production.",
            app_env,
        )


def sanitize_json(obj: Any) -> Any:
    """
    Recursively sanitize objects for JSON serialization.
    Handles numpy types and invalid float values.
    """
    if obj is None:
        return None
    
    # Handle numpy types
    if isinstance(obj, (np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle Python float
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    
    # Handle containers
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    
    return obj


# ============================================================================
# WEB SEARCH
# ============================================================================

def web_search(query: str, max_results: int = None) -> str:
    """
    Search the web using DuckDuckGo HTML interface.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        Formatted string of search results or empty string on failure
    """
    if max_results is None:
        max_results = config.WEB_SEARCH_MAX_RESULTS
        
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        from urllib.parse import quote_plus
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        
        response = requests.get(
            search_url, 
            headers=headers, 
            timeout=config.WEB_SEARCH_TIMEOUT
        )
        
        if response.status_code != 200:
            logger.warning(f"Web search returned status {response.status_code}")
            return ""
        
        html = response.text
        results = []
        
        # Extract snippets
        snippet_pattern = r'<a class="result__snippet"[^>]*>(.*?)</a>'
        snippets = re.findall(snippet_pattern, html, re.DOTALL)
        
        for snippet in snippets[:max_results]:
            clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
            if clean_snippet:
                results.append(f"• {clean_snippet}")
        
        if results:
            logger.info(f"🌐 Web search found {len(results)} results for: {query[:50]}...")
            return "\n".join(results)
        
        # Fallback to titles if no snippets
        title_pattern = r'<a class="result__a"[^>]*>(.*?)</a>'
        titles = re.findall(title_pattern, html, re.DOTALL)
        
        for title in titles[:max_results]:
            clean_title = re.sub(r'<[^>]+>', '', title).strip()
            if clean_title:
                results.append(f"• {clean_title}")
        
        return "\n".join(results) if results else ""
        
    except requests.exceptions.Timeout:
        logger.warning("Web search timed out")
        return ""
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return ""


# ============================================================================
# SERVICE INITIALIZATION HELPERS
# ============================================================================

def wait_for_ollama(max_attempts: int = 30, delay: float = 2.0) -> bool:
    """Wait for Ollama service to become available"""
    logger.info("🔄 Checking Ollama service readiness...")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(
                f"{config.OLLAMA_BASE_URL}/api/version", 
                timeout=5
            )
            if response.status_code == 200:
                version = response.json().get("version", "unknown")
                logger.info(f"✅ Ollama service is ready (version: {version})")
                return True
        except requests.exceptions.RequestException:
            pass
        
        logger.info(f"⏳ Waiting for Ollama... (attempt {attempt + 1}/{max_attempts})")
        time.sleep(delay)
    
    logger.error("❌ Ollama service not ready after timeout")
    return False


def ensure_model(model_name: str) -> bool:
    """Ensure the required LLM model is available, pulling if necessary"""
    try:
        logger.info(f"🔄 Checking for model: '{model_name}'")
        
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
        
        models = response.json().get("models", [])
        available_names = {m.get("name", "") for m in models}
        available_base = {m.get("name", "").split(":")[0] for m in models}
        
        base_name = model_name.split(":")[0]
        
        if model_name in available_names or base_name in available_base:
            logger.info(f"✅ Model '{model_name}' is available")
            return True
        
        logger.warning(f"⚠️ Model '{model_name}' not found, pulling...")
        pull_response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/pull",
            json={"name": model_name},
            timeout=1800  # 30 min timeout for large models
        )
        
        if pull_response.status_code == 200:
            logger.info(f"✅ Model '{model_name}' pulled successfully")
            return True
        
        logger.error(f"❌ Failed to pull model: {pull_response.text}")
        return False
        
    except Exception as e:
        logger.error(f"🔥 Error ensuring model: {e}")
        return False


def test_database_connection(db_pool) -> bool:
    """Test database connectivity and core schema availability."""
    conn = None
    try:
        conn = db_pool.getconn()

        with conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            if not cur.fetchone():
                logger.error("❌ pgvector extension not found!")
                return False

            cur.execute("SELECT to_regclass('public.documents');")
            has_documents = cur.fetchone()[0] is not None

            doc_count = 0
            if has_documents:
                cur.execute("SELECT COUNT(*) FROM documents;")
                doc_count = cur.fetchone()[0]

        logger.info(f"✅ Database connected. Documents: {doc_count}")
        return True

    except Exception as e:
        logger.error(f"🔥 Database connection failed: {e}")
        return False
    finally:
        if conn is not None:
            db_pool.putconn(conn)


# ============================================================================
# FILE PROCESSING
# ============================================================================

def extract_text_from_file(file_content: bytes, filename: str, mime_type: str) -> str:
    """
    Extract text content from various file types.
    
    Supported formats:
    - Text files (.txt)
    - CSV files (.csv)
    - JSON files (.json)
    - PDF files (.pdf) - requires PyMuPDF or pdfplumber
    - Word documents (.docx) - requires python-docx
    - Images - requires pytesseract (OCR)
    """
    filename_lower = filename.lower()
    
    try:
        # Plain text
        if mime_type == "text/plain" or filename_lower.endswith(".txt"):
            return file_content.decode("utf-8", errors="ignore")
        
        # CSV
        if mime_type == "text/csv" or filename_lower.endswith(".csv"):
            return file_content.decode("utf-8", errors="ignore")
        
        # JSON
        if mime_type == "application/json" or filename_lower.endswith(".json"):
            try:
                data = json.loads(file_content.decode("utf-8"))
                return json.dumps(data, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                return file_content.decode("utf-8", errors="ignore")
        
        # PDF
        if mime_type == "application/pdf" or filename_lower.endswith(".pdf"):
            # Try PyMuPDF first (faster)
            try:
                import fitz
                pdf_doc = fitz.open(stream=file_content, filetype="pdf")
                text = "\n".join(page.get_text() for page in pdf_doc)
                pdf_doc.close()
                return text.strip()
            except ImportError:
                pass
            
            # Fallback to pdfplumber
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                    text = "\n".join(
                        page.extract_text() or "" 
                        for page in pdf.pages
                    )
                return text.strip()
            except ImportError:
                return "[Error: No PDF reader available. Install PyMuPDF or pdfplumber]"
        
        # Word documents
        if filename_lower.endswith((".docx", ".doc")):
            try:
                from docx import Document
                doc = Document(io.BytesIO(file_content))
                text = "\n".join(para.text for para in doc.paragraphs)
                return text.strip()
            except ImportError:
                return "[Error: python-docx not installed]"
        
        # Images (OCR)
        if mime_type and mime_type.startswith("image"):
            try:
                image = Image.open(io.BytesIO(file_content))
                text = pytesseract.image_to_string(image)
                return text.strip()
            except Exception as e:
                return f"[Error reading image: {e}]"
        
        return f"[Unsupported file type: {mime_type or filename}]"
        
    except Exception as e:
        logger.error(f"🔥 Error extracting text from {filename}: {e}")
        return f"[Error reading file: {e}]"


# ============================================================================
# LLM INTERACTION
# ============================================================================

def build_system_prompt() -> str:
    """Build the system prompt for the PLC Assistant"""
    return """You are a knowledgeable PLC & Industrial Automation Assistant.

EXPERTISE AREAS:
• PLC Programming: Ladder Logic, Structured Text, Function Block Diagram, Instruction List, Sequential Function Chart
• Industrial Protocols: Modbus (RTU/TCP), PROFINET, EtherNet/IP, OPC UA, PROFIBUS, CANopen, BACnet
• Automation Systems: SCADA, HMI, DCS, MES integration
• Motion Control: Servo drives, VFDs, stepper motors, positioning
• Safety Systems: Safety PLCs, emergency stops, light curtains, IEC 61508/62443
• Troubleshooting: Diagnostic techniques, error analysis, preventive maintenance

RESPONSE GUIDELINES:
1. Always respond in English, regardless of the input language
2. Be precise and technical when discussing automation topics
3. Include relevant specifications, standards, or protocols when applicable
4. Provide step-by-step guidance for troubleshooting questions
5. Mention safety considerations where relevant
6. If you don't know something, say so clearly"""


def ask_llm_directly(
    llm,
    question: str,
    file_content: str = "",
    filename: str = "",
    mode: str = "fast",
    chat_history: List[Dict] = None,
    web_context: str = ""
) -> Dict[str, Any]:
    """
    Send question directly to LLM without RAG.
    Used for Fast mode responses.
    """
    start_time = time.perf_counter()
    
    # Build conversation history
    history_str = ""
    if chat_history:
        for msg in chat_history[-10:]:  # Last 10 messages for context
            role = "User" if msg.get("sender") == "user" else "Assistant"
            text = msg.get("text", "")[:500]  # Truncate long messages
            history_str += f"{role}: {text}\n"
    
    # Build file section
    file_section = ""
    if file_content:
        max_chars = config.DEEP_MODE_CHARS if mode == "deep" else config.FAST_MODE_CHARS
        truncated = len(file_content) > max_chars
        content = file_content[:max_chars] if truncated else file_content
        
        file_section = f"""
=== UPLOADED FILE: {filename} ===
{content}
{"[... content truncated ...]" if truncated else ""}
==="""
    
    # Build web search section
    web_section = ""
    if web_context:
        web_section = f"""
=== WEB SEARCH RESULTS ===
{web_context}
==="""
    
    # Build the prompt
    system_prompt = build_system_prompt()
    
    prompt = f"""{system_prompt}

{"=== CONVERSATION HISTORY ===" + chr(10) + history_str + "===" if history_str else ""}
{file_section}
{web_section}

USER QUESTION: {question}

Provide a helpful, detailed response in English:"""

    try:
        response = llm.invoke(prompt)
        elapsed = time.perf_counter() - start_time
        
        return {
            "reply": response,
            "processing_time": elapsed,
            "mode": mode
        }
    except Exception as e:
        logger.error(f"🔥 LLM error: {e}")
        return {
            "reply": f"I encountered an error processing your request: {str(e)}",
            "processing_time": time.perf_counter() - start_time,
            "mode": mode
        }


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown"""
    logger.info("🚀 Starting application...")

    app_env = get_app_env()
    app.state.app_env = app_env

    try:
        validate_runtime_security_config(app_env)
    except Exception as e:
        logger.error("🔥 Invalid runtime security config: %s", e)
        raise RuntimeError("Backend startup aborted due to invalid security configuration") from e

    # Resolve database config
    try:
        database_url, database_source = resolve_database_url()
        app.state.database_url = database_url
        app.state.database_url_source = database_source
        logger.info("✅ Database URL resolved via %s", database_source)
        logger.info("   DB target: %s", redact_database_url(database_url))
    except Exception as e:
        logger.error("🔥 Invalid database configuration: %s", e)
        raise RuntimeError("Backend startup aborted due to invalid database configuration") from e

    # Initialize database pool and schema
    try:
        # init_db_pool() creates and returns SimpleConnectionPool as defined in backend/app/db.py
        db_pool = init_db_pool(database_url)
        app.state.db_pool = db_pool
        logger.info("✅ Database connection pool created via init_db_pool()")

        ensure_schema(db_pool)
        logger.info("✅ Database migration check complete")

        if not test_database_connection(db_pool):
            raise RuntimeError("Database health check failed during startup")
    except Exception as e:
        logger.error(f"🔥 Failed to initialize DB/migrations: {e}", exc_info=True)
        raise RuntimeError("Backend startup aborted due to database initialization error") from e
    
    # Initialize LLM
    app.state.llm = None
    if wait_for_ollama() and ensure_model(config.OLLAMA_MODEL):
        try:
            app.state.llm = OllamaLLM(
                model=config.OLLAMA_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=config.LLM_TEMPERATURE,
                timeout=config.LLM_TIMEOUT,
                num_predict=config.LLM_NUM_PREDICT,  # Limit output tokens
            )
            set_llm(app.state.llm)  # Share LLM with utils module
            logger.info(f"✅ LLM loaded: {config.OLLAMA_MODEL} (max {config.LLM_NUM_PREDICT} tokens)")
        except Exception as e:
            logger.error(f"🔥 Failed to load LLM: {e}")
    
    # Initialize embedder (use singleton from embed_logic)
    app.state.embedder = None
    try:
        app.state.embedder = get_embedder()
        logger.info(f"✅ Embedder loaded: {config.EMBED_MODEL_NAME}")
    except Exception as e:
        logger.error(f"🔥 Failed to load embedder: {e}")

    # Optional bootstrap embedding so fresh deployments can answer from bundled docs.
    if should_auto_embed_knowledge() and app.state.embedder is not None:
        try:
            auto_embed_result = auto_embed_knowledge_if_empty(
                db_pool=app.state.db_pool,
                embedder=app.state.embedder,
                collection=config.DEFAULT_COLLECTION,
                knowledge_dir=get_auto_embed_knowledge_dir(),
                batch_size=get_auto_embed_batch_size(),
                chunk_size=get_auto_embed_chunk_size(),
                chunk_overlap=get_auto_embed_chunk_overlap(),
                sync_if_not_empty=True,
                skip_known_sources=not should_auto_embed_force_rescan(),
            )
            logger.info("📚 Knowledge auto-embed result: %s", auto_embed_result)
        except Exception as e:
            logger.error(f"🔥 Knowledge auto-embed failed: {e}", exc_info=True)

    # Optional fallback seed from golden_qa.json.
    if should_auto_seed() and app.state.embedder is not None:
        try:
            seed_result = seed_golden_qa_if_empty(
                db_pool=app.state.db_pool,
                embedder=app.state.embedder,
                collection=config.DEFAULT_COLLECTION,
                json_path=get_default_golden_qa_path(
                    (os.getenv("GOLDEN_QA_PATH", "") or "").strip()
                ),
            )
            logger.info(f"🌱 Golden QA seed result: {seed_result}")
        except Exception as e:
            logger.error(f"🔥 Golden QA auto-seed failed: {e}", exc_info=True)
    
    logger.info("🎉 Application startup complete")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("👋 Shutting down...")
    if hasattr(app.state, 'db_pool') and app.state.db_pool:
        app.state.db_pool.closeall()
        logger.info("Database pool closed")


# Create FastAPI app
app = FastAPI(
    lifespan=lifespan,
    title="PLC Assistant API",
    description="""
    Universal PLC & Industrial Automation Assistant with RAG capabilities.
    
    ## Modes
    - **Fast**: Direct LLM response for general questions (~5-15s)
    - **Deep**: RAG-powered response using documentation (~30-60s)
    
    ## Features
    - Multi-file support (PDF, DOCX, images, etc.)
    - Web search integration
    - Chat history context
    - Voice transcription
    """,
    version="3.0.0"
)

app.include_router(auth_router)
app.include_router(chat_router)

# CORS middleware - supports Railway and local development
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"
).split(",")
# Normalize and support Railway wildcard subdomains via regex.
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]
RAILWAY_ORIGIN_REGEX = r"^https://.*\.railway\.app$"
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=RAILWAY_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"]
)


# ============================================================================
# REQUEST TRACING MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests for tracing/debugging"""
    request_id = str(uuid4())
    request.state.request_id = request_id
    
    # Log request start
    logger.info(f"[{request_id[:8]}] {request.method} {request.url.path}")
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """Handle custom application exceptions"""
    request_id = getattr(request.state, 'request_id', None)
    logger.warning(f"[{request_id}] AppException: {exc.code} - {exc.message}")
    return create_error_response(
        code=exc.code,
        message=exc.message,
        status_code=exc.status_code,
        request_id=request_id,
        details=exc.details
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Convert HTTPException to unified error format"""
    request_id = getattr(request.state, 'request_id', None)
    return create_error_response(
        code=f"HTTP_{exc.status_code}",
        message=str(exc.detail),
        status_code=exc.status_code,
        request_id=request_id
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions"""
    request_id = getattr(request.state, 'request_id', None)
    logger.error(f"[{request_id}] Unhandled exception: {exc}", exc_info=True)
    return create_error_response(
        code=ErrorCode.INTERNAL_ERROR,
        message="An unexpected error occurred. Please try again.",
        status_code=500,
        request_id=request_id
    )


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChatResponse(BaseModel):
    reply: str
    processing_time: Optional[float] = None
    retrieval_time: Optional[float] = None
    context_count: Optional[int] = None
    ragas: Optional[dict] = None
    sources: Optional[List[dict]] = None


class HealthResponse(BaseModel):
    status: str
    services: dict
    timestamp: str
    version: str = "3.0.0"


# ============================================================================
# API ENDPOINTS
# ============================================================================

# NOTE: init_db_pool() is already called in lifespan() — do NOT duplicate here

@app.get("/", tags=["Info"])
def root():
    """API information and documentation"""
    return {
        "name": "PLC Assistant API",
        "version": "3.0.0",
        "description": "Universal PLC & Industrial Automation Assistant",
        "modes": {
            "fast": {
                "description": "Direct LLM response for general questions",
                "response_time": "~5-15 seconds",
                "use_for": ["General PLC concepts", "Quick troubleshooting", "Syntax help"]
            },
            "deep": {
                "description": "RAG-powered response using embedded documentation",
                "response_time": "~30-60 seconds",
                "use_for": ["Specific documentation lookups", "Detailed specifications", "Accuracy-critical queries"]
            }
        },
        "endpoints": {
            "health": "GET /health",
            "chat": "POST /api/chat",
            "agent_chat": "POST /api/agent-chat",
            "stream": "POST /api/chat/stream",
            "transcribe": "POST /api/transcribe",
            "collections": "GET /api/collections",
            "stats": "GET /api/stats"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check(request: Request):
    """Check service health status"""
    services = {
        "database": False,
        "llm": False,
        "embedder": False
    }
    
    # Check database
    try:
        if request.app.state.db_pool:
            conn = request.app.state.db_pool.getconn()
            request.app.state.db_pool.putconn(conn)
            services["database"] = True
    except Exception:
        pass
    
    # Check LLM and embedder
    services["llm"] = request.app.state.llm is not None
    services["embedder"] = request.app.state.embedder is not None
    
    status = "healthy" if all(services.values()) else "degraded"
    
    return HealthResponse(
        status=status,
        services=services,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )


@app.post("/api/agent-chat", tags=["Chat"])
def agent_chat(
    message: str = Form(""),
    file: UploadFile = File(None),
    mode: str = Form("fast"),
    chat_history: str = Form("[]"),
    log_eval: bool = Form(False),
    enable_ragas: bool = Form(False),
    fast_ragas: Optional[bool] = Form(None),
    ground_truth: str = Form(""),
    use_rerank: Any = Form(None),
    use_rank: Any = Form(None),
):
    """
    Advanced chat endpoint with mode selection and file support.
    
    Parameters:
    - message: The user's question
    - file: Optional file upload (PDF, images, etc.)
    - mode: Response mode - "fast" or "deep"
    - chat_history: JSON array of previous messages for context
    """
    start_time = time.perf_counter()
    
    # Parse chat history
    try:
        history = json.loads(chat_history) if chat_history else []
    except json.JSONDecodeError:
        history = []
    
    # Validate mode - only "fast" and "deep" allowed
    if mode not in ["fast", "deep"]:
        mode = "fast"
    
    logger.info(f"🎯 Request received - Mode: {mode}, Message: {message[:50]}...")
    
    # Process uploaded file
    file_text = ""
    if file:
        file_content = file.file.read()
        mime_type, _ = mimetypes.guess_type(file.filename)
        
        # Redirect audio files to transcription endpoint
        if mime_type and mime_type.startswith("audio"):
            return JSONResponse(
                status_code=400,
                content={"error": "Please use /api/transcribe for audio files"}
            )
        
        file_text = extract_text_from_file(file_content, file.filename, mime_type)
        logger.info(f"📄 Extracted {len(file_text)} chars from {file.filename}")
    
    # ========================================
    # FAST MODE - Direct LLM
    # ========================================
    if mode == "fast":
        # Determine if web search would be helpful
        web_context = ""
        search_triggers = [
            "latest", "current", "today", "news", "price",
            "2024", "2025", "update", "release", "announce"
        ]
        
        if any(trigger in message.lower() for trigger in search_triggers):
            logger.info(f"🌐 Performing web search for: {message[:50]}...")
            web_context = web_search(message)
        
        result = ask_llm_directly(
            llm=app.state.llm,
            question=message,
            file_content=file_text,
            filename=file.filename if file else "",
            mode=mode,
            chat_history=history,
            web_context=web_context
        )
        
        response = {
            "reply": result.get("reply", ""),
            "processing_time": time.perf_counter() - start_time,
            "retrieval_time": 0,
            "context_count": 0,
            "contexts": [],
            "sources": [],
            "mode": mode,
            "web_searched": bool(web_context),
            "file_processed": file.filename if file else None
        }
        
        return JSONResponse(content=sanitize_json(response))
    
    # ========================================
    # DEEP MODE - RAG Pipeline
    # ========================================
    
    # Determine reranking strategy
    parsed_rerank = to_bool(use_rerank) or to_bool(use_rank)
    if parsed_rerank is None:
        parsed_rerank = os.getenv("USE_RERANK_DEFAULT", "true").lower() in ("1", "true", "yes")
    
    reranker_cls = EnhancedFlashrankRerankRetriever if parsed_rerank else NoRerankRetriever
    
    # Build context from history
    history_context = ""
    if history:
        for msg in history[-6:]:
            role = "User" if msg.get("sender") == "user" else "Assistant"
            history_context += f"{role}: {msg.get('text', '')[:200]}\n"
    
    # Prepare query
    retrieval_query = message
    if file_text:
        max_chars = config.DEEP_MODE_CHARS
        truncated = file_text[:max_chars] if len(file_text) > max_chars else file_text
        retrieval_query = f"{message}\n\n--- File Content ({file.filename}) ---\n{truncated}"
    
    # Execute RAG pipeline
    result = answer_question(
        question=retrieval_query,
        db_pool=app.state.db_pool,
        llm=app.state.llm,
        embedder=app.state.embedder,
        collection=config.DEFAULT_COLLECTION,
        retriever_class=PostgresVectorRetriever,
        reranker_class=reranker_cls,
    )
    
    contexts = result.get("contexts_list") or result.get("contexts") or []
    reply_text = result.get("llm_answer", "") or result.get("reply", "")
    
    # Fallback if no relevant context found
    if "could not find relevant" in reply_text.lower() or not reply_text.strip():
        logger.info("⚠️ No relevant context in Deep mode, falling back to direct LLM")
        result = ask_llm_directly(
            llm=app.state.llm,
            question=message,
            file_content=file_text,
            filename=file.filename if file else "",
            mode=mode,
            chat_history=history
        )
        reply_text = result.get("reply", "")
    
    total_time = time.perf_counter() - start_time
    logger.info(f"📊 Deep mode completed in {total_time:.2f}s")
    
    response = {
        "reply": reply_text,
        "processing_time": total_time,
        "retrieval_time": result.get("retrieval_time"),
        "context_count": result.get("context_count"),
        "contexts": contexts,
        "sources": result.get("sources", []),
        "mode": mode,
        "use_rerank": parsed_rerank,
        "file_processed": file.filename if file else None
    }
    
    return JSONResponse(content=sanitize_json(response))


@app.post("/api/transcribe", tags=["Audio"])
def transcribe(file: UploadFile = File(...)):
    """Transcribe audio file to text using Whisper"""
    import tempfile
    
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise HTTPException(
            status_code=503, 
            detail="Whisper not available. Install faster-whisper."
        )
    
    # Use cached model for faster subsequent requests
    if not hasattr(app.state, 'whisper_model') or app.state.whisper_model is None:
        logger.info("Loading Whisper model (small.en)...")
        app.state.whisper_model = WhisperModel("small.en", device="cpu", compute_type="float32")
    
    # Save to temp file
    suffix = "." + file.filename.split('.')[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    
    try:
        segments, _ = app.state.whisper_model.transcribe(tmp_path, language="en", beam_size=1)
        transcript = "".join(s.text for s in segments)
        return {"text": transcript.strip()}
    finally:
        # Cleanup temp file
        import os
        os.unlink(tmp_path)


@app.post("/api/chat-image", response_model=ChatResponse, tags=["Chat"])
def chat_image(
    request: Request,
    file: UploadFile = File(...),
    message: str = Form("")
):
    """Chat with an image using OCR"""
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    ocr_text = pytesseract.image_to_string(image)
    
    combined_question = f"{message}\n\n[Image OCR Text]:\n{ocr_text}".strip()
    
    result = answer_question(
        question=combined_question,
        db_pool=request.app.state.db_pool,
        llm=request.app.state.llm,
        embedder=request.app.state.embedder,
        collection=config.DEFAULT_COLLECTION,
        retriever_class=PostgresVectorRetriever,
        reranker_class=EnhancedFlashrankRerankRetriever,
    )
    
    return ChatResponse(**sanitize_json(result))

@app.get("/api/chat/history", tags=["Chat"])
def chat_history(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user=Depends(get_current_user),
):
    db_pool = request.app.state.db_pool
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    history = get_user_chat_history(
        db_pool=db_pool,
        user_id=current_user["id"],
        limit=limit,
        offset=offset,
    )

    return {
        "user_id": current_user["id"],
        "count": len(history),
        "items": history,
    }

@app.get("/api/collections", tags=["Data"])
def get_collections(request: Request):
    """List available document collections"""
    db_pool = request.app.state.db_pool
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT collection 
                FROM documents 
                ORDER BY collection;
            """)
            collections = [row[0] for row in cur.fetchall()]
        return {"collections": collections}
    finally:
        db_pool.putconn(conn)


@app.get("/api/stats", tags=["Data"])
def get_stats(request: Request):
    """Get document statistics"""
    db_pool = request.app.state.db_pool
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    collection,
                    COUNT(*) as document_count,
                    ROUND(AVG(LENGTH(content))::numeric, 2) as avg_content_length,
                    MIN(LENGTH(content)) as min_content_length,
                    MAX(LENGTH(content)) as max_content_length
                FROM documents 
                GROUP BY collection
                ORDER BY collection;
            """)
            stats = []
            for row in cur.fetchall():
                stats.append({
                    "collection": row[0],
                    "document_count": row[1],
                    "avg_content_length": float(row[2]) if row[2] else 0,
                    "min_content_length": row[3],
                    "max_content_length": row[4]
                })
        return {"statistics": stats}
    finally:
        db_pool.putconn(conn)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "5000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1
    )
