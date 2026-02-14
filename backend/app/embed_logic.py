import os
import re
import json
import logging
from typing import List, Any, Dict, Optional, Set
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

# Singleton embedder instance
_embedder = None


def _resolve_embed_device() -> str:
    requested = (os.getenv("EMBED_DEVICE", "auto") or "auto").strip().lower()

    if requested == "cpu":
        return "cpu"

    if requested.startswith("cuda"):
        if torch is None or not torch.cuda.is_available():
            logging.warning("EMBED_DEVICE=%s requested, but CUDA is unavailable. Falling back to CPU.", requested)
            return "cpu"
        return "cuda:0" if requested == "cuda" else requested

    # auto
    if torch is not None and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"

def get_embedder():
    """
    Returns a singleton SentenceTransformer embedder.
    Uses BAAI/bge-m3 model (matching embed.py and .env configuration).
    """
    global _embedder
    if _embedder is None:
        model_name = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
        cache_folder = os.getenv("MODEL_CACHE", "/data/models")
        device = _resolve_embed_device()
        logging.info(f"[get_embedder] Loading embedder: {model_name} (device={device})")
        _embedder = SentenceTransformer(model_name, cache_folder=cache_folder, device=device)
        logging.info("[get_embedder] Embedder loaded successfully")
    return _embedder


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text before chunking.
    Preserves paragraph structure (single newlines) for better semantic chunking.
    """
    # Normalize horizontal whitespace (spaces/tabs -> single space)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Collapse excessive newlines (3+ -> 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove page headers/footers (common patterns)
    text = re.sub(r'(Page\s*\d+|\d+\s*of\s*\d+)', '', text, flags=re.IGNORECASE)
    
    # Remove Docling/Phoenix-specific patterns
    text = re.sub(r'--- PAGE \d+ ---', '', text)
    text = re.sub(r'\d{6}_en_\d{2,}', '', text)
    text = re.sub(r'PHOENIX CONTACT \d+/\d+', '', text)
    
    # Remove garbage characters (keep ASCII printable + Thai + newlines)
    text = re.sub(r'[^\x20-\x7E\n\u0E00-\u0E7F]', '', text)
    
    # Remove TOC fillers (5+ consecutive dots/dashes/underscores)
    text = re.sub(r'[.\-_]{5,}', ' ', text)
    
    # Clean up spaces around newlines
    text = re.sub(r' *\n *', '\n', text)
    
    # Final horizontal whitespace cleanup
    text = re.sub(r'[ \t]+', ' ', text)
    

    return text.strip()


def is_valid_chunk(chunk: Document) -> tuple[bool, str]:
    """
    Check if chunk is worth keeping.
    Returns (is_valid: bool, reason: str)
    
    Filters:
    1. Too short (<50 chars with low alpha)
    2. Table of Contents entries (page references)
    3. Low alphabetic ratio (<15% after removing fillers)
    4. Excessive whitespace (>50%)
    5. Repetitive characters (10+ same char)
    6. High special characters (>40% punctuation)
    7. Header/page number only
    """
    content = chunk.page_content.strip()
    
    # 1. Too short (<50 chars) - but allow high-alpha short content
    if len(content) < 50:
        alpha_ratio = sum(1 for c in content if c.isalpha()) / len(content) if content else 0
        if alpha_ratio > 0.70:  # Short but meaningful
            return True, "ok"
        return False, f"too_short ({len(content)} chars)"
    
    # 2. TABLE OF CONTENTS detection - REJECT these!
    # Pattern: "3.4.1 Something........... 45" or "Chapter 3....... 15"
    if re.search(r'\d+\.\d+.*\.{3,}\s*\d+', content):
        return False, "toc_entry"
    
    # TOC in markdown table format: "| Something ... | 15 |" or "| 3.1.6 [G] Expansion boards...48 |"
    if re.search(r'\|.*\.{3,}.*\d+\s*\|', content):
        return False, "toc_table_entry"
    
    # Simple page reference patterns in tables: "| Standards | 15 |" with mostly numbers + short text
    if '|' in content:
        # Extract cell contents
        cells = [c.strip() for c in content.split('|') if c.strip()]
        # If most cells are just numbers or very short text, it's likely TOC
        page_number_cells = sum(1 for c in cells if re.match(r'^\d{1,4}$', c))
        if len(cells) > 0 and page_number_cells >= len(cells) // 2:
            # Check if content is suspiciously short on actual text
            total_text = ' '.join(cells)
            if len(total_text) < 100:
                return False, "toc_page_reference"
    
    # 3. Low alphabetic ratio (<15% letters, excluding fillers)
    content_no_fillers = re.sub(r'[.\-_=]{2,}', '', content)
    if len(content_no_fillers) > 10:
        alpha_count = sum(1 for c in content_no_fillers if c.isalpha())
        alpha_ratio = alpha_count / len(content_no_fillers)
        if alpha_ratio < 0.15:
            return False, f"low_alpha ({alpha_ratio:.1%})"
    
    # 4. Excessive whitespace (>50%)
    whitespace_ratio = sum(1 for c in content if c.isspace()) / len(content) if content else 0
    if whitespace_ratio > 0.50:
        return False, f"excessive_whitespace ({whitespace_ratio:.1%})"
    
    # 5. Repetitive garbage chars (exclude dots/dashes/underscores/spaces)
    if re.search(r'([^.\-_\s])\1{9,}', content):
        return False, "repetitive_chars"
    
    # 6. High special characters (>40%) - excluding common formatting
    non_formatting_special = sum(1 for c in content if not c.isalnum() and not c.isspace() and c not in '.-_=|')
    special_ratio = non_formatting_special / len(content) if content else 0
    if special_ratio > 0.40:
        return False, f"high_special ({special_ratio:.1%})"
    
    # 7. Page headers only
    if re.match(r'^(page\s*\d+|\d+\s*of\s*\d+|chapter\s*\d+)$', content.lower()):
        return False, "header_only"

    return True, "ok"


_PAGE_KEYS = {
    "page",
    "page_no",
    "page_number",
    "page_num",
    "page_id",
    "pageindex",
    "page_index",
    "pageidx",
    "page_idx",
    "start_page",
    "end_page",
}
_ZERO_BASED_PAGE_KEYS = {"pageindex", "page_index", "pageidx", "page_idx"}


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+", value.strip())
        if match:
            try:
                return int(match.group(0))
            except Exception:
                return None
    return None


def _as_mapping(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return value
    for attr in ("model_dump", "dict"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                mapped = fn()
                if isinstance(mapped, dict):
                    return mapped
            except Exception:
                pass
    raw = getattr(value, "__dict__", None)
    if isinstance(raw, dict):
        return raw
    return None


def _normalize_page_candidate(raw_page: Optional[int], key_name: str) -> Optional[int]:
    if raw_page is None:
        return None
    if key_name in _ZERO_BASED_PAGE_KEYS and raw_page >= 0:
        raw_page += 1
    if raw_page <= 0:
        return None
    return raw_page


def _walk_for_page_candidate(value: Any, seen: Set[int], depth: int = 0) -> Optional[int]:
    if value is None or depth > 10:
        return None

    obj_id = id(value)
    if obj_id in seen:
        return None
    seen.add(obj_id)

    mapping = _as_mapping(value)
    if mapping is not None:
        ordered_items = sorted(
            mapping.items(),
            key=lambda kv: 0 if str(kv[0]).lower() in _PAGE_KEYS else 1,
        )
        for key, item in ordered_items:
            key_lower = str(key).lower()
            if key_lower in _PAGE_KEYS:
                candidate = _normalize_page_candidate(_coerce_int(item), key_lower)
                if candidate is not None:
                    return candidate
            nested = _walk_for_page_candidate(item, seen, depth + 1)
            if nested is not None:
                return nested
        return None

    if isinstance(value, (list, tuple)):
        for item in value:
            nested = _walk_for_page_candidate(item, seen, depth + 1)
            if nested is not None:
                return nested
        return None

    return None


def _extract_page_from_text(text: str) -> Optional[int]:
    if not text:
        return None

    snippet = text[:1000]
    patterns = [
        r"---\s*PAGE\s*(\d{1,5})\s*---",
        r"\bpage\s*[:#-]?\s*(\d{1,5})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, snippet, flags=re.IGNORECASE)
        if match:
            page = _coerce_int(match.group(1))
            if page is not None and page > 0:
                return page
    return None


def extract_page_number(doc_metadata: dict, chunk_text: str = "") -> int:
    """
    Extract page number from Docling metadata with robust fallback strategy.
    Returns 0 only when no page number can be inferred.
    """
    metadata = doc_metadata or {}

    direct_candidate = _normalize_page_candidate(
        _coerce_int(metadata.get("page")),
        "page",
    )
    if direct_candidate is not None:
        return direct_candidate

    recursive_candidate = _walk_for_page_candidate(metadata, seen=set())
    if recursive_candidate is not None:
        return recursive_candidate

    text_candidate = _extract_page_from_text(chunk_text)
    if text_candidate is not None:
        return text_candidate

    return 0


def enhance_metadata(metadata: dict, chunk_content: str) -> dict:
    """Add useful metadata to chunk"""
    enhanced_meta = metadata.copy()
    enhanced_meta.update({
        "char_count": len(chunk_content),
        "word_count": len(chunk_content.split()),
    })

    source = enhanced_meta.get("source")
    if isinstance(source, str) and source:
        enhanced_meta["source"] = os.path.basename(source)

    page = _coerce_int(enhanced_meta.get("page"))
    enhanced_meta["page"] = page if page is not None and page > 0 else 0

    return enhanced_meta


def get_file_label(source: str) -> str:
    """
    Extract clean filename label from source path.
    Examples:
        'MELSEC-F_manual.pdf' -> 'MELSEC-F_manual'
        '/path/to/PLC_guide.pdf' -> 'PLC_guide'
    """
    filename = os.path.basename(source)
    label = os.path.splitext(filename)[0]
    return label


def split_table_by_rows(table_text: str, max_chars: int = 800) -> List[str]:
    """
    Split a markdown table by rows while preserving headers.
    Each chunk will have the header rows prepended.
    
    Args:
        table_text: Markdown table string
        max_chars: Maximum characters per chunk
    
    Returns:
        List of table chunks, each with headers preserved
    """
    lines = table_text.strip().split('\n')
    if len(lines) < 2:
        return [table_text]  # Not a valid table
    
    # Find header and separator (first 2 lines typically)
    # Header: | Col1 | Col2 |
    # Separator: |------|------|
    header_lines = []
    data_lines = []
    
    for i, line in enumerate(lines):
        if i < 2 and ('|' in line):
            header_lines.append(line)
        elif '|' in line:
            data_lines.append(line)
    
    if not header_lines:
        return [table_text]  # No headers found, return as-is
    
    header_text = '\n'.join(header_lines)
    header_len = len(header_text) + 1  # +1 for newline
    
    # Split data rows into chunks
    chunks = []
    current_chunk_lines = []
    current_len = header_len
    
    for row in data_lines:
        row_len = len(row) + 1  # +1 for newline
        
        # If adding this row exceeds limit, flush current chunk
        if current_len + row_len > max_chars and current_chunk_lines:
            chunk_text = header_text + '\n' + '\n'.join(current_chunk_lines)
            chunks.append(chunk_text)
            current_chunk_lines = []
            current_len = header_len
        
        current_chunk_lines.append(row)
        current_len += row_len
    
    # Flush remaining rows
    if current_chunk_lines:
        chunk_text = header_text + '\n' + '\n'.join(current_chunk_lines)
        chunks.append(chunk_text)
    
    return chunks if chunks else [table_text]


def extract_tables_and_prose(text: str) -> tuple[List[str], str]:
    """
    Separate markdown tables from prose text.
    
    Returns:
        (list of table strings, remaining prose text)
    """
    # Pattern to match markdown tables (consecutive lines starting with |)
    table_pattern = re.compile(r'((?:^\|.*\|$\n?)+)', re.MULTILINE)
    
    tables = []
    prose = text
    
    for match in table_pattern.finditer(text):
        table_text = match.group(1).strip()
        # Only consider it a table if it has at least 2 rows (header + data)
        if table_text.count('\n') >= 1 and '|' in table_text:
            tables.append(table_text)
            prose = prose.replace(match.group(1), '\n')  # Remove table from prose
    
    return tables, prose


def create_pdf_chunks(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Create chunks from Docling-extracted documents with filtering.
    Extracts key-value pairs separately and labels chunks with document title.
    
    Args:
        docs: List of Document objects from Docling
        chunk_size: Maximum characters per chunk (default: 800)
        chunk_overlap: Overlap between chunks (default: 150)
    """
    all_chunks = []
    filtered_count = 0
    kv_pattern = re.compile(r'^(?P<key>[A-Za-z0-9\(\)\/\s\.,-]{5,80}?)\s{2,}(?P<value>.+?)$', re.MULTILINE)
    
    # Configurable chunk settings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " "]
    )

    for doc in docs:
        page_content = doc.page_content
        page_metadata = doc.metadata or {}
        source = page_metadata.get('source', 'unknown')
        # Extract real page number from Docling metadata (DOC_CHUNKS export)
        page_number = extract_page_number(page_metadata, page_content)
        file_label = get_file_label(source)
        
        # Key-Value Extraction Logic
        kv_matches = kv_pattern.findall(page_content)
        for key, value in kv_matches:
            key_clean = key.strip()
            value_clean = value.strip()
            
            # Stronger validation: require meaningful content
            # 1. Combined length > 25 chars (filters short garbage)
            # 2. Value must have real content (not just a label word)
            # 3. Reject section headers and step instructions
            combined_len = len(key_clean) + len(value_clean)
            value_has_content = len(value_clean) > 8 or any(c.isdigit() for c in value_clean)
            
            # Filter out garbage patterns
            is_section_header = value_clean.startswith('##') or value_clean.startswith('#')
            is_step_instruction = re.match(r'^-\s*\d+\s+', key_clean)  # "- 4 Fit the..."
            is_garbage = is_section_header or bool(is_step_instruction)
            
            if combined_len > 25 and value_has_content and not is_garbage:
                kv_content = f"{file_label}: {key_clean}: {value_clean}"
                kv_meta = {"source": os.path.basename(source), "page": page_number, "chunk_type": "spec_pair"}
                if extra_metadata:
                    kv_meta.update(extra_metadata)
                kv_chunk = Document(
                    page_content=kv_content,
                    metadata=enhance_metadata(
                        kv_meta,
                        kv_content
                    )
                )
                # Apply filtering
                is_valid, reason = is_valid_chunk(kv_chunk)
                if is_valid:
                    all_chunks.append(kv_chunk)
                else:
                    filtered_count += 1
        
        # Remove extracted key-value pairs from content using regex
        remaining_content = kv_pattern.sub('', page_content)
        
        # === TABLE-AWARE SPLITTING ===
        # Extract tables and process them separately with row-based splitting
        tables, prose_content = extract_tables_and_prose(remaining_content)
        
        # Process tables with header-preserving splits
        for table in tables:
            table_chunks = split_table_by_rows(table, max_chars=chunk_size)
            for table_chunk_text in table_chunks:
                labeled_content = f"{file_label}: {table_chunk_text}"
                table_meta = {"source": os.path.basename(source), "page": page_number, "chunk_type": "table"}
                if extra_metadata:
                    table_meta.update(extra_metadata)
                table_chunk = Document(
                    page_content=labeled_content,
                    metadata=enhance_metadata(
                        table_meta,
                        labeled_content
                    )
                )
                is_valid, reason = is_valid_chunk(table_chunk)
                if is_valid:
                    all_chunks.append(table_chunk)
                else:
                    filtered_count += 1
        
        # Process remaining prose content
        prose_content = clean_text(prose_content)
        
        # Create prose chunks
        if prose_content and len(prose_content.strip()) > 50:
            prose_chunks = text_splitter.create_documents([prose_content])
            for chunk in prose_chunks:
                # Prepend file label to chunk content
                labeled_content = f"{file_label}: {chunk.page_content}"
                chunk.page_content = labeled_content
                prose_meta = {"source": os.path.basename(source), "page": page_number, "chunk_type": "prose"}
                if extra_metadata:
                    prose_meta.update(extra_metadata)
                chunk.metadata = enhance_metadata(prose_meta, labeled_content)
                # Apply filtering
                is_valid, reason = is_valid_chunk(chunk)
                if is_valid:
                    all_chunks.append(chunk)
                else:
                    filtered_count += 1

    logging.info(f"✅ Created {len(all_chunks)} chunks (filtered out {filtered_count} trash chunks)")
    return all_chunks


def create_json_qa_chunks(
    file_path: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """Create chunks from JSON QA file"""
    chunks = []
    file_label = get_file_label(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        for pair in qa_pairs:
            # Prepend file label for consistency with PDF chunks
            content = f"{file_label}: Question: {pair.get('question', '')}\nAnswer: {pair.get('answer', '')}"
            metadata_base = {"source": os.path.basename(file_path), "chunk_type": "golden_qa"}
            if extra_metadata:
                metadata_base.update(extra_metadata)
            metadata = enhance_metadata(metadata_base, content)
            chunks.append(Document(page_content=content, metadata=metadata))
        logging.info(f"✅ Created {len(chunks)} chunks from Golden QA Set.")
    except Exception as e:
        logging.error(f"🔥 Failed to process JSON file {file_path}: {e}")
    return chunks


def get_embedding_instruction(chunk_type: str) -> str:
    """Customize instruction based on chunk type for better embedding quality"""
    instructions = {
        "golden_qa": "Represent this authoritative question-answer pair for search: ",
        "spec_pair": "Represent this technical specification value for search: ",
        "table": "Represent this technical data table for search: ",
        "prose": "Represent this technical documentation paragraph for search: "
    }
    return instructions.get(chunk_type, "Represent this sentence for searching relevant passages: ")
