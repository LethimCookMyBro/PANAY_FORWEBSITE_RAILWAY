"""
Batch embedding CLI script for documents.
Uses embed_logic.py for chunking/filtering logic.

Usage (run from host via docker compose exec):
    # From Git Bash/MINGW on Windows, use // to prevent path mangling:
    docker compose exec backend python embed.py /data/PLCTEST

    # From PowerShell/CMD on Windows or Linux/Mac:
    docker compose exec backend python embed.py /data/PLCTEST

    # Dry-run (preview without embedding)
    docker compose exec backend python embed.py /data/PLCTEST --dry-run

    # Custom chunk settings
    docker compose exec backend python embed.py /data/PLCTEST --chunk-size 500 --chunk-overlap 100

    # Custom collection & smaller batches (for low VRAM)
    docker compose exec backend python embed.py /data/PLCTEST --collection my_docs --batch-size 500
"""
import os
import glob
import argparse
import logging
import hashlib
import json
import gc
from typing import List

# Force Docling to use CPU if EMBED_USE_CPU is set (prevents OOM on 8GB GPUs)
if os.getenv("EMBED_USE_CPU", "false").lower() in ("true", "1", "yes"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["DOCLING_DEVICE"] = "cpu"
    print("🔧 Forcing CPU mode for Docling (EMBED_USE_CPU=true)")

from dotenv import load_dotenv
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
import psycopg2
from psycopg2.extras import execute_values
import torch
from tqdm import tqdm

# Import all chunking/filtering logic from embed_logic
from app.embed_logic import (
    create_pdf_chunks,
    create_json_qa_chunks, 
    get_embedding_instruction,
)

# ==== Load config ====
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_URL = os.getenv("DATABASE_URL")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")


def get_device():
    """Get the best available device (GPU if available)"""
    if torch.cuda.is_available():
        device = "cuda:0"
        logging.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logging.info("⚠️ GPU not available, using CPU")
    return device


def get_files(paths: List[str]) -> List[str]:
    """Get all PDF and JSON files from paths"""
    all_files = []
    for path in paths:
        if os.path.isfile(path):
            if path.lower().endswith(('.pdf', '.json')):
                all_files.append(path)
        elif os.path.isdir(path):
            all_files.extend(glob.glob(os.path.join(path, "**/*.pdf"), recursive=True))
            all_files.extend(glob.glob(os.path.join(path, "**/*.json"), recursive=True))
    return all_files


def flush_chunks(chunks_to_embed: List[Document], embedder, conn, collection: str) -> int:
    """Embed and save a batch of chunks using batch INSERT (5-10x faster)"""
    if not chunks_to_embed:
        return 0
    
    cur = conn.cursor()
    
    # Apply correct instruction per chunk type
    texts = []
    for chunk in chunks_to_embed:
        chunk_type = chunk.metadata.get("chunk_type", "prose")
        instruction = get_embedding_instruction(chunk_type)
        texts.append(instruction + chunk.page_content)
    
    logging.info(f"   🔄 Embedding {len(chunks_to_embed)} chunks...")
    # Use normalized embeddings for better cosine similarity (BGE-M3 recommendation)
    # batch_size=32 works well when Docling uses CPU (leaves GPU memory for embedding)
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
    
    # Prepare batch data
    batch_data = []
    for chunk, embedding in zip(chunks_to_embed, embeddings):
        text = chunk.page_content
        vector = embedding.tolist()
        hash_ = hashlib.sha256(text.encode()).hexdigest()
        metadata_json = json.dumps(chunk.metadata)
        batch_data.append((text, vector, collection, hash_, metadata_json))
    
    # Batch INSERT (5-10x faster than individual inserts)
    try:
        execute_values(
            cur,
            """INSERT INTO documents (content, embedding, collection, hash, metadata)
               VALUES %s ON CONFLICT (hash) DO NOTHING""",
            batch_data,
            template="(%s, %s, %s, %s, %s)"
        )
        inserted = cur.rowcount
    except Exception as e:
        logging.error(f"❌ Batch insert error: {e}")
        inserted = 0
    
    conn.commit()
    cur.close()
    logging.info(f"   ✅ Committed {inserted} new chunks")
    
    # Free up GPU memory to prevent fragmentation/OOM
    del embeddings
    del texts
    gc.collect()
    torch.cuda.empty_cache()
    
    return inserted


def get_already_processed(conn, collection: str) -> set:
    """
    Get set of already processed filenames for resume capability.
    Uses basename only for consistent matching.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT metadata->>'source' FROM documents WHERE collection=%s",
        (collection,)
    )
    # metadata['source'] now always stores basename, so this matches correctly
    already_processed = {row[0] for row in cur.fetchall() if row[0]}
    cur.close()
    return already_processed


def main():
    parser = argparse.ArgumentParser(description="Embed documents into a Postgres vector database (Docling).")
    parser.add_argument("files", nargs="+", help="Path(s) to PDF/JSON file(s) or folder(s) to embed.")
    parser.add_argument("--collection", default="plcnext", help="Collection name.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of chunks per embedding batch.")
    parser.add_argument("--chunk-size", type=int, default=800, help="Max characters per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Overlap between chunks.")
    parser.add_argument("--model-cache", default="/app/models", help="Model cache directory.")
    parser.add_argument("--dry-run", action="store_true", help="Parse files but don't embed or save.")
    args = parser.parse_args()
    
    # Get all files
    all_files = get_files(args.files)
    
    if not all_files:
        logging.error("No PDF/JSON files found!")
        exit(1)
    
    logging.info(f"📚 Found {len(all_files)} files to process")
    
    if args.dry_run:
        logging.info("🔍 DRY RUN MODE - will parse but not embed or save")
    
    # Get device
    device = get_device()
    
    # Initialize
    BATCH_SIZE = args.batch_size
    pending_chunks = []
    total_embedded = 0
    total_chunks_created = 0
    total_files_processed = 0
    
    # Load embedding model once (skip in dry-run)
    embedder = None
    if not args.dry_run:
        logging.info(f"📥 Loading embedding model: {EMBED_MODEL}")
        embedder = SentenceTransformer(EMBED_MODEL, device=device, cache_folder=args.model_cache)
        logging.info(f"✅ Model loaded on {device}")
    
    # Connect to database once (skip in dry-run)
    conn = None
    already_processed = set()
    if not args.dry_run:
        conn = psycopg2.connect(DB_URL)
        already_processed = get_already_processed(conn, args.collection)
        if already_processed:
            logging.info(f"⏭️ Found {len(already_processed)} already processed files in collection '{args.collection}'")
    
    try:
        for file_idx, file_path in enumerate(tqdm(all_files, desc="📁 Processing files", unit="file")):
            if not os.path.exists(file_path):
                logging.warning(f"File not found: {file_path}")
                continue
            
            # Skip already processed files (compare basenames consistently)
            filename = os.path.basename(file_path)
            if filename in already_processed:
                logging.info(f"⏭️ [{file_idx + 1}/{len(all_files)}] Skipping {filename} (already embedded)")
                continue
            
            chunks = []
            
            # Process based on file type
            if file_path.lower().endswith('.json'):
                chunks = create_json_qa_chunks(file_path)
            elif file_path.lower().endswith('.pdf'):
                try:
                    loader = DoclingLoader(file_path=file_path, export_type=ExportType.MARKDOWN)
                    pages = loader.load()
                    chunks = create_pdf_chunks(
                        pages,
                        chunk_size=args.chunk_size,
                        chunk_overlap=args.chunk_overlap
                    )
                except Exception as e:
                    logging.error(f"❌ Failed to process PDF {file_path}: {e}")
                    continue
            
            total_chunks_created += len(chunks)
            total_files_processed += 1
            
            # Log progress
            logging.info(f"📄 [{file_idx + 1}/{len(all_files)}] {filename}: {len(chunks)} chunks")
            
            if args.dry_run:
                continue  # Skip embedding in dry-run
            
            pending_chunks.extend(chunks)
            
            # Flush when we have enough pending chunks
            while len(pending_chunks) >= BATCH_SIZE:
                batch = pending_chunks[:BATCH_SIZE]
                pending_chunks = pending_chunks[BATCH_SIZE:]
                total_embedded += flush_chunks(batch, embedder, conn, args.collection)
        
        # Flush remaining chunks
        if pending_chunks and not args.dry_run:
            logging.info(f"📦 Flushing final {len(pending_chunks)} chunks...")
            total_embedded += flush_chunks(pending_chunks, embedder, conn, args.collection)
        
        # Summary
        if args.dry_run:
            logging.info(f"🔍 DRY RUN COMPLETE: Would create {total_chunks_created} chunks from {total_files_processed} files")
        else:
            logging.info(f"🎉 Done! Embedded {total_embedded} chunks from {total_files_processed} files")
    
    except Exception as e:
        logging.error(f"❌ Error: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
