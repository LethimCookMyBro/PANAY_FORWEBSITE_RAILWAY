import os
import argparse
import logging
from dotenv import load_dotenv
from langchain_docling import DoclingLoader

# ==== Import logic ====
from app.embed_logic import (
    create_pdf_chunks, create_json_qa_chunks, embed_chunks
)

# ==== Load config ====
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_URL = os.getenv("DATABASE_URL")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed documents into a Postgres vector database.")
    parser.add_argument("files", nargs="+", help="Path(s) to the PDF or JSON file(s) to embed.")
    parser.add_argument("--collection", default="plcnext", help="Collection name.")
    args = parser.parse_args()
    
    all_chunks = []
    for file_path in args.files:
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            continue
        
        if file_path.lower().endswith('.json'):
            all_chunks.extend(create_json_qa_chunks(file_path))
        elif file_path.lower().endswith('.pdf'):
            try:
                loader = DoclingLoader(file_path=file_path)
                pages = loader.load()
                all_chunks.extend(create_pdf_chunks(pages))
            except Exception as e:
                logging.error(f"Failed to load/chunk PDF {file_path}: {e}")
    
    if all_chunks:
        embed_chunks(all_chunks, args.collection, EMBED_MODEL, DB_URL)
    else:
        logging.info("No chunks generated.")
