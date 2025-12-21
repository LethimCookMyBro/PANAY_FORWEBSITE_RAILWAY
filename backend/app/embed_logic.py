import os
import re
import json
import hashlib
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_docling import DoclingLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import psycopg2

_embedder = None

def clean_text(text: str) -> str:
    text = re.sub(r'--- PAGE \d+ ---', '', text)
    text = re.sub(r'\d{6}_en_\d{2,}', '', text)
    text = re.sub(r'PHOENIX CONTACT \d+/\d+', '', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text.strip()

def enhance_metadata(metadata: dict, chunk_content: str) -> dict:
    """เพิ่ม Metadata ที่มีประโยชน์เข้าไปใน Chunk"""
    enhanced_meta = metadata.copy()
    enhanced_meta.update({
        "char_count": len(chunk_content),
        "word_count": len(chunk_content.split()),
        "language": "en", # สมมติว่าเป็นภาษาอังกฤษ
        "domain": "plcnext_automation"
    })
    return enhanced_meta

def create_pdf_chunks(docs: List[Document]) -> List[Document]:
    all_chunks = []
    kv_pattern = re.compile(r'^(?P<key>[A-Za-z0-9\(\)\/\s\.,-]{5,80}?)\s{2,}(?P<value>.+?)$', re.MULTILINE)
    
    # ปรับปรุง Chunk Strategy ตามข้อเสนอแนะ
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", ", ", " "]
    )

    for doc in docs:
        page_content = doc.page_content
        page_metadata = doc.metadata or {}
        source = page_metadata.get('source', 'unknown')
        page_number = page_metadata.get('page', 0)
        
        # Key-Value Extraction Logic
        kv_matches = kv_pattern.findall(page_content)
        for key, value in kv_matches:
            if len(key.strip()) > 5 and len(value.strip()) > 3:
                kv_content = f"{key.strip()}: {value.strip()}"
                kv_metadata = enhance_metadata(
                    {"source": source, "page": page_number, "chunk_type": "spec_pair"},
                    kv_content
                )
                all_chunks.append(Document(page_content=kv_content, metadata=kv_metadata))
        
        # Remove extracted key-value pairs from prose content
        prose_content = page_content
        for key, value in kv_matches:
            original_line = f"{key}{' ' * (len(key) - key.strip().__len__())}{value}"
            prose_content = prose_content.replace(original_line, "")
        
        prose_content = clean_text(prose_content)
        
        # Create prose chunks
        if prose_content and len(prose_content.strip()) > 50:
            prose_chunks = text_splitter.create_documents([prose_content])
            for chunk in prose_chunks:
                chunk.metadata = enhance_metadata(
                    {"source": source, "page": page_number, "chunk_type": "prose"},
                    chunk.page_content
                )
                all_chunks.append(chunk)

    logging.info(f"✅ Created {len(all_chunks)} chunks from PDF with enhanced metadata.")
    return all_chunks

def create_json_qa_chunks(file_path: str) -> List[Document]:
    chunks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        for pair in qa_pairs:
            content = f"Question: {pair.get('question', '')}\nAnswer: {pair.get('answer', '')}"
            metadata = enhance_metadata(
                {"source": os.path.basename(file_path), "chunk_type": "golden_qa"},
                content
            )
            chunks.append(Document(page_content=content, metadata=metadata))
        logging.info(f"✅ Created {len(chunks)} chunks from Golden QA Set.")
    except Exception as e:
        logging.error(f"🔥 Failed to process JSON file {file_path}: {e}")
    return chunks

def get_embedding_instruction(chunk_type: str) -> str:
    """ปรับแต่ง instruction ตามประเภทของ Chunk"""
    instructions = {
        "golden_qa": "Represent this authoritative question-answer pair for search: ",
        "spec_pair": "Represent this technical specification value for search: ",
        "prose": "Represent this technical documentation paragraph for search: "
    }
    return instructions.get(chunk_type, "Represent this sentence for searching relevant passages: ")

def embed_chunks(chunks: List[Document], collection: str, embed_model: str, db_url: str):
    if not chunks: return
    try:
        embedder = SentenceTransformer(embed_model, cache_folder='/app/models')
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        successful_inserts = 0
        for chunk in chunks:
            try:
                text = chunk.page_content
                chunk_type = chunk.metadata.get("chunk_type", "prose")
                instruction = get_embedding_instruction(chunk_type)
                text_to_embed = instruction + text
                
                vector = embedder.encode(text_to_embed).tolist()
                hash_ = hashlib.sha256(text.encode()).hexdigest()
                metadata_json = json.dumps(chunk.metadata)
                
                cur.execute(
                    "INSERT INTO documents (content, embedding, collection, hash, metadata) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (hash) DO NOTHING;", 
                    (text, vector, collection, hash_, metadata_json)
                )
                successful_inserts += 1
            except Exception as e:
                logging.error(f"🔥 Error embedding chunk: {e}")
                continue
        
        conn.commit()
        cur.close()
        conn.close()
        logging.info(f"✅ Successfully embedded {successful_inserts}/{len(chunks)} chunks into collection '{collection}'")
    except Exception as e:
        logging.error(f"🔥 DB embedding error: {e}", exc_info=True)

def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(model_name, cache_folder="/app/models")
    return _embedder