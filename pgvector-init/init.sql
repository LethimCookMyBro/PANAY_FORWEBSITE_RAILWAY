CREATE EXTENSION IF NOT EXISTS vector;

-- 2. สร้างตาราง documents สำหรับเก็บข้อมูล
-- โครงสร้างตารางนี้อ้างอิงจากโค้ดในไฟล์ embed.py และ retriever.py
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    collection VARCHAR(255) NOT NULL,
    hash VARCHAR(64) UNIQUE NOT NULL, -- ใช้สำหรับป้องกันข้อมูลซ้ำซ้อน (sha256)
    -- ขนาดของ Vector (1024) ต้องตรงกับโมเดลที่ใช้ (BAAI/bge-m3)
    embedding VECTOR(1024),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 3. สร้าง Index เพื่อเพิ่มความเร็วในการค้นหาข้อมูล Vector (แนะนำ)
-- การสร้าง Index จะช่วยให้การค้นหาเอกสารที่คล้ายกันทำได้เร็วขึ้นมาก
CREATE INDEX IF NOT EXISTS idx_hnsw_embedding ON documents USING hnsw (embedding vector_l2_ops);

-- 4. สร้าง Index อื่นๆ ที่อาจเป็นประโยชน์
CREATE INDEX IF NOT EXISTS idx_collection ON documents (collection);