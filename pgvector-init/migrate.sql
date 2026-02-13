-- Idempotent migration script for existing databases (Railway or local)
-- Safe to run multiple times

-- Add metadata column to chat_messages if missing
ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;

-- Add created_at column to documents if missing
ALTER TABLE documents ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now();

-- Verify
DO $$
BEGIN
    RAISE NOTICE 'Migration complete: chat_messages.metadata and documents.created_at columns ensured.';
END $$;
