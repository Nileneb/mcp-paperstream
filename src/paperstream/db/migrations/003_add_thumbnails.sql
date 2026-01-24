-- Migration 003: Add Thumbnail Support
-- =====================================
-- Adds thumbnail columns to papers table

-- Add thumbnail storage columns
ALTER TABLE papers ADD COLUMN thumbnail_path TEXT DEFAULT NULL;

ALTER TABLE papers ADD COLUMN thumbnail_base64 TEXT DEFAULT NULL;

ALTER TABLE papers
ADD COLUMN thumbnail_generated_at TIMESTAMP DEFAULT NULL;

-- Create index for papers with/without thumbnails
CREATE INDEX IF NOT EXISTS idx_papers_thumbnail ON papers (thumbnail_path);