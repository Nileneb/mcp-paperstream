-- Migration 002: Update Rules Schema
-- ====================================
-- Adds semantic phrase support to rules table

-- Add new columns for semantic matching
ALTER TABLE rules ADD COLUMN positive_phrases TEXT DEFAULT '[]';

ALTER TABLE rules ADD COLUMN negative_phrases TEXT DEFAULT '[]';

ALTER TABLE rules ADD COLUMN threshold REAL DEFAULT 0.65;

-- Update existing rules to have default threshold
UPDATE rules SET threshold = 0.65 WHERE threshold IS NULL;

-- Note: After migration, run load_default_rules() to populate with enhanced rules