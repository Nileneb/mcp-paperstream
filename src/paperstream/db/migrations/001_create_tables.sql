-- ============================================
-- MCP-PaperStream Database Schema
-- Migration: 001_create_tables
-- Author: System
-- ============================================

-- TABLE: papers
-- Speichert alle Papers die evaluiert werden sollen
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT UNIQUE NOT NULL,
    title TEXT,
    authors TEXT,
    journal TEXT,
    publication_date TEXT,
    pdf_url TEXT,
    pdf_local_path TEXT,
    
    -- Status: pending, downloading, processing, ready, failed
    status TEXT DEFAULT 'pending',
    downloaded_at TIMESTAMP,
    processed_at TIMESTAMP,
    
    source TEXT DEFAULT 'n8n',
    priority INTEGER DEFAULT 5,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status);
CREATE INDEX IF NOT EXISTS idx_papers_priority ON papers(priority DESC);
CREATE INDEX IF NOT EXISTS idx_papers_paper_id ON papers(paper_id);


-- TABLE: paper_sections
-- Pro Paper: Abschnitte mit Embeddings und Voxel-Daten
CREATE TABLE IF NOT EXISTS paper_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    section_name TEXT NOT NULL,
    section_text TEXT,
    
    page_number INTEGER,
    image_path TEXT,
    embedding BLOB,
    voxel_data TEXT,  -- JSON: 8x8x12 voxel grid
    
    -- RGB color for visualization
    color_r REAL,
    color_g REAL,
    color_b REAL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sections_paper ON paper_sections(paper_id);
CREATE INDEX IF NOT EXISTS idx_sections_name ON paper_sections(section_name);


-- TABLE: rules
-- Von BioBERT generierte Validierungs-Regeln
CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id TEXT UNIQUE NOT NULL,
    question TEXT NOT NULL,
    
    positive_phrases TEXT,  -- JSON array
    negative_phrases TEXT,  -- JSON array
    pos_embedding BLOB,     -- numpy float32 array
    neg_embedding BLOB,     -- numpy float32 array
    
    threshold REAL DEFAULT 0.75,
    
    is_active BOOLEAN DEFAULT 1,
    created_by TEXT DEFAULT 'system',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_rules_active ON rules(is_active);
CREATE INDEX IF NOT EXISTS idx_rules_rule_id ON rules(rule_id);


-- TABLE: validation_jobs
-- Jobs die an Android Devices verteilt werden
CREATE TABLE IF NOT EXISTS validation_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT UNIQUE NOT NULL,
    paper_id TEXT NOT NULL,
    section_id INTEGER NOT NULL,
    rule_id TEXT NOT NULL,
    
    -- Status: pending, assigned, completed, failed, expired
    status TEXT DEFAULT 'pending',
    assigned_to TEXT,
    assigned_at TIMESTAMP,
    completed_at TIMESTAMP,
    expires_at TIMESTAMP,
    
    -- Result (from assigned device)
    is_match BOOLEAN,
    similarity REAL,
    confidence REAL,
    
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE,
    FOREIGN KEY (section_id) REFERENCES paper_sections(id) ON DELETE CASCADE,
    FOREIGN KEY (rule_id) REFERENCES rules(rule_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON validation_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_paper ON validation_jobs(paper_id);
CREATE INDEX IF NOT EXISTS idx_jobs_assigned ON validation_jobs(assigned_to);
CREATE INDEX IF NOT EXISTS idx_jobs_expires ON validation_jobs(expires_at);


-- TABLE: device_registry
-- Registrierte Android Devices
CREATE TABLE IF NOT EXISTS device_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT UNIQUE NOT NULL,
    device_name TEXT,
    device_model TEXT,
    os_version TEXT,
    app_version TEXT,
    
    -- Performance metrics
    jobs_completed INTEGER DEFAULT 0,
    avg_processing_time REAL,
    accuracy REAL,
    
    -- Status
    is_active BOOLEAN DEFAULT 1,
    last_seen TIMESTAMP,
    last_assigned TIMESTAMP,
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_devices_active ON device_registry(is_active);
CREATE INDEX IF NOT EXISTS idx_devices_last_assigned ON device_registry(last_assigned);


-- TABLE: validation_results
-- Ergebnisse von Android Devices
CREATE TABLE IF NOT EXISTS validation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    section_id INTEGER NOT NULL,
    
    is_match BOOLEAN NOT NULL,
    similarity REAL NOT NULL,
    confidence REAL NOT NULL,
    
    points_earned INTEGER,
    time_taken_ms INTEGER,
    
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (job_id) REFERENCES validation_jobs(job_id) ON DELETE CASCADE,
    FOREIGN KEY (device_id) REFERENCES device_registry(device_id),
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_results_paper_rule ON validation_results(paper_id, rule_id);
CREATE INDEX IF NOT EXISTS idx_results_job ON validation_results(job_id);
CREATE INDEX IF NOT EXISTS idx_results_device ON validation_results(device_id);


-- TABLE: paper_consensus
-- Aggregierte Validierungs-Ergebnisse pro Paper + Rule
CREATE TABLE IF NOT EXISTS paper_consensus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    
    is_match BOOLEAN,
    avg_similarity REAL,
    avg_confidence REAL,
    vote_count INTEGER DEFAULT 0,
    agreement_ratio REAL,
    
    found_in_sections TEXT,  -- JSON array of section_ids
    
    is_validated BOOLEAN DEFAULT 0,
    min_votes_required INTEGER DEFAULT 3,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(paper_id, rule_id)
);

CREATE INDEX IF NOT EXISTS idx_consensus_paper ON paper_consensus(paper_id);
CREATE INDEX IF NOT EXISTS idx_consensus_validated ON paper_consensus(is_validated);


-- TABLE: leaderboard
-- Gamification: Spieler-Rangliste
CREATE TABLE IF NOT EXISTS leaderboard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT UNIQUE NOT NULL,
    player_name TEXT,
    
    total_points INTEGER DEFAULT 0,
    papers_validated INTEGER DEFAULT 0,
    matches_found INTEGER DEFAULT 0,
    accuracy REAL,
    streak INTEGER DEFAULT 0,
    max_streak INTEGER DEFAULT 0,
    
    achievements TEXT,  -- JSON array
    
    rank INTEGER,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (device_id) REFERENCES device_registry(device_id)
);

CREATE INDEX IF NOT EXISTS idx_leaderboard_points ON leaderboard(total_points DESC);
CREATE INDEX IF NOT EXISTS idx_leaderboard_rank ON leaderboard(rank);


-- TABLE: processing_queue
-- Async Processing Queue für Papers
CREATE TABLE IF NOT EXISTS processing_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    task_type TEXT NOT NULL,  -- download, extract, embed, voxelize
    
    status TEXT DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_queue_status ON processing_queue(status, priority DESC);
