# 🔧 MCP-Server TODO: PaperRun Backend

## 🎯 Ziel
n8n Agent sendet Papers und Regeln → MCP-Server speichert in SQLite → Android Devices holen ab und validieren

---

## ✅ Implementierungs-Status

### Phase 1: Database & Core API ✅ DONE
- [x] SQLite Schema erstellt (`db/migrations/001_create_tables.sql`)
- [x] Database Manager implementiert (`db/database.py`)
- [x] Models definiert (`db/models.py`)
- [x] POST /api/papers/submit (n8n Webhook)
- [x] POST /api/rules/create (BioBERT Embeddings)

### Phase 2: BioBERT Rule Generator ✅ DONE
- [x] RuleHandler mit BioBERT Integration
- [x] Positive/Negative Phrase Embeddings
- [x] 6 Default Rules definiert

### Phase 3: Paper Processing Pipeline ✅ DONE
- [x] PDF Download (async mit aiohttp)
- [x] PDF → PNG Rendering (PyMuPDF)
- [x] Text Extraction & Section Detection
- [x] BioBERT Embedding Generation
- [x] Voxel Converter (768-dim → 8x8x12)

### Phase 4: Job Distribution & Android API ✅ DONE
- [x] GET /api/jobs/next (mit Embeddings)
- [x] POST /api/validation/submit
- [x] Device Registration & Tracking
- [x] Fair Job Distribution

### Phase 5: Consensus Engine ✅ DONE
- [x] Ergebnis-Aggregation
- [x] Mehrheitsentscheidung
- [x] Agreement Ratio
- [x] Min 3 Votes für Validierung

### Phase 6: Unity SSE & Gamification ✅ DONE
- [x] GET /api/stream/unity (SSE)
- [x] Events: paper_validated, leaderboard_update, etc.
- [x] Leaderboard mit Punktesystem

### Phase 7: Vision Encoder ⏳ OPTIONAL
- [ ] BLIP-2/SigLIP Integration (für Bild-basierte Validierung)

---

## 📊 Datenbank-Schema (Implementiert)

### Priorität: ⚡ KRITISCH - Zuerst implementieren!

```sql
-- TABLE: papers
-- Speichert alle Papers die evaluiert werden sollen
CREATE TABLE papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT UNIQUE NOT NULL,
    title TEXT,
    authors TEXT,
    journal TEXT,
    publication_date TEXT,
    pdf_url TEXT,
    pdf_local_path TEXT,
    
    status TEXT DEFAULT 'pending',
    downloaded_at TIMESTAMP,
    processed_at TIMESTAMP,
    
    source TEXT DEFAULT 'n8n',
    priority INTEGER DEFAULT 5,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_papers_status ON papers(status);
CREATE INDEX idx_papers_priority ON papers(priority DESC);


-- TABLE: paper_sections
-- Pro Paper: Abschnitte mit Embeddings und Voxel-Daten
CREATE TABLE paper_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    section_name TEXT NOT NULL,
    section_text TEXT,
    
    page_number INTEGER,
    image_path TEXT,
    embedding BLOB,
    voxel_data TEXT,
    
    color_r REAL,
    color_g REAL,
    color_b REAL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);

CREATE INDEX idx_sections_paper ON paper_sections(paper_id);


-- TABLE: rules
-- Von BioBERT generierte Regeln
CREATE TABLE rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id TEXT UNIQUE NOT NULL,
    question TEXT NOT NULL,
    
    positive_phrases TEXT,
    negative_phrases TEXT,
    pos_embedding BLOB,
    neg_embedding BLOB,
    
    threshold REAL DEFAULT 0.75,
    
    is_active BOOLEAN DEFAULT 1,
    created_by TEXT DEFAULT 'system',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- TABLE: validation_jobs
-- Jobs die an Android Devices verteilt werden
CREATE TABLE validation_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT UNIQUE NOT NULL,
    paper_id TEXT NOT NULL,
    section_id INTEGER NOT NULL,
    rule_id TEXT NOT NULL,
    
    status TEXT DEFAULT 'pending',
    assigned_to TEXT,
    assigned_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    is_match BOOLEAN,
    similarity REAL,
    confidence REAL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id),
    FOREIGN KEY (section_id) REFERENCES paper_sections(id),
    FOREIGN KEY (rule_id) REFERENCES rules(rule_id)
);

CREATE INDEX idx_jobs_status ON validation_jobs(status);
CREATE INDEX idx_jobs_paper ON validation_jobs(paper_id);


-- TABLE: device_registry
-- Registrierte Android Devices
CREATE TABLE device_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT UNIQUE NOT NULL,
    device_name TEXT,
    device_model TEXT,
    os_version TEXT,
    app_version TEXT,
    
    jobs_completed INTEGER DEFAULT 0,
    avg_processing_time REAL,
    accuracy REAL,
    
    is_active BOOLEAN DEFAULT 1,
    last_seen TIMESTAMP,
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- TABLE: validation_results
-- Ergebnisse von Android Devices
CREATE TABLE validation_results (
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
    
    FOREIGN KEY (job_id) REFERENCES validation_jobs(job_id),
    FOREIGN KEY (device_id) REFERENCES device_registry(device_id),
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);

CREATE INDEX idx_results_paper_rule ON validation_results(paper_id, rule_id);


-- TABLE: paper_consensus
-- Aggregierte Validierungs-Ergebnisse
CREATE TABLE paper_consensus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    
    is_match BOOLEAN,
    avg_similarity REAL,
    avg_confidence REAL,
    vote_count INTEGER,
    agreement_ratio REAL,
    
    found_in_sections TEXT,
    
    is_validated BOOLEAN DEFAULT 0,
    min_votes_required INTEGER DEFAULT 3,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(paper_id, rule_id)
);


-- TABLE: leaderboard
-- Gamification: Spieler-Rangliste
CREATE TABLE leaderboard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT UNIQUE NOT NULL,
    player_name TEXT,
    
    total_points INTEGER DEFAULT 0,
    papers_validated INTEGER DEFAULT 0,
    matches_found INTEGER DEFAULT 0,
    accuracy REAL,
    
    achievements TEXT,
    
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (device_id) REFERENCES device_registry(device_id)
);
```

---

## 🔌 API Endpoints

### Priorität: ⚡ KRITISCH

#### 1. n8n Webhook: Paper einreichen
```
POST /api/papers/submit
Content-Type: application/json

Request Body:
{
  "paper_id": "PMC12345",
  "title": "Randomized Trial of...",
  "pdf_url": "https://...",
  "priority": 7,
  "source": "n8n"
}

Response:
{
  "status": "accepted",
  "paper_id": "PMC12345",
  "message": "Paper queued for processing"
}
```

**TODO:**
- [ ] Endpoint erstellen in src/paperstream/handlers/paper_handler.py
- [ ] PDF Download starten (async)
- [ ] In SQLite speichern
- [ ] Processing-Queue triggern


#### 2. n8n Webhook: Regel erstellen
```
POST /api/rules/create

Request:
{
  "rule_id": "is_rct",
  "question": "Ist das eine RCT?",
  "positive_phrases": ["randomized controlled trial", "RCT"],
  "negative_phrases": ["observational", "retrospective"],
  "threshold": 0.75
}
```

**TODO:**
- [ ] Endpoint in src/paperstream/handlers/rule_handler.py
- [ ] BioBERT Embeddings generieren
- [ ] In SQLite + Cache speichern


#### 3. Android: Jobs abholen
```
GET /api/jobs/next?device_id=android_abc123&limit=5

Response: Array von Jobs mit Voxel-Daten und Regel-Embeddings
```

**TODO:**
- [ ] Job-Assignierung Logik (fair distribution)
- [ ] Device-Tracking (last_seen updaten)
- [ ] Komprimierung (gzip) für Embeddings


#### 4. Android: Ergebnis submitten
```
POST /api/validation/submit

Request:
{
  "device_id": "android_abc123",
  "results": [
    {
      "job_id": "job_001",
      "is_match": true,
      "similarity": 0.89,
      "points_earned": 89
    }
  ]
}
```

**TODO:**
- [ ] Ergebnisse in validation_results speichern
- [ ] Consensus berechnen (wenn genug Votes)
- [ ] Leaderboard updaten


#### 5. Unity: Real-time Updates (SSE)
```
GET /api/stream/unity

Server-Sent Events mit paper_validated, leaderboard_update
```

---

## 📋 TODO-Liste (Priorisiert)

### Phase 1: Database & Core API (Tag 1-2) ⚡

- [ ] SQLite Schema erstellen
  - [ ] migrations/001_create_tables.sql schreiben
  - [ ] Schema in database.py initialisieren
  
- [ ] Models definieren (db/models.py)
  - [ ] Paper, PaperSection, Rule, ValidationJob models
  
- [ ] n8n Webhook Endpoints
  - [ ] POST /api/papers/submit
  - [ ] POST /api/rules/create


### Phase 2: BioBERT Rule Generator (Tag 2-3) 🧬

- [ ] RuleGenerator Klasse (handlers/rule_handler.py)
  - [ ] BioBERT Integration für Embeddings
  - [ ] Positive/Negative Phrase Embedding
  - [ ] Caching in SQLite
  
- [ ] Default Rules laden
  - [ ] 6 Standard-Regeln beim Server-Start


### Phase 3: Paper Processing Pipeline (Tag 3-5) 🔧

- [ ] PDF Download (pipeline/paper_processor.py)
  - [ ] Async download mit aiohttp
  - [ ] Lokale Speicherung in /data/papers/
  
- [ ] PDF → PNG Rendering
  - [ ] PyMuPDF Integration
  - [ ] 512x512 Resolution
  
- [ ] Vision Encoder (pipeline/vision_encoder.py)
  - [ ] BLIP-2 Q-Former ODER SigLIP
  - [ ] GPU-Nutzung (RTX 3060)
  
- [ ] Text Extraction & Section Detection
  - [ ] PyMuPDF text extraction
  - [ ] Regex für Sections (Abstract, Methods, Results)
  
- [ ] Voxel Converter (pipeline/voxel_converter.py)
  - [ ] Aus paperrun_game_logic.py portieren
  - [ ] 768-dim → 8x8x12 Voxel Grid
  - [ ] JSON Export für Unity


### Phase 4: Job Distribution & Android API (Tag 5-6) 📱

- [ ] Job Creation (handlers/job_handler.py)
  - [ ] Pro Paper + Rule + Section = 1 Job
  - [ ] Priority Queue
  
- [ ] Android Endpoints
  - [ ] GET /api/jobs/next
  - [ ] POST /api/validation/submit
  - [ ] GET /api/rules/active
  
- [ ] Device Management (handlers/device_handler.py)
  - [ ] Device Registration
  - [ ] Performance Metrics


### Phase 5: Consensus & VectorDB (Tag 6-7) 🎯

- [ ] Consensus Engine (pipeline/consensus_engine.py)
  - [ ] Ergebnisse aggregieren (3+ Devices)
  - [ ] Mehrheitsentscheidung
  - [ ] Agreement Ratio berechnen
  
- [ ] VectorDB Integration
  - [ ] ChromaDB oder Qdrant
  - [ ] Validierte Rules als Metadata


### Phase 6: Unity SSE & Gamification (Tag 7-8) 🎮

- [ ] SSE Endpoint (api/sse.py)
  - [ ] /api/stream/unity für Live-Updates
  - [ ] Events: paper_validated, leaderboard_update
  
- [ ] Leaderboard (game/leaderboard.py)
  - [ ] Top 100 Players
  - [ ] Achievements System


### Phase 7: Testing & Optimization (Tag 8-10) ✅

- [ ] Unit Tests
  - [ ] Alle Handler testen
  - [ ] Pipeline Tests mit Mock-Data
  
- [ ] Integration Tests
  - [ ] End-to-End: n8n → SQLite → Android → Consensus
  - [ ] Load Testing (1000 Papers/Tag)
  
- [ ] Performance Optimization
  - [ ] Batch Processing
  - [ ] GPU Memory Management


---

## 🔧 Technische Details

### Embedding Storage (BLOB)

```python
import numpy as np

# Speichern
embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

# Laden
embedding = np.frombuffer(row['embedding'], dtype=np.float32)
```

### Fair Job Distribution

```python
# Least-Recently-Assigned Device bekommt Job
SELECT device_id 
FROM device_registry 
WHERE is_active = 1 
ORDER BY last_assigned ASC 
LIMIT 1
```

---

## ⏱️ Zeitschätzung

| Phase | Tage | Kritikalität |
|-------|------|--------------|
| Database & API | 2 | ⚡⚡⚡ |
| BioBERT | 1 | ⚡⚡⚡ |
| Paper Pipeline | 3 | ⚡⚡ |
| Android API | 2 | ⚡⚡⚡ |
| Consensus | 1 | ⚡⚡ |
| Unity SSE | 1 | ⚡ |
| Testing | 2 | ⚡⚡ |
| **TOTAL** | **12 Tage** | |

---

## 🎯 Erfolgs-Kriterien

- ✅ n8n kann Papers submitten → erscheinen in DB
- ✅ Papers werden automatisch prozessiert (PDF → Voxel)
- ✅ Android kann 5 Jobs abholen in < 1 Sekunde
- ✅ Validation Results werden korrekt aggregiert
- ✅ Consensus nach 3 Votes funktioniert
- ✅ Unity bekommt Live-Updates via SSE
- ✅ 1000 Papers/Tag verarbeitbar (RTX 3060)
