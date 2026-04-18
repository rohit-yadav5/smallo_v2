CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,
    title TEXT,
    raw_text TEXT NOT NULL,
    summary TEXT,
    importance_score REAL DEFAULT 0,
    confidence_score REAL DEFAULT 1,
    source TEXT,
    project_reference TEXT,
    status TEXT DEFAULT 'active',
    session_id TEXT DEFAULT 'legacy',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    entity_type TEXT NOT NULL,
    category TEXT NOT NULL,
    domain TEXT NOT NULL,
    parent_entity_id TEXT,
    importance_score REAL DEFAULT 0,
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_entity_id) REFERENCES entities(id)
);

CREATE TABLE IF NOT EXISTS entity_relations (
    id TEXT PRIMARY KEY,
    source_entity_id TEXT,
    target_entity_id TEXT,
    relation_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_entity_id) REFERENCES entities(id),
    FOREIGN KEY (target_entity_id) REFERENCES entities(id)
);

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT,
    entity_id TEXT,
    PRIMARY KEY (memory_id, entity_id),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_relations (
    id TEXT PRIMARY KEY,
    source_memory_id TEXT,
    target_memory_id TEXT,
    relation_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_memory_id) REFERENCES memories(id),
    FOREIGN KEY (target_memory_id) REFERENCES memories(id)
);

CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id TEXT PRIMARY KEY,
    vector_id TEXT,
    model_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_lifecycle (
    memory_id TEXT PRIMARY KEY,
    stage TEXT,
    consolidated_into TEXT,
    archived_at TIMESTAMP,
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

CREATE INDEX idx_memory_type ON memories(memory_type);
CREATE INDEX idx_project_reference ON memories(project_reference);
CREATE INDEX idx_created_at ON memories(created_at);
CREATE INDEX idx_entity_name ON entities(name);
CREATE INDEX idx_entity_type ON entities(entity_type);
CREATE INDEX idx_entity_domain ON entities(domain);
CREATE INDEX idx_entity_category ON entities(category);
CREATE INDEX idx_memory_entities_entity ON memory_entities(entity_id);
CREATE INDEX idx_memory_entities_memory ON memory_entities(memory_id);
CREATE INDEX idx_memory_lifecycle_stage ON memory_lifecycle(stage);
CREATE INDEX IF NOT EXISTS idx_entities_usage_count ON entities(usage_count);
CREATE INDEX IF NOT EXISTS idx_entities_parent ON entities(parent_entity_id);
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_model ON memory_embeddings(model_name);