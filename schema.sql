-- SMARTER.AI - Memory Schema
-- Three domain-agnostic tables: users, sessions, messages
-- Two application-specific tables: module_interactions, professional_problems

-- 1. USERS
-- NTD role: Identifies returning learners.
-- Without last_seen, RS cannot be computed (needs days_elapsed).
CREATE TABLE IF NOT EXISTS users (
    user_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    last_seen       TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata        JSONB NOT NULL DEFAULT '{}'
);

-- 2. SESSIONS
-- NTD role: Stores profession (enables professional_apps tracking for SS),
-- recommended_modules (knowledge window persistence),
-- session_state (conversation resumption).
CREATE TABLE IF NOT EXISTS sessions (
    session_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    started_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    last_active         TIMESTAMP NOT NULL DEFAULT NOW(),
    topic               TEXT,
    hours_budget        FLOAT,
    profession          TEXT,
    learning_format     TEXT,
    recommended_modules JSONB,
    session_state       TEXT NOT NULL DEFAULT 'slot_filling',
    professional_context TEXT
);

-- 3. MESSAGES
-- NTD role: Stores full message text for response_depth computation
-- (avg word count of substantive user responses feeds SS).
-- message_type and module_ref enable per-module engagement analysis.
CREATE TABLE IF NOT EXISTS messages (
    message_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content         TEXT NOT NULL,
    message_type    TEXT NOT NULL DEFAULT 'general',
    module_ref      JSONB,
    metadata        JSONB NOT NULL DEFAULT '{}'
);

-- 4. MODULE_INTERACTIONS — THE CORE NTD TABLE
-- NTD role:
--   question_count    = encoding events for SS
--   concepts_asked    = concept breadth for SS (JSONB array, distinct only)
--   professional_apps = associative links for SS
--   first_asked_at / last_asked_at = RS decay computation
-- If using Ebbinghaus, only last_asked_at would be needed. NTD demands ALL fields.
CREATE TABLE IF NOT EXISTS module_interactions (
    interaction_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    module_idx      INTEGER NOT NULL,
    module_pdf_name TEXT NOT NULL,
    first_asked_at  TIMESTAMP NOT NULL DEFAULT NOW(),
    last_asked_at   TIMESTAMP NOT NULL DEFAULT NOW(),
    question_count  INTEGER NOT NULL DEFAULT 0,
    concepts_asked  JSONB NOT NULL DEFAULT '[]',
    professional_apps JSONB NOT NULL DEFAULT '[]'
);

-- 5. PROFESSIONAL_PROBLEMS
-- NTD role: When a learner connects a module to a professional problem,
-- it creates an associative encoding event that increases SS
-- through additional encoding pathways.
CREATE TABLE IF NOT EXISTS professional_problems (
    problem_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    problem_text    TEXT NOT NULL,
    related_modules JSONB NOT NULL DEFAULT '[]',
    related_concepts JSONB NOT NULL DEFAULT '[]',
    resolution_status TEXT NOT NULL DEFAULT 'open'
);

-- ============================================================
-- INDEXES
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_sessions_user
    ON sessions(user_id);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_module_interactions_session
    ON module_interactions(session_id);

CREATE INDEX IF NOT EXISTS idx_professional_problems_session
    ON professional_problems(session_id);