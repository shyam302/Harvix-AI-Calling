-- Runs once, when the postgres volume is first created.
-- The bot also runs CREATE TABLE IF NOT EXISTS on startup (ari_app/db.py), so this
-- file and db.py must stay in sync.

CREATE TABLE IF NOT EXISTS users (
    id            BIGSERIAL PRIMARY KEY,
    phone         TEXT NOT NULL UNIQUE,          -- caller id / SIP user (e.g. 1001)
    name          TEXT NOT NULL,
    pin_hash      TEXT NOT NULL,                 -- scrypt$salt$hash, never the PIN itself
    language      TEXT,
    profile       JSONB NOT NULL DEFAULT '{}'::jsonb,   -- answers to registration_questions.json
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_call_at  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS calls (
    id            BIGSERIAL PRIMARY KEY,
    session_id    TEXT NOT NULL UNIQUE,
    channel_id    TEXT,
    user_id       BIGINT REFERENCES users(id) ON DELETE SET NULL,
    caller_number TEXT,
    dialed_number TEXT,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at      TIMESTAMPTZ,
    duration_s    REAL,
    turns         INT,
    language      TEXT,
    status        TEXT,                           -- registered | new_registration | guest | verified | pin_failed
    recording_path  TEXT,                         -- relative to RECORDINGS_DIR (Opus .ogg)
    recording_bytes BIGINT
);

CREATE TABLE IF NOT EXISTS call_messages (
    id         BIGSERIAL PRIMARY KEY,
    call_id    BIGINT NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,                     -- user | assistant
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS complaints (
    id            BIGSERIAL PRIMARY KEY,
    ticket_no     TEXT UNIQUE,                    -- 100000 + id, read out to the caller
    call_id       BIGINT REFERENCES calls(id) ON DELETE SET NULL,
    user_id       BIGINT REFERENCES users(id) ON DELETE SET NULL,
    caller_number TEXT,
    summary       TEXT,                           -- what the caller said to start the complaint
    details       JSONB NOT NULL DEFAULT '{}'::jsonb,  -- answers to complaint_questions.json
    status        TEXT NOT NULL DEFAULT 'open',
    resolution    TEXT,                           -- written on the admin page; read out to the caller
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS calls_user_idx ON calls(user_id, started_at DESC);
CREATE INDEX IF NOT EXISTS calls_caller_idx ON calls(caller_number, started_at DESC);
CREATE INDEX IF NOT EXISTS call_messages_call_idx ON call_messages(call_id, id);
