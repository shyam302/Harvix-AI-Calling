"""
PostgreSQL: registered callers and call history.

Tables (also in docker/postgres/init.sql; created here too if missing):
  users          one row per registered caller, looked up by phone (caller id)
  calls          one row per call: who, when, how long, outcome
  call_messages  every line said in a call (user / assistant)

The database is optional: if DATABASE_URL is empty or Postgres is down, every
function here logs once and returns None/False, and calls continue as guests.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            BIGSERIAL PRIMARY KEY,
    phone         TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    pin_hash      TEXT NOT NULL,
    language      TEXT,
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
    status        TEXT
);
CREATE TABLE IF NOT EXISTS call_messages (
    id         BIGSERIAL PRIMARY KEY,
    call_id    BIGINT NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS calls_user_idx ON calls(user_id, started_at DESC);
CREATE INDEX IF NOT EXISTS calls_caller_idx ON calls(caller_number, started_at DESC);
CREATE INDEX IF NOT EXISTS call_messages_call_idx ON call_messages(call_id, id);
ALTER TABLE users ADD COLUMN IF NOT EXISTS profile JSONB NOT NULL DEFAULT '{}'::jsonb;
CREATE TABLE IF NOT EXISTS complaints (
    id            BIGSERIAL PRIMARY KEY,
    ticket_no     TEXT UNIQUE,
    call_id       BIGINT REFERENCES calls(id) ON DELETE SET NULL,
    user_id       BIGINT REFERENCES users(id) ON DELETE SET NULL,
    caller_number TEXT,
    summary       TEXT,
    details       JSONB NOT NULL DEFAULT '{}'::jsonb,
    status        TEXT NOT NULL DEFAULT 'open',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS complaints_user_idx ON complaints(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS complaints_call_idx ON complaints(call_id);
ALTER TABLE calls ADD COLUMN IF NOT EXISTS recording_path TEXT;
ALTER TABLE calls ADD COLUMN IF NOT EXISTS recording_bytes BIGINT;
"""


@dataclass
class User:
    id: int
    phone: str
    name: str
    pin_hash: str
    language: str | None


# PIN hashing (scrypt; the PIN itself is never stored or logged)

def hash_pin(pin: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(pin.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_pin(pin: str, stored: str) -> bool:
    try:
        algo, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    if algo != "scrypt":
        return False
    digest = hashlib.scrypt(
        pin.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1, dklen=32
    )
    return hmac.compare_digest(digest.hex(), digest_hex)


# connection pool

_pool: Any = None
_pool_lock = asyncio.Lock()
_unavailable_logged = False


async def init(database_url: str) -> bool:
    """Open the pool and create tables. Returns False (and logs) if unavailable."""
    global _pool, _unavailable_logged
    if not database_url:
        log.info("DATABASE_URL not set: caller registration and call history are off")
        return False
    async with _pool_lock:
        if _pool is not None:
            return True
        import asyncpg

        last_exc: Exception | None = None
        for attempt in range(3):  # Postgres may still be starting (docker compose up)
            try:
                _pool = await asyncpg.create_pool(
                    database_url, min_size=1, max_size=10, command_timeout=10, timeout=10
                )
                async with _pool.acquire() as con:
                    await con.execute(_SCHEMA)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if _pool is not None:
                    await _pool.close()
                _pool = None
                if attempt < 2:
                    await asyncio.sleep(2)
        if last_exc is not None:
            exc = last_exc
            if not _unavailable_logged:
                _unavailable_logged = True
                log.error(
                    "PostgreSQL unavailable (%r): calls continue without registration/history. "
                    "Start it with: cd docker && docker compose up -d postgres",
                    exc,
                )
            return False
    log.info("PostgreSQL ready (users, calls, call_messages)")
    return True


async def close() -> None:
    """Close the pool; if a query is still running after 3 s, drop the connections."""
    global _pool
    pool, _pool = _pool, None
    if pool is None:
        return
    try:
        await asyncio.wait_for(pool.close(), timeout=3)
    except (asyncio.TimeoutError, Exception):
        pool.terminate()


def available() -> bool:
    return _pool is not None


async def _run(fn_name: str, coro_fn, default=None):
    """Run a query; on any DB error log it and return ``default`` (the call goes on)."""
    if _pool is None:
        return default
    try:
        async with _pool.acquire() as con:
            return await coro_fn(con)
    except Exception as exc:
        log.warning("DB %s failed: %s", fn_name, exc)
        return default


# users

async def get_user_by_phone(phone: str | None) -> User | None:
    if not phone:
        return None

    async def q(con):
        row = await con.fetchrow(
            "SELECT id, phone, name, pin_hash, language FROM users WHERE phone = $1", phone
        )
        return User(**dict(row)) if row else None

    return await _run("get_user_by_phone", q)


async def create_user(
    *, phone: str, name: str, pin: str, language: str | None = None
) -> User | None:
    pin_hash = hash_pin(pin)

    async def q(con):
        row = await con.fetchrow(
            """
            INSERT INTO users (phone, name, pin_hash, language)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (phone) DO UPDATE
               SET name = EXCLUDED.name, pin_hash = EXCLUDED.pin_hash,
                   language = EXCLUDED.language, updated_at = now()
            RETURNING id, phone, name, pin_hash, language
            """,
            phone,
            name,
            pin_hash,
            language,
        )
        return User(**dict(row))

    return await _run("create_user", q)


# call history

async def start_call(
    *,
    session_id: str,
    channel_id: str,
    caller_number: str | None,
    dialed_number: str | None,
) -> int | None:
    async def q(con):
        return await con.fetchval(
            """
            INSERT INTO calls (session_id, channel_id, caller_number, dialed_number)
            VALUES ($1, $2, $3, $4) RETURNING id
            """,
            session_id,
            channel_id,
            caller_number,
            dialed_number,
        )

    return await _run("start_call", q)


async def set_call_user(call_id: int | None, user_id: int, status: str) -> None:
    if call_id is None:
        return

    async def q(con):
        await con.execute(
            "UPDATE calls SET user_id = $2, status = $3 WHERE id = $1", call_id, user_id, status
        )
        await con.execute("UPDATE users SET last_call_at = now() WHERE id = $1", user_id)

    await _run("set_call_user", q)


async def add_message(call_id: int | None, role: str, content: str) -> None:
    if call_id is None or not content:
        return

    async def q(con):
        await con.execute(
            "INSERT INTO call_messages (call_id, role, content) VALUES ($1, $2, $3)",
            call_id,
            role,
            content,
        )

    await _run("add_message", q)


async def end_call(
    call_id: int | None,
    *,
    duration_s: float,
    turns: int,
    language: str | None,
    status: str | None,
) -> None:
    if call_id is None:
        return

    async def q(con):
        await con.execute(
            """
            UPDATE calls SET ended_at = now(), duration_s = $2, turns = $3,
                   language = $4, status = COALESCE($5, status)
             WHERE id = $1
            """,
            call_id,
            duration_s,
            turns,
            language,
            status,
        )

    await _run("end_call", q)


async def recent_calls(user_id: int, limit: int = 3) -> list[dict]:
    """Last few calls with their first user line, for "last time you asked about..." context."""

    async def q(con):
        rows = await con.fetch(
            """
            SELECT c.started_at,
                   (SELECT content FROM call_messages m
                     WHERE m.call_id = c.id AND m.role = 'user' ORDER BY m.id LIMIT 1) AS first_ask
              FROM calls c
             WHERE c.user_id = $1 AND c.ended_at IS NOT NULL
             ORDER BY c.started_at DESC LIMIT $2
            """,
            user_id,
            limit,
        )
        return [dict(r) for r in rows]

    return await _run("recent_calls", q, default=[])


# recordings + browsing (recordings page)

async def set_call_recording(call_id: int | None, rel_path: str, size: int) -> None:
    if call_id is None:
        return

    async def q(con):
        await con.execute(
            "UPDATE calls SET recording_path = $2, recording_bytes = $3 WHERE id = $1",
            call_id,
            rel_path,
            size,
        )

    await _run("set_call_recording", q)


async def list_calls(*, limit: int = 50, offset: int = 0, search: str = "") -> list[dict]:
    """Newest calls first, with the caller's name and a transcript snippet."""
    like = f"%{search.strip()}%" if search.strip() else None

    async def q(con):
        rows = await con.fetch(
            """
            SELECT c.id, c.session_id, c.caller_number, c.started_at, c.ended_at,
                   c.duration_s, c.turns, c.language, c.status,
                   c.recording_path IS NOT NULL AS has_recording, c.recording_bytes,
                   u.name AS user_name
              FROM calls c LEFT JOIN users u ON u.id = c.user_id
             WHERE $3::text IS NULL
                OR c.caller_number ILIKE $3 OR u.name ILIKE $3
                OR EXISTS (SELECT 1 FROM call_messages m
                            WHERE m.call_id = c.id AND m.content ILIKE $3)
             ORDER BY c.started_at DESC
             LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
            like,
        )
        return [dict(r) for r in rows]

    return await _run("list_calls", q, default=[])


async def get_call(call_id: int) -> dict | None:
    async def q(con):
        row = await con.fetchrow(
            """
            SELECT c.*, u.name AS user_name, u.profile::text AS user_profile FROM calls c
              LEFT JOIN users u ON u.id = c.user_id WHERE c.id = $1
            """,
            call_id,
        )
        if row is None:
            return None
        msgs = await con.fetch(
            "SELECT role, content, created_at FROM call_messages WHERE call_id = $1 ORDER BY id",
            call_id,
        )
        out = dict(row)
        out["messages"] = [dict(m) for m in msgs]
        comps = await con.fetch(
            "SELECT ticket_no, status, summary, details::text AS details, created_at "
            "FROM complaints WHERE call_id = $1 ORDER BY id",
            call_id,
        )
        out["complaints"] = [dict(c) for c in comps]
        return out

    return await _run("get_call", q)


async def update_user_profile(user_id: int, fields: dict[str, str]) -> None:
    """Merge answers into users.profile (JSON), e.g. {"city": "Pune"}."""
    import json

    async def q(con):
        await con.execute(
            "UPDATE users SET profile = profile || $2::jsonb, updated_at = now() WHERE id = $1",
            user_id,
            json.dumps(fields, ensure_ascii=False),
        )

    await _run("update_user_profile", q)


# complaints

async def create_complaint(
    *,
    call_id: int | None,
    user_id: int | None,
    caller_number: str | None,
    summary: str,
    details: dict[str, str],
) -> str | None:
    """Save a complaint; returns its ticket number (100000 + id), or None on failure."""
    import json

    async def q(con):
        async with con.transaction():
            cid = await con.fetchval(
                """
                INSERT INTO complaints (call_id, user_id, caller_number, summary, details)
                VALUES ($1, $2, $3, $4, $5::jsonb) RETURNING id
                """,
                call_id,
                user_id,
                caller_number,
                summary,
                json.dumps(details, ensure_ascii=False),
            )
            ticket = str(100000 + cid)
            await con.execute("UPDATE complaints SET ticket_no = $2 WHERE id = $1", cid, ticket)
            return ticket

    return await _run("create_complaint", q)


async def complaints_for_call(call_id: int) -> list[dict]:
    async def q(con):
        rows = await con.fetch(
            """
            SELECT ticket_no, status, summary, details::text AS details, created_at
              FROM complaints WHERE call_id = $1 ORDER BY id
            """,
            call_id,
        )
        return [dict(r) for r in rows]

    return await _run("complaints_for_call", q, default=[])


async def open_complaints(user_id: int | None, caller_number: str | None, limit: int = 5) -> list[dict]:
    """Complaints that are not closed, newest first (by user, else by caller number)."""
    if user_id is None and not caller_number:
        return []

    async def q(con):
        rows = await con.fetch(
            """
            SELECT ticket_no, status, summary, details::text AS details, created_at
              FROM complaints
             WHERE status <> 'closed'
               AND (($1::bigint IS NOT NULL AND user_id = $1)
                    OR ($1::bigint IS NULL AND caller_number = $2))
             ORDER BY created_at DESC
             LIMIT $3
            """,
            user_id,
            caller_number,
            limit,
        )
        out = []
        for r in rows:
            d = dict(r)
            try:
                import json

                d["details"] = json.loads(d["details"] or "{}")
            except ValueError:
                d["details"] = {}
            out.append(d)
        return out

    return await _run("open_complaints", q, default=[])
