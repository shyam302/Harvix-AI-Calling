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
ALTER TABLE complaints ADD COLUMN IF NOT EXISTS resolution TEXT;
CREATE TABLE IF NOT EXISTS phones (
    extension  TEXT PRIMARY KEY,          -- SIP login added from the web page (PJSIP in astdb)
    note       TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
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
            SELECT ticket_no, status, summary, details::text AS details, created_at,
                   resolution
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


# admin pages (/customers): customers + complaints

COMPLAINT_STATUSES = ("open", "in_progress", "resolved", "closed")


def _decode_json(d: dict, *keys: str) -> dict:
    import json

    for k in keys:
        v = d.get(k)
        if isinstance(v, str):
            try:
                d[k] = json.loads(v)
            except ValueError:
                d[k] = {}
    return d


async def list_customers(*, search: str = "", limit: int = 100, offset: int = 0) -> list[dict]:
    like = f"%{search.strip()}%" if search.strip() else None

    async def q(con):
        rows = await con.fetch(
            """
            SELECT u.id, u.phone, u.name, u.language, u.profile::text AS profile,
                   u.created_at, u.last_call_at,
                   (SELECT count(*) FROM complaints c WHERE c.user_id = u.id) AS complaints,
                   (SELECT count(*) FROM complaints c
                     WHERE c.user_id = u.id AND c.status <> 'closed') AS open_complaints,
                   (SELECT count(*) FROM calls k WHERE k.user_id = u.id) AS calls
              FROM users u
             WHERE $3::text IS NULL OR u.name ILIKE $3 OR u.phone ILIKE $3
                OR u.profile::text ILIKE $3
             ORDER BY COALESCE(u.last_call_at, u.created_at) DESC
             LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
            like,
        )
        return [_decode_json(dict(r), "profile") for r in rows]

    return await _run("list_customers", q, default=[])


async def get_customer(user_id: int) -> dict | None:
    async def q(con):
        row = await con.fetchrow(
            """
            SELECT id, phone, name, language, profile::text AS profile,
                   created_at, updated_at, last_call_at
              FROM users WHERE id = $1
            """,
            user_id,
        )
        if row is None:
            return None
        out = _decode_json(dict(row), "profile")
        comps = await con.fetch(
            """
            SELECT ticket_no, status, summary, details::text AS details, created_at,
                   updated_at, call_id, resolution
              FROM complaints WHERE user_id = $1 ORDER BY created_at DESC
            """,
            user_id,
        )
        out["complaints"] = [_decode_json(dict(c), "details") for c in comps]
        calls = await con.fetch(
            """
            SELECT id, started_at, duration_s, turns, status,
                   recording_path IS NOT NULL AS has_recording
              FROM calls WHERE user_id = $1 ORDER BY started_at DESC LIMIT 20
            """,
            user_id,
        )
        out["calls"] = [dict(c) for c in calls]
        return out

    return await _run("get_customer", q)


async def list_complaints(
    *, status: str = "", search: str = "", limit: int = 100, offset: int = 0
) -> list[dict]:
    like = f"%{search.strip()}%" if search.strip() else None
    st = status if status in COMPLAINT_STATUSES else None

    async def q(con):
        rows = await con.fetch(
            """
            SELECT c.ticket_no, c.status, c.summary, c.details::text AS details,
                   c.created_at, c.updated_at, c.call_id, c.caller_number, c.resolution,
                   c.user_id, u.name AS customer_name
              FROM complaints c LEFT JOIN users u ON u.id = c.user_id
             WHERE ($3::text IS NULL OR c.status = $3)
               AND ($4::text IS NULL OR c.ticket_no ILIKE $4 OR c.caller_number ILIKE $4
                    OR u.name ILIKE $4 OR c.details::text ILIKE $4 OR c.summary ILIKE $4)
             ORDER BY c.created_at DESC
             LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
            st,
            like,
        )
        return [_decode_json(dict(r), "details") for r in rows]

    return await _run("list_complaints", q, default=[])


async def complaint_counts() -> dict[str, int]:
    async def q(con):
        rows = await con.fetch("SELECT status, count(*) AS n FROM complaints GROUP BY status")
        return {r["status"]: r["n"] for r in rows}

    return await _run("complaint_counts", q, default={})


async def set_complaint_status(ticket_no: str, status: str) -> bool:
    if status not in COMPLAINT_STATUSES:
        return False

    async def q(con):
        res = await con.execute(
            "UPDATE complaints SET status = $2, updated_at = now() WHERE ticket_no = $1",
            ticket_no,
            status,
        )
        return res.endswith(" 1")

    return bool(await _run("set_complaint_status", q, default=False))


async def set_complaint_resolution(ticket_no: str, resolution: str) -> bool:
    """Message for the customer (read out by the bot when they ask about this ticket)."""
    text = (resolution or "").strip()[:1000] or None

    async def q(con):
        res = await con.execute(
            "UPDATE complaints SET resolution = $2, updated_at = now() WHERE ticket_no = $1",
            ticket_no,
            text,
        )
        return res.endswith(" 1")

    return bool(await _run("set_complaint_resolution", q, default=False))


# ---- customer management (admin page /customers) ------------------------------------

class CustomerError(ValueError):
    """Validation problem shown to the admin (duplicate phone, bad PIN…)."""


def _clean_profile(profile: dict | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (profile or {}).items():
        k = str(k).strip().lower().replace(" ", "_")
        v = "" if v is None else str(v).strip()
        if k and v:
            out[k[:40]] = v[:200]
    return out


def _check_pin(pin: str, length: int) -> str:
    pin = (pin or "").strip()
    if not pin.isdigit() or len(pin) != length:
        raise CustomerError(f"PIN must be exactly {length} digits")
    return pin


async def admin_create_customer(
    *, phone: str, name: str, pin: str, pin_length: int, language: str | None = None,
    profile: dict | None = None,
) -> int:
    """New customer from the admin page. Returns the user id."""
    import json

    phone, name = (phone or "").strip(), (name or "").strip()
    if not phone:
        raise CustomerError("Phone (caller id) is required")
    if not name:
        raise CustomerError("Name is required")
    pin = _check_pin(pin, pin_length)
    if _pool is None:
        raise CustomerError("Database not connected")
    async with _pool.acquire() as con:
        if await con.fetchval("SELECT 1 FROM users WHERE phone = $1", phone):
            raise CustomerError(f"A customer with phone {phone} already exists")
        return await con.fetchval(
            """
            INSERT INTO users (phone, name, pin_hash, language, profile)
            VALUES ($1, $2, $3, $4, $5::jsonb) RETURNING id
            """,
            phone,
            name[:80],
            hash_pin(pin),
            (language or "").strip() or None,
            json.dumps(_clean_profile(profile), ensure_ascii=False),
        )


async def admin_update_customer(
    user_id: int, *, phone: str | None = None, name: str | None = None,
    language: str | None = None, profile: dict | None = None,
) -> bool:
    """Edit details. ``profile`` replaces the whole profile (keys left empty are removed)."""
    import json

    if _pool is None:
        raise CustomerError("Database not connected")
    async with _pool.acquire() as con:
        if phone is not None:
            phone = phone.strip()
            if not phone:
                raise CustomerError("Phone (caller id) can't be empty")
            if await con.fetchval(
                "SELECT 1 FROM users WHERE phone = $1 AND id <> $2", phone, user_id
            ):
                raise CustomerError(f"Another customer already has phone {phone}")
        if name is not None and not name.strip():
            raise CustomerError("Name can't be empty")
        res = await con.execute(
            """
            UPDATE users SET
                phone    = COALESCE($2, phone),
                name     = COALESCE($3, name),
                language = CASE WHEN $4::text IS NULL THEN language ELSE NULLIF($4, '') END,
                profile  = COALESCE($5::jsonb, profile),
                updated_at = now()
             WHERE id = $1
            """,
            user_id,
            phone,
            name.strip()[:80] if name is not None else None,
            language.strip() if language is not None else None,
            json.dumps(_clean_profile(profile), ensure_ascii=False) if profile is not None else None,
        )
        return res.endswith(" 1")


async def admin_set_pin(user_id: int, pin: str, pin_length: int) -> bool:
    """New PIN (also "forgot PIN"): only the scrypt hash is stored."""
    pin = _check_pin(pin, pin_length)

    async def q(con):
        res = await con.execute(
            "UPDATE users SET pin_hash = $2, updated_at = now() WHERE id = $1",
            user_id,
            hash_pin(pin),
        )
        return res.endswith(" 1")

    return bool(await _run("admin_set_pin", q, default=False))


async def admin_delete_customer(user_id: int) -> bool:
    """Remove a customer. Their calls/complaints stay (user_id becomes NULL = guest)."""

    async def q(con):
        res = await con.execute("DELETE FROM users WHERE id = $1", user_id)
        return res.endswith(" 1")

    return bool(await _run("admin_delete_customer", q, default=False))


# ---- phone accounts added from the web page (ari_app/phones.py) -----------------------

async def web_phones() -> dict[str, dict]:
    async def q(con):
        rows = await con.fetch("SELECT extension, note, created_at FROM phones")
        return {r["extension"]: dict(r) for r in rows}

    return await _run("web_phones", q, default={})


async def add_web_phone(extension: str, note: str | None = None) -> None:
    async def q(con):
        await con.execute(
            "INSERT INTO phones (extension, note) VALUES ($1, $2) ON CONFLICT (extension) DO NOTHING",
            extension,
            note,
        )

    await _run("add_web_phone", q)


async def remove_web_phone(extension: str) -> None:
    async def q(con):
        await con.execute("DELETE FROM phones WHERE extension = $1", extension)

    await _run("remove_web_phone", q)


async def customers_by_phone(phones: list[str]) -> dict[str, dict]:
    async def q(con):
        rows = await con.fetch("SELECT id, phone, name FROM users WHERE phone = ANY($1::text[])", phones)
        return {r["phone"]: {"id": r["id"], "name": r["name"]} for r in rows}

    return await _run("customers_by_phone", q, default={})
