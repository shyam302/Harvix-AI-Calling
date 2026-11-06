# Databases: RAG in Redis, callers and call history in PostgreSQL

Branch: `db_support`

## What changed for a caller

1. **The call starts.** The bot looks up the caller id (for example `1001`) in `users`. It asks nothing.
   - **Registered:** "Welcome back, Shyam! How can I help you today?" The LLM is told the caller's name and that they're a registered customer, so it uses the name now and then.
   - **Not registered:** the normal greeting.
2. **The conversation is normal** for everyone.
3. **The caller wants to lodge a complaint.** The bot detects words like "complaint", "complain", "lodge", "शिकायत" or "shikayat".
   - **Not registered:** "First I need to register you. What's your name?" → a PIN of `PIN_LENGTH` digits → the same PIN again to confirm → saved. The bot then goes back to the complaint. This happens at most once per call. If the caller gives up, the call continues as a guest.
   - **Registered, with `AUTH_VERIFY_PIN=true`:** the bot asks for their PIN first. With the default `false`, nothing is asked.
4. **The PIN can be spoken or typed on the keypad**, whichever comes first.
   - Spoken: "one two three four", "1 2 3 4", "double five six seven", "ek do teen char" and "एक दो तीन चार" all work.
   - Keypad: four digits finish the entry automatically. `#` ends it early and `*` clears.
5. **Every line said in the call**, by the caller and the bot, is saved to `call_messages`. Start time, end time, duration, turns, language and outcome go to `calls`.

**PIN safety:** the PIN is stored only as a salted scrypt hash. It is never spoken back, logged, kept in chat history, sent to the LLM or written to `call_messages`. Logs show only "PIN entered on keypad" or "keypad key received (n buffered)".

## Call recordings

Every call is recorded with both voices: a snoop channel with `spy=both`, recorded by Asterisk. At hang-up the WAV is compressed to **Opus** (mono, 16 kbps, about 60 KB per minute, about 16 times smaller than the WAV) and saved to:

```
recordings/YYYY/MM/DD/<HHMMSS>_<caller>_<session>.ogg     (git-ignored)
```

The path and size are stored in `calls.recording_path` and `calls.recording_bytes`.

**Browse and play:** open **http://127.0.0.1:8765/recordings** while the bot runs. The page has a call list (search by number, name or anything said), an audio player with a speed button and download, and the transcript shown as a chat.

| Setting | Default | |
|---|---|---|
| `CALL_RECORDING_ENABLED` | `true` | |
| `RECORDINGS_DIR` | `recordings` | relative to the project root |
| `RECORDING_OPUS_KBPS` | `16` | 12 is still clear for speech; 24 or more for music |

Code: `ari_app/call_recording.py` (paths, ffmpeg Opus), `CallSession._start_call_recording()` / `_finish_call_recording()`, routes in `server.py`, page in `ari_app/static/recordings.html`.

The Docker Asterisk image has no `/var/spool/asterisk/recording` folder. `docker/render.sh` now creates it; without it, every ARI recording fails with "No such file or directory".

## Architecture

```
                    ┌────────────── Python bot ──────────────┐
caller ─ Asterisk ─►│ call_session.py                        │
                    │   _identify_caller()  ── users ─────────┼─► PostgreSQL
                    │   _complaint_gate()   ── register/verify┼─► (users, calls,
                    │   ctx.message_listener ─ every line ────┼─►  call_messages)
                    │   _knowledge_context() ─ rag.py ────────┼─► Redis Stack
                    └─────────────────────────────────────────┘   (vector index)
```

| File | Role |
|---|---|
| `ari_app/db.py` | asyncpg pool, creates the tables, user lookup and creation, PIN hashing, call history. If Postgres is down, everything returns None and calls carry on as guests. |
| `ari_app/registration.py` | Complaint intent, `parse_pin()`, `extract_name()` (LLM, with simple patterns as fallback), and the English/Hindi lines the bot says. |
| `ari_app/rag_redis.py` | `RedisKnowledgeBase`, the same interface as the local store; HNSW cosine index `rag_idx`. Used when `RAG_STORE=redis`; if Redis can't be reached, the local store is used. |
| `ari_app/call_session.py` | `_identify_caller`, `_complaint_gate`, `_register_new_caller`, `_verify_returning_caller`, `_ask_pin` (voice and keypad race), `_hear_once`, `on_dtmf`, and the DB call start/end. |
| `ari_app/ari_loop.py` | Opens the DB at startup and passes `ChannelDtmfReceived` events to the session. |
| `docker/docker-compose.yml` | Adds the `redis` (redis-stack-server) and `postgres` services with persistent volumes, bound to `127.0.0.1` only. |
| `docker/postgres/init.sql` | The table definitions, kept in sync with `db.py`. |
| `ari_app/call_recording.py`, `ari_app/static/recordings.html` | Recordings on disk, and the browse/play page. |

## Tables

- **`users`:** `id`, `phone` (unique caller id), `name`, `pin_hash`, `language`, `created_at`, `updated_at`, `last_call_at`. More fields can be added here later.
- **`calls`:** `session_id`, `channel_id`, `user_id`, `caller_number`, `dialed_number`, `started_at`, `ended_at`, `duration_s`, `turns`, `language`, `status`, `recording_path`, `recording_bytes`. Status is one of `guest`, `registered`, `new_registration`, `verified`, `pin_failed`.
- **`call_messages`:** `call_id`, `role` (`user` or `assistant`), `content`, `created_at`.

## Run it

```bash
cd docker
docker compose up -d redis postgres        # plus asterisk if you use the Docker Asterisk
cd ..
pip install -r requirements.txt            # adds redis, asyncpg
python -m ari_app.rag ingest               # loads knowledge/ into Redis (also happens at bot start)
python -m ari_app.server
```

Settings in the project `.env`:

```
RAG_STORE=redis
REDIS_URL=redis://:callbot@127.0.0.1:6379/0
DATABASE_URL=postgresql://callbot:callbot@127.0.0.1:5432/callbot
REGISTRATION_ENABLED=true
AUTH_VERIFY_PIN=false
PIN_LENGTH=4
```

The passwords must match `REDIS_PASSWORD` and `POSTGRES_*` in `docker/.env`. Change both before using this anywhere other than your own machine.

## Look at the data

```bash
docker exec -it callbot-postgres psql -U callbot
  SELECT id, phone, name, created_at, last_call_at FROM users;
  SELECT id, caller_number, user_id, status, turns, duration_s, started_at FROM calls ORDER BY id DESC LIMIT 10;
  SELECT role, content FROM call_messages WHERE call_id = 12 ORDER BY id;

docker exec -it callbot-redis redis-cli -a callbot FT.INFO rag_idx   # num_docs = chunks
```

## Tested (2026-11-06, scripted calls through Docker Asterisk)

- New caller asks a normal question: normal answer, no registration.
- New caller says "I want to lodge a complaint…": the bot asks for a name (it heard "Shyam Kumar"), then a spoken PIN, then confirmation. The user is saved and the LLM takes the complaint using the name.
- The same flow with the PIN typed on the keypad (`SendDTMF`): saved.
- The same caller calls again: "Welcome back, Shyam Kumar!" with nothing asked.
- `call_messages` holds the greeting and the conversation. No PIN appears anywhere in the database or the logs.
- Call recording: a 22 s call gave a 21 KB `.ogg` (the WAV was 338 KB). Transcribing it back gave both the bot's and the caller's words. `/api/calls`, `/api/calls/{id}`, `/audio` and `/recordings` all return 200, and an unknown id returns 404.
- Redis RAG returns the same pages as the local index (scores within about 0.01); a search takes about 10 ms, mostly spent embedding the question.

## Not done yet

- The complaint itself isn't saved as a structured record (only in `call_messages`). A `complaints` table with category, status and ticket number would be the next step.
- `db.recent_calls()` exists but isn't used yet. It could give the LLM "last time you called about…".
- There's no way to reset a forgotten PIN.
- The recordings page has no login. It's bound to 127.0.0.1, so add authentication before exposing it on a network.
- There's no automatic deletion of old recordings (a retention policy).
- Callers are identified by caller id only, so anyone calling from that number is treated as that user unless `AUTH_VERIFY_PIN=true`.
