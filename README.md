# Harvix AI Calling

A voice bot for Asterisk. You dial an extension from a SIP phone, and a bot answers and holds a conversation. It runs on your own machine: Whisper for speech-to-text, any OpenAI-compatible LLM server (I use vLLM with Gemma), and Supertonic for text-to-speech. No paid speech APIs.

```
SIP phone ──► Asterisk ──ARI──► Python app ──► Whisper ──► LLM ──► Supertonic ──► back to the call
```

I built it to test AI phone calls on a home LAN, and that's still the setting it's tuned for. It works, but it's a hobby project, not a contact-center product.

## What it does

- **Answers in the caller's language.** No "press 1 for English". Whisper detects the language each turn, and the bot replies in that language. Hindi and English get the most testing. Supertonic speaks about 30 languages; others fall back to a generic voice that sounds noticeably worse.
- **Knows when you've stopped talking.** Asterisk's built-in silence detection kept recording for 10–40 seconds on a noisy line. Instead, the app taps the caller's audio over RTP and runs Silero VAD on it, so a turn ends about 0.7 s after you stop.
- **Lets you interrupt.** If you talk over the bot, it stops mid-sentence and listens.
- **Remembers the call.** The conversation history is kept until hang-up, so "what was my name again?" works.
- **Answers from your own documents.** Put PDF, Word (.docx), TXT or Markdown files in `knowledge/`, and the bot answers from them when a question matches: prices, policies, opening hours, whatever you give it. Questions in Hindi still match English documents. See [Knowledge base](#knowledge-base) below.
- **Looks things up.** If you ask for something it can't reliably know (opening hours, exchange rates, flight times…), it does a quick web search before answering. Small talk doesn't trigger a search.
- **Knows returning callers.** Callers are looked up by caller id in PostgreSQL. A registered caller is greeted by name ("Welcome back, Shyam!"), and the bot uses the name during the call. Nobody is asked anything up front.
- **Registers callers only when needed.** When an unregistered caller wants to lodge a complaint, the bot asks for their name and a 4-digit PIN (spoken, or typed on the keypad), confirms it, saves them, and then takes the complaint. Only a salted hash of the PIN is stored, and it's never logged or sent to the LLM.
- **Records and logs every call.** Both voices are recorded and compressed to Opus (about 60 KB per minute), and every line said is saved. A small web page lets you search calls, play them and read the transcript.
- **Gives cleaner audio.** TTS output is loudness-normalized, and there's a 16 kHz version for phones that use G.722.

There's also a `/health` endpoint, and an optional log file (`LOG_FILE`) that records per-turn timing: speech-to-text, LLM, web search, TTS and playback.

## Requirements

- Linux with Asterisk 22 (PJSIP + ARI): either the Docker setup in `docker/` (easiest) or your own install. I've run 22.9 from source and 22.11 in Docker.
- Docker, for Redis (document search) and PostgreSQL (callers and call history). Both are optional: without them the bot uses a local index and treats every caller as a guest.
- Python 3.11 and `ffmpeg`.
- An NVIDIA GPU is strongly recommended. On an RTX 3070 laptop GPU, Whisper `medium` takes about 0.5 s per turn and Supertonic about 0.2 s.
- An LLM server with an OpenAI-style `/v1` API: vLLM, Ollama, llama.cpp server, etc. Small models are fine; this was tuned on `google/gemma-4-E2B-it`.
- A SIP softphone for testing: Linphone or Zoiper on the same Wi-Fi.

## Setup

The short version is below. [SETUP.txt](SETUP.txt) has every step, including the Asterisk permissions that usually trip people up.

```bash
git clone https://github.com/shyam302/Harvix-AI-Calling.git
cd Harvix-AI-Calling

python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # then edit: ARI_PASSWORD, VLLM_BASE_URL, VLLM_MODEL
```

1. Start Asterisk, Redis and PostgreSQL in Docker:

   ```bash
   cd docker && cp .env.example .env    # edit passwords; they must match the project .env
   docker compose up -d                 # asterisk + redis + postgres
   cd ..
   ```

   See [docker/README.md](docker/README.md). If you'd rather use a system Asterisk, copy `asterisk_snippets/` into `/etc/asterisk/` instead, include them from `pjsip.conf` and `extensions.conf`, and change the example passwords first. With the Docker Asterisk, set `ASTERISK_SOUNDS_EN=<project>/docker/data/sounds` in `.env` so Asterisk can find the bot's audio.
2. Make sure your LLM server answers: `curl $VLLM_BASE_URL/models`.
3. Start the bot:

   ```bash
   python -m ari_app.server
   curl -s http://127.0.0.1:8765/health
   ```

4. Register a softphone as `1001` with the server set to your machine's LAN IP, port 5060, **UDP** (Linphone defaults to TLS, which fails with "IOError"). Then dial **7001**.

`python tts/supertonic_demo.py` is a quick way to check that TTS works before you make a call.

## Configuration

Everything is set in `.env`; [.env.example](.env.example) lists every option with comments. The ones you'll most likely change:

| Variable | What it's for |
|---|---|
| `ARI_USERNAME` / `ARI_PASSWORD` | Must match `ari_user_callbot.conf` |
| `VLLM_BASE_URL` / `VLLM_MODEL` | Your LLM server (URL ends in `/v1`) |
| `CALL_PRIMARY_LANG` | Language of the greeting (`en` or `hi`); after that the bot follows the caller |
| `WHISPER_MODEL` | `medium` is a good balance; `large-v3` is more accurate but slower |
| `SUPERTONIC_VOICE_FEMALE` / `_MALE` | e.g. `F3`, `M1` |
| `REPLY_MAX_SENTENCES` | Longer replies sound more natural but take longer to play |
| `VAD_END_SILENCE_MS` | How long a pause ends your turn (default 700). Raise it if the bot cuts you off |
| `BARGE_IN_ENABLED` | Set to `false` if the bot keeps interrupting itself (see below) |
| `WEB_SEARCH_PROVIDER` | `fast` (free, DuckDuckGo), `serpapi` or `google` (need API keys) |
| `RAG_STORE` / `REDIS_URL` | `redis` keeps the document index in Redis; `local` keeps it in files |
| `DATABASE_URL` | PostgreSQL for callers and call history; leave it empty to turn both off |
| `REGISTRATION_ENABLED` / `AUTH_VERIFY_PIN` / `PIN_LENGTH` | Registration before a complaint; whether registered callers must say their PIN first |
| `CALL_RECORDING_ENABLED` / `RECORDINGS_DIR` / `RECORDING_OPUS_KBPS` | Whole-call recordings and where they go |

Never commit `.env`.

## Knowledge base

Drop files into `knowledge/` (subfolders are fine): `.pdf`, `.docx`, `.txt`, `.md`. On startup the bot splits them into chunks and embeds them with `intfloat/multilingual-e5-small`; it downloads about 470 MB the first time. Only new or changed files are re-indexed, and the folder is re-checked every 30 seconds while the bot runs, so you don't need to restart after adding a file.

On each turn, the caller's question is matched against the chunks. If something scores above `RAG_MIN_SCORE` (default 0.82), the top `RAG_TOP_K` chunks are given to the model and it answers from them, without a web search. Otherwise the call carries on as normal.

To check what a question would pull up:

```bash
python -m ari_app.rag ingest                       # build / update the index
python -m ari_app.rag search "what does the premium plan cost?"
```

With `RAG_STORE=redis`, the chunks and their vectors live in Redis (a RediSearch HNSW index), which keeps lookups at a few milliseconds as the collection grows. If Redis can't be reached, the bot falls back to the local index in `knowledge/.rag_index/`.

Scanned PDFs (images with no text layer) won't work, because there's no OCR. Contents of `knowledge/` are git-ignored, so private documents don't get pushed.

## Database support

Two databases run in Docker next to Asterisk. Both are optional: if one is down, the bot logs it once and carries on.

| Database | What it holds | Setting | Without it |
|---|---|---|---|
| **Redis Stack** | Knowledge-base chunks and their vectors (a RediSearch HNSW index) | `RAG_STORE=redis`, `REDIS_URL` | The local index in `knowledge/.rag_index/` |
| **PostgreSQL** | Registered callers, every call, every line said | `DATABASE_URL` | Every caller is a guest; no history or recordings list |

Start them with `cd docker && docker compose up -d redis postgres`. The tables are created automatically when the bot starts.

### Tables

| Table | One row per | Main columns |
|---|---|---|
| `users` | registered caller | `phone` (caller id, unique), `name`, `pin_hash`, `language`, `profile` (JSON answers), `last_call_at` |
| `calls` | call | `caller_number`, `user_id`, `started_at`, `duration_s`, `turns`, `language`, `status`, `recording_path` |
| `call_messages` | line said in a call | `call_id`, `role` (`user` or `assistant`), `content` |
| `complaints` | complaint | `ticket_no`, `call_id`, `user_id`, `caller_number`, `summary` (what the caller first said), `details` (JSON answers), `status` (`open`) |

`calls.status` is one of `guest`, `registered`, `new_registration`, `verified` or `pin_failed`.

### Callers and registration

- **Returning callers:** the caller id (for example `1001`) is looked up in `users` when the call starts. If it's found, the bot greets them by name and keeps using the name. Nobody is asked anything up front.
- **Registration happens only when it's needed:** when an unregistered caller wants to lodge a complaint ("complaint", "lodge", "शिकायत"…). The bot asks for their name, then a PIN twice. The caller can say the PIN or type it: four digits finish the entry, `#` ends it early, `*` clears. Then the bot goes back to the complaint.
- **Extra questions:** after the PIN, the bot asks the questions in [registration_questions.json](registration_questions.json) (by default city and an alternate phone number) and saves each answer to `users.profile` as soon as it's given. A question can be:
  - `text`: the LLM pulls the answer out of the reply,
  - `digits`: spoken, or typed with `#` at the end,
  - `choice`: one of a list of `options`.

  Callers can say "skip". To add a field, edit the JSON and restart the bot; no database change is needed.
- **PIN safety:** only a salted scrypt hash is stored. The PIN is never spoken back, logged, kept in chat history or sent to the LLM. Phone-number answers aren't logged either.
- **PIN checks:** `AUTH_VERIFY_PIN=true` makes registered callers say their PIN before a complaint. If the PIN fails, no complaint is taken.

### Complaints

After registration (or straight away for a registered caller), the bot asks the questions in [complaint_questions.json](complaint_questions.json), saves them as one row in `complaints`, and reads out a ticket number digit by digit ("…your ticket number is 1 0 0 0 0 3"). Then the conversation carries on normally.

- **Default questions:** `category` (a choice of order, delivery, payment, product quality, customer service or other), `order_number` (optional, spoken or typed) and `description` (type `free`: the caller's own words, kept as said).
- **Format:** the same as the registration questions, plus the `free` type. Edit the file and restart the bot to change them.
- **Guests too:** callers who skip registration can still lodge a complaint; it's saved against their caller number.
- **Where to see them:** in the database, or on the call's page at `/recordings`, which shows the ticket and the answers.

### Call history and recordings

- Every call gets a row in `calls`, and every line said goes to `call_messages`.
- Both voices are recorded and compressed to Opus (16 kbps mono, about 60 KB per minute, around 16 times smaller than the WAV), saved as `recordings/YYYY/MM/DD/<HHMMSS>_<caller>_<session>.ogg`. Change the folder with `RECORDINGS_DIR`, or turn recording off with `CALL_RECORDING_ENABLED=false`. The folder is git-ignored.
- To browse them, open **http://127.0.0.1:8765/recordings** while the bot is running. You can search by number, name or anything that was said, then play the call (with a speed control), download it, and read the transcript and the caller's profile. The page has no login and only listens on localhost, so keep it that way or add authentication.

### Looking at the data

```bash
docker exec -it callbot-postgres psql -U callbot
  SELECT id, phone, name, profile, last_call_at FROM users;
  SELECT id, caller_number, status, turns, round(duration_s) AS secs, started_at
    FROM calls ORDER BY id DESC LIMIT 10;
  SELECT role, content FROM call_messages WHERE call_id = 12 ORDER BY id;

docker exec -it callbot-redis redis-cli -a callbot FT.INFO rag_idx    # num_docs = chunks
```

[docs/DB_SUPPORT.md](docs/DB_SUPPORT.md) has the design, test results and what's not done yet.

## Known issues

- **Free web search is unreliable.** DuckDuckGo and similar engines rate-limit by IP, so searches sometimes fail. When that happens, the bot says it couldn't find the answer and stops searching for a minute, but answers are only as good as the search snippets. For anything serious, use a SerpAPI or Google Programmable Search key.
- **Barge-in relies on the phone's echo cancellation.** If the bot's own voice leaks back through your mic, it can interrupt itself. Raise `BARGE_IN_THRESHOLD` or `BARGE_IN_MIN_SPEECH_MS`, or turn barge-in off.
- **G.722 (HD audio) can sound slow and garbled.** G.722 advertises an 8 kHz clock although the audio is 16 kHz, and some softphones get this wrong. The Docker setup therefore offers `CODECS=ulaw,alaw,g722`. To try HD audio, set `CODECS=g722,ulaw,alaw` in `docker/.env` and enable G.722 in the softphone, then switch back if playback sounds off.
- **Small LLMs make things up.** The prompt pushes the model to search rather than guess, but a 2B model still gets things wrong now and then. Don't use this for anything where a wrong answer matters.
- **Callers are identified by caller id.** Anyone calling from a registered number is treated as that person unless `AUTH_VERIFY_PIN=true`. There's no PIN reset yet.
- **Complaints can't be followed up yet.** Tickets are created with status `open`, but there's no way to update them, and the bot can't yet answer "what's the status of ticket 100003?".
- **Recordings are kept forever.** There's no retention or automatic clean-up.
- **Only tested on a LAN.** Calling over the internet or Tailscale needs `external_media_address` set up in PJSIP (there are notes in the snippet), and you should lock down SIP and ARI first.
- **Asterisk installed twice.** If you have both the distro Asterisk package and a source build, make sure the binary loads its own `libasteriskpj`. A mismatched PJSIP library crashed Asterisk on every SIP REGISTER here.

## Security

The configs in `asterisk_snippets/` and `docker/.env.example` use example passwords (`1001`/`1001`, `callbot` for Redis and PostgreSQL). Change them before the box is reachable from anywhere you don't trust. Keep ARI, Redis, PostgreSQL and the bot's web server (recordings page) bound to `127.0.0.1`. The Docker setup does this. Only open SIP (5060/udp) and RTP (10000–20000/udp) to the networks that need them.

## Project layout

```
ari_app/              the bot: ARI event loop, call session, STT, LLM, TTS, live VAD, web search, RAG,
                      database (db.py), registration, call recording, recordings page (static/)
docker/               Asterisk + Redis + PostgreSQL in Docker
knowledge/            your documents for the knowledge base (git-ignored)
recordings/           compressed call recordings (git-ignored)
asterisk_snippets/    example configs for a system Asterisk
docs/                 call flow walkthrough (CALL_FLOW.md), database notes (DB_SUPPORT.md)
registration_questions.json   extra questions asked when a caller registers
complaint_questions.json      questions asked when a caller lodges a complaint
tts/                  TTS test scripts
```

[docs/CALL_FLOW.md](docs/CALL_FLOW.md) walks through one call from dialing to hang-up, naming the function that does each step. [ARCHITECTURE.md](ARCHITECTURE.md) is the older overview.

## License

MIT. See [LICENSE](LICENSE).
