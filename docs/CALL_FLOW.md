# How a call flows through the code

This follows one phone call from start to finish: you dial 7001, the bot looks you up, greets you, you talk, it answers, you hang up, and the call is saved. Each step names the file and function that does the work, so you can open the code next to this guide. Line numbers are approximate and will drift as the code changes.

---

## The big picture

```
 Phone (Linphone)
    │  SIP (port 5060) + RTP audio
    ▼
 Asterisk (Docker or system)
    │  dialplan: 7001 → Stasis(callbot)
    │
    │  ARI = HTTP commands + a WebSocket of events  (port 8088)
    ▼
 Python bot  (python -m ari_app.server)
    ├── ari_loop.py      listens for events, creates one CallSession per call
    ├── call_session.py  runs the conversation: listen → think → speak, repeat
    ├── live_vad.py      hears when the caller starts and stops talking
    ├── stt.py           speech → text (faster-whisper)
    ├── rag.py           searches your documents in knowledge/
    ├── web_search.py    searches the internet when needed
    ├── llm.py           text → reply (Gemma on vLLM)
    ├── conversation.py  cleans the reply for speaking
    ├── tts.py           reply → WAV file (Supertonic) that Asterisk plays
    ├── db.py            PostgreSQL: users, calls, call_messages
    ├── registration.py  complaint intent, spoken name/PIN, registration lines
    ├── rag_redis.py     RAG chunks + vectors in Redis Stack
    └── call_recording.py whole-call recording → Opus file on disk

 Docker (docker/docker-compose.yml): asterisk · redis (RAG) · postgres (users + history)
 Web page: http://127.0.0.1:8765/recordings (list, play, transcript)
```

**Key idea:** Asterisk handles the phone line and the audio. Python never touches the phone directly. It sends commands to Asterisk over HTTP (answer, play this file, hang up) and receives events from Asterisk over a WebSocket (a call arrived, a playback finished, the caller hung up).

---

## Stage 0: Before any call, the bot starts up

You run:
```bash
python -m ari_app.server
```

### `ari_app/server.py`

| Code | What it does |
|---|---|
| `main()` (~line 155) | Starts **uvicorn**, a web server, on `127.0.0.1:8765`. It only serves `/health`; the calls themselves don't go through it. |
| `lifespan()` (~line 71) | Runs once at startup, before any call can come in. |
| `_configure_logging()` | Sets up logging to the console and `logs/callbot.log`. |
| `load_settings()` (from `config.py`) | Reads every setting from `.env`: ARI login, vLLM address, voices, timings. |
| `configure_inference_pool()` (`inference_pool.py`) | Creates a thread pool and limits how many STT and TTS jobs run at once (for example 3 Whisper jobs), so several calls don't overload the GPU. |
| `_warmup_models()` (~line 86) | Loads every model **now**, so the first caller doesn't wait: Supertonic TTS, Whisper STT, the gender model, Silero VAD, and the RAG embedding model (which also indexes `knowledge/`). |
| `get_knowledge_base()` (`rag.py`) | With `RAG_STORE=redis` this returns `RedisKnowledgeBase` (`rag_redis.py`), which indexes `knowledge/` into Redis. If Redis can't be reached, it uses the local numpy index. |
| `check_vllm_reachable()` (`llm.py`) | Checks that the LLM server answers. If it doesn't, the bot logs an error but keeps running. |
| `asyncio.create_task(run_ari_forever())` | Starts the main event loop from `ari_loop.py` in the background. |

### `ari_app/ari_loop.py`: `run_ari_forever()` (~line 44)

1. `db.init(DATABASE_URL)` opens the PostgreSQL pool and creates the tables if they're missing (`users`, `calls`, `call_messages`). If Postgres is down, it logs once and every call runs as a guest with no history.
2. Builds the ARI addresses:
   - REST: `http://127.0.0.1:8088/ari` (for sending commands)
   - WebSocket: `ws://127.0.0.1:8088/ari/events?app=callbot&api_key=user:pass` (for receiving events)
3. Opens **one shared `httpx.AsyncClient`**, reused for every HTTP call to Asterisk.
4. **Connects the WebSocket.** From this moment Asterisk knows an app named `callbot` exists (`ari show apps` lists it).
5. Loops forever: `async for raw in ws:` → `json.loads` → `dispatch(ev)`.
6. If the connection drops (Asterisk restarted, for example), it waits 2 s and reconnects. A 401 error means the ARI user or password is wrong, so it exits with a hint.

The bot is now idle, waiting for events.

---

## Stage 1: You dial 7001

This part happens entirely inside Asterisk:

1. Linphone sends a SIP `INVITE` for `7001` to Asterisk.
2. `pjsip.conf` checks the caller's username and password, and uses endpoint `1001` with `context=from-internal-app`.
3. `extensions.conf`:
   ```
   exten => 7001,1,NoOp(Voice LLM via ARI)
    same => n,Stasis(callbot)
   ```
   `Stasis(callbot)` means "hand this call to the ARI app named callbot".
4. Asterisk sends a **`StasisStart`** event over the WebSocket. It contains the channel (the call): its `id`, the caller number and the dialed number.

---

## Stage 2: The bot receives the call

### `ari_loop.py`: `dispatch()` → `StasisStart` branch (~line 73)

```python
if et == "StasisStart":
    cid = ch.get("id")
    if cid.startswith(INTERNAL_CHANNEL_PREFIXES):
        return          # our own helper channels (see Stage 3), not a real caller
    caller_number, caller_name, dialed = stasis_channel_peer_info(ch)
    ctx = CallContext.create(...)          # memory for this call
    registry.register(ctx)                 # shows up in /health
    session = CallSession(...)             # the worker that runs the call
    sessions[cid] = session
    asyncio.create_task(_task())           # run_loop() in the background
```

- **`CallContext`** (`call_context.py`) is the call's *memory*: chat history (`add_user`, `add_assistant`, `history_for_llm`), current language (`preferred_lang`), caller gender guesses, the last web search result, and the database ids (`call_db_id`, `user_id`, `user_status`). One per call. Its `message_listener` hook is called for every line added to the history, which is how the call log reaches the database.
- **`CallSession`** (`call_session.py`, class at ~line 210) is the call's *worker*. It holds the channel id, the HTTP client and the settings, and has all the "do something" methods.
- **`asyncio.create_task`** runs each call as its own async task. That's how several calls run at the same time without blocking each other.
- The `finally:` in `_task()` removes the session from `sessions` and the registry when the call ends, however it ends.

### The other events `dispatch()` handles

| Event from Asterisk | What the bot does |
|---|---|
| `PlaybackFinished` | `session.on_playback_finished()` marks that audio clip as done, so `play_sound()` can continue. |
| `RecordingFinished` / `RecordingFailed` | `session.on_recording_finished()` is only used by the fallback recording path (Stage 4b). |
| `ChannelDtmfReceived` | A keypad key was pressed. `session.on_dtmf(digit)` buffers it (used for the PIN). |
| `ChannelDestroyed` | The caller hung up. `session.mark_channel_dead()` stops the conversation loop. |

**How waiting works.** When the bot plays audio, it creates an `asyncio.Future` (a "promise") and waits on it. When `PlaybackFinished` arrives, `dispatch` fulfils that promise. That's how an HTTP command ("play") gets paired with a later WebSocket event ("finished").

---

## Stage 3: Setting up the call

### `call_session.py`: `run_loop()` (~line 1995), the heart of the call

```python
await self.answer()                  # POST /channels/{id}/answer → the call connects
await self._setup_live_vad()         # start live "is the caller talking?" detection
await self._db_call_started()        # INSERT INTO calls … → ctx.call_db_id
await self._start_call_recording()   # record caller + bot for the whole call
opening = await self._identify_caller()   # registered? greet by name
await self.say_text(opening, is_opening=True)
while not self._closed:
    ... one conversation turn ...
```

### `answer()` (~line 380)
A single ARI command: `POST /channels/{id}/answer`. The phone shows the call as connected.

### `_setup_live_vad()` (~line 455): copying the caller's audio to Python

Asterisk normally keeps the audio to itself. To hear the caller live, the bot builds a small pipeline:

```
caller ──► Snoop channel (copy of what the caller says, spy=in)
                 │
          mixing bridge  cbvad-xxxx
                 │
          ExternalMedia channel ──RTP/UDP──► 127.0.0.1:<port>  (Python LiveVad)
```

| ARI call | Purpose |
|---|---|
| `live_vad.open_listener()` | Opens a UDP port in Python that receives the audio. |
| `POST /channels/{id}/snoop` (`spy=in`) | Makes `cbsnoop-…`, a copy of the caller's voice only, without the bot's. |
| `POST /channels/externalMedia` | Makes `cbmedia-…`, which sends audio as RTP packets (`slin16` = 16 kHz raw PCM) to the Python port. |
| `POST /bridges` + `addChannel` | Joins the two, so the snooped audio flows into the external media channel. |

These helper channels also enter Stasis and trigger `StasisStart`. That's why `dispatch()` ignores channel ids that start with `cbsnoop-`, `cbmedia-` or `cbrec-` (`INTERNAL_CHANNEL_PREFIXES` in `live_vad.py`).

If any step fails, VAD is turned off and the bot falls back to Asterisk's own recording (Stage 4b).

### `live_vad.py`: the `LiveVad` class (~line 62)

- `datagram_received()` is called for every RTP packet (~20 ms of audio). It strips the 12-byte RTP header and feeds the samples to `_feed()`.
- `_feed()` / `_on_window()` cut the audio into 512-sample windows (32 ms) and ask **Silero VAD** "is this speech?", which returns a probability.
- It has **modes**:
  - `listen`: waiting for the caller's turn. It records speech, and when there's silence for `VAD_END_SILENCE_MS` (700 ms) it sets the `ended` event.
  - `barge`: the bot is speaking. If the caller starts talking, it sets `speech_started`, which cuts the bot off.
  - `off`: ignore audio.
- `_tick()` is a wall-clock timer. Asterisk stops sending packets during silence, so "no packets" must also count as silence.
- `utterance_pcm()` returns the recorded speech, including 0.5 s from just before speech started, so the first syllable isn't lost.

### `_db_call_started()` (~line 1747): the call row
`db.start_call()` inserts a row into `calls` (session id, channel, caller number, dialed number, start time) and keeps its id in `ctx.call_db_id`. From then on, `_log_message_to_db()` (~line 1739), the `CallContext.message_listener` hook, writes each history line to `call_messages` in a background task, so the call never waits for the database.

### `_start_call_recording()` (~line 1650): record the whole call
```
caller + bot ──► Snoop channel cbrec-xxxx (spy=both) ──► POST /channels/cbrec-xxxx/record
                                                         name=call-<session>, format=wav
```
Asterisk writes the WAV into its spool folder for as long as the call lasts. It's saved and compressed at hang-up (Stage 10). If it can't start, the call carries on without a recording.

### `_identify_caller()` (~line 1948): who is calling?
```python
user = await db.get_user_by_phone(self.caller_number)
if user:                                   # registered
    ctx.user_id, ctx.caller_name = user.id, user.name
    ctx.user_status = "registered"
    return "Welcome back, {name}! How can I help you today?"
return voice_select.opening_greeting_for_call(...)   # unknown: normal greeting
```
**No questions are asked at the start.** For a registered caller, `ctx.caller_name` and `caller_context_line()` ("registered customer") go into every LLM prompt, so the bot uses their name naturally (see `build_system_prompt()` in `llm.py`). Registration only happens when the caller wants to lodge a complaint (Stage 5b).

---

## Stage 4: Listening to the caller (one turn)

Every pass through the `while not self._closed:` loop in `run_loop()` is one turn of the conversation.

### `_prepare_to_listen()` (~line 408)
- Waits for any bot audio still playing to finish (`_play_futs`).
- Sleeps a short gap (`PAUSE_AFTER_TTS_MS` + `PAUSE_BEFORE_LISTEN_MS`) so the end of the bot's own voice isn't recorded as the caller.
- If the caller already **barged in**, it skips the wait and listens right away.

### 4a. `_capture_turn_vad()` (~line 541), the normal path

```python
vad.to_listen()                                       # switch to listen mode
await asyncio.wait_for(vad.ended.wait(), timeout=45)  # wait until they stop talking
pcm = vad.utterance_pcm()
vad.arm("barge")                                      # watch for interruptions while we think and speak
# write pcm to a temporary .wav file
return path, seconds
```

- If the caller said nothing, it returns `(None, 0.0)`. `run_loop` calls `_skip_turn_no_usable_speech()` and listens again. After 8 empty turns in a row (`RECORD_MAX_CONSECUTIVE_EMPTY`), the bot gives up.
- If **no audio packets arrived at all**, the stream is broken. It tears VAD down and returns `None`, which switches to 4b.

### 4b. Fallback: Asterisk recording
`start_recording()` → `POST /channels/{id}/record` (Asterisk stops after `RECORD_MAX_SILENCE_SECONDS` of silence) → `wait_recording()` waits for the `RecordingFinished` event → `fetch_recording_wav()` downloads `GET /recordings/stored/{name}/file` → `delete_stored_recording()`.

It's slower and less accurate than VAD, but it always works.

### Filters before STT
- A WAV that's too short (< `RECORD_MIN_DURATION_SECONDS`) is skipped. `_is_likely_line_noise_clip()` stops echo from the bot's own audio counting as an empty turn.
- `_pause_before_response()` waits a tiny optional pause so the reply doesn't feel abrupt.

---

## Stage 5: Speech → text

### `call_session._transcribe_turn()` (~line 757)
Runs two jobs **in parallel**:
- `stt.transcribe_wav()` turns the speech into text, and
- a pitch estimate of the caller's voice (used for gender-aware Hindi grammar).

### `stt.py`: `transcribe_wav()` (~line 277)
- Converts the WAV to 16 kHz mono if needed (`_ffmpeg_to_16k_mono_wav`).
- Runs **faster-whisper** (`medium`, float16 on CUDA) in the inference thread pool, so the async loop isn't blocked.
- Returns `(text, language, language_probability)`. Whisper detects the language automatically.

### Back in `run_loop()`
| Code | Why |
|---|---|
| `_drop_likely_silence_hallucination()` (~line 154) | Whisper sometimes "hears" phrases like "Thank you for watching" in silence. Known junk is dropped. |
| `_start_gender_model_in_background()` | A small model guesses the caller's gender from the audio without slowing the reply. |
| `ctx.set_lang_from_user()` → `lang.detect_reply_lang()` | Decides the **reply language**. It switches only when the caller clearly speaks another language (enough words, Whisper confident enough), or asks for one ("speak Hindi"). English is the default. |
| `log.info("turn=… user: …")` | The line you see in the log with what the caller said. |

---

## Stage 5b: Complaint? Register first, then take it

### `_complaint_gate(user_text)`
It runs on every turn, right after STT, and does something **at most once per call**:

```python
if not registration.wants_complaint(user_text):   # "complaint", "lodge", "शिकायत", …
    return False                                   # normal turn
if ctx.user_id is None:                 # unknown caller → register (can be skipped)
    await self._register_new_caller()   # name, PIN, registration_questions.json
elif AUTH_VERIFY_PIN:                   # known caller → check PIN; fail = no complaint
    await self._verify_returning_caller(user)
await self._take_complaint(user_text)   # complaint_questions.json → complaints table
return True                             # turn handled: no LLM reply this time
```

### `_take_complaint(user_text)`
- Asks each question in `complaint_questions.json` through `_ask_question()`:
  - `choice`: the LLM maps "it's about delivery" to `delivery`.
  - `digits`: spoken, or typed with `#`.
  - `free`: the caller's own words.
  - `text`: a short value the LLM extracts.
- Saves everything with `db.create_complaint()`. The ticket is `100000 + id`.
- Reads it out digit by digit: "Your complaint is registered. Your ticket number is 1 0 0 0 0 3."
- Adds a short note to chat history, "(Complaint registered, ticket 100003, category: delivery…)", so the LLM knows about it for the rest of the call.

### `_register_new_caller()` (~line 1866)
1. **Name:** `_ask("…What's your name?")` → `_hear_once()` → `registration.extract_name()`. This is a small JSON call to the LLM ("Yeah hi, this is Shyam here" → "Shyam"), with simple patterns as a fallback. Up to 3 tries.
2. **PIN:** `_ask_pin()`, then `_ask_pin("say it once more to confirm")`. If the two don't match, it asks again.
3. **Save:** `db.create_user(phone, name, pin)` stores a **scrypt hash** (`db.hash_pin`), and `db.set_call_user(call_id, user_id, "new_registration")` links the call.
4. **Profile:** `_ask_profile_questions()` asks `registration_questions.json` (city, alternate phone…) and saves each answer at once with `db.update_user_profile()` into `users.profile` (JSON).
5. If the caller gives up, `"No problem, we can finish the registration later."` and the call continues as a guest.

Registration prompts are spoken with `record_in_session=False` and `trim=False`: they're never added to chat history and are never shortened. The answers (name and PIN) are never added to history either.

### `_ask_pin()` (~line 1818): spoken **or** typed
```python
self._dtmf_digits.clear()
await self.say_text(prompt)                    # keys pressed during the prompt still count
voice = create_task(self._hear_once(stt_prompt="PIN digits: …"))
keys  = create_task(self._wait_dtmf_pin(n))    # n keypad digits, or "#"
done = await asyncio.wait({voice, keys}, return_when=FIRST_COMPLETED)
```
- **Keypad:** Asterisk sends `ChannelDtmfReceived` → `dispatch()` → `on_dtmf()` (~line 1622) appends the digit and wakes `_wait_dtmf_pin()`. With 4 digits the entry is done (`#` ends early, `*` clears). The voice task is cancelled (`_cancel_listen`).
- **Voice:** `_hear_once()` (~line 1766) records one answer and runs Whisper with a digits hint, then `registration.parse_pin()` turns "one two three four", "1 2 3 4", "double five…" or "ek do teen char" into `1234`.
- The PIN is never logged or spoken back. Logs show only "PIN entered on keypad" or "keypad key received (n buffered)".

### `_verify_returning_caller()` (~line 1925)
Only when `AUTH_VERIFY_PIN=true`: up to 3 tries, `db.verify_pin(pin, user.pin_hash)` (constant-time compare), status `verified` or `pin_failed`.

---

## Stage 6: Finding facts (documents first, then the web)

```python
web_context = await self._knowledge_context(user_text)      # 1. your documents
if web_context is None:
    web_context, ... = await self._maybe_fetch_web_context(...)  # 2. obvious live questions
```

### 1. `_knowledge_context()` (~line 1084) → `rag.py` / `rag_redis.py`
- `kb.maybe_refresh_in_background()` re-indexes `knowledge/` if files changed. Only changed files are re-embedded.
- Short follow-ups ("and the price?") get the previous question added, so the search knows the topic.
- `kb.search()` embeds the question with `multilingual-e5-small` and keeps matches above `RAG_MIN_SCORE`:
  - **Redis** (`RAG_STORE=redis`): a `KNN` query on the HNSW index `rag_idx` (cosine) inside Redis, plus a small word-overlap bonus. Chunks are stored as hashes `rag:chunk:<id>` (text, source, page, embedding).
  - **Local:** a numpy dot product against every chunk, plus a BM25 keyword bonus.
- `rag.format_context()` turns the matching chunks into reference text for the LLM.

### 2. `_maybe_fetch_web_context()` (~line 1343) → `web_search.py`
Only for obviously real-time topics (gold price, petrol, news, weather…) caught by keyword rules. It can reuse a recent search for follow-up questions.

### 3. The model decides (inside `_reply_with_lookup`, Stage 7)
For everything else, the LLM itself judges whether it needs to look something up.

---

## Stage 7: Thinking of a reply (LLM)

### `_reply_with_lookup()` (~line 1198)

```python
lookup = create_task(self._model_requested_lookup(user_text))   # "do I need to search?"
draft  = create_task(llm.reply_text(**kw, web_context=None))    # normal reply, at the same time
query = await lookup
if not query:
    return await draft          # no search needed: the draft is ready, no extra time
draft.cancel()
web_context = await self._search_for_model(query)               # search silently
return await llm.reply_text(**kw, web_context=web_context)      # answer with the facts
```

Running both at once means turns that need no search cost no extra time.

| Function | File | What it does |
|---|---|---|
| `_reply_kwargs()` | call_session.py ~1114 | Collects everything the LLM needs: history, language, caller info, limits. |
| `llm.web_lookup_query()` | llm.py ~178 | A tiny LLM call that answers `NO` or `SEARCH: <query>`. |
| `_search_for_model()` | call_session.py ~1161 | Runs `_web_search()` (DuckDuckGo with fallbacks) with a 5 s budget. If it fails, it tells the LLM to say honestly that it couldn't find it, not to guess. |
| `llm.reply_text()` | llm.py ~318 | The main LLM call to vLLM's `/chat/completions`. |
| `_chat_messages()` / `build_system_prompt()` | llm.py ~277 / ~125 | Builds the prompt: phone-agent personality, rules (short replies, the right language), caller info, plus the RAG or web facts "as if you simply know them". |
| `_alternating_history()` | llm.py ~255 | Gemma requires turns to alternate user/assistant/user… This merges or fixes the history so it does. |

### Cleaning the reply (back in `run_loop`)
- `llm.strip_search_lines()` removes any leftover "SEARCH:" text.
- `conversation.strip_leading_filler()` removes "Umm," and similar at the start.
- `conversation.strip_robotic_phrases()` removes phrases like "As an AI…".
- `conversation.align_hindi_bot_grammar()` fixes Hindi verb gender (for Hindi only).

---

## Stage 8: Speaking the reply (TTS + playback)

### `say_text()` (~line 1487)

1. **Trim:** `conversation.trim_reply_for_phone()` keeps at most `REPLY_MAX_SENTENCES` / `REPLY_MAX_CHARS`. Phone replies must be short. Fixed prompts pass `trim=False`.
2. **Split:** `conversation.split_tts_phrases()` cuts the reply into chunks of about one sentence each.
3. **Choose a voice and language:** `tts.pick_supertonic_lang()`, `_agent_voice_pair()`.
4. **Pipeline loop:** while chunk *i* plays, chunk *i+1* is already being made:
   ```python
   for i in range(n):
       synthesize chunk i (or wait for the one started earlier)
       start synthesizing chunk i+1 in the background
       interrupted = await self._play_wav(part_paths[i])
       if interrupted: break        # caller barged in
   ```
5. **Remember only what was actually heard.** If the caller cut in, the history gets the spoken part plus " …", so the LLM doesn't think it said things the caller never heard.
6. Logs `TTS pipeline chunks=… synth=… play=…`.

### `tts.synthesize_to_wav()` (tts.py ~476)
- `conversation.prepare_text_for_tts()` makes text speakable: numbers into words, phone numbers digit by digit, emoji removed.
- **Supertonic** (ONNX on the GPU) generates the voice.
- `_ffmpeg_to_asterisk_wav()` converts it for the phone line: trims trailing silence, normalises loudness, band-limits, writes 8 kHz `.wav` and 16 kHz `.wav16` (for G.722).
- The file lands in `ASTERISK_SOUNDS_EN/custom/tts-<random>.wav`. **Asterisk must be able to read this folder.** That's why Docker mounts `docker/data/sounds`.

### `_play_wav()` (~line 931) → `play_sound()` (~line 678)
```python
r = await self._post(f"/channels/{id}/play", json={"media": "sound:custom/tts-abc"})
fut = loop.create_future(); self._play_futs[pb_id] = fut
done = await asyncio.wait({fut, barge}, return_when=FIRST_COMPLETED)
```
- Asterisk plays the file to the caller. Note there's no file extension: Asterisk picks the best format itself.
- The bot waits for **whichever comes first**:
  - `PlaybackFinished` arrives → `dispatch` → `on_playback_finished()` → `fut` is done, so the bot continues;
  - **or** the VAD detects the caller talking (`speech_started`), so the bot sends `DELETE /playbacks/{id}` to stop playback and returns `True` (barge-in).
- `tts.remove_tts_outputs()` then deletes the WAV files.

---

## Stage 9: End of the turn

Back in `run_loop()`:
```python
self.ctx.add_user(user_text)          # history + call_messages row (via message_listener)
self.ctx.add_assistant(reply)         # history + call_messages row: what the caller heard
_log_turn_latency_breakdown(...)      # "stt=0.5s llm=0.6s tts=0.3s … bottleneck=llm"
```
`finally: wav_path.unlink()` deletes the caller's temporary recording. Then the loop goes back to **Stage 4** and listens again.

### Error handling inside a turn
| Exception | What happens |
|---|---|
| `httpx.HTTPError` | The channel is gone (usually a hangup), so the loop ends. |
| `APITimeoutError` / `APIConnectionError` | vLLM is unreachable. The bot says a "sorry, can't think right now" phrase and keeps the call. |
| any other `Exception` | Logged. The bot says "Sorry, something went wrong. Goodbye." and ends the call. |

---

## Stage 10: Hang-up and cleanup

**When the caller hangs up:**
1. Asterisk sends `ChannelDestroyed`.
2. `dispatch()` → `session.mark_channel_dead()` sets `_closed = True` and cancels any waits.
3. `run_loop()` leaves the `while` loop.

**When the bot ends the call** (too many silent turns, a fatal error), `run_loop()` breaks out of the loop itself.

**Either way, `run_loop()`'s outer `finally:` runs:**
```python
await self._teardown_live_vad()      # close the UDP port, DELETE snoop/externalMedia channels and the bridge
await self.hangup()                  # DELETE /channels/{id} (does nothing if already gone)
await self._db_call_ended()          # wait for pending message writes; UPDATE calls: ended_at, duration, turns, status
await self._finish_call_recording()  # save the recording (below)
```

### `_finish_call_recording()` (~line 1685)
1. `POST /recordings/live/call-<session>/stop` stops the recording, and `DELETE` removes the `cbrec-` snoop channel.
2. Downloads the WAV with `GET /recordings/stored/call-<session>/file`, retrying briefly because Asterisk finalises the file a moment after the stop.
3. `call_recording.compress_to_opus()` runs ffmpeg: mono, `libopus`, `RECORDING_OPUS_KBPS` (16 kbps), `-application voip`. In the test, a 22 s call went from **338 KB WAV to 21 KB Opus**, about 16 times smaller (roughly 60 KB per minute).
4. Saves it to `RECORDINGS_DIR/YYYY/MM/DD/<HHMMSS>_<caller>_<session>.ogg`, and `db.set_call_recording()` stores the relative path and size in `calls`.
5. Deletes Asterisk's stored WAV.

Then `_task()` in `ari_loop.py` removes the session from `sessions` and the registry. The in-memory `CallContext` is discarded, but the call, its transcript and its recording stay in PostgreSQL and on disk.

---

## After the call: the recordings page

`server.py` (the same uvicorn app as `/health`) serves:

| Route | What it does |
|---|---|
| `GET /recordings` | The page (`ari_app/static/recordings.html`): a searchable call list, an audio player, and the transcript as a chat. |
| `GET /api/calls?q=&limit=&offset=` | `db.list_calls()`: newest first, with the registered name. `q` searches the caller number, the name, and anything said. |
| `GET /api/calls/{id}` | `db.get_call()`: call details and all messages. The disk path is never sent to the browser. |
| `GET /api/calls/{id}/audio[?download=true]` | Streams the `.ogg`. `call_recording.resolve_inside()` refuses any path outside `RECORDINGS_DIR`. |

Open `http://127.0.0.1:8765/recordings` while the bot is running. It's bound to localhost only.

---

## One turn on a timeline

```
caller speaks ───────┐ 0.7 s silence
                     ▼
 VAD "ended" → WAV → Whisper (~0.5 s) → RAG/web check → LLM (~0.6 s)
                                                          │
            Supertonic chunk 1 (~0.3 s) → Asterisk plays chunk 1 ─► caller hears
                          chunk 2 synthesized meanwhile ─► plays next
                                                          │
       (caller talks over it? → VAD barge → DELETE playback → listen now)
                                                          ▼
                                              back to listening
```

## Where to look when something goes wrong

| Symptom | Look at |
|---|---|
| Call drops right after dialing 7001 | Bot not running or not connected (`ari show apps`); `run_ari_forever()` log lines |
| Connected but silence | `ASTERISK_SOUNDS_EN` doesn't match the folder Asterisk reads; Asterisk log "File … does not exist" |
| Bot doesn't hear you | `_setup_live_vad()` log "live VAD on…"; "no caller audio on the VAD stream" means an RTP/NAT problem |
| Slow replies | The `turn=N latency breakdown` log line shows which part (stt / llm / tts / web) was slow |
| Wrong language | `set_lang_from_user()` and the `lang.py` thresholds (`LANG_SWITCH_MIN_WORDS`, `LANG_SWITCH_MIN_PROB`) |
| Wrong or invented facts | `_knowledge_context()` (RAG score), `web_lookup_query()`, and the prompt in `build_system_prompt()` |
| Registered caller not greeted by name | `DATABASE_URL`, the "PostgreSQL ready" log line, and whether `users.phone` matches the caller id exactly (`pjsip show endpoints`) |
| Registration never starts | `registration.wants_complaint()` keywords; the "complaint from unregistered caller" log line |
| PIN not understood | Log "PIN not understood"; try the keypad; `registration.parse_pin()` |
| No recording / "could not start" | Asterisk needs `/var/spool/asterisk/recording` (created by `docker/render.sh`); `CALL_RECORDING_ENABLED`; the "call recording saved" log line |
| Recordings page says "Database not connected" | `docker compose up -d postgres`, then restart the bot |
