# Running everything in Docker (one container per function)

Branch: `dockerize`

```
                         ┌──────────── host network (127.0.0.1) ─────────────┐
 phone ──SIP/RTP──► asterisk ◄──ARI──► bot ──HTTP──► stt     :8101  Whisper (GPU)
                         │             │  └─HTTP──► tts     :8102  Supertonic + ffmpeg (GPU)
                         │             │  └─HTTP──► embed   :8103  e5 embeddings (CPU)
                         │             │  └─HTTP──► gender  :8104  gender model (CPU)
                         │             ├──────────► redis   :6379  RAG vectors
                         │             ├──────────► postgres:5432  users · calls · complaints
                         │             └──HTTP────► vLLM (remote, or the optional "llm" container)
                         └─ plays sounds written by bot (docker/data/sounds)
 browser ──► web :8765  /recordings · /customers   (reads postgres + recordings/)
```

| Container | Image (target in `docker/Dockerfile`) | What it does | GPU | Port |
|---|---|---|---|---|
| `asterisk` | `andrius/asterisk:22.11.0` | SIP, RTP, ARI | – | 5060/udp, 8088 |
| `redis` | `redis/redis-stack-server` | RAG vector index | – | 6379 |
| `postgres` | `postgres:17-alpine` | users, calls, messages, complaints | – | 5432 |
| `stt` | `harvix/stt` (`stt`) | speech-to-text, faster-whisper `medium` | ✅ | 8101 |
| `tts` | `harvix/tts` (`tts`) | Supertonic + phone-audio conversion | ✅ | 8102 |
| `embed` | `harvix/ml` (`ml`) | multilingual-e5-small embeddings | CPU | 8103 |
| `gender` | `harvix/ml` (`ml`) | caller gender (Wav2Vec2) | CPU | 8104 |
| `bot` | `harvix/app` (`app`) | the call logic: ARI, live VAD, LLM, DB, recordings | – | 8766 (`/health`) |
| `web` | `harvix/app` (`app`) | `/recordings`, `/customers` pages + API | – | 8765 |
| `llm` (optional) | `vllm/vllm-openai` | local Gemma instead of the remote vLLM | ✅ | 8000 |

All ports are bound to `127.0.0.1`, except Asterisk's SIP/RTP for the phones.

## Start / stop

```bash
cd docker
cp .env.example .env              # first time (Asterisk/DB passwords, codecs)
docker compose build              # first time, ~10–20 min (CUDA libraries are large)
docker compose up -d              # everything
docker compose ps                 # all should be "healthy" / "running"
docker compose logs -f bot        # follow the calls
docker compose down               # stop everything (data in volumes/folders is kept)
```

The bot needs the project's `../.env`, which is the same file used when running on the host, for vLLM, ARI and all the call settings. The compose file only adds the container-specific values on top: the service URLs, `ASTERISK_SOUNDS_EN=/sounds` and the ports.

**Don't run the bot twice.** If `python -m ari_app.server` is running on the host, stop it before `docker compose up -d bot`. Both would connect to the same ARI app and answer the same calls.

To run only the infrastructure in Docker and the bot on the host (the old way):

```bash
docker compose up -d asterisk redis postgres
cd .. && python -m ari_app.server        # loads the models in-process
```

## How the split works

The bot code is unchanged. Each AI module checks for a URL and, when it's set, sends its work to that container instead of loading the model ([ari_app/remote.py](../ari_app/remote.py)):

| Variable (set for `bot` in docker-compose.yml) | Module | Without it |
|---|---|---|
| `STT_SERVICE_URL=http://127.0.0.1:8101` | `ari_app/stt.py` `transcribe_wav()` | Whisper in-process |
| `TTS_SERVICE_URL=http://127.0.0.1:8102` | `ari_app/tts.py` `synthesize_to_wav()` | Supertonic in-process |
| `EMBED_SERVICE_URL=http://127.0.0.1:8103` | `ari_app/rag.py` `_embed()` | sentence-transformers in-process |
| `GENDER_SERVICE_URL=http://127.0.0.1:8104` | `ari_app/gender_model.py` `classify_caller_gender()` | Wav2Vec2 in-process |

The service containers run the **same** `ari_app` code for their function, behind a small FastAPI app in [services/](../services/):

| Service | API |
|---|---|
| `stt_service.py` | `POST /transcribe` (multipart `audio`, `model`, `language`, `initial_prompt`) → `{text, language, probability}` |
| `tts_service.py` | `POST /synthesize` (JSON, the same arguments as `synthesize_to_wav`) → `{wav, wav16}` (base64) |
| `embed_service.py` | `POST /embed` `{texts, kind, model}` → `{vectors}` |
| `gender_service.py` | `POST /classify` (multipart `audio`, `min_confidence`) → `{gender, p_female, p_male}` |
| every service | `GET /health` → `{ready, info}`, which is also the Docker healthcheck |

At startup the bot waits until every configured service says `ready`, then connects to ARI. It starts in about 1 second because it loads no models.

`web` is the same image as `bot` with `ARI_ENABLED=false`: it serves the pages and API, but loads no models and takes no calls.

## Changing a container, step by step (example: TTS)

There are three levels, from easiest to most work. The same steps apply to `stt`, `embed` and `gender`.

### Level 1: voice, speed or loudness (settings only, no rebuild)

The **bot** chooses the voice and speed and sends them with each request; loudness and the audio chain are applied inside `tts`. Edit `../.env`, then recreate both:

```bash
# ../.env
SUPERTONIC_VOICE_FEMALE=F2      # voices: F1–F5, M1–M5
SUPERTONIC_SPEED_EN=1.05
TTS_LOUDNESS_LUFS=-16
```
```bash
docker compose up -d bot tts     # recreates the containers so they read the new .env
```

### Level 2: the other built-in engine (Edge TTS)

`ari_app/tts.py` already supports `TTS_BACKEND=edge` (Microsoft neural voices; needs internet).

1. Add `edge-tts>=6.1.0` to `docker/requirements/tts.txt`.
2. In `docker-compose.yml`, `tts:` block, add `environment: { TTS_BACKEND: edge }`.
3. Run `docker compose build tts && docker compose up -d tts`.

Voices are set with `EDGE_TTS_VOICE_EN` and `EDGE_TTS_VOICE_HI` in `../.env`.

### Level 3: a completely different model (for example Piper)

The bot only depends on the **contract**, so keep it and put any engine behind it:

> `POST /synthesize` with JSON `{text, voice_en, session_lang, speed, tail_pad_s, …}` → `{"wav": base64 8 kHz mono WAV, "wav16": base64 16 kHz WAV or null}`, and `GET /health` → `{"ready": true}`

1. **Packages:** add the engine to `docker/requirements/tts.txt` (for example `piper-tts`).
2. **Code:** in `services/tts_service.py`, load the model in `lifespan()` and synthesize in `/synthesize`. Convert the output with the existing phone chain so it sounds right on the line:

   ```python
   from ari_app.tts import _ffmpeg_to_asterisk_wav, wideband_path
   ...
   VOICE.synthesize(body["text"], wav_file)                    # raw.wav at the model's own rate
   _ffmpeg_to_asterisk_wav(raw, out, tail_pad_s=body.get("tail_pad_s", 0.0))   # -> 8 kHz + 16 kHz twin
   ```
3. **Model file:** mount it in the `tts:` block (`volumes: - ${HOME}/models/piper:/models:ro`) and pass its path through `environment:`.
4. **Rebuild and test the container on its own,** before any call:

   ```bash
   docker compose build tts && docker compose up -d tts
   docker compose logs -f tts                      # wait for "Application startup complete"
   curl -s 127.0.0.1:8102/health                   # {"ready":true,...}
   curl -s -H 'Content-Type: application/json' -d '{"text":"Hello, this is the new voice.","voice_en":"F3"}' \
        127.0.0.1:8102/synthesize | python3 -c "import sys,json,base64; open('test.wav','wb').write(base64.b64decode(json.load(sys.stdin)['wav']))"
   ffplay test.wav
   ```
5. **Call 7001.** The bot isn't rebuilt or restarted.

**If you change the request or response format,** also update the bot side: the `if remote.enabled("tts")` block at the top of `synthesize_to_wav()` in `ari_app/tts.py`. Then run `docker compose build bot && docker compose up -d bot`.

### The general recipe

| Step | What |
|---|---|
| 1. Logic | `ari_app/<module>.py` and/or `services/<name>_service.py` |
| 2. Packages | `docker/requirements/<name>.txt`, or its stage in `docker/Dockerfile` |
| 3. Env, ports, model mounts | the `<name>:` block in `docker/docker-compose.yml` |
| 4. Rebuild only it | `docker compose build <name> && docker compose up -d <name>` |
| 5. Check | `docker compose logs -f <name>`, `curl 127.0.0.1:<port>/health` |
| 6. Roll back | `git checkout -- <files>`, then step 4 again |

| Container | Port | Logic | Wrapper | Packages |
|---|---|---|---|---|
| `stt` | 8101 | `ari_app/stt.py` | `services/stt_service.py` | `requirements/stt.txt` |
| `tts` | 8102 | `ari_app/tts.py` | `services/tts_service.py` | `requirements/tts.txt` |
| `embed` | 8103 | `ari_app/rag.py` | `services/embed_service.py` | `requirements/ml.txt` |
| `gender` | 8104 | `ari_app/gender_model.py` | `services/gender_service.py` | `requirements/ml.txt` |
| `bot`, `web` | 8766, 8765 | the rest of `ari_app/` | – | `requirements/app.txt` |

## Testing with a phone

1. Check that all containers are up:

   ```bash
   docker compose ps                 # stt/tts/embed/gender "healthy", bot/web "Up"
   docker exec callbot-asterisk asterisk -rx "ari show apps"   # must list: callbot
   ```
2. Register Linphone or Zoiper. Server: this PC's IP (`ip -4 -br addr`) on port **5060**, transport **UDP**, user `1001`, password from `PHONE_1001_PASSWORD` in `docker/.env`. Check that it registered:

   ```bash
   docker exec callbot-asterisk asterisk -rx "pjsip show contacts"
   ```
3. Dial **7001**.
4. Watch it live:

   ```bash
   docker compose logs -f bot                       # the conversation
   docker compose logs -f stt tts                   # requests hitting each service
   ```
5. Afterwards, play the call and read the transcript at http://127.0.0.1:8765/recordings.

If there's **no audio on the phone**, check that `ASTERISK_SOUNDS_EN=/sounds` (bot) and `./data/sounds` (asterisk) are the same folder. They are by default. If the **bot doesn't answer**, check `ari show apps` and `docker compose logs bot`.

## Models and data

| What | Where | Mounted into |
|---|---|---|
| Whisper, e5, gender model | `~/.cache/huggingface` (host) | `stt`, `embed`, `gender` |
| Supertonic voices | `~/.cache/supertonic3` (host) | `tts` |
| TTS sound files | `docker/data/sounds` | `bot` (writes), `asterisk` (plays) |
| Call recordings | `recordings/` | `bot` (writes), `web` (reads) |
| Knowledge documents | `knowledge/` | `bot` |
| Logs | `logs/callbot.log` | `bot` |
| PostgreSQL / Redis data | Docker volumes `postgres-data`, `redis-data` | `postgres`, `redis` |
| Phone accounts added on the web page | Docker volume `asterisk-db` (astdb) | `asterisk` |

Models are **not** baked into the images. They're read from your host caches, so nothing is downloaded twice and the images stay smaller. On a fresh machine, the first start downloads them into those folders.

## Images and resources

| Image | Size | Built from |
|---|---|---|
| `harvix/stt` | 3.8 GB | python:3.11-slim + CUDA 12 libs (cuBLAS, cuDNN 9…) + faster-whisper |
| `harvix/tts` | 4.0 GB | the same CUDA layer + Supertonic + onnxruntime-gpu 1.26 + ffmpeg |
| `harvix/ml` | 2.1 GB | CPU PyTorch + transformers + sentence-transformers + librosa |
| `harvix/app` | 1.7 GB | call logic only (httpx, openai, asyncpg, redis, faster-whisper for the Silero VAD) |

`stt` and `tts` share the base and CUDA layers, so the real disk use is less than the sum. GPU memory in use: `stt` about 2.1 GB, `tts` about 0.9 GB.

Only the CUDA **driver** comes from the host (NVIDIA container runtime); the CUDA libraries are inside the images. To move `embed` to the GPU, build `ml` with CUDA PyTorch (change the `--index-url` in the `ml` stage) and add a `deploy.resources` GPU block to `embed`.

## Tested (2026-11-11)

- Each service on its own:
  - `stt` transcribed a 12 s clip in 0.87 s on CUDA.
  - `tts` synthesized a sentence in 0.97 s (8 kHz + 16 kHz).
  - `embed` returned a 384-dimension normalized vector in 0.13 s.
  - `gender` classified a clip as male (0.92).
- A full call through the containers: question → RAG and LLM answer → "That's all, thank you" → "Shall I end the call?" → "Yes" → goodbye and hang-up. The transcript was saved in PostgreSQL and a 43 KB Opus recording was saved. Requests served during the call: stt 3, tts 7, gender 2, embed 1.
- `web`: `/customers`, `/recordings` and the API all return 200.

Not tested: the optional `llm` (vLLM) container. It downloads a large image, and Gemma E2B next to Whisper and Supertonic is tight on an 8 GB GPU.

## Performance tuning (measured on this PC: RTX 3070 8 GB, 16 cores, 14 GB RAM)

| Change | Where | Effect |
|---|---|---|
| No web search for questions about the caller's own data ("who changed **my** name", "where is **my** parcel") | `registration.about_caller()` → `_model_requested_lookup()` | A turn went from 3.1 s of useless search to 0 s |
| No web search for questions about the business ("what courses do **you** offer", "**your** prices / hours"); answered from the knowledge base | `registration.about_business()` | 3.2 s → knowledge-base hit in 25 ms |
| Web search time budget 5 s → 3.5 s | `.env` `WEB_LOOKUP_BUDGET_SECONDS` | A failed search can't stall a turn as long |
| Pitch analysis warmed up at startup (librosa/numba JIT) | `server.py` lifespan | First-turn STT step 0.70 s → 0.40 s |
| Model containers: 4 CPU threads each, `MALLOC_ARENA_MAX=2` | `x-model-env` in `docker-compose.yml` | RAM of stt+tts+gender+embed 3.7 GB → 2.0 GB; a gender check no longer takes all 16 cores mid-call |
| Models load offline from the cache (`HF_HUB_OFFLINE=1`) | `x-model-env` | No HuggingFace calls at startup; whole stack healthy in ~17 s; works without internet |
| Log rotation 3 × 10 MB for every container | `x-logging` | Container logs can't fill the disk (it was 97% full) |
| Healthchecks for `bot` (`/health`) and `web` | `docker-compose.yml` | `docker compose ps` shows real status for every container |

**Result:** from the moment the caller stops talking to the first word of the reply takes about **1.9 s**:

| Step | Time |
|---|---|
| End-of-speech detection | 0.7 s |
| STT | 0.4 s |
| LLM | 0.4 s |
| First TTS chunk | 0.45 s |

The rest of a turn is the reply playing.

**Not changed, but adjustable if you want:**
- `VAD_END_SILENCE_MS=700`: 600 saves 0.1 s per turn, but the bot is more likely to cut in during short pauses.
- Whisper `medium`: `large-v3-turbo` is more accurate for a little more GPU time.
- `REPLY_MAX_SENTENCES=2`: 1 makes replies, and calls, shorter.

**After a model change,** the model must already be in `~/.cache`, because the containers run offline. Download it once on the host, or set `HF_HUB_OFFLINE=0` in `x-model-env` for one start.

## Problems hit while building (and fixed)

- **PyPI timeouts** on the 600–700 MB NVIDIA wheels: pip now uses a 300 s timeout and 10 retries, and a BuildKit pip cache, so a failed build resumes.
- **librosa / numba "cannot cache function"** as the non-root user: `NUMBA_CACHE_DIR=/tmp/numba` in the `ml` and `app` images.
- **`web` crashed writing `logs/callbot.log`:** `LOG_FILE=""` for `web`.
- **Disk:** the images need about 10 GB; Docker's root disk had 5 GB free, so the unused build cache was pruned.
