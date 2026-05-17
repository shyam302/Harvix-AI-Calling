# Harvix AI Calling

Open-source **voice call assistant** for Asterisk: callers dial a SIP extension and talk to an AI that listens, thinks, and speaks back in natural Hindi or English.

Built with **Asterisk ARI**, **faster-whisper**, **vLLM** (or any OpenAI-compatible API), and **Supertonic** (local neural TTS — no cloud TTS required).

```
SIP phone → Asterisk → Python (ARI) → Whisper → LLM → Supertonic → play audio on call
```

## Features

- **Phone-native** — half-duplex turns tuned for real calls (silence detection, pauses, short replies)
- **Local STT & TTS** — Whisper on GPU/CPU; Supertonic ONNX voices (e.g. F3 female, M1 male)
- **Hindi + English** — auto language per call; say *“continue”* or speak English to switch
- **Per-call memory** — conversation context until hangup
- **Gender-aware voice** — optional pitch detection to match caller with bot voice & persona
- **Health endpoint** — `GET /health` while the bot runs

## Stack

| Layer | Technology |
|-------|------------|
| Telephony | Asterisk 22.x, PJSIP, Stasis app `callbot` |
| Orchestration | Python 3.11, FastAPI + uvicorn, ARI WebSocket |
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| LLM | vLLM / Ollama / OpenAI-compatible HTTP API |
| Text-to-speech | [Supertonic](https://github.com/supertone-inc/supertonic) |

## Requirements

- Linux server with Asterisk
- Python 3.11+, `ffmpeg`
- GPU recommended for Whisper (`large-v3`); CPU is fine for Supertonic
- A reachable LLM server (`VLLM_BASE_URL` ending in `/v1`)

## Quick start

```bash
git clone https://github.com/shyam302/Harvix-AI-Calling.git
cd Harvix-AI-Calling

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: ARI_PASSWORD, VLLM_BASE_URL, VLLM_MODEL, voices, etc.

python tts/supertonic_demo.py   # optional: test TTS before first call
```

1. Install Asterisk configs from `asterisk_snippets/` — see **[SETUP.txt](SETUP.txt)**
2. Start the bot:

```bash
python -m ari_app.server
curl -s http://127.0.0.1:8765/health
```

3. Register a SIP softphone, dial **7001** (default extension in snippets).

## Configuration

Copy **[.env.example](.env.example)** to `.env`. Important variables:

| Variable | Purpose |
|----------|---------|
| `ARI_*` | Asterisk REST interface (must match `ari_user_callbot.conf`) |
| `VLLM_BASE_URL` / `VLLM_MODEL` | LLM API |
| `WHISPER_MODEL` | e.g. `large-v3` |
| `SUPERTONIC_VOICE_FEMALE` / `MALE` | e.g. `F3`, `M1` |
| `CALL_PRIMARY_LANG` | `hi` or `en` |
| `OPENING_GREETING` | First line played after answer |

Never commit `.env` — it contains secrets.

## Documentation

| Document | Contents |
|----------|----------|
| [SETUP.txt](SETUP.txt) | Full install, Asterisk, permissions, troubleshooting |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, call flow, Python modules |

## Security

- Change **default passwords** in `asterisk_snippets/` before production
- Bind ARI HTTP to `127.0.0.1` unless the bot runs on a trusted host
- Open firewall ports only as needed: SIP `5060/udp`, RTP `10000-20000/udp`
- Use strong `ARI_PASSWORD` and keep the repo private if testing with real credentials

## Project layout

```
ari_app/              Application (ARI loop, call session, STT, LLM, TTS)
asterisk_snippets/    Example Asterisk configs (edit before deploy)
tts/                  Supertonic demo script
```

## License

Specify a license (e.g. MIT) when you publish. No license file is included yet.
