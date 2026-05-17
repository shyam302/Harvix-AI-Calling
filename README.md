# Asterisk Callbot

Voice phone assistant built on **Asterisk ARI**: SIP call → record speech → **Whisper** STT → **vLLM** → **Supertonic** TTS → play reply. Supports Hindi and English with per-call conversation memory.

## Features

- Asterisk Stasis app (`callbot`) — dial an extension to talk to the bot
- Local speech-to-text (faster-whisper) and local TTS (Supertonic ONNX)
- OpenAI-compatible LLM backend (vLLM, Ollama, etc.)
- Hindi / English with session language locking
- Optional caller pitch → male/female voice and persona (F3 / M1)
- In-call context (history until hangup)

## Requirements

- Linux, Asterisk 18+ (22.x tested)
- Python 3.11+, `ffmpeg`
- GPU recommended for Whisper; CPU OK for Supertonic TTS
- Reachable vLLM HTTP API (`/v1/chat/completions`)

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/Asterisk_Call.git
cd Asterisk_Call

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — set ARI_PASSWORD, VLLM_BASE_URL, etc.

python tts/supertonic_demo.py   # optional TTS smoke test
```

Configure Asterisk and run the app — **full steps:** [SETUP.txt](SETUP.txt)

```bash
python -m ari_app.server
curl -s http://127.0.0.1:8765/health
```

Register a SIP phone (see snippets), dial extension **7001** (default in repo).

## Documentation

| File | Description |
|------|-------------|
| [SETUP.txt](SETUP.txt) | Install Asterisk snippets, `.env`, permissions, troubleshooting |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Components, call flow, module map |
| [.env.example](.env.example) | All configuration variables (copy to `.env`) |

## Security (public deployers)

- **Do not commit** `.env` — it is gitignored
- Change default passwords in `asterisk_snippets/` before production
- Keep ARI HTTP on `127.0.0.1` unless you know what you are exposing
- Use strong `ARI_PASSWORD` and firewall SIP/RTP appropriately

## License

Add your license file (e.g. MIT) if you publish this repository.
