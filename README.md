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
- **Looks things up.** If you ask for something it can't reliably know (opening hours, exchange rates, flight times…), it does a quick web search before answering. Small talk doesn't trigger a search.
- **Gives cleaner audio.** TTS output is loudness-normalized, and there's a 16 kHz version for phones that use G.722.

There's also a `/health` endpoint, and an optional log file (`LOG_FILE`) that records per-turn timing: speech-to-text, LLM, web search, TTS and playback.

## Requirements

- Linux with Asterisk 22 (PJSIP + ARI). I run 22.9 built from source.
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

1. Copy the files in `asterisk_snippets/` into `/etc/asterisk/` and include them from `pjsip.conf` and `extensions.conf`. Change the example passwords first.
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

Never commit `.env`.

## Known issues

- **Free web search is unreliable.** DuckDuckGo and similar engines rate-limit by IP, so searches sometimes fail. When that happens, the bot says it couldn't find the answer and stops searching for a minute, but answers are only as good as the search snippets. For anything serious, use a SerpAPI or Google Programmable Search key.
- **Barge-in relies on the phone's echo cancellation.** If the bot's own voice leaks back through your mic, it can interrupt itself. Raise `BARGE_IN_THRESHOLD` or `BARGE_IN_MIN_SPEECH_MS`, or turn barge-in off.
- **HD audio needs G.722 negotiated.** Put `g722` first in the endpoint's `allow=` line and enable G.722 in the softphone; otherwise calls stay at narrowband (8 kHz) quality.
- **Small LLMs make things up.** The prompt pushes the model to search rather than guess, but a 2B model still gets things wrong now and then. Don't use this for anything where a wrong answer matters.
- **Only tested on a LAN.** Calling over the internet or Tailscale needs `external_media_address` set up in PJSIP (there are notes in the snippet), and you should lock down SIP and ARI first.
- **Asterisk installed twice.** If you have both the distro Asterisk package and a source build, make sure the binary loads its own `libasteriskpj`. A mismatched PJSIP library crashed Asterisk on every SIP REGISTER here.

## Security

The configs in `asterisk_snippets/` use example passwords (`1001`/`1001`). Change them before the box is reachable from anywhere you don't trust. Keep ARI bound to `127.0.0.1`. Only open SIP (5060/udp) and RTP (10000–20000/udp) to the networks that need them.

## Project layout

```
ari_app/              the bot: ARI event loop, call session, STT, LLM, TTS, live VAD, web search
asterisk_snippets/    example Asterisk configs
tts/                  TTS test scripts
```

[ARCHITECTURE.md](ARCHITECTURE.md) explains how a call flows through the modules.

## License

MIT. See [LICENSE](LICENSE).
