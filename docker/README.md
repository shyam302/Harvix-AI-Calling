# Docker setup

Everything the call bot needs runs here, one container per function. The full guide is [docs/DOCKER.md](../docs/DOCKER.md); this page is the quick reference.

| Container | Does | GPU | Port (localhost) |
|---|---|---|---|
| `asterisk` | SIP, RTP, ARI ([andrius/asterisk](https://hub.docker.com/r/andrius/asterisk) 22.11) | – | SIP 5060/udp, ARI 8088 |
| `redis` | knowledge-base vectors | – | 6379 |
| `postgres` | users, calls, complaints | – | 5432 |
| `stt` | Whisper speech-to-text | ✅ | 8101 |
| `tts` | Supertonic text-to-speech | ✅ | 8102 |
| `embed` | e5 embeddings for the knowledge base | CPU | 8103 |
| `gender` | caller gender model | CPU | 8104 |
| `bot` | the call logic (`python -m ari_app.server` inside) | – | 8766 (`/health`) |
| `web` | `/recordings`, `/customers` pages | – | 8765 |
| `llm` (optional) | local vLLM (`--profile llm`) | ✅ | 8000 |

```
docker/
  docker-compose.yml   all containers
  Dockerfile           images for stt, tts, ml (embed+gender), app (bot+web)
  requirements/        Python packages per image (stt.txt, tts.txt, ml.txt, app.txt)
  .env                 Asterisk/DB settings (git-ignored); copy from .env.example
  .env.example         all settings with defaults
  render.sh            fills ${VARS} in asterisk/*.conf from .env at container start
  asterisk/            Asterisk config templates
    pjsip.conf         phones 1001/1002, SIP port, codecs, Tailscale/NAT IP
    extensions.conf    BOT_EXTENSION (7001) -> Stasis(callbot), 1001 <-> 1002
    http.conf, ari.conf  ARI on 127.0.0.1:8088, user/password (must match ../.env)
    rtp.conf           RTP port range
  postgres/init.sql    database tables (the bot also creates them if missing)
  data/                runtime, git-ignored
    sounds/            TTS files: written by bot, played by asterisk
    logs/              Asterisk logs
../services/           the small HTTP apps inside stt, tts, embed, gender
```

Two settings files:
- `docker/.env`: Asterisk, Redis and PostgreSQL (ports, passwords, codecs).
- The project's `../.env`: everything the bot does (vLLM, voices, VAD, RAG, PIN…), the same file used when running the bot on the host.

## First run

**1. Stop the system Asterisk,** if you have one. Both want ports 5060 and 8088:

```bash
sudo systemctl stop asterisk && sudo systemctl disable asterisk
```

**2. Settings files** (first time only):

```bash
cd /mnt/data/projects/Asterisk_Call/docker
cp .env.example .env                 # Asterisk / DB passwords, codecs
cp ../.env.example ../.env           # bot settings, if you don't have one yet; set VLLM_BASE_URL
```

**3. Build and start everything.** The first build takes 10–20 minutes because the CUDA libraries are large; the GPU containers need the NVIDIA container runtime.

```bash
docker compose build
docker compose up -d
docker compose ps                    # stt/tts/embed/gender/bot/web: healthy
```

**Don't also run `python -m ari_app.server` on the host.** The `bot` container is the bot; two of them would answer the same calls.

**4. Phone:** register Linphone or Zoiper as `1001` with the password from `PHONE_1001_PASSWORD`, using **UDP** to this PC's IP on port 5060. Then dial **7001**.

**5. Watch:**
- `docker compose logs -f bot` (the conversation)
- http://127.0.0.1:8765/recordings and http://127.0.0.1:8765/customers

## Everyday commands

```bash
docker compose up -d                       # start (also brings back anything stopped)
docker compose down                        # stop everything (data is kept)
docker compose ps                          # status
docker compose logs -f bot                 # follow calls (stt, tts, … for the services)
docker compose up -d bot tts               # after editing ../.env (recreates those containers)
docker compose restart asterisk            # after editing files in asterisk/ (re-renders them)
docker compose build tts && docker compose up -d tts      # after changing one service's code
docker exec -it callbot-asterisk asterisk -rvvv            # Asterisk console
docker exec callbot-asterisk asterisk -rx "pjsip show contacts"   # who is registered
```

For changing one container step by step (for example a new TTS model), see [docs/DOCKER.md](../docs/DOCKER.md#changing-a-container-step-by-step-example-tts).

## Bot on the host instead (old way)

Run only the infrastructure in Docker and the bot with its models in-process on the host:

```bash
docker compose up -d asterisk redis postgres
cd .. && conda activate py311 && python -m ari_app.server
```

In this mode, set `ASTERISK_SOUNDS_EN=/mnt/data/projects/Asterisk_Call/docker/data/sounds` in `../.env` so Asterisk can find the bot's audio. The `bot` container sets this itself.

## Calling over Tailscale

Set `EXTERNAL_IP` in `docker/.env` to this PC's Tailscale IP (`tailscale ip -4`), then run `docker compose up -d asterisk`.

## Going back to the system Asterisk

```bash
docker compose down
sudo systemctl enable --now asterisk
```

Then run the bot on the host, with `ASTERISK_SOUNDS_EN` set back to `/var/lib/asterisk/sounds/en` (or the line deleted).

## Troubleshooting

| Symptom | Check |
|---|---|
| A container exits with "address in use" | the system Asterisk, or a host bot, is still running (`sudo ss -ulpn \| grep 5060`, `pgrep -af ari_app.server`) |
| Call drops right after dialling 7001 | the bot isn't connected: `docker exec callbot-asterisk asterisk -rx "ari show apps"` must list `callbot`; then `docker compose logs bot` |
| Call connects but the bot is silent | `tts` unhealthy (`docker compose logs tts`), or the sounds folder isn't shared (`./data/sounds` for asterisk = `/sounds` in bot) |
| `stt` / `tts` won't start | GPU not visible in containers: `docker info \| grep -i nvidia` and `nvidia-smi` |
| A model "not found" at startup | the containers load models offline from `~/.cache`; download it once on the host, or set `HF_HUB_OFFLINE=0` in `x-model-env` for one start |
| One-way audio from outside the LAN | set `EXTERNAL_IP` (see Tailscale) and allow UDP 5060 and 10000–10100 in the firewall |
| Slow G.722 / garbled audio | keep `CODECS=ulaw,alaw,g722` in `docker/.env` (ulaw first) |
