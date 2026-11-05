# Asterisk in Docker

This runs Asterisk 22.11 from [andrius/asterisk](https://hub.docker.com/r/andrius/asterisk) with the call bot's config, so you don't need the source build in `/usr/sbin/asterisk`. The Python bot still runs on the host as before.

```
docker/
  docker-compose.yml
  asterisk/          config files mounted into the container
    pjsip.conf       phones 1001/1002 (password = user)
    extensions.conf  7001 -> Stasis(callbot), 1001 <-> 1002
    http.conf        ARI HTTP on 127.0.0.1:8088
    ari.conf         ARI user callbot / 1001 (must match .env)
    rtp.conf         RTP ports 10000-10100
  data/              created at runtime, git-ignored
    sounds/          bot TTS WAVs (Asterisk plays from here)
    logs/            Asterisk logs
```

It uses `network_mode: host`. SIP and RTP sit directly on the machine's IP, with no NAT or port mapping, and the bot's live-VAD audio stream to `127.0.0.1` keeps working.

## First run

**1. Stop the system Asterisk.** Both want ports 5060 and 8088:

```bash
sudo systemctl stop asterisk
sudo systemctl disable asterisk      # so it doesn't come back after reboot
```

**2. Point the bot at the shared sounds folder.** In the project `.env`:

```
ASTERISK_SOUNDS_EN=/mnt/data/projects/Asterisk_Call/docker/data/sounds
```

The bot writes TTS files to `<that>/custom/` and Asterisk, inside the container, sees them at `/var/lib/asterisk/sounds/en/custom/`.

**3. Start Asterisk:**

```bash
cd /mnt/data/projects/Asterisk_Call/docker
docker compose up -d
docker compose logs -f asterisk        # Ctrl+C to stop watching
```

**4. Start the bot** as usual, in another terminal:

```bash
conda activate py311
cd /mnt/data/projects/Asterisk_Call
python -m ari_app.server
```

**5. Phone:** register Zoiper/Linphone as `1001` / `1001` (UDP) to this PC's IP, then dial `7001`.

## Everyday commands

```bash
docker compose up -d                 # start
docker compose down                  # stop
docker compose restart asterisk      # after editing files in asterisk/
docker exec -it callbot-asterisk asterisk -rvvv          # Asterisk console
docker exec callbot-asterisk asterisk -rx "pjsip show endpoints"   # who is registered
docker exec callbot-asterisk asterisk -rx "pjsip reload"           # reload SIP only
```

## Calling over Tailscale

In `asterisk/pjsip.conf`, uncomment and set both lines under `[transport-udp]` to this PC's Tailscale IP (`tailscale ip -4`), then `docker compose restart asterisk`.

## Going back to the system Asterisk

```bash
docker compose down
sudo systemctl enable --now asterisk
```

Then set `ASTERISK_SOUNDS_EN` back to `/var/lib/asterisk/sounds/en`, or delete the line.

## Troubleshooting

- **Container exits right away / "address in use":** the system Asterisk is still running (`sudo ss -ulpn | grep 5060`).
- **Bot can't connect to 8088:** check `docker exec callbot-asterisk asterisk -rx "http show status"`.
- **Call connects but the bot is silent:** `ASTERISK_SOUNDS_EN` isn't pointing at `docker/data/sounds`.
- **One-way audio from outside the LAN:** set the `external_*` addresses (see Tailscale above) and allow UDP 5060 and 10000-10100 in the firewall.
