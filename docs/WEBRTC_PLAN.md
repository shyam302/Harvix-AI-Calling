# Plan: our own calling app (WebRTC)

Status: **planned, not started**. Written 2026-11-02.

## Goal

A "phone line" app of our own: someone opens a web page (later a mobile app), taps **Call**, and talks to the Harvix bot. They don't install Linphone or Zoiper, and there's no SIP account to set up.

## The approach: browser → Asterisk over WebRTC → the bot we already have

```
Browser / mobile app (SIP.js + WebRTC)
   │  SIP signalling over WSS (wss://<host>:8089/ws)
   │  audio over SRTP (DTLS), Opus or G.722
   ▼
Asterisk 22 (PJSIP endpoint with webrtc=yes)
   │  dial 7001 → Stasis(callbot)   ← same as today
   ▼
ari_app (unchanged): live VAD, Whisper, LLM, RAG, web search, Supertonic
```

Why this way: the bot already speaks ARI to Asterisk, so a WebRTC caller is just another channel to it. Barge-in, VAD, RAG and HD audio all keep working with **zero bot changes**. Asterisk handles the hard parts (ICE, DTLS-SRTP, codecs).

Alternative, kept for later: skip Asterisk and run WebRTC straight into Python (aiortc, or LiveKit/pipecat). It's lower latency and has fewer moving parts per call, but we'd have to rewrite audio I/O, VAD feed and playback, and give up Asterisk features (SIP trunks, real phone numbers, IVR). Only worth it if Asterisk becomes the bottleneck.

## What this machine already has (checked 2026-11-02)

| Piece | Status |
|---|---|
| `res_http_websocket`, `res_pjsip_transport_websocket` | ✅ built |
| `res_srtp` (DTLS-SRTP) | ✅ built |
| `codec_g722` (wideband) | ✅ built |
| `codec_opus` | ❌ missing. Browsers do G.722/PCMU too, so it's optional, but Opus is better on bad networks |
| TLS on Asterisk HTTP (`tlsenable` in `http.conf`) | ❌ not set |
| TLS certificate | ❌ none (no Let's Encrypt) |
| TURN server (coturn) | ❌ not installed. Needed for callers outside the LAN |

## Phases

### Phase 1: Asterisk accepts WebRTC calls (LAN only)
1. Get a certificate: `mkcert` for LAN testing (install its root CA on the test phone), or Let's Encrypt once we have a domain. Browsers refuse microphone access on plain `http://`, except on `localhost`.
2. `http.conf`: `tlsenable=yes`, `tlsbindaddr=0.0.0.0:8089`, `tlscertfile`, `tlsprivatekey`.
3. `pjsip.conf`:
   - transport `type=transport, protocol=wss, bind=0.0.0.0`
   - endpoint template with `webrtc=yes` (sets `use_avpf`, `media_encryption=dtls`, `ice_support`, `rtcp_mux`, `dtls_auto_generate_cert`)
   - `allow=opus,g722,ulaw`
   - `context=from-internal-app`, so it can dial 7001
4. Optional: build `codec_opus` (Asterisk's `menuselect`, or the binary codec_opus for 22).
5. Test with the SIP.js demo page or <https://tryit.jssip.net> → register → dial 7001.

### Phase 2: our web app (the "phone")
- A single page: **Call** button, mute, hang up, call timer, status ("Connecting…", "Bot is speaking", "Listening").
- **SIP.js** (or JsSIP) handles registration and the call; `getUserMedia` provides the mic, with echo cancellation and noise suppression ON. Barge-in depends on echo cancellation.
- Served over HTTPS. The easiest place is the existing FastAPI server (`ari_app/server.py`, port 8765) behind the same cert, or nginx.
- **Live transcript (nice to have):** the bot already knows each turn's text. Add a small WebSocket on the FastAPI server that pushes `{caller: …, bot: …}` per turn, keyed by call id, so the page can show subtitles.
- Works as a **PWA** on Android Chrome (Add to Home Screen), so a phone app before we write one.

### Phase 3: accounts and security
- No hard-coded SIP passwords in the page. A backend endpoint (`POST /api/session`) issues **short-lived SIP credentials** per user or session. Asterisk realtime or `pjsip_wizard` user, or one shared endpoint with a per-session token checked in the dialplan.
- Pass who is calling into the call via a SIP header (`X-User-Id`) → channel var → ARI `StasisStart`, so the bot can greet by name and load that user's context.
- Rate-limit calls per user; fail2ban on Asterisk auth failures; firewall to 443/8089 + the RTP range.

### Phase 4: calls from outside the LAN
- **coturn** TURN server (UDP 3478, TLS 5349) with time-limited credentials; give the ICE servers to SIP.js.
- `pjsip`: `external_media_address`, `external_signaling_address`, `local_net`; `rtp.conf` `icesupport=yes`, `stunaddr`.
- A real domain + Let's Encrypt cert.

### Phase 5: native mobile app (only if the PWA isn't enough)
- **Flutter** (`sip_ua` + `flutter_webrtc`) or **React Native** (`react-native-webrtc` + JsSIP).
- Reasons to go native: background calls, Bluetooth headsets, CallKit (iOS) / ConnectionService (Android) so incoming calls ring like a normal phone, and push notifications (FCM/APNs) to wake the app for an incoming call.

## Things to watch
- **Echo:** browsers' echo cancellation is good, but test barge-in on laptop speakers; it may need a higher `BARGE_IN_THRESHOLD` for web callers.
- **Codec path:** Opus 48 kHz → Asterisk → `slin16` for our VAD / `wav16` recordings, so quality should match or beat G.722 softphones.
- **iOS Safari:** WebRTC works, but the mic needs a user tap first, and audio stops when the page is backgrounded. Another reason for native in Phase 5.
- **Certificates:** an expired cert means "can't connect" with no clear error; automate renewal.

## Rough effort
| Phase | Effort |
|---|---|
| 1. Asterisk WebRTC on LAN | half a day |
| 2. Web phone page (+ transcript) | 1–2 days |
| 3. Accounts / security | 1–2 days |
| 4. Outside-LAN (TURN, domain) | 1 day |
| 5. Native app | 1–2 weeks |
