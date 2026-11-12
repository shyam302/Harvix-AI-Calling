# Voice chat in the browser

Branch: `webchat` (built on `dockerize`)

The same assistant as the phone line, in a web page. There are two ways to open it:
- **From the portal** (http://127.0.0.1:8765/), or the standalone `/customers` and `/recordings` pages: the **💬 Chat** button at the bottom right slides it in as a panel. Closing the panel (✕) or switching sections keeps the conversation going.
- **Full page:** http://127.0.0.1:8766/chat (served by the `bot` container).

**Two buttons, independent of each other:**

| Button | On | Off |
|---|---|---|
| 🎤 **Mic** | the bot listens: talk instead of typing (live VAD, barge-in) | type only; the browser releases the microphone |
| 🔊 **Speaker** | replies are **spoken** (typed or talked) | replies are text only (faster: no speech is generated) |

Turning the mic on also turns the speaker on; you can switch the speaker off again to read silently while talking. Both settings are remembered for the browser tab.

**Chat history** is kept for as long as the browser tab is open, in the tab's sessionStorage, which all portal pages share:
- It survives page switches (Customers ↔ Recordings), refreshes, and closing and reopening the side panel.
- When the chat reconnects, the last 20 messages are sent to the bot (`{"type":"start","history":[…]}` → `WebChatSession.resume()`), so it **remembers** the conversation, not just the screen. "What did I just ask you?" works after a page switch.
- **End** keeps the history, and typing again continues it.
- **New conversation** clears it.
- If you entered a phone number, the PIN is asked again on each reconnect, for security.

**The chat starts text-only.** Type, and the bot answers in text: no microphone, and no speech is generated, so replies are faster. Click **🎤** to switch the microphone **on**: you can then talk, and the bot also answers **aloud**. Click 🎤 again to switch it **off**; the browser releases the microphone (its indicator goes off) and you're back to text. Text-only mode never times out on "silence".

| | Phone (Asterisk) | Web chat |
|---|---|---|
| Caller audio | RTP from Asterisk → live VAD | browser mic → WebSocket → the same live VAD |
| Bot audio | WAV played by Asterisk | 16 kHz WAV sent to the browser, played in the page |
| Keypad (PIN) | phone DTMF | on-screen **Keypad** |
| Who is calling | caller id | optional **phone number** field on the start screen |
| Everything else | same code | same code |

**Same behaviour as calls:**
- STT, LLM, TTS, the knowledge base and Tavily web search;
- the PIN check for registered numbers;
- the open-complaint offer, ticket status, registration and complaints with confirmation;
- "Shall I end the chat?";
- transcripts in PostgreSQL (shown on `/recordings`, without audio).

Extras in the browser:
- **Type** instead of speaking (typed text skips speech-to-text and takes exactly the same route).
- **Mute** the mic.
- See the conversation as text. PIN and number answers spoken aloud are shown as dots.
- **Barge-in:** talk over the bot and its playback stops.

## How it works

- **Speaker switch:** `{"type":"speaker","on":…}` → `set_speaker()`. With the speaker off, `_synthesize_phrase` and `_play_wav` do nothing.
- **Voice switch:** the page sends `{"type":"voice","on":true|false}`, which calls `WebChatSession.set_voice()`.
  - **Off:** `_synthesize_phrase` / `_play_wav` do nothing, and listening waits only for typed text.
  - **On:** live VAD and spoken replies, exactly as on a call.
- **Portal side panel:** a small snippet at the end of `customers.html` and `recordings.html` opens `http://<host>:8766/chat?embed=1` in an iframe (`allow="microphone; autoplay"`). `?embed=1` makes the layout compact.


`ari_app/webchat.py`: `WebChatSession(CallSession)` overrides only the methods that talked to Asterisk:
- `answer`
- `_setup_live_vad`: a `LiveVad` fed by `feed_pcm()` instead of RTP
- `_play_wav`: sends the WAV and waits for the browser's `played`, or for a barge-in
- `_capture_turn_vad`: speech **or** a typed message, whichever comes first
- `hangup`
- the recording methods (no audio file for chats)

`run_loop()` and every conversation rule are inherited unchanged.

The WebSocket is `/ws/chat?phone=<optional>`, in `server.py`; the protocol is documented at the top of `webchat.py`. The page is `ari_app/static/chat.html`: AudioWorklet mic capture → 16 kHz PCM16, an `<audio>` element per reply chunk so the browser's echo cancellation keeps working, plus the keypad and the text box.

## Using it from other devices

Browsers only allow the **microphone on `localhost` or HTTPS**:
- On this PC, `http://127.0.0.1:8766/chat` works.
- From a phone or another PC on the network, the page must be served over **HTTPS**, for example through a reverse proxy (Caddy or nginx) with a certificate, or a Tailscale HTTPS name. The bot also has to listen beyond 127.0.0.1, so set `WEB_ADMIN_PASSWORD` first.
- Without HTTPS, other devices can still **type** but not talk.

## Tested (2026-11-12)

- **Voice in Chrome** (fake microphone file): "What courses do you offer?" → captured by the VAD, transcribed, answered, played. No page errors.
- **Typed:**
  - date → answered from the clock ("Thursday, November 12th, 2026");
  - refund policy → honest "I don't have that information… register a request";
  - "That's all" → "Shall I end the chat?" → "Yes" → goodbye and the chat ends.
- **Registered number + PIN on the on-screen keypad:** "Welcome back, Asha Test… PIN" → 1 2 3 4 → verified → "that's all" → goodbye by name.

## Fixed on the way (also affects phone calls)

"What's today's date?" was sent to a web search, which returned a page from a US time zone (the 11th instead of the 12th). Date and time questions are now always answered from the clock (`registration.asks_date_time()`), while "the date **of** the next match" is still searched.
