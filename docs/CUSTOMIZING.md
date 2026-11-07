# Customizing the bot: prompts, questions and complaints

This covers:

1. Which file to edit to change what the bot says or how it behaves.
2. How to add registration (and complaint) questions.
3. PIN check at call start, open complaints and resolution notes.
4. How keypad (DTMF) input works, line by line.
5. How the bot ends the call (hang-up), line by line.
6. Phone echo, and short answers like "skip".

After any change, **restart the bot**: press Ctrl+C once, then run `python -m ari_app.server` in the `py311` environment. Line numbers are approximate.

---

## 1. Changing prompts

There are three kinds of "prompt":

- **Instructions to the LLM**, which shape everything the AI says.
- **Fixed sentences**, which the bot always says word for word.
- **`.env` settings**, such as reply length and creativity.

### 1a. The bot's personality and conversation rules

**File: [ari_app/llm.py](../ari_app/llm.py)**

| What | Where | Change it when… |
|---|---|---|
| Main instructions (English): tone, short replies, use the caller's name, search instead of guessing | `_PHONE_AGENT_CORE_EN` (line ~16) | you want a different personality, business focus or rules |
| The same in Hindi | `_PHONE_AGENT_CORE_HI` (line ~40) | the same, for Hindi calls |
| Persona: female, male or neutral wording | `_PERSONA_EN`, `_PERSONA_HI`, `_PERSONA_OTHER` (line ~62) | you change the agent's gender or style |
| "Reply in <language>" rule | `_reply_language_rule()` (line ~87) | language behaviour |
| How the system prompt is assembled: caller name, "registered customer", open complaints, document and web facts | `build_system_prompt()` (line ~125) | you want to give the LLM new context |
| When to search the web | `_WEB_LOOKUP_SYSTEM` (line ~169) | it searches too often or not often enough |

**Example:** to make the bot a support agent for your company, start `_PHONE_AGENT_CORE_EN` with:

```python
_PHONE_AGENT_CORE_EN = (
    "You are Priya from Harvix Support, talking with a customer on a live phone call. "
    "You help with orders, deliveries, payments and complaints. "
    ...
```

Keep the rules that matter on the phone: short replies, no lists or emoji, and never reading out URLs.

### 1b. Fixed sentences (always said as written)

| What | File and place |
|---|---|
| Registration, PIN, complaint, ticket and "existing complaint" lines, in English and Hindi | [ari_app/registration.py](../ari_app/registration.py), the `_LINES` dictionary (line ~324): `"en": {...}` and `"hi": {...}` |
| Opening greeting (unregistered callers) | `.env`: `OPENING_GREETING=Hello, welcome to Harvix!` and `OPENING_GREETING_DYNAMIC=false`. Or edit the random greetings in `_dynamic_opening_en()` in [voice_select.py](../ari_app/voice_select.py) (line ~49) |
| Greeting for registered callers | `"welcome_back"` in `_LINES` |
| Greeting for registered callers **with open complaints** | `"offer_one"` and `"offer_many"` in `_LINES` (see section 3) |
| "Sorry, could you say that again?" and "can't think right now" | `pick_retry_phrase()` and `pick_llm_unavailable_phrase()` in [conversation.py](../ari_app/conversation.py) (line ~28) |

**Placeholders:** keep `{name}`, `{n}`, `{k}`, `{ticket}`, `{topic}`, `{count}`, `{date}`, `{status}` and `{order}` exactly as written; the code fills them in. A missing placeholder is fine. A misspelled one, such as `{nmae}`, makes that line fail.

### 1c. Words the bot listens for

| What | File and place |
|---|---|
| Words that start a complaint ("complaint", "lodge", "शिकायत", "shikayat"…) | `_COMPLAINT` in [registration.py](../ari_app/registration.py) (line ~147) |
| Asking **about** complaints ("how many", "status", "already", "track", "kya hua"…) vs **lodging** one ("lodge", "file", "register", "new", "darj"…) | `_STATUS_ASK`, `_TICKET`, `_LODGE` and `complaint_intent()` in [registration.py](../ari_app/registration.py) |
| Questions **about** the process ("how does the complaint system work?", "how do I lodge a complaint?"), which go to the LLM, unless the caller also says "I want / please / can you lodge…" | `_HOW_QUESTION`, `_WANTS` in [registration.py](../ari_app/registration.py) |
| **Backing out** of a complaint ("I don't want to lodge a complaint", "cancel", "never mind", "not now", "rehne do", "nahi chahiye") | `wants_cancel()`, `_NEG_WANT`, `_ABOUT_COMPLAINT`, `_CANCEL_PHRASE` in [registration.py](../ari_app/registration.py) |
| Yes / no on the confirmation ("yes", "correct", "haan", "sahi" / "no", "wrong", "galat") | `_YES`, `_NO` in [registration.py](../ari_app/registration.py) |
| Sound-alikes of "skip" on phone audio ("keep", "kip", "ship"…, only in replies of 3 words or fewer) | `_SKIP_SOUNDALIKE` in [registration.py](../ari_app/registration.py) |
| "Existing / status / update / purani…" vs "new / another / naya…" | `_EXISTING` and `_NEW` in [registration.py](../ari_app/registration.py) (line ~159) |
| Words that mean skip ("skip", "no", "nahi", "नहीं"…) | `_SKIP` in [registration.py](../ari_app/registration.py) |
| Spoken digit words ("one", "ek", "एक", "double"…) | `_DIGIT_WORDS` in [registration.py](../ari_app/registration.py) |

These are regular expressions. To add a word, add it inside the `( … | … )` group, separated by `|`.

### 1d. Instructions for small helper tasks

| What | File and place |
|---|---|
| Picking the name out of "yeah hi, this is Shyam here" | `extract_name()` in [registration.py](../ari_app/registration.py) (line ~108) |
| Picking answers out of question replies | `extract_answer()` in [registration.py](../ari_app/registration.py) (line ~273) |
| How document (RAG) facts are presented to the LLM | `format_context()` in [rag.py](../ari_app/rag.py) (line ~313) |
| How web search results are presented | `format_context_for_llm()` in [web_search.py](../ari_app/web_search.py) (line ~598) |

### 1e. `.env` settings (no code)

| Setting | Effect |
|---|---|
| `REPLY_MAX_SENTENCES=2`, `REPLY_MAX_CHARS=220` | How long AI replies can be (fixed sentences are never cut) |
| `LLM_TEMPERATURE=0.7` | Lower is more predictable; higher is more varied |
| `LLM_MAX_TOKENS=160` | A hard cap on reply length |
| `CALL_PRIMARY_LANG=en` | Language of the greeting |
| `VAD_END_SILENCE_MS=700` | How long a pause ends the caller's turn |
| `VAD_MIN_SPEECH_MS=100` | Shortest sound that counts as an answer (see "Short answers" below) |

---

## 2. Adding questions

### 2a. Registration questions

**File: [registration_questions.json](../registration_questions.json).** No code or database changes are needed. The questions are asked in the order listed, right after the name and PIN are saved, and each answer is saved to `users.profile` (JSON) as soon as it's given.

Add an entry to the `"questions"` list:

```json
{
  "questions": [
    { "key": "city", ... },
    { "key": "alternate_phone", ... },

    {
      "key": "email",
      "type": "text",
      "required": false,
      "ask": {
        "en": "What is your email address? You can also say skip.",
        "hi": "आपका ईमेल पता क्या है? आप स्किप भी बोल सकते हैं।"
      }
    },
    {
      "key": "customer_type",
      "type": "choice",
      "options": ["individual", "business"],
      "required": true,
      "ask": { "en": "Are you calling as an individual or for a business?" }
    },
    {
      "key": "pincode",
      "type": "digits",
      "min_len": 6,
      "max_len": 6,
      "ask": { "en": "What is your area pincode? Say it, or type it and press hash." }
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `key` | The name it's saved under: lowercase with underscores, e.g. `date_of_birth` |
| `type` | `text`: a short answer the LLM extracts (city, email, company)<br>`digits`: a number, spoken or typed with `#` at the end, with `min_len` / `max_len`<br>`choice`: one of `options`; the LLM maps "it's for my company" to `business`<br>`free`: the caller's exact words (long answers) |
| `ask.en` | The question in English (required) |
| `ask.hi` | The question in Hindi (optional; used on Hindi calls) |
| `required` | `false` lets the caller say "skip". `true` asks twice, then moves on without an answer |

**Tips**
- Keep it to 2–4 questions; each one adds about 10 seconds to the call. The bot announces the count automatically ("Just 3 quick questions more").
- For optional questions, mention "you can say skip" in the question.
- For digits, mention "type it and press hash", because the keypad is more reliable than speech for long numbers.
- A trailing comma or a missing quote breaks the file, and the bot then asks **no** questions and logs an error. Check it before restarting:

  ```bash
  python -c "import json; json.load(open('registration_questions.json')); print('ok')"
  ```

**Where the answers go**

```sql
SELECT name, profile->>'city' AS city, profile->>'email' AS email FROM users;
```

They also show under "Profile" on the recordings page (`http://127.0.0.1:8765/recordings`).

### 2b. Complaint questions and confirmation

**File: [complaint_questions.json](../complaint_questions.json).** It uses exactly the same format; the answers go to `complaints.details` (JSON). The defaults are `category` (choice), `order_number` (optional digits) and `description` (`free`). After the last question the bot **reads the complaint back and asks for confirmation** before saving:

> "Let me read that back. Category: delivery. Order number: 4 5 6 7. You said: the parcel arrived three days late. Shall I register this complaint? Please say yes or no."

| Caller says | Result |
|---|---|
| "yes", "correct", "haan", "sahi hai" | saved, and the ticket number is read out |
| "no", "that's wrong", "galat" | "Okay, let's correct it." The questions are asked again, then the readback again. A second "no" means it's **not** registered |
| unclear twice | not registered ("Okay, I haven't registered it…") |
| hangs up before confirming | nothing is saved |

Required questions (`"required": true`, like the description) don't accept "skip": the bot says "I need this to register the complaint" and asks again. The readback wording is in `_LINES`: `confirm_intro`, `confirm_field`, `confirm_said`, `confirm_ask`, `confirm_again`, `confirm_fix`, `confirm_cancel` and `required_answer`.

---

## 3. PIN check, asking about complaints, and returning callers

### PIN at the start of the call (`VERIFY_PIN_ON_CALL=true`, the default)

A registered caller is asked for their PIN **before** anything personal is said:

1. "Welcome back, Siam! For your security, please say or type your 4 digit PIN."
2. **Correct PIN** (spoken or typed): "Thank you, you're verified." Then the open-complaint offer below, or "How can I help you today?". The caller is **never asked for a ticket or complaint number**; the bot tells them.
3. **Wrong PIN** (up to 3 tries): "Sorry, I couldn't verify your PIN, so I can't share your account details on this call. I can still help with general questions." For the rest of the call the bot doesn't use the name, doesn't mention tickets, and refuses status questions and new complaints ("…because the PIN wasn't verified"). It also never re-registers the number, which would overwrite the real customer's PIN.

Set `VERIFY_PIN_ON_CALL=false` in `.env` to greet registered callers by name without a PIN. `AUTH_VERIFY_PIN=true` is a separate, older option that asks for the PIN only before a complaint; it's skipped for callers already verified at the start.

Code: `CallSession._identify_caller()` calls `_verify_returning_caller(user, first_key="verify_start", fail_key="verify_start_failed")`. The lines are `verify_start`, `verify_start_failed`, `welcome_verified`, `offer_one_v`, `offer_many_v` and `verify_needed` in `_LINES`.

### Asking about complaints

**Only a clear request starts a complaint:** "I want to lodge / file / register a complaint", "I have a complaint", "I want to complain", "a new complaint", "shikayat darj karni hai". Just mentioning complaints, or asking how they work, is answered normally by the LLM.

**The caller can back out at any time** (during registration, the questions or the confirmation): "I don't want to lodge a complaint", "cancel", "never mind", "not now", "rehne do". The bot says "Okay, I won't register a complaint. What else can I help you with?", saves nothing, and the caller can still lodge one later in the same call. So a description like "I don't want this product, it's broken" doesn't cancel anything, a negated "want" only counts as a cancel when it's about the complaint itself (lodge, complaint, ticket…). Short phrases like "cancel" and "never mind" always count. In code, `_hear_once()` raises `ComplaintCancelled`, and `_complaint_gate()` catches it.

**At any point in a call**, "How many complaints do I have?", "What's the status of my complaint?" or "I already registered a complaint, what happened?" read out the caller's open tickets, or "You don't have any open complaints right now". These questions **never** start a new complaint. If a caller asks one in the middle of the complaint questions, the intake stops and the status is given instead. Only "I want to lodge / file / register a complaint", "I have a complaint" and similar start a new one.



When a **registered caller** calls, the bot looks up their complaints that aren't closed (`status <> 'closed'`), newest first.

### What the caller hears

| Situation | Greeting |
|---|---|
| No open complaints | "Welcome back, Shyam Kumar! How can I help you today?" |
| One open complaint | "Welcome back, Shyam Kumar! I see you have an open complaint, ticket 1 0 0 0 0 3, about delivery. Would you like an update on it, or do you want to lodge a new complaint?" |
| Several open complaints | "…I see you have 2 open complaints, the latest is ticket 1 0 0 0 0 5, about payment. Would you like an update on them, or…" |

### What happens with the reply

| The caller says… | Detected as | The bot… |
|---|---|---|
| "update", "status", "existing", "my old complaint", "purani wali", "पिछली" | `existing` | reads each open ticket (up to 3): "Ticket 1 0 0 0 0 3, about delivery, raised on 6 November, is still open. The order number on it is 4 5 6 7. Our team is working on it. Is there anything else I can help you with?" |
| "a new one", "another complaint", "naya", "नई शिकायत", or any clear "I want to lodge a complaint…" | `new` | goes straight to the complaint questions (with a PIN check first if `AUTH_VERIFY_PIN=true`), then reads out the new ticket number |
| anything else ("what courses do you offer?") | neither | answers normally. The offer is dropped, but the LLM still knows about the open tickets |

If both are mentioned ("not the old one, a new one"), the one said last wins.

**For the rest of the call** the LLM is given the open tickets ("open complaints: 100003 (delivery, open, 06 Nov)"), so questions like "what was my ticket number again?" are answered correctly later on.

### Where it lives in the code

| Part | File and function |
|---|---|
| Fetching open complaints | `db.open_complaints()` in [db.py](../ari_app/db.py) |
| Choosing the greeting | `CallSession._identify_caller()` in [call_session.py](../ari_app/call_session.py) |
| Reading the reply | `registration.complaint_choice()` (keywords `_EXISTING`, `_NEW`) |
| Handling the choice | `CallSession._handle_complaint_choice()` |
| Reading the ticket status | `CallSession._tell_complaint_status()` |
| The sentences | `_LINES`: `offer_one`, `offer_many`, `status_item`, `status_order`, `status_end`, `status_open`, `status_in_progress`, `status_resolved` |

### Resolution note: a message from your team to the customer

On **http://127.0.0.1:8765/customers**, each ticket has a **"Resolution / message to customer"** box. Whatever you save there is read out by the bot, word for word, when that customer asks about the complaint:

> "Ticket 1 0 0 0 0 3, about payment, raised on 6 November, is being worked on. **Here is the update from our team: We found the double charge and have started a refund. It will reach your account in three to five working days.**"

Write it the way you'd say it on the phone: short sentences, no bullet points, links or abbreviations. It's stored in `complaints.resolution` (up to 1000 characters) and is also given to the LLM, so follow-up questions about it are answered correctly. The LLM is told it **cannot change, close or resolve** a complaint itself. If a caller asks "mark it as resolved", it says the team will update the ticket.

### Ticket status values

A new ticket is `open`. The easy way to change it is the **status dropdown on http://127.0.0.1:8765/customers** (Complaints tab, or a customer's page). It saves immediately, and the bot reads the new status on the customer's next call. You can also use SQL:

```sql
UPDATE complaints SET status = 'in_progress', updated_at = now() WHERE ticket_no = '100003';
UPDATE complaints SET status = 'resolved',    updated_at = now() WHERE ticket_no = '100003';
UPDATE complaints SET status = 'closed',      updated_at = now() WHERE ticket_no = '100003';  -- no longer offered
```

| `status` | Spoken as | Offered at call start |
|---|---|---|
| `open` | "is still open" | yes |
| `in_progress` | "is being worked on" | yes |
| `resolved` | "is resolved" | yes (until closed) |
| `closed` | – | no |
| anything else | the word itself, e.g. `on_hold` → "on hold" | yes |

To add a status with its own wording, add `"status_<name>": "…"` in `_LINES` for both `en` and `hi`.

---

## 4. Keypad (DTMF) input: how it works, line by line

When a caller presses a key, the phone sends it as a **DTMF** event (RFC 2833, not as sound). Asterisk passes it to the bot over the ARI WebSocket.

### Step 1: the event arrives ([ari_loop.py](../ari_app/ari_loop.py), line ~161)

```python
if et == "ChannelDtmfReceived":
    cid = (ev.get("channel") or {}).get("id")
    if cid and cid in sessions:
        sessions[cid].on_dtmf(ev.get("digit") or "")   # "1", "#", "*" …
```

### Step 2: the digit is buffered ([call_session.py](../ari_app/call_session.py), `on_dtmf()`, line ~1665)

```python
def on_dtmf(self, digit):
    if digit == "#" and not any(d.isdigit() for d in self._dtmf_digits):
        return                      # stray "#" (end key of the previous entry)
    if digit == "*":
        self._dtmf_digits.clear()   # * = start again
    else:
        self._dtmf_digits.append(digit)
    log.info("keypad key received (%s buffered)", len(self._dtmf_digits))  # never the digit
    self._dtmf_event.set()          # wake up whoever is waiting for keys
```

### Step 3: waiting for a complete entry (`_wait_dtmf_pin()` line ~1691, `_wait_dtmf_number()` line ~1679)

- **PIN** (`_wait_dtmf_pin(n)`): done when `n` digits are typed (4 by default), or `#` after at least one digit. It gives up after 6 s without a key press.
- **Other numbers** (`_wait_dtmf_number(min, max)`, for example the order number or alternate phone): done at `#`, or when `max_len` digits are typed. After 8 s without a key, it accepts what was typed if the length is between `min_len` and `max_len`.

### Step 4: voice and keypad race each other (`_ask_pin()` line ~1897, `_ask_question()` line ~1945)

```python
self._dtmf_digits.clear()                      # keys pressed during the question still count
await self.say_text(prompt, ..., trim=False)   # "…say it, or type it and press hash"
voice = asyncio.create_task(self._hear_once(stt_prompt=_PIN_STT_PROMPT))   # listen
keys  = asyncio.create_task(self._wait_dtmf_pin(n))                        # watch keypad
done, _ = await asyncio.wait({voice, keys}, return_when=asyncio.FIRST_COMPLETED)
if keys in done and keys.result():             # keypad finished first
    pin = keys.result()
    await self._cancel_listen(voice)           # stop listening, re-arm barge-in VAD
else:                                          # speech finished first
    answer = await voice
    pin = registration.parse_pin(answer, n)    # "one two three four" -> "1234"
    if pin is None and self._dtmf_digits:      # mic timed out while they were typing
        pin = await keys
```

Whichever finishes first wins. The caller can say "one two three four", "1 2 3 4", "ek do teen char" or "double five…" (`parse_pin()` / `parse_digits()` in [registration.py](../ari_app/registration.py) at line ~50 and ~323, using the `_DIGIT_WORDS` table at line ~25), **or** type it.

### Step 5: what happens with it

- **Registration:** two entries must match → `db.hash_pin()` (scrypt) → `users.pin_hash`.
- **PIN check:** `db.verify_pin(pin, user.pin_hash)` (a constant-time compare).
- **Number questions:** saved to `users.profile` / `complaints.details`.

**Privacy:** digits are never logged, spoken back, put in chat history or sent to the LLM. The logs show only "PIN entered on keypad", "keypad key received (3 buffered)" and "<10 digits>".

### Changing it

| You want… | Change |
|---|---|
| A 6-digit PIN | `.env` `PIN_LENGTH=6` |
| A keypad number in a new question | add a `"type": "digits"` question with `min_len` / `max_len` to the JSON file |
| A different "finish" or "clear" key | `on_dtmf()` / `_wait_dtmf_*()` (`"#"` and `"*"`) |
| Longer typing time | `idle_s=` in `_wait_dtmf_pin()` (6 s) / `_wait_dtmf_number()` (8 s) |

### Testing without a phone

In a scripted call, `SendDTMF(1234#)` presses the keys (see the end of this doc).

---

## 5. Ending the call (hang-up): how it works, line by line

The bot ends the call itself, but **always asks first**.

### Step 1: does the caller sound done? (`registration.wants_to_end()`, line ~578)

Matches `_CLOSING` (line ~560): "bye", "that's all", "nothing else", "no thanks", "I'm done", "end the call", "bas", "dhanyavaad", "shukriya", "अलविदा"…

- **"thank you"** alone (4 words or fewer) counts; "Thank you, but what about my refund?" does not.
- **A plain "no"** counts **only** right after the bot asked "anything else?" (the bot's last line is checked).

### Step 2: ask before hanging up (`CallSession._maybe_end_call()`, line ~2297)

```python
if not registration.wants_to_end(user_text, self._last_prompt):
    return None                                    # normal turn
answer = await self._ask(line("end_ask"))          # "Shall I end the call now? Please say yes,
                                                   #  or tell me what else you need."
decision = registration.end_reply(answer)          # 'end' | 'stay' | 'other'
if decision == "end":
    await self.say_text(line("goodbye", name=...)) # "Thank you for calling, Siam. … goodbye!"
    self._ended_by_bot = True
    return "end"
if decision == "stay":
    await self.say_text(line("end_stay"))          # "Sure, what else can I help you with?"
    return "stay"
return answer                                      # they asked something new instead
```

`end_reply()` (line ~593):
- **end:** yes, end, bye, hang up, done, "haan kar do".
- **stay:** a short "no".
- **other:** anything longer ("Actually, what are your hours?"), which is answered in the same turn.

### Step 3: the main loop acts on it (`run_loop()`, line ~2735)

```python
ending = await self._maybe_end_call(user_text)
if ending == "end":
    break                 # leave the conversation loop
if ending == "stay":
    continue              # listen again
if ending:
    user_text = ending    # process their new request normally
```

### Step 4: the actual hang-up (`run_loop()` `finally:`, line ~2897, and `hangup()`, line ~344)

```python
finally:
    await self._teardown_live_vad()      # remove snoop/externalMedia channels + bridge
    await self.hangup()                  # DELETE /ari/channels/{id}  → the phone hangs up
    await self._db_call_ended()          # calls row: end time, duration, turns
    await self._finish_call_recording()  # compress + save the recording
```

`hangup()` sends `DELETE /channels/{id}` to Asterisk, which ends the call on the phone. `say_text()` only returns after the goodbye has finished playing, so the caller always hears it in full.

### Other ways a call ends

| Case | What happens |
|---|---|
| Caller hangs up | Asterisk sends `ChannelDestroyed` → `mark_channel_dead()` (line ~331) → loop stops → the same `finally:` cleanup (hangup is skipped) |
| Silence | After `RECORD_MAX_CONSECUTIVE_EMPTY` (8) empty listens, `_skip_turn_no_usable_speech()` (line ~805) says "I can't hear you, so I'll end the call now… Goodbye!" (`silence_goodbye`), then the loop ends |
| Error | "Sorry, something went wrong. Goodbye." then hang-up |

### Changing it

| You want… | Change |
|---|---|
| More or different "I'm done" words | `_CLOSING`, `_THANKS`, `_PLAIN_NO` in [registration.py](../ari_app/registration.py) |
| Different question or goodbye | `end_ask`, `end_stay`, `goodbye`, `silence_goodbye` in `_LINES` |
| Hang up without asking | in `_maybe_end_call()`, skip the `_ask(...)` and set `decision = "end"` (not recommended) |
| Wait longer before giving up on silence | `.env` `RECORD_MAX_CONSECUTIVE_EMPTY` and `VAD_NO_SPEECH_TIMEOUT_SECONDS` |

---

## 6. Phone echo

Many phones send a little of the bot's own voice back up the line. In real calls this made the bot "hear" the end of its own question ("…in a few sentences", "…otherwise say skip") and treat it as the caller's answer. It then asked the question again, or rejected a real "skip". Three things now prevent that:

1. **Questions can't be interrupted** (`say_text(..., trim=False)` sets `_allow_barge = False`, line ~1521). Echo can't cut a question short.
2. **Fresh listening after each question** (`_hear_once()`, line ~1821): a 0.3 s pause, then `vad.arm("listen")`, which throws away anything heard while the bot was talking.
3. **The echo is removed from the answer** (`registration.strip_echo()`, line ~517): if the answer starts with the **end** of the bot's question (or 3+ words from it), that part is cut. "otherwise say skip. Skip." becomes "Skip.", and "In a few sentences." alone is ignored. The bot keeps listening **without asking again**. Answers matching the question's `options` ("customer service") are never treated as echo.

The log shows what happened each time: `heard: '…'` or `ignored '…' (echo of the question); still listening`.

---

## Short answers like "skip", "no" or "haan"

Phone audio also changes the words: in a real call, "skip" was transcribed as "**Let's keep.**" Short replies (three words or fewer) that sound like skip ("keep", "kip", "ship") are therefore accepted as skip on optional questions.

One-word answers are also very short: "skip" is about 190 ms of speech. The voice detector used to need at least 200 ms before treating sound as an answer, so short words were often ignored, and the bot waited 8 seconds and asked again. The minimum is now `VAD_MIN_SPEECH_MS=100`.

- **Short words still ignored:** lower it to `80`.
- **Background noise starting turns:** raise it to `150`. Noise that does get through becomes an empty transcript and the bot just keeps listening.
- **Barge-in is separate:** interrupting the bot while it speaks still needs 400 ms (`BARGE_IN_MIN_SPEECH_MS`), so short noises won't cut it off.

---

## Testing a change without a phone

You can place a scripted call from inside Asterisk. It plays recorded clips as the caller and sets a caller id:

```bash
A="docker exec callbot-asterisk asterisk -rx"
$A 'dialplan add extension 7997,1,Set(CALLERID(num)=1001) into from-internal-app'
$A 'dialplan add extension 7997,2,Goto(from-internal-app,7001,1) into from-internal-app'
# clips in docker/data/sounds/custom/<name>.wav; Wait() gives the bot time to talk
$A 'dialplan add extension s,1,Wait(8) into test-caller'
$A 'dialplan add extension s,2,Playback(custom/my-question) into test-caller'
$A 'dialplan add extension s,3,Wait(15) into test-caller'
$A 'dialplan add extension s,4,Hangup() into test-caller'
$A 'channel originate Local/7997@from-internal-app/n extension s@test-caller'
```

To send keypad digits instead of speech, use `SendDTMF(1234#)` in place of `Playback(…)`. Then follow `logs/callbot.log`. These dialplan additions disappear when the Asterisk container restarts.
