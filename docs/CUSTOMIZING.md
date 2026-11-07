# Customizing the bot: prompts, questions and complaints

This covers three things:

1. Which file to edit to change what the bot says or how it behaves.
2. How to add registration (and complaint) questions.
3. How returning callers with open complaints are handled.

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

### 2b. Complaint questions

**File: [complaint_questions.json](../complaint_questions.json).** It uses exactly the same format; the answers go to `complaints.details` (JSON). The defaults are `category` (choice), `order_number` (optional digits) and `description` (`free`). After the last question, the complaint is saved and the ticket number is read out.

---

## 3. Returning callers with open complaints

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
| "a new one", "another complaint", "naya", "नई शिकायत" | `new` | goes straight to the complaint questions (with a PIN check first if `AUTH_VERIFY_PIN=true`), then reads out the new ticket number |
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

### Ticket status values

A new ticket is `open`. Change it in the database, and the bot will read it out:

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

## Short answers like "skip", "no" or "haan"

One-word answers are very short: "skip" is about 190 ms of speech. The voice detector used to need at least 200 ms before treating sound as an answer, so short words were often ignored, and the bot waited 8 seconds and asked again. The minimum is now `VAD_MIN_SPEECH_MS=100`.

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
