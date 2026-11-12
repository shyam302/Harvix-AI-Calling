"""Categorize complaints with the same LLM the call bot uses (vLLM / Gemma).

Reads complaints from an Excel file (or, later, from Postgres), asks the model
to pick one category for each, and writes the result.

Usage
-----
    # Excel in -> Excel out (adds "category" and "category_reason" columns)
    python categorize_complaints.py complaints.xlsx
    python categorize_complaints.py complaints.xlsx --column "Complaint" --out result.xlsx

    # Postgres: read the bot's complaints table and save the category back
    python categorize_complaints.py --from-db
    python categorize_complaints.py --from-db --only-new      # skip already categorized

Model settings come from .env (VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY),
the database from DATABASE_URL - the same values the call bot uses.

EDIT THE TWO SECTIONS BELOW (CATEGORIES and PROMPT) to suit your data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ===========================================================================
# 1. CATEGORIES - edit freely. Name -> short description shown to the model.
# ===========================================================================
CATEGORIES: dict[str, str] = {
    "Billing": "wrong charges, refunds, invoices, payment problems",
    "Technical": "service not working, errors, outages, slow performance",
    "Delivery": "late, missing or damaged orders / shipments",
    "Account": "login, password, profile, registration, KYC",
    "Staff Behaviour": "rude or unhelpful staff, poor service attitude",
    "Product Quality": "defective or poor quality product",
    "Other": "anything that does not fit the categories above",
}

# Category used when the model's answer can't be matched to the list.
FALLBACK_CATEGORY = "Other"

# ===========================================================================
# 2. PROMPT - edit freely. {categories} and {complaint} are filled in.
# ===========================================================================
SYSTEM_PROMPT = """You are a customer-support analyst. You classify customer \
complaints into exactly one category from a fixed list. The complaint may be \
in English, Hindi or a mix. Answer only with JSON."""

USER_PROMPT = """Categories:
{categories}

Complaint:
\"\"\"{complaint}\"\"\"

Pick the single best category from the list above (use the exact name).
Reply with JSON only, like:
{{"category": "<category name>", "reason": "<one short sentence>"}}"""

# Model call settings
TEMPERATURE = 0.0
MAX_TOKENS = 120
# ===========================================================================


def load_env(path: Path = Path(__file__).with_name(".env")) -> None:
    """Load KEY=VALUE lines from .env without overriding real env vars."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def make_client():
    from openai import OpenAI

    return OpenAI(
        base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.getenv("VLLM_API_KEY") or "EMPTY",
        timeout=float(os.getenv("VLLM_TIMEOUT_SECONDS", "120")),
    )


def _match_category(name: str) -> str:
    name = (name or "").strip().lower()
    for cat in CATEGORIES:
        if cat.lower() == name:
            return cat
    for cat in CATEGORIES:  # loose match, e.g. "billing issue"
        if cat.lower() in name or name in cat.lower():
            return cat
    return FALLBACK_CATEGORY


def categorize(client, model: str, complaint: str) -> tuple[str, str]:
    """Return (category, reason) for one complaint text."""
    complaint = (complaint or "").strip()
    if not complaint:
        return FALLBACK_CATEGORY, "empty complaint"
    cats = "\n".join(f"- {name}: {desc}" for name, desc in CATEGORIES.items())
    resp = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(categories=cats, complaint=complaint)},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    if data:
        return _match_category(str(data.get("category", ""))), str(data.get("reason", "")).strip()
    return _match_category(text), ""  # model ignored JSON; try the raw text


# ---------------------------------------------------------------- Excel ----
def run_excel(args, client, model: str) -> None:
    import pandas as pd

    src = Path(args.excel)
    df = pd.read_excel(src, sheet_name=args.sheet or 0)
    col = args.column or _guess_column(df)
    if col not in df.columns:
        sys.exit(f"Column {col!r} not found. Columns: {list(df.columns)}")
    print(f"Reading {len(df)} rows from column {col!r}")

    cats, reasons = [], []
    for i, text in enumerate(df[col].fillna("").astype(str), 1):
        try:
            cat, why = categorize(client, model, text)
        except Exception as exc:  # keep going; mark the row
            cat, why = "ERROR", str(exc)[:200]
        cats.append(cat)
        reasons.append(why)
        print(f"[{i}/{len(df)}] {cat:<16} {text[:70]!r}")

    df["category"] = cats
    df["category_reason"] = reasons
    out = Path(args.out) if args.out else src.with_name(src.stem + "_categorized.xlsx")
    df.to_excel(out, index=False)
    print(f"\nSaved: {out}")
    print(df["category"].value_counts().to_string())


def _guess_column(df) -> str:
    """Pick the text column: name looks like complaint text (not an ID),
    and among those the one with the longest text on average."""
    def avg_len(c):
        return df[c].fillna("").astype(str).str.len().mean()

    named = [
        c for c in df.columns
        if re.search(r"complaint|description|issue|details|summary|text|remark", str(c), re.I)
        and not re.search(r"\b(id|no|number|date|code)\b", str(c), re.I)
    ]
    return max(named or list(df.columns), key=avg_len)


# ------------------------------------------------------------- Postgres ----
def run_db(args, client, model: str) -> None:
    """Read the bot's complaints table and store the category back in it."""
    import asyncio

    import asyncpg

    async def main() -> None:
        conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
        try:
            await conn.execute("ALTER TABLE complaints ADD COLUMN IF NOT EXISTS category TEXT")
            sql = "SELECT id, ticket_no, summary, details FROM complaints"
            if args.only_new:
                sql += " WHERE category IS NULL"
            rows = await conn.fetch(sql + " ORDER BY id")
            print(f"{len(rows)} complaints to categorize")
            for i, r in enumerate(rows, 1):
                details = r["details"]
                if isinstance(details, str):
                    details = json.loads(details or "{}")
                text = " ".join(filter(None, [r["summary"] or "", *map(str, (details or {}).values())]))
                cat, _ = categorize(client, model, text)
                await conn.execute("UPDATE complaints SET category=$1 WHERE id=$2", cat, r["id"])
                print(f"[{i}/{len(rows)}] {r['ticket_no']}: {cat}")
        finally:
            await conn.close()

    asyncio.run(main())


def main() -> None:
    load_env()
    p = argparse.ArgumentParser(description="Categorize complaints with the LLM.")
    p.add_argument("excel", nargs="?", help="input .xlsx file")
    p.add_argument("--column", help="column holding the complaint text (auto-detected)")
    p.add_argument("--sheet", help="sheet name (default: first sheet)")
    p.add_argument("--out", help="output .xlsx (default: <input>_categorized.xlsx)")
    p.add_argument("--from-db", action="store_true", help="use the Postgres complaints table")
    p.add_argument("--only-new", action="store_true", help="with --from-db: skip categorized rows")
    p.add_argument("--model", help="override VLLM_MODEL")
    args = p.parse_args()

    client = make_client()
    model = args.model or os.getenv("VLLM_MODEL", "google/gemma-4-E2B-it")
    if args.from_db:
        run_db(args, client, model)
    elif args.excel:
        run_excel(args, client, model)
    else:
        p.error("give an Excel file or --from-db")


if __name__ == "__main__":
    main()
