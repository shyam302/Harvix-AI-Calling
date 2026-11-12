"""Memory of past (already categorized) complaints, used as context for the LLM.

You load your previous complaints once - with their correct category - and
they are saved to a local JSON file or to Postgres. Every time a new
complaint is categorized, the most similar past complaints are given to the
model as examples, so it follows the way you categorized before.

Where the context is stored
---------------------------
    file (default) : complaint_context.json next to this script
    db             : Postgres table "complaint_examples" (uses DATABASE_URL)

Command line
------------
    # load past complaints (Excel needs a complaint column and a category column)
    python complaint_context.py load past_complaints.xlsx
    python complaint_context.py load past.xlsx --text-column "Complaint" --category-column "Category"
    python complaint_context.py load past.xlsx --db          # save into Postgres instead
    python complaint_context.py load past_complaints.json    # JSON works too:
        # [{"complaint": "charged twice", "category": "Billing"}, ...]

    python complaint_context.py stats                        # how many per category
    python complaint_context.py search "I was billed twice"  # see what the LLM would get
    python complaint_context.py clear                        # delete all saved context

From Python
-----------
    from complaint_context import ContextStore
    store = ContextStore()            # or ContextStore(backend="db")
    store.add("app keeps crashing", "Technical")
    store.similar("app not opening", k=5)   # -> [(text, category), ...]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

# ===========================================================================
# Settings - edit freely
# ===========================================================================
CONTEXT_FILE = Path(__file__).with_name("complaint_context.json")
DB_TABLE = "complaint_examples"
DEFAULT_EXAMPLES = 5        # how many similar past complaints go to the LLM
MAX_EXAMPLE_CHARS = 300     # long complaints are cut to this length in the prompt
# ===========================================================================

_WORD = re.compile(r"\w+", re.UNICODE)
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "i", "my", "me", "to", "of", "and",
    "in", "for", "on", "it", "this", "that", "not", "has", "have", "be", "with",
    "please", "hai", "ho", "ka", "ki", "ke", "se", "me", "mera", "meri", "mujhe", "ko",
}


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 1]


def _load_env() -> None:
    env = Path(__file__).with_name(".env")
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class ContextStore:
    """Past complaints + categories, with simple similarity search (TF-IDF)."""

    def __init__(self, backend: str = "file") -> None:
        if backend not in ("file", "db"):
            raise ValueError("backend must be 'file' or 'db'")
        self.backend = backend
        self._items: list[tuple[str, str]] | None = None  # cache of (text, category)
        self._index = None

    # ------------------------------------------------------------ storage --
    def _read_all(self) -> list[tuple[str, str]]:
        if self.backend == "file":
            if not CONTEXT_FILE.exists():
                return []
            data = json.loads(CONTEXT_FILE.read_text(encoding="utf-8") or "[]")
            return [(d["text"], d["category"]) for d in data]
        return asyncio.run(self._db_read())

    def _write_new(self, items: list[tuple[str, str]]) -> int:
        """Save items that are not stored yet. Returns how many were added."""
        if self.backend == "db":
            return asyncio.run(self._db_insert(items))
        existing = self._read_all()
        seen = {t for t, _ in existing}
        new = [(t, c) for t, c in items if t not in seen and not seen.add(t)]
        rows = [{"text": t, "category": c} for t, c in existing + new]
        CONTEXT_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        return len(new)

    async def _db_conn(self):
        import asyncpg

        _load_env()
        conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
        await conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {DB_TABLE} (
                   id BIGSERIAL PRIMARY KEY,
                   text TEXT NOT NULL UNIQUE,
                   category TEXT NOT NULL,
                   created_at TIMESTAMPTZ NOT NULL DEFAULT now())"""
        )
        return conn

    async def _db_read(self) -> list[tuple[str, str]]:
        conn = await self._db_conn()
        try:
            rows = await conn.fetch(f"SELECT text, category FROM {DB_TABLE} ORDER BY id")
            return [(r["text"], r["category"]) for r in rows]
        finally:
            await conn.close()

    async def _db_insert(self, items: list[tuple[str, str]]) -> int:
        conn = await self._db_conn()
        try:
            before = await conn.fetchval(f"SELECT count(*) FROM {DB_TABLE}")
            await conn.executemany(
                f"INSERT INTO {DB_TABLE}(text, category) VALUES($1, $2) ON CONFLICT (text) DO NOTHING",
                items,
            )
            return await conn.fetchval(f"SELECT count(*) FROM {DB_TABLE}") - before
        finally:
            await conn.close()

    def clear(self) -> None:
        if self.backend == "file":
            CONTEXT_FILE.unlink(missing_ok=True)
        else:
            async def _clear():
                conn = await self._db_conn()
                try:
                    await conn.execute(f"TRUNCATE {DB_TABLE}")
                finally:
                    await conn.close()
            asyncio.run(_clear())
        self._items = self._index = None

    # --------------------------------------------------------------- API ---
    def all(self) -> list[tuple[str, str]]:
        if self._items is None:
            self._items = self._read_all()
        return self._items

    def add(self, text: str, category: str) -> bool:
        return self.add_many([(text, category)]) == 1

    def add_many(self, items) -> int:
        clean = [(str(t).strip(), str(c).strip()) for t, c in items if str(t).strip() and str(c).strip()]
        added = self._write_new(clean)
        self._items = self._index = None  # reload on next use
        return added

    def add_from_file(self, path, text_column=None, category_column=None, sheet=None) -> int:
        """Load past complaints from .xlsx/.xls, .csv or .json.

        JSON format: a list of objects, e.g.
            [{"complaint": "charged twice", "category": "Billing"}, ...]
        (key names are detected the same way as Excel column names)."""
        import pandas as pd

        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(data, dict):  # {"complaints": [...]} is fine too
                data = next((v for v in data.values() if isinstance(v, list)), [])
            df = pd.DataFrame(data)
        elif suffix == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, sheet_name=sheet or 0)
        tcol = text_column or _pick(df, r"complaint|description|issue|details|summary|text|remark")
        ccol = category_column or _pick(df, r"categor|type|class|label", text=False)
        for name, col in (("text", tcol), ("category", ccol)):
            if col not in df.columns:
                sys.exit(f"{name} column {col!r} not found. Columns: {list(df.columns)}")
        print(f"Using text column {tcol!r} and category column {ccol!r}")
        df = df[[tcol, ccol]].dropna()
        return self.add_many(zip(df[tcol].astype(str), df[ccol].astype(str)))

    def similar(self, text: str, k: int = DEFAULT_EXAMPLES) -> list[tuple[str, str]]:
        """The k past complaints most similar to `text` (TF-IDF cosine)."""
        items = self.all()
        if not items or k <= 0:
            return []
        if self._index is None:
            docs = [Counter(_tokens(t)) for t, _ in items]
            df = Counter(w for d in docs for w in d)
            n = len(docs)
            idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}
            vecs = []
            for d in docs:
                v = {w: c * idf[w] for w, c in d.items()}
                norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
                vecs.append({w: x / norm for w, x in v.items()})
            self._index = (idf, vecs)
        idf, vecs = self._index
        q = {w: c * idf.get(w, 0.0) for w, c in Counter(_tokens(text)).items()}
        qn = math.sqrt(sum(x * x for x in q.values())) or 1.0
        scored = [
            (sum(x * v.get(w, 0.0) for w, x in q.items()) / qn, i) for i, v in enumerate(vecs)
        ]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(reverse=True)
        return [items[i] for _, i in scored[:k]]

    def prompt_block(self, text: str, k: int = DEFAULT_EXAMPLES) -> str:
        """Text to put in the LLM prompt. Empty string if nothing similar."""
        examples = self.similar(text, k)
        if not examples:
            return ""
        lines = [f'- "{t[:MAX_EXAMPLE_CHARS]}" -> {c}' for t, c in examples]
        return "Past complaints and how they were categorized:\n" + "\n".join(lines)

    def stats(self) -> Counter:
        return Counter(c for _, c in self.all())


def _pick(df, pattern: str, text: bool = True) -> str:
    cols = [c for c in df.columns if re.search(pattern, str(c), re.I)]
    if text:
        cols = [c for c in cols if not re.search(r"\b(id|no|number|date|code)\b", str(c), re.I)]
        return max(cols or list(df.columns), key=lambda c: df[c].fillna("").astype(str).str.len().mean())
    return cols[0] if cols else ""


def main() -> None:
    p = argparse.ArgumentParser(description="Manage past-complaint context for the categorizer.")
    p.add_argument("--db", action="store_true", help="use Postgres instead of the local JSON file")
    sub = p.add_subparsers(dest="cmd", required=True)
    ld = sub.add_parser("load", help="load past complaints from Excel, CSV or JSON")
    ld.add_argument("file", help=".xlsx, .csv or .json")
    ld.add_argument("--text-column")
    ld.add_argument("--category-column")
    ld.add_argument("--sheet")
    sub.add_parser("stats", help="count saved complaints per category")
    sr = sub.add_parser("search", help="show similar past complaints")
    sr.add_argument("text")
    sr.add_argument("-k", type=int, default=DEFAULT_EXAMPLES)
    sub.add_parser("clear", help="delete all saved context")
    args, rest = p.parse_known_args()
    if "--db" in rest:  # allow --db after the sub-command too
        args.db = True

    store = ContextStore("db" if args.db else "file")
    where = f"Postgres table {DB_TABLE}" if args.db else str(CONTEXT_FILE)
    if args.cmd == "load":
        added = store.add_from_file(args.file, args.text_column, args.category_column, args.sheet)
        print(f"Added {added} new complaints to {where} (total {len(store.all())})")
    elif args.cmd == "stats":
        s = store.stats()
        print(f"{sum(s.values())} complaints in {where}")
        for cat, n in s.most_common():
            print(f"  {cat:<20} {n}")
    elif args.cmd == "search":
        for t, c in store.similar(args.text, args.k):
            print(f"{c:<18} {t[:100]}")
    elif args.cmd == "clear":
        store.clear()
        print(f"Cleared {where}")


if __name__ == "__main__":
    main()
