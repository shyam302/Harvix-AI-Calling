"""
Knowledge base (RAG) over your own documents: PDF, DOCX, TXT, MD.

Drop files into RAG_DOCS_DIR (default ``knowledge/``). On startup, and whenever the
folder changes, new or edited files are split into chunks and embedded with a
multilingual model (intfloat/multilingual-e5-small), so a caller asking in Hindi
still finds an English document. The index lives in ``<docs>/.rag_index/`` and only
changed files are re-embedded.

RAG_STORE=redis keeps the chunks in Redis Stack instead (see rag_redis.py); the
local store is the fallback when Redis is unreachable.

Each caller turn is matched against the chunks (dense similarity, plus BM25 keyword
scores to help with names, codes and numbers). Strong matches go to the LLM as
reference text; when nothing relevant is found, the call carries on as before
(general knowledge or web lookup).

CLI:
    python -m ari_app.rag ingest           # build / update the index
    python -m ari_app.rag search "query"   # see what a question would retrieve
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".pdf", ".docx", ".txt", ".md")
_INDEX_DIRNAME = ".rag_index"
_RESCAN_INTERVAL_S = 30.0


@dataclass
class RagHit:
    text: str
    source: str
    page: int | None
    score: float


# --- document loading --------------------------------------------------------------
def _load_pdf(path: Path) -> list[tuple[str, int | None]]:
    import fitz  # pymupdf

    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, 1):
            text = page.get_text("text")
            if text.strip():
                pages.append((text, i))
    return pages


def _load_docx(path: Path) -> list[tuple[str, int | None]]:
    import docx

    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return [("\n".join(parts), None)] if parts else []


def _load_text(path: Path) -> list[tuple[str, int | None]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [(text, None)] if text.strip() else []


def load_document(path: Path) -> list[tuple[str, int | None]]:
    """Return ``[(text, page_number_or_None), ...]`` for one file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    return _load_text(path)


def _chunk(text: str, *, size: int, overlap: int) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "। ", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )
    clean = re.sub(r"[ \t]+", " ", text)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return [c.strip() for c in splitter.split_text(clean) if len(c.strip()) >= 20]


_TOKEN = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


# --- index -------------------------------------------------------------------------
class KnowledgeBase:
    def __init__(
        self,
        docs_dir: Path,
        *,
        model_name: str = "intfloat/multilingual-e5-small",
        chunk_size: int = 700,
        chunk_overlap: int = 120,
    ) -> None:
        self.docs_dir = docs_dir
        self.index_dir = docs_dir / _INDEX_DIRNAME
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._model: Any = None
        self._lock = threading.Lock()  # guards index swap + model load
        self._refreshing = threading.Lock()
        self.chunks: list[dict[str, Any]] = []
        self.embeddings = np.zeros((0, 1), dtype=np.float32)
        self.manifest: dict[str, list[float]] = {}  # relpath -> [mtime, size]
        self._bm25: Any = None
        self._last_scan = 0.0

    # model
    def _embedder(self) -> Any:
        with self._lock:
            if self._model is None:
                import torch
                from sentence_transformers import SentenceTransformer

                device = "cuda" if torch.cuda.is_available() else "cpu"
                log.info("RAG: loading embedding model %s on %s", self.model_name, device)
                self._model = SentenceTransformer(self.model_name, device=device)
            return self._model

    def _embed(self, texts: list[str], *, kind: str) -> np.ndarray:
        from ari_app import remote

        if remote.enabled("embed"):  # Docker: embed container (services/embed_service.py)
            vecs = remote.post_json_sync(
                "embed", "/embed", {"texts": texts, "kind": kind, "model": self.model_name}
            )["vectors"]
            return np.asarray(vecs, dtype=np.float32)
        # e5 models expect "query: " / "passage: " prefixes.
        prefixed = [f"{kind}: {t}" for t in texts]
        vecs = self._embedder().encode(
            prefixed, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)

    # persistence
    def _load_saved(self) -> None:
        meta = self.index_dir / "meta.json"
        if not meta.is_file():
            return
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            if info.get("model") != self.model_name:
                log.info("RAG: embedding model changed; rebuilding index")
                return
            chunks = json.loads((self.index_dir / "chunks.json").read_text(encoding="utf-8"))
            emb = np.load(self.index_dir / "embeddings.npy")
        except (OSError, ValueError) as exc:
            log.warning("RAG: could not read saved index (%s); rebuilding", exc)
            return
        if len(chunks) != len(emb):
            log.warning("RAG: saved index is inconsistent; rebuilding")
            return
        self.chunks, self.embeddings = chunks, emb.astype(np.float32)
        self.manifest = info.get("manifest", {})
        self._rebuild_bm25()

    def _save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        (self.index_dir / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8"
        )
        np.save(self.index_dir / "embeddings.npy", self.embeddings)
        (self.index_dir / "meta.json").write_text(
            json.dumps({"model": self.model_name, "manifest": self.manifest}, indent=1),
            encoding="utf-8",
        )

    def _rebuild_bm25(self) -> None:
        if not self.chunks:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([_tokens(c["text"]) for c in self.chunks])

    # ingestion
    def _scan(self) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        if not self.docs_dir.is_dir():
            return found
        for root, dirs, files in os.walk(self.docs_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in files:
                if name.startswith(".") or not name.lower().endswith(SUPPORTED_SUFFIXES):
                    continue
                path = Path(root) / name
                st = path.stat()
                found[str(path.relative_to(self.docs_dir))] = [st.st_mtime, float(st.st_size)]
        return found

    def refresh(self) -> bool:
        """Embed new/changed files, drop deleted ones. Returns True if the index changed."""
        if not self._refreshing.acquire(blocking=False):
            return False  # another refresh is already running
        try:
            if not self.chunks and not self.manifest:
                self._load_saved()
            current = self._scan()
            self._last_scan = time.monotonic()
            changed = [p for p, sig in current.items() if self.manifest.get(p) != sig]
            removed = [p for p in self.manifest if p not in current]
            if not changed and not removed:
                return False

            t0 = time.perf_counter()
            stale = set(changed) | set(removed)
            keep = [i for i, c in enumerate(self.chunks) if c["source"] not in stale]
            chunks = [self.chunks[i] for i in keep]
            emb: np.ndarray | None = self.embeddings[keep] if keep else None

            new_chunks: list[dict[str, Any]] = []
            for rel in sorted(changed):
                try:
                    pages = load_document(self.docs_dir / rel)
                except Exception as exc:
                    log.warning("RAG: could not read %s: %s", rel, exc)
                    continue
                n = 0
                for text, page in pages:
                    for piece in _chunk(text, size=self.chunk_size, overlap=self.chunk_overlap):
                        new_chunks.append({"text": piece, "source": rel, "page": page})
                        n += 1
                log.info("RAG: %s -> %s chunks", rel, n)
            if new_chunks:
                new_emb = self._embed([c["text"] for c in new_chunks], kind="passage")
                chunks += new_chunks
                emb = new_emb if emb is None else np.vstack([emb, new_emb])
            if emb is None:
                emb = np.zeros((0, 1), dtype=np.float32)

            with self._lock:
                self.chunks, self.embeddings = chunks, emb
                self.manifest = current
                self._rebuild_bm25()
            self._save()
            log.info(
                "RAG: index updated (%s changed, %s removed) -> %s chunks from %s files in %.1fs",
                len(changed),
                len(removed),
                len(self.chunks),
                len(self.manifest),
                time.perf_counter() - t0,
            )
            return True
        finally:
            self._refreshing.release()

    def warmup(self) -> None:
        """Index the folder and load the embedding model before the first call."""
        from ari_app import remote

        if remote.enabled("embed"):
            remote.wait_ready("embed")
            self.refresh()
            return
        self.refresh()
        self._embedder()

    def maybe_refresh_in_background(self) -> None:
        """Pick up files added while the bot runs, without blocking a call."""
        if time.monotonic() - self._last_scan < _RESCAN_INTERVAL_S:
            return
        self._last_scan = time.monotonic()
        threading.Thread(target=self.refresh, name="rag-refresh", daemon=True).start()

    # retrieval
    def search(
        self, query: str, *, top_k: int = 3, min_score: float = 0.82
    ) -> list[RagHit]:
        """Chunks relevant to ``query``; empty when nothing clears ``min_score``."""
        with self._lock:
            chunks, emb, bm25 = self.chunks, self.embeddings, self._bm25
        if not chunks or not (query or "").strip():
            return []
        dense = emb @ self._embed([query], kind="query")[0]

        # Small BM25 bonus for exact names / codes / numbers the embedding may blur.
        bonus = np.zeros(len(chunks), dtype=np.float32)
        if bm25 is not None:
            raw = np.asarray(bm25.get_scores(_tokens(query)), dtype=np.float32)
            if raw.max() > 0:
                bonus = 0.03 * raw / raw.max()
        score = dense + bonus

        order = np.argsort(-score)[: max(top_k * 3, top_k)]
        hits = []
        for i in order:
            if dense[i] < min_score:
                continue
            c = chunks[int(i)]
            hits.append(RagHit(c["text"], c["source"], c.get("page"), float(score[i])))
            if len(hits) >= top_k:
                break
        return hits


def format_context(hits: list[RagHit]) -> str:
    """Reference text for the system prompt."""
    lines = [
        "\nReference information from our own documents (trust this over general "
        "knowledge; answer from it in short spoken sentences, never mention files, "
        "pages or 'documents', and if it does not contain the answer, say you are not "
        "sure rather than guessing):"
    ]
    for i, h in enumerate(hits, 1):
        text = re.sub(r"\s+", " ", h.text)
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)


_kb: KnowledgeBase | None = None
_kb_lock = threading.Lock()


def get_knowledge_base(docs_dir: str | Path, *, model_name: str) -> KnowledgeBase:
    global _kb
    with _kb_lock:
        if _kb is None:
            path = Path(docs_dir)
            if not path.is_absolute():
                # Relative to the project root, not wherever the bot was started.
                path = Path(__file__).resolve().parent.parent / path
            store = os.getenv("RAG_STORE", "local").strip().lower()
            if store == "redis":
                from ari_app.rag_redis import make_redis_kb

                _kb = make_redis_kb(
                    path,
                    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                    model_name=model_name,
                )
            if _kb is None:
                _kb = KnowledgeBase(path, model_name=model_name)
        return _kb


def _main() -> None:
    import argparse

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Knowledge base for the call bot")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="build / update the index")
    s = sub.add_parser("search", help="show what a question retrieves")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    kb = get_knowledge_base(
        os.getenv("RAG_DOCS_DIR", "knowledge"),
        model_name=os.getenv("RAG_EMBED_MODEL", "intfloat/multilingual-e5-small"),
    )
    kb.refresh()
    print(f"{len(kb.chunks)} chunks in {type(kb).__name__} from {kb.docs_dir}")
    if args.cmd == "search":
        min_score = float(os.getenv("RAG_MIN_SCORE", "0.82"))
        hits = kb.search(args.query, top_k=args.k, min_score=0.0)
        for h in hits:
            mark = "USED " if h.score >= min_score else "below"
            page = f" p.{h.page}" if h.page else ""
            print(f"\n[{mark} {h.score:.3f}] {h.source}{page}\n{h.text[:400]}")


if __name__ == "__main__":
    _main()
