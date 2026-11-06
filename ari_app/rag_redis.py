"""
RAG chunks in Redis Stack (RediSearch vector index) instead of local numpy files.

Same interface as rag.KnowledgeBase (refresh / warmup / maybe_refresh_in_background /
search), picked when RAG_STORE=redis. Documents still come from the knowledge/
folder; only the chunks and their embeddings live in Redis:

  rag:chunk:<id>     HASH  text, source, page, embedding (float32 bytes)
  rag:src:<relpath>  SET   chunk ids belonging to one file (to delete on change)
  rag:manifest       HASH  relpath -> "mtime,size" (what is already indexed)
  rag:meta           HASH  model, dim
  rag_idx            RediSearch HNSW index over rag:chunk:* (cosine)

Search is a KNN query inside Redis (~1 ms), plus a small keyword bonus for exact
names / codes / numbers, like the local store's BM25 bonus.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ari_app.rag import (
    KnowledgeBase,
    RagHit,
    _chunk,
    _tokens,
    load_document,
)

log = logging.getLogger(__name__)

INDEX = "rag_idx"
PREFIX = "rag:chunk:"
MANIFEST = "rag:manifest"
META = "rag:meta"


class RedisKnowledgeBase(KnowledgeBase):
    def __init__(self, docs_dir: Path, *, redis_url: str, model_name: str, **kw: Any) -> None:
        super().__init__(docs_dir, model_name=model_name, **kw)
        import redis

        self.r = redis.Redis.from_url(redis_url, decode_responses=False)
        self.redis_url = redis_url
        self._count = 0

    # `len(kb.chunks)` is used in startup logs
    @property
    def chunks(self) -> list:  # type: ignore[override]
        return [None] * self._count

    @chunks.setter
    def chunks(self, _value: list) -> None:
        pass  # parent __init__ assigns []; chunks live in Redis

    # index management
    def _dim(self) -> int:
        m = self._embedder()
        fn = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
        return int(fn())

    def _ensure_index(self) -> None:
        meta = {k.decode(): v.decode() for k, v in self.r.hgetall(META).items()}
        if meta and meta.get("model") != self.model_name:
            log.info("RAG(redis): embedding model changed; clearing the index")
            self._drop_all()
        try:
            self.r.ft(INDEX).info()
            return
        except Exception:
            pass
        from redis.commands.search.field import NumericField, TagField, TextField, VectorField
        from redis.commands.search.index_definition import IndexDefinition, IndexType

        dim = self._dim()
        self.r.ft(INDEX).create_index(
            [
                TextField("text"),
                TagField("source", separator="|"),
                NumericField("page"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"},
                ),
            ],
            definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH),
        )
        self.r.hset(META, mapping={"model": self.model_name, "dim": dim})
        log.info("RAG(redis): created index %s (dim=%s)", INDEX, dim)

    def _drop_all(self) -> None:
        try:
            self.r.ft(INDEX).dropindex(delete_documents=True)
        except Exception:
            pass
        for key in self.r.scan_iter(match=b"rag:*"):
            self.r.delete(key)

    def _remove_source(self, rel: str) -> None:
        src_key = f"rag:src:{rel}"
        ids = self.r.smembers(src_key)
        pipe = self.r.pipeline()
        for cid in ids:
            pipe.delete(PREFIX.encode() + cid)
        pipe.delete(src_key)
        pipe.hdel(MANIFEST, rel)
        pipe.execute()

    def _count_chunks(self) -> int:
        try:
            return int(self.r.ft(INDEX).info()["num_docs"])
        except Exception:
            return 0

    # ingestion
    def refresh(self) -> bool:
        if not self._refreshing.acquire(blocking=False):
            return False
        try:
            self._ensure_index()
            current = self._scan()
            self._last_scan = time.monotonic()
            saved = {
                k.decode(): [float(x) for x in v.decode().split(",")]
                for k, v in self.r.hgetall(MANIFEST).items()
            }
            changed = [p for p, sig in current.items() if saved.get(p) != sig]
            removed = [p for p in saved if p not in current]
            if not changed and not removed:
                self._count = self._count_chunks()
                return False

            t0 = time.perf_counter()
            for rel in removed + changed:
                self._remove_source(rel)
            for rel in sorted(changed):
                try:
                    pages = load_document(self.docs_dir / rel)
                except Exception as exc:
                    log.warning("RAG(redis): could not read %s: %s", rel, exc)
                    continue
                pieces = [
                    (piece, page)
                    for text, page in pages
                    for piece in _chunk(text, size=self.chunk_size, overlap=self.chunk_overlap)
                ]
                if pieces:
                    emb = self._embed([p for p, _ in pieces], kind="passage")
                    tag = hashlib.sha1(rel.encode()).hexdigest()[:12]
                    pipe = self.r.pipeline()
                    for i, ((piece, page), vec) in enumerate(zip(pieces, emb)):
                        cid = f"{tag}:{i}"
                        pipe.hset(
                            PREFIX + cid,
                            mapping={
                                "text": piece,
                                "source": rel,
                                "page": page or 0,
                                "embedding": vec.astype(np.float32).tobytes(),
                            },
                        )
                        pipe.sadd(f"rag:src:{rel}", cid)
                    pipe.execute()
                self.r.hset(MANIFEST, rel, ",".join(str(x) for x in current[rel]))
                log.info("RAG(redis): %s -> %s chunks", rel, len(pieces))
            self._count = self._count_chunks()
            log.info(
                "RAG(redis): index updated (%s changed, %s removed) -> %s chunks in %.1fs",
                len(changed),
                len(removed),
                self._count,
                time.perf_counter() - t0,
            )
            return True
        finally:
            self._refreshing.release()

    # retrieval
    def search(self, query: str, *, top_k: int = 3, min_score: float = 0.82) -> list[RagHit]:
        if not (query or "").strip() or self._count == 0:
            return []
        from redis.commands.search.query import Query

        vec = self._embed([query], kind="query")[0].astype(np.float32).tobytes()
        k = max(top_k * 3, top_k)
        q = (
            Query(f"*=>[KNN {k} @embedding $vec AS dist]")
            .sort_by("dist")
            .return_fields("text", "source", "page", "dist")
            .paging(0, k)
            .dialect(2)
        )
        res = self.r.ft(INDEX).search(q, query_params={"vec": vec})

        q_tokens = set(_tokens(query))
        scored = []
        for doc in res.docs:
            dense = 1.0 - float(doc.dist)  # cosine distance -> similarity
            text = doc.text if isinstance(doc.text, str) else doc.text.decode()
            overlap = len(q_tokens & set(_tokens(text))) / max(len(q_tokens), 1)
            scored.append((dense, dense + 0.03 * overlap, doc, text))
        scored.sort(key=lambda t: -t[1])

        hits = []
        for dense, score, doc, text in scored:
            if dense < min_score:
                continue
            page = int(float(doc.page)) or None
            source = doc.source if isinstance(doc.source, str) else doc.source.decode()
            hits.append(RagHit(text, source, page, score))
            if len(hits) >= top_k:
                break
        return hits


def make_redis_kb(docs_dir: Path, *, redis_url: str, model_name: str) -> KnowledgeBase | None:
    """Redis store if reachable, else None (caller falls back to the local store)."""
    try:
        import redis

        redis.Redis.from_url(redis_url, socket_connect_timeout=2).ping()
    except Exception as exc:
        log.error(
            "RAG: Redis unreachable at %s (%s); using the local index instead. "
            "Start it with: cd docker && docker compose up -d redis",
            _safe_url(redis_url),
            exc,
        )
        return None
    log.info("RAG: using Redis vector store at %s", _safe_url(redis_url))
    return RedisKnowledgeBase(docs_dir, redis_url=redis_url, model_name=model_name)


def _safe_url(url: str) -> str:
    """Hide the password when logging a redis:// URL."""
    if "@" in url:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://***@{rest.split('@', 1)[1]}"
    return url

