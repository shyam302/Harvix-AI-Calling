"""
Live search: duckduckgo-search → crawl4ai (Playwright) → BeautifulSoup fallback
→ LangChain text chunking for LLM context.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import httpx

from ari_app.web_search import (
    SearchHit,
    _search_duckduckgo_sync,
    build_search_query,
    detect_search_topic,
)

log = logging.getLogger(__name__)

_MAX_CRAWL_TEXT = 2800
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
}
_CRICBUZZ_HOME = "https://www.cricbuzz.com/"
_CRICBUZZ_LIVE = "https://m.cricbuzz.com/live-cricket-scores"
_CRICBUZZ_LIVE_DESKTOP = "https://www.cricbuzz.com/cricket-match/live-scores"
_CRICBUZZ_SCHEDULE = (
    "https://www.cricbuzz.com/cricket-schedule/upcoming-series/league"
)
_CRICBUZZ_API_HOME = "https://www.cricbuzz.com/api/html/homepage-scag"


def _langchain_chunks(text: str, *, source: str, max_chunks: int = 2) -> list[str]:
    """Split crawled page text for the phone LLM (compact, factual chunks)."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) < 80:
        return [text] if text else []
    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=40,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        docs = splitter.split_documents(
            [Document(page_content=text, metadata={"source": source})]
        )
        return [d.page_content.strip() for d in docs[:max_chunks] if d.page_content.strip()]
    except ImportError:
        log.warning("langchain not installed; using raw crawl text")
        return [text[:600]]


def _extract_with_beautifulsoup(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "svg"]):
        tag.decompose()
    parts = [s.strip() for s in soup.stripped_strings if s.strip()]
    text = " ".join(parts)
    return re.sub(r"\s+", " ", text)[:_MAX_CRAWL_TEXT]


async def _fetch_html(url: str, *, timeout_seconds: float) -> str:
    async with httpx.AsyncClient(
        timeout=timeout_seconds,
        follow_redirects=True,
        headers=_HTTP_HEADERS,
    ) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.text


async def _crawl_with_crawl4ai(url: str, *, timeout_seconds: float) -> str:
    from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

    page_timeout_ms = int(max(5.0, timeout_seconds) * 1000)
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=page_timeout_ms,
        word_count_threshold=8,
        remove_overlay_elements=True,
        process_iframes=False,
    )
    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url, config=config)
    if not result.success:
        raise RuntimeError(result.error_message or "crawl4ai failed")
    text = (result.markdown or result.cleaned_html or "").strip()
    if not text and result.html:
        text = _extract_with_beautifulsoup(result.html)
    return re.sub(r"\s+", " ", text)[:_MAX_CRAWL_TEXT]


async def _fetch_page_text(url: str, *, timeout_seconds: float) -> str:
    """crawl4ai (Playwright) first, then httpx + BeautifulSoup."""
    try:
        return await asyncio.wait_for(
            _crawl_with_crawl4ai(url, timeout_seconds=timeout_seconds),
            timeout=timeout_seconds + 2.0,
        )
    except Exception as exc:
        log.info("crawl4ai failed for %s (%s); trying BeautifulSoup", url[:60], exc)
    try:
        html = await _fetch_html(url, timeout_seconds=timeout_seconds)
        return _extract_with_beautifulsoup(html)
    except Exception as exc:
        log.warning("BeautifulSoup fetch failed for %s: %s", url[:60], exc)
        return ""


# Cricbuzz match-card layout markers (mobile + desktop). These class fragments
# survive minor template tweaks and are what BeautifulSoup keys off of.
_CB_MATCH_HEADER_HINTS = (
    "vs",
    "match",
    "qualifier",
    "eliminator",
    "semi-final",
    "semifinal",
    "final",
    "test",
    "odi",
    "t20",
    "ipl",
    "wpl",
    "ranji",
)
_CB_STATUS_HINTS = (
    "stumps",
    "lunch",
    "tea",
    "drinks",
    "innings break",
    "rain",
    "won by",
    "won the toss",
    "elected to",
    "yet to bat",
    "trail by",
    "lead by",
    "abandoned",
    "no result",
    "match drawn",
    "match tied",
)
# Nav / heading text that contains "live" / "scorecard" but is NOT a match line.
# We use this list to reject false positives before passing snippets to the LLM.
_CB_NAV_BLACKLIST = (
    "live scores",
    "live now",
    "live cricket score",
    "live commentary",
    "scorecard",
    "schedule",
    "fixtures",
    "results",
    "cricket schedule",
    "cricbuzz.com",
    "points table",
    "international",
    "domestic",
    "news",
    "videos",
    "photos",
    "stats",
    "rankings",
    "interviews",
)


def _clean_inline(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _looks_like_match_line(line: str) -> bool:
    low = line.lower().strip()
    if not low or len(low) < 8 or len(line) > 280:
        return False

    # Reject pure navigation labels (no team-style content).
    if any(low.startswith(nav) or low == nav for nav in _CB_NAV_BLACKLIST):
        return False
    # Need at least one of: "A vs B", a score with digits, or a clear match status.
    has_versus = bool(
        re.search(r"\b([A-Za-z][A-Za-z\.\-]+)\s+vs?\s+([A-Za-z][A-Za-z\.\-]+)", line)
    )
    has_score = bool(re.search(r"\b\d+\s*[/\-]\s*\d+", line)) or bool(
        re.search(r"\b\d{1,3}\s*(?:runs|wkts?|wickets?|overs?)\b", low)
    )
    has_status = any(k in low for k in _CB_STATUS_HINTS)
    return has_versus or has_score or has_status


def _is_live_match_line(line: str) -> bool:
    low = line.lower()
    if re.search(r"\b\d+\s*[/\-]\s*\d+", line):
        return True  # live score format
    if any(k in low for k in ("won by", "innings break", "stumps", "rain")):
        return True
    if " live " in f" {low} " or low.endswith(" live") or low.startswith("live "):
        return True
    return False


def _parse_cricbuzz_html(html: str) -> tuple[list[str], list[str]]:
    """
    Return ``(live_lines, upcoming_lines)`` extracted from Cricbuzz HTML.

    The mobile + desktop site both use ``cb-`` prefixed CSS classes that group
    each match card. We walk those cards and emit a one-line summary per match
    so the LLM can speak a concrete answer instead of a clarifying question.
    """
    if not html:
        return [], []
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return [], []

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    live: list[str] = []
    upcoming: list[str] = []

    # 1. Desktop / responsive match-card containers.
    for card in soup.select(
        "a.cb-mtch-blk, div.cb-mtch-lst, div.cb-mtch-crd, "
        "div.cb-lv-scrs-col, li.cb-mtch-lst-itm"
    ):
        text = _clean_inline(card.get_text(" "))
        if not text:
            continue
        header_tag = card.select_one(
            ".cb-billing-plans-text, .cb-billing-plans-text-2, "
            ".cb-billing-plans-text-3, h3, .cb-mtch-blk-state-txt"
        )
        header = _clean_inline(header_tag.get_text(" ")) if header_tag else ""
        snippet = text if not header else f"{header} — {text}"
        snippet = _clean_inline(snippet)
        if len(snippet) > 220:
            snippet = snippet[:217].rstrip() + "..."
        if not _looks_like_match_line(snippet):
            continue
        bucket = live if _is_live_match_line(snippet) else upcoming
        bucket.append(snippet)

    # 2. Mobile site uses simpler list rows — pick up "<TeamA> vs <TeamB>" lines.
    if not live and not upcoming:
        for raw_line in soup.get_text("\n").splitlines():
            line = _clean_inline(raw_line)
            if not _looks_like_match_line(line):
                continue
            if _is_live_match_line(line):
                live.append(line)
            else:
                upcoming.append(line)
            if len(live) + len(upcoming) >= 12:
                break

    def _dedupe(items: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    return _dedupe(live)[:5], _dedupe(upcoming)[:8]


async def _fetch_html_safe(url: str, *, timeout_seconds: float) -> str:
    try:
        return await _fetch_html(url, timeout_seconds=timeout_seconds)
    except Exception as exc:
        log.info("Cricbuzz httpx fetch failed for %s: %s", url[:60], exc)
        return ""


async def _cricbuzz_fast_hits(
    *,
    num_results: int,
    timeout_seconds: float,
    intent: str = "any",
) -> list[SearchHit]:
    """
    Fast cricket snapshot directly from Cricbuzz pages.

    Mobile site renders without JS, so plain httpx+BeautifulSoup is 5-10x
    faster than Playwright for live-score lookups (~0.5s vs ~3s).
    ``intent`` may be ``live``, ``schedule``, or ``any`` to bias the URLs we hit.
    """
    timeout = max(2.0, min(timeout_seconds, 5.0))
    intent = (intent or "any").lower()

    if intent == "schedule":
        urls = [_CRICBUZZ_SCHEDULE, _CRICBUZZ_LIVE, _CRICBUZZ_LIVE_DESKTOP]
    elif intent == "live":
        urls = [_CRICBUZZ_LIVE, _CRICBUZZ_LIVE_DESKTOP, _CRICBUZZ_HOME]
    else:
        urls = [_CRICBUZZ_LIVE, _CRICBUZZ_LIVE_DESKTOP, _CRICBUZZ_SCHEDULE]

    htmls = await asyncio.gather(
        *(_fetch_html_safe(u, timeout_seconds=timeout) for u in urls)
    )

    live: list[str] = []
    upcoming: list[str] = []
    pages_parsed = 0
    for url, html in zip(urls, htmls):
        if not html:
            continue
        cur_live, cur_up = _parse_cricbuzz_html(html)
        if cur_live or cur_up:
            pages_parsed += 1
        live.extend(cur_live)
        upcoming.extend(cur_up)

    # Dedupe across sources.
    def _dedupe(items: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    live = _dedupe(live)[:5]
    upcoming = _dedupe(upcoming)[:8]

    hits: list[SearchHit] = []
    if live:
        hits.append(
            SearchHit(
                title="Cricbuzz live matches",
                snippet="LIVE: " + " | ".join(live),
                link=_CRICBUZZ_LIVE,
            )
        )
    if upcoming:
        hits.append(
            SearchHit(
                title="Cricbuzz upcoming / next match",
                snippet="UPCOMING: " + " | ".join(upcoming),
                link=_CRICBUZZ_SCHEDULE,
            )
        )

    # Plain-text fallback when the HTML structure changed and the parser missed.
    if not hits:
        for url, html in zip(urls, htmls):
            if not html:
                continue
            try:
                text = _extract_with_beautifulsoup(html)
            except Exception:
                text = ""
            if text:
                hits.append(
                    SearchHit(
                        title="Cricbuzz snapshot",
                        snippet=text[:600],
                        link=url,
                    )
                )
                break

    if not hits:
        return []

    # Cap at requested width but keep at least live + upcoming sections.
    limit = max(2, min(num_results, 3))
    hits = hits[:limit]
    log.info(
        "Cricbuzz fastpath intent=%s pages=%s live=%s upcoming=%s",
        intent,
        pages_parsed,
        len(live),
        len(upcoming),
    )
    return hits


async def _enrich_hits_with_crawl(
    hits: list[SearchHit],
    *,
    max_pages: int,
    timeout_seconds: float,
) -> list[SearchHit]:
    if not hits or max_pages <= 0:
        return hits

    urls = [h.link for h in hits if h.link][:max_pages]
    if not urls:
        return hits

    tasks = [_fetch_page_text(u, timeout_seconds=timeout_seconds) for u in urls]
    pages = await asyncio.gather(*tasks, return_exceptions=True)

    url_to_text: dict[str, str] = {}
    for url, out in zip(urls, pages):
        if isinstance(out, str) and len(out) > 80:
            url_to_text[url] = out
            log.info("Crawled %s chars from %s", len(out), url[:55])

    enriched: list[SearchHit] = []
    for h in hits:
        body = url_to_text.get(h.link, "")
        if not body:
            enriched.append(h)
            continue
        chunks = _langchain_chunks(body, source=h.link, max_chunks=2)
        merged_snippet = (h.snippet + " | " if h.snippet else "") + " ".join(chunks)
        enriched.append(
            SearchHit(
                title=h.title,
                snippet=merged_snippet[:900],
                link=h.link,
            )
        )
    return enriched


async def search_live_crawl(
    query: str,
    *,
    num_results: int = 3,
    max_pages_to_crawl: int | None = None,
    timeout_seconds: float = 12.0,
) -> list[SearchHit]:
    """
    1) duckduckgo-search for URLs/snippets
    2) crawl4ai + Playwright on top ranked pages
    3) LangChain chunking into snippet text for vLLM
    """
    topic = detect_search_topic(query)
    if topic == "general" and re.search(r"gold|गोल्ड|सोन", query, re.I):
        topic = "gold"

    t0 = time.perf_counter()
    # Cricket / IPL queries: prefer direct Cricbuzz snapshot first.
    if topic == "cricket":
        intent = _cricket_intent_from_text(query)
        cricbuzz_hits = await _cricbuzz_fast_hits(
            num_results=num_results,
            timeout_seconds=timeout_seconds,
            intent=intent,
        )
        if cricbuzz_hits:
            return cricbuzz_hits

    hits = await asyncio.to_thread(
        _search_duckduckgo_sync,
        query,
        num_results=num_results,
        timeout_seconds=min(timeout_seconds, 10.0),
        topic=topic,
    )
    if not hits:
        log.info("Live crawl: no DuckDuckGo hits for %r", build_search_query(query)[:50])
        return []

    max_pages = max_pages_to_crawl
    if max_pages is None:
        max_pages = max(1, _int_env("CRAWL_MAX_PAGES", 1))
    page_timeout = _float_env("CRAWL_PAGE_TIMEOUT_SECONDS", timeout_seconds)

    hits = await _enrich_hits_with_crawl(
        hits,
        max_pages=max_pages,
        timeout_seconds=page_timeout,
    )
    log.info(
        "Live crawl topic=%s query=%r hits=%s pages=%s in %.2fs",
        topic,
        build_search_query(query)[:50],
        len(hits),
        max_pages,
        time.perf_counter() - t0,
    )
    return hits


_CRICKET_SCHEDULE_PAT = re.compile(
    r"(next|upcoming|schedule|fixture|kab|when|"
    r"नेक्स्ट|अगला|अगली|कब|शेड्यूल|फिक्स्चर)",
    re.IGNORECASE,
)
_CRICKET_LIVE_PAT = re.compile(
    r"(live|score|result|right now|currently|"
    r"लाइव|स्कोर|रिज़ल्ट|रिजल्ट|अभी)",
    re.IGNORECASE,
)


def _cricket_intent_from_text(text: str) -> str:
    """Pick the most relevant Cricbuzz URL bucket for the query."""
    t = (text or "")
    if _CRICKET_SCHEDULE_PAT.search(t):
        return "schedule"
    if _CRICKET_LIVE_PAT.search(t):
        return "live"
    return "any"


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default
