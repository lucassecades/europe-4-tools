#!/usr/bin/env python3
"""
SEDIA REST API helpers for EU Funding & Tenders Portal topic data.

Official endpoint (Section 5 - Topic Details service):
  POST https://api.tech.ec.europa.eu/search-api/prod/rest/search?apiKey=SEDIA&text="<IDENTIFIER>"
  - text must be QUOTED for exact-match lookup
  - method must be POST (GET returns HTTP 405)
  - pageSize=20 to get all language variants
  - prefer metadata where language=['en']
"""

import json
import re
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup


_SEDIA_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Origin": "https://ec.europa.eu",
}

_TYPE_CLEAN = re.compile(r"^HORIZON\s*(MSCA\s*)?", re.IGNORECASE)


def extract_topic_id(url: str) -> str | None:
    m = re.search(r"topic-details/([^/?#]+)", str(url), re.IGNORECASE)
    return m.group(1) if m else None


# ── SEDIA fetch ───────────────────────────────────────────────────────────────

def sedia_fetch_metadata(topic_id: str) -> dict | None:
    """
    Return the English metadata dict for an exact topic identifier, or None.
    Fetches pageSize=20 to capture all language variants, then prefers English.
    """
    quoted = urllib.parse.quote(f'"{topic_id}"')
    url = (
        f"https://api.tech.ec.europa.eu/search-api/prod/rest/search"
        f"?apiKey=SEDIA&text={quoted}&pageSize=20"
    )
    try:
        req = urllib.request.Request(url, data=b"", headers=_SEDIA_HEADERS, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8", "ignore"))
        results = data.get("results", [])
        matches = [
            r["metadata"] for r in results
            if topic_id in r.get("metadata", {}).get("identifier", [])
        ]
        if not matches:
            return None
        en = next((m for m in matches if "en" in (m.get("language") or [])), None)
        return en if en else matches[0]
    except Exception:
        return None


# ── Field extractors ──────────────────────────────────────────────────────────

def extract_budget(meta: dict, topic_id: str) -> str:
    """
    Parse budgetOverview JSON and return the budget for this specific topic,
    e.g. '€17.0M (2 grants)'.
    """
    raw = meta.get("budgetOverview", "")
    bstr = raw[0] if isinstance(raw, list) and raw else str(raw)
    if not bstr:
        return ""
    try:
        bdata = json.loads(bstr)
        for ccm2_entries in bdata.get("budgetTopicActionMap", {}).values():
            for entry in ccm2_entries:
                if not entry.get("action", "").startswith(topic_id):
                    continue
                total = sum(
                    float(str(v).replace(",", ""))
                    for v in entry.get("budgetYearMap", {}).values() if v
                )
                grants = entry.get("expectedGrants", "")
                amount = f"€{total / 1_000_000:.1f}M"
                if grants:
                    amount += f" ({grants} grant{'s' if int(grants) > 1 else ''})"
                return amount
    except Exception:
        pass
    return ""


def extract_type(meta: dict) -> str:
    """Return clean action type, e.g. 'Innovation Actions'."""
    raw = meta.get("typesOfAction", [])
    val = raw[0] if isinstance(raw, list) and raw else str(raw)
    return _TYPE_CLEAN.sub("", val).strip()


# ── Description summariser ────────────────────────────────────────────────────

def _section_blocks(header_tag, max_items: int = 4) -> list[str]:
    """
    Collect text blocks following a section header, stopping at the next header.
    Handles <p>, <ul><li>, <ol><li>.
    """
    items = []
    for sibling in header_tag.next_siblings:
        if (hasattr(sibling, "attrs")
                and sibling.attrs.get("class") == ["topicdescriptionkind"]):
            break
        if not hasattr(sibling, "name") or sibling.name is None:
            continue
        if sibling.name in ("ul", "ol"):
            for li in sibling.find_all("li"):
                text = li.get_text(" ", strip=True)
                if text and len(text) > 15:
                    items.append(text)
                    if len(items) >= max_items:
                        return items
        elif sibling.name == "p":
            text = sibling.get_text(" ", strip=True)
            if text and len(text) > 15:
                for sentence in re.split(r"(?<=[.!?])\s+", text)[:2]:
                    if len(sentence) > 15:
                        items.append(sentence)
                        if len(items) >= max_items:
                            return items
    return items


def summarise_description(meta: dict, max_words: int = 250) -> str:
    """
    Build a structured ~250-word summary from descriptionByte HTML:
      • First 3 Expected Outcome items (bullets or paragraphs)
      • First 3 Scope items (bullets or sentences)
    Falls back to plain text truncation if no sections found.
    """
    raw = meta.get("descriptionByte", "")
    html = raw[0] if isinstance(raw, list) and raw else str(raw)
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    headers = soup.find_all("p", class_="topicdescriptionkind")

    sections: dict[str, object] = {
        tag.get_text(strip=True).rstrip(":").lower(): tag
        for tag in headers
    }

    _OUTCOME_KEYS = {"expected outcome", "expectedoutcome", "expected outcomes"}
    _SCOPE_KEYS   = {"scope"}

    outcome_hdr = next((sections[k] for k in sections if k in _OUTCOME_KEYS), None)
    scope_hdr   = next((sections[k] for k in sections if k in _SCOPE_KEYS), None)

    # Positional fallback when headers are in another language
    if not outcome_hdr and len(headers) >= 1:
        outcome_hdr = headers[0]
    if not scope_hdr and len(headers) >= 2:
        scope_hdr = headers[1]

    outcome_parts = _section_blocks(outcome_hdr, max_items=3) if outcome_hdr else []
    scope_parts   = _section_blocks(scope_hdr,   max_items=3) if scope_hdr   else []

    parts = []
    if outcome_parts:
        parts.append("Expected outcomes: " + " | ".join(outcome_parts))
    if scope_parts:
        parts.append("Scope: " + " | ".join(scope_parts))

    summary = " ".join(parts)

    if not summary.strip():
        summary = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()

    words = summary.split()
    if len(words) > max_words:
        summary = " ".join(words[:max_words]) + "..."

    return summary.strip()


# ── Main entry point ──────────────────────────────────────────────────────────

def fetch_topic_data(url: str) -> dict:
    """
    Fetch description, budget and type for a topic URL.
    Returns dict with keys: description, budget, type (all str, may be empty).
    """
    result = {"description": "", "budget": "", "type": ""}
    topic_id = extract_topic_id(url)
    if not topic_id:
        return result

    meta = sedia_fetch_metadata(topic_id)
    if not meta:
        return result

    result["description"] = summarise_description(meta)
    result["budget"]      = extract_budget(meta, topic_id)
    result["type"]        = extract_type(meta)
    return result
