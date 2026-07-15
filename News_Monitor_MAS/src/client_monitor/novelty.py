"""Headline-hash syndication dedup without LLMs."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SOURCE_SUFFIX_PATTERN = re.compile(
    r"\s[-–|]\s(reuters|bloomberg|associated press|ap news|cnbc|wall street journal|wsj)\s*$",
    re.IGNORECASE,
)


def normalize_headline(headline: str | None) -> str:
    """Normalize headline text for stable hashing."""
    if not headline:
        return ""
    text = headline.strip().lower()
    text = SOURCE_SUFFIX_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def headline_hash(headline: str | None) -> str:
    """SHA256 hex digest of normalized headline."""
    normalized = normalize_headline(headline)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def mark_syndication(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag primary vs syndicated repeat rows within one run."""
    seen: set[str] = set()
    marked: list[dict[str, Any]] = []
    for row in rows:
        digest = headline_hash(str(row.get("headline") or ""))
        is_primary = digest not in seen and digest != headline_hash("")
        if digest and digest != headline_hash(""):
            seen.add(digest)
        enriched = {
            **row,
            "headline_hash": digest,
            "is_primary_story": is_primary,
            "is_syndicated_repeat": not is_primary and digest != headline_hash(""),
        }
        marked.append(enriched)
    return marked


class SeenHeadlineStore:
    """Optional rolling store for cross-bucket headline repeats."""

    def __init__(self, path: Path, ttl_hours: int = 48) -> None:
        self.path = path
        self.ttl_hours = ttl_hours

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_headlines (
                headline_hash TEXT PRIMARY KEY,
                first_seen_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        return conn

    def mark_cross_run(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add ``is_repeat_headline`` when hash was seen in a prior bucket."""
        conn = self.connect()
        cutoff = datetime.now(UTC) - timedelta(hours=self.ttl_hours)
        conn.execute("DELETE FROM seen_headlines WHERE first_seen_at < ?", (cutoff.isoformat(),))
        existing = {
            row[0]
            for row in conn.execute("SELECT headline_hash FROM seen_headlines").fetchall()
        }

        updated: list[dict[str, Any]] = []
        for row in rows:
            digest = str(row.get("headline_hash") or "")
            is_repeat = bool(digest and digest in existing)
            updated.append({**row, "is_repeat_headline": is_repeat})

        now = datetime.now(UTC).isoformat()
        for row in updated:
            digest = str(row.get("headline_hash") or "")
            if digest:
                conn.execute(
                    "INSERT OR IGNORE INTO seen_headlines "
                    "(headline_hash, first_seen_at) VALUES (?, ?)",
                    (digest, now),
                )
        conn.commit()
        conn.close()
        return updated
