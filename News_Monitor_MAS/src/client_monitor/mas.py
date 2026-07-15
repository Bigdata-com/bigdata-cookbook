"""Media Attention Score (MAS) from smart-search co-mention volumes."""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.client_monitor.query import QuerySpec
from src.client_monitor.volumes import fetch_entity_volumes
from src.client_monitor.window import TimeWindow

logger = logging.getLogger(__name__)

BASELINE_DAYS = 30
BASELINE_LAG_DAYS = 1


def sigmoid(value: float) -> float:
    if value > 500:
        return 1.0
    if value < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def compute_mas(v_now: int, lambda_bucket: float) -> dict[str, float]:
    """Compute MAR, Z, and MAS for one entity."""
    mar = (v_now + 1) / (lambda_bucket + 1)
    z_score = (v_now - lambda_bucket) / math.sqrt(lambda_bucket + 1)
    if v_now == 0:
        mas = 0.0
    else:
        k = 5.0
        magnitude = math.log1p(v_now) / math.log1p(lambda_bucket + 10)
        surprise = sigmoid(z_score / k)
        mas = min(100.0, 100.0 * surprise * magnitude)
    return {"mar": round(mar, 2), "z_score": round(z_score, 2), "mas": round(mas, 2)}


def scale_lambda_from_total(total_volume: int, window_minutes: int, span_days: int) -> float:
    """Convert total volume over ``span_days`` into expected count per bucket."""
    total_minutes = span_days * 24 * 60
    rate_per_minute = total_volume / total_minutes if total_minutes > 0 else 0.0
    return rate_per_minute * window_minutes


@dataclass(frozen=True)
class BaselineStore:
    """SQLite cache for 30-day baseline totals per query hash and entity."""

    path: Path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS baselines (
                query_hash TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                volume_30d INTEGER NOT NULL,
                lambda_bucket REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (query_hash, entity_id)
            )
            """
        )
        conn.commit()
        return conn

    def load(self, conn: sqlite3.Connection, query_hash: str) -> dict[str, float]:
        rows = conn.execute(
            "SELECT entity_id, lambda_bucket FROM baselines WHERE query_hash = ?",
            (query_hash,),
        ).fetchall()
        return {str(entity_id): float(lambda_bucket) for entity_id, lambda_bucket in rows}

    def upsert(
        self,
        conn: sqlite3.Connection,
        query_hash: str,
        entity_volumes: dict[str, int],
        *,
        window_minutes: int,
    ) -> dict[str, float]:
        now = datetime.now(UTC).isoformat()
        lambdas: dict[str, float] = {}
        rows: list[tuple[str, str, int, float, str]] = []
        for entity_id, volume in entity_volumes.items():
            lambda_bucket = scale_lambda_from_total(volume, window_minutes, BASELINE_DAYS)
            lambdas[entity_id] = lambda_bucket
            rows.append((query_hash, entity_id, volume, lambda_bucket, now))
        conn.executemany(
            "INSERT OR REPLACE INTO baselines "
            "(query_hash, entity_id, volume_30d, lambda_bucket, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        return lambdas


def baseline_window(reference_end: datetime) -> TimeWindow:
    """Lagged 30-day baseline window ending ``BASELINE_LAG_DAYS`` before reference."""
    end = reference_end.astimezone(UTC) - timedelta(days=BASELINE_LAG_DAYS)
    start = end - timedelta(days=BASELINE_DAYS)
    minutes = BASELINE_DAYS * 24 * 60
    return TimeWindow(start=start, end=end, minutes=minutes)


def ensure_baselines(
    spec: QuerySpec,
    entity_ids: list[str],
    store: BaselineStore,
    *,
    force_refresh: bool = False,
) -> dict[str, float]:
    """Load or refresh baseline ``lambda_bucket`` values for a query spec."""
    query_hash = spec.query_hash()
    conn = store.connect()
    if not force_refresh:
        existing = store.load(conn, query_hash)
        if existing:
            conn.close()
            return existing

    logger.info("Refreshing MAS baselines for %s (%s)", spec.monitor_topic, query_hash)
    base_window = baseline_window(spec.window.end)
    baseline_spec = QuerySpec(
        monitor_topic=spec.monitor_topic,
        search_mode=spec.search_mode,
        text=spec.text,
        topic_filter=spec.topic_filter,
        category=spec.category,
        window=base_window,
    )
    volumes = fetch_entity_volumes(baseline_spec, entity_ids)
    lambdas = store.upsert(conn, query_hash, volumes, window_minutes=spec.window.minutes)
    conn.close()
    return lambdas


def score_topic(
    spec: QuerySpec,
    entity_ids: list[str],
    v_now: dict[str, int],
    lambda_by_entity: dict[str, float],
    id_to_name: dict[str, str],
) -> list[dict[str, Any]]:
    """Score all entities for one monitor topic."""
    rows: list[dict[str, Any]] = []
    for entity_id in entity_ids:
        now = int(v_now.get(entity_id, 0))
        lambda_bucket = float(lambda_by_entity.get(entity_id, 0.0))
        metrics = compute_mas(now, lambda_bucket)
        rows.append(
            {
                "RP_ENTITY_ID": entity_id,
                "COMPANY_NAME": id_to_name.get(entity_id, ""),
                "monitor_topic": spec.monitor_topic,
                "search_mode": spec.search_mode.value,
                "V_NOW": now,
                "LAMBDA_BUCKET": round(lambda_bucket, 6),
                "MAR": metrics["mar"],
                "Z_SCORE": metrics["z_score"],
                "MAS": metrics["mas"],
                "window_start": spec.window.start_iso,
                "window_end": spec.window.end_iso,
                "window_minutes": spec.window.minutes,
            }
        )

    rows.sort(key=lambda row: (row["V_NOW"], row["Z_SCORE"]))
    count = len(rows)
    for index, row in enumerate(rows):
        row["PCT_RANK"] = round(100.0 * index / max(count - 1, 1), 1)

    rows.sort(key=lambda row: row["MAS"], reverse=True)
    return rows
