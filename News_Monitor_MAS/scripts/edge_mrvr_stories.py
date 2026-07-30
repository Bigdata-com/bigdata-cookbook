"""RavenPack Edge MRVR story pull — entity-scoped web news.

Edge 1.0 does **not** accept ``url`` as a dataset field. URLs are resolved via
``api.get_document_url(rp_document_id)``.

Dataset filters always include ``rp_provider_id=MRVR``. Optional
``--min-entity-relevance 90`` adds an entity_relevance floor.

Universe (pick one)::

    --universe us_sml.csv              # CSV with RP_ENTITY_ID [, COMPANY_NAME]
    --tickers AAPL,MSFT,NVDA           # map tickers → RP ids via Edge
    --entity-ids 0157B1,4A6F00         # raw RP entity ids
    --missed-csv MissedStories.csv     # tickers from Ticker column (recover default)

Time range (UTC)::

    --window-minutes 15                # end=now, start=now-15m  (default for pull/feed)
    --window-end 2026-07-29T15:00:00Z --window-minutes 60
    --start 2026-07-01T00:00:00Z --end 2026-07-02T00:00:00Z

Examples::

    uv run python scripts/edge_mrvr_stories.py pull \\
      --universe us_sml.csv --limit-entities 50 --window-minutes 15

    uv run python scripts/edge_mrvr_stories.py pull \\
      --tickers AAPL,MSFT --start 2026-07-28T12:00:00Z --end 2026-07-28T12:15:00Z

    uv run python scripts/edge_mrvr_stories.py recover --missed-csv MissedStories.csv
    uv run python scripts/edge_mrvr_stories.py feed --universe us_sml.csv --max-buckets 1
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from ravenpackapi import Dataset, RPApi

logger = logging.getLogger("edge_mrvr_stories")

OUTPUT_COLUMNS = (
    "timestamp_utc",
    "company_name",
    "title",
    "source_name",
    "url",
    "entity_relevance",
    "entity_sentiment",
    "title_similarity_days",
    "rp_document_id",
)
# Edge 1.0 does not expose `url` as an analytics field; resolve via document endpoint.
INTERNAL_FIELDS = [
    "timestamp_utc",
    "entity_name",
    "title",
    "source_name",
    "rp_document_id",
    "rp_entity_id",
    "entity_relevance",
    "entity_sentiment",
    "title_similarity_days",
]
PROVIDER_MRVR = "MRVR"
DATASET_NAME = "publicAI-edge-mrvr-stories"


def load_api_key() -> str:
    load_dotenv(Path.cwd() / ".env", override=False)
    import os

    key = os.getenv("RAVENPACK_API_KEY") or os.getenv("RP_API_KEY")
    if not key:
        msg = "RAVENPACK_API_KEY (or RP_API_KEY) is not set"
        raise SystemExit(msg)
    return key


def make_api() -> RPApi:
    api = RPApi(api_key=load_api_key(), product="edge")
    api.log_curl_commands = False  # avoid leaking API key into logs
    return api


def read_missed_stories(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    # Drop empty trailing columns from trailing semicolons
    df = df.loc[:, [c for c in df.columns if c]]
    required = {"Ticker", "Headline", "Publisher", "Pub Date"}
    missing = required - set(df.columns)
    if missing:
        msg = f"MissedStories missing columns: {missing}"
        raise ValueError(msg)
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Headline"] = df["Headline"].astype(str)
    df["Pub Date"] = pd.to_datetime(df["Pub Date"], errors="coerce")
    return df.dropna(subset=["Ticker", "Headline"]).reset_index(drop=True)


def unique_tickers(missed: pd.DataFrame) -> list[str]:
    return sorted({t for t in missed["Ticker"].tolist() if t and t != "TICKER"})


def parse_utc(value: str) -> datetime:
    """Parse an ISO / Edge-friendly UTC timestamp."""
    text = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def resolve_time_window(
    *,
    start: str | None,
    end: str | None,
    window_end: str | None,
    window_minutes: int,
) -> tuple[datetime, datetime]:
    """Resolve ``[start, end)`` in UTC.

    Priority:
    1. ``--start`` + ``--end``
    2. ``--window-end`` (or now) minus ``--window-minutes``
    """
    if start is not None and end is not None:
        start_dt = parse_utc(start)
        end_dt = parse_utc(end)
        if start_dt >= end_dt:
            msg = f"--start ({start_dt.isoformat()}) must be before --end ({end_dt.isoformat()})"
            raise SystemExit(msg)
        return start_dt, end_dt
    if start is not None or end is not None:
        raise SystemExit("Provide both --start and --end, or use --window-minutes / --window-end")
    if window_minutes <= 0:
        raise SystemExit("--window-minutes must be > 0")
    end_dt = parse_utc(window_end) if window_end else datetime.now(UTC).replace(microsecond=0)
    start_dt = end_dt - timedelta(minutes=window_minutes)
    return start_dt, end_dt


def load_universe_csv(path: Path) -> pd.DataFrame:
    """Load a universe CSV with RP_ENTITY_ID or rp_entity_id."""
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip() for c in frame.columns]
    cols = {c.lower(): c for c in frame.columns}
    id_col = cols.get("rp_entity_id")
    if id_col is None:
        msg = f"{path} must contain RP_ENTITY_ID (found: {list(frame.columns)})"
        raise SystemExit(msg)
    name_col = cols.get("company_name") or cols.get("entity_name")
    ticker_col = cols.get("ticker")
    out = pd.DataFrame(
        {
            "rp_entity_id": frame[id_col].astype(str).str.strip().str.upper(),
            "entity_name": frame[name_col].astype(str) if name_col else "",
            "ticker": frame[ticker_col].astype(str).str.upper().str.strip() if ticker_col else "",
            "map_source": "universe_csv",
        }
    )
    out = out.loc[out["rp_entity_id"].astype(bool)].drop_duplicates(subset=["rp_entity_id"])
    return out.reset_index(drop=True)


def resolve_entity_mapping(
    api: RPApi,
    *,
    output_dir: Path,
    universe: Path | None,
    tickers: str | None,
    entity_ids: str | None,
    missed_csv: Path | None,
    limit_entities: int,
    reuse_mapping: bool,
    us_sml_path: Path | None = Path("us_sml.csv"),
) -> pd.DataFrame:
    """Build ticker/entity mapping from CLI universe options."""
    mapping_path = output_dir / "entity_mapping.csv"
    if reuse_mapping and mapping_path.exists():
        logger.info("Reusing entity mapping from %s", mapping_path)
        mapping = pd.read_csv(mapping_path)
        mapping["rp_entity_id"] = mapping["rp_entity_id"].astype(str)
        if "ticker" in mapping.columns:
            mapping["ticker"] = mapping["ticker"].astype(str).str.upper()
        if limit_entities > 0:
            mapping = mapping.head(limit_entities)
        return mapping.reset_index(drop=True)

    if universe is not None:
        mapping = load_universe_csv(universe)
    elif entity_ids:
        ids = [x.strip().upper() for x in entity_ids.split(",") if x.strip()]
        mapping = pd.DataFrame(
            {
                "ticker": [""] * len(ids),
                "rp_entity_id": ids,
                "entity_name": [""] * len(ids),
                "map_source": ["cli_entity_ids"] * len(ids),
            }
        )
    elif tickers:
        ticker_list = [x.strip().upper() for x in tickers.split(",") if x.strip()]
        mapping = map_tickers(api, ticker_list, us_sml_path=us_sml_path)
    elif missed_csv is not None and missed_csv.exists():
        missed = read_missed_stories(missed_csv)
        mapping = map_tickers(api, unique_tickers(missed), us_sml_path=us_sml_path)
    else:
        raise SystemExit(
            "Provide a universe via --universe, --tickers, --entity-ids, or --missed-csv"
        )

    if limit_entities > 0:
        mapping = mapping.head(limit_entities).reset_index(drop=True)
    mapping.to_csv(mapping_path, index=False)
    return mapping


def map_tickers(
    api: RPApi,
    tickers: list[str],
    *,
    us_sml_path: Path | None = Path("us_sml.csv"),
    trusted_ids: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Map tickers to Edge rp_entity_id with US-listing disambiguation.

    Prefer (in order): trusted audit ids, listing-qualified COMP matches that land
    in ``us_sml.csv``, then bare COMP candidates in ``us_sml``, else best COMP hit.
    """
    us_ids: set[str] = set()
    us_name: dict[str, str] = {}
    if us_sml_path is not None and us_sml_path.exists():
        us = pd.read_csv(us_sml_path)
        us_ids = set(us["RP_ENTITY_ID"].astype(str))
        us_name = dict(
            zip(us["RP_ENTITY_ID"].astype(str), us["COMPANY_NAME"].astype(str), strict=True)
        )

    trusted_ids = trusted_ids or {}
    rows: list[dict[str, str]] = []

    for ticker in tickers:
        ticker_u = ticker.upper()
        chosen_id = ""
        chosen_name = ""
        source = ""

        if ticker_u in trusted_ids:
            chosen_id = str(trusted_ids[ticker_u])
            chosen_name = us_name.get(chosen_id, "")
            source = "trusted"

        if not chosen_id:
            for listing in (f"XNAS:{ticker_u}", f"XNYS:{ticker_u}", f"ARCX:{ticker_u}"):
                mapping = api.get_entity_mapping(
                    [{"ticker": ticker_u, "listing": listing, "entity_type": "COMP"}]
                )
                if not mapping.matched:
                    continue
                hit = mapping.matched[0]
                candidates = [hit, *(hit.candidates or [])]
                for cand in candidates:
                    cid = str(getattr(cand, "id", None) or getattr(cand, "rp_entity_id", "") or "")
                    cname = str(getattr(cand, "name", "") or "")
                    if cid and cid in us_ids:
                        chosen_id, chosen_name, source = cid, cname or us_name[cid], f"listing:{listing}"
                        break
                if chosen_id:
                    break
                # keep first listing hit as fallback
                chosen_id = str(hit.id)
                chosen_name = str(hit.name or "")
                source = f"listing_fallback:{listing}"
                break

        if not chosen_id:
            mapping = api.get_entity_mapping([{"ticker": ticker_u, "entity_type": "COMP"}])
            if mapping.matched:
                hit = mapping.matched[0]
                for cand in [hit, *(hit.candidates or [])]:
                    cid = str(getattr(cand, "id", None) or getattr(cand, "rp_entity_id", "") or "")
                    cname = str(getattr(cand, "name", "") or "")
                    if cid and cid in us_ids:
                        chosen_id, chosen_name, source = cid, cname or us_name[cid], "bare_us_sml"
                        break
                if not chosen_id:
                    chosen_id = str(hit.id)
                    chosen_name = str(hit.name or "")
                    source = "bare_fallback"

        if not chosen_id:
            logger.warning("No Edge entity mapping for ticker=%s", ticker_u)
            continue

        rows.append(
            {
                "ticker": ticker_u,
                "rp_entity_id": chosen_id,
                "entity_name": chosen_name or us_name.get(chosen_id, ""),
                "map_source": source,
            }
        )

    return pd.DataFrame(rows)


def build_filters(
    entity_ids: list[str],
    *,
    min_entity_relevance: float | None,
) -> dict[str, Any]:
    clauses: list[dict[str, Any]] = [
        {"rp_provider_id": {"$in": [PROVIDER_MRVR]}},
        {"rp_entity_id": {"$in": entity_ids}},
    ]
    if min_entity_relevance is not None:
        clauses.append({"entity_relevance": {"$gte": min_entity_relevance}})
    return {"$and": clauses}


def ensure_dataset(
    api: RPApi,
    dataset_id_path: Path | None,
    *,
    entity_ids: list[str] | None = None,
    min_entity_relevance: float | None = None,
    force_new: bool = False,
) -> Dataset:
    if (
        not force_new
        and dataset_id_path is not None
        and dataset_id_path.exists()
        and entity_ids is None
    ):
        dataset_id = dataset_id_path.read_text(encoding="utf-8").strip()
        if dataset_id:
            logger.info("Reusing Edge dataset %s", dataset_id)
            return api.get_dataset(dataset_id=dataset_id)

    filters: dict[str, Any]
    if entity_ids:
        filters = build_filters(entity_ids, min_entity_relevance=min_entity_relevance)
        name = f"{DATASET_NAME}-universe"
    else:
        filters = {"rp_provider_id": {"$in": [PROVIDER_MRVR]}}
        name = DATASET_NAME

    ds = api.create_dataset(
        Dataset(
            name=name,
            product="edge",
            product_version="1.0",
            frequency="granular",
            fields=INTERNAL_FIELDS,
            filters=filters,
        )
    )
    logger.info("Created Edge dataset %s", ds.id)
    if dataset_id_path is not None and entity_ids is None:
        dataset_id_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_id_path.write_text(str(ds.id), encoding="utf-8")
    return ds


def records_to_frame(records: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        if hasattr(record, "as_dict"):
            payload = record.as_dict()
        elif isinstance(record, dict):
            payload = record
        else:
            payload = {}
            for key in INTERNAL_FIELDS:
                if hasattr(record, key):
                    payload[key] = getattr(record, key)
                else:
                    # Result objects often expose lower-case attrs
                    payload[key] = getattr(record, key.lower(), None)
        rows.append({k: payload.get(k) for k in INTERNAL_FIELDS})
    if not rows:
        return pd.DataFrame(columns=[*INTERNAL_FIELDS, "url"])
    frame = pd.DataFrame(rows)
    frame["url"] = None
    return frame


def query_window(
    ds: Dataset,
    *,
    start: datetime,
    end: datetime,
    entity_ids: list[str],
    min_entity_relevance: float | None,
) -> pd.DataFrame:
    filters = build_filters(entity_ids, min_entity_relevance=min_entity_relevance)
    start_s = start.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
    end_s = end.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
    logger.info(
        "Edge JSON %s → %s (%d entities, relevance=%s)",
        start_s,
        end_s,
        len(entity_ids),
        min_entity_relevance,
    )
    results = ds.json(
        start_date=start_s,
        end_date=end_s,
        fields=INTERNAL_FIELDS,
        filters=filters,
    )
    return records_to_frame(results)


def fill_missing_urls(api: RPApi, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "url" not in out.columns:
        out["url"] = None
    cache: dict[str, str] = {}
    if "rp_document_id" not in out.columns:
        return out
    daily_limit_hit = False
    for doc_id in out["rp_document_id"].dropna().astype(str).unique():
        if not doc_id:
            continue
        if daily_limit_hit:
            cache[doc_id] = ""
            continue
        try:
            cache[doc_id] = str(api.get_document_url(doc_id) or "")
        except Exception as exc:  # noqa: BLE001 — best-effort URL fill
            msg = str(exc)
            logger.warning("get_document_url failed for %s: %s", doc_id, exc)
            cache[doc_id] = ""
            if "Requests-per-day" in msg or "429" in msg:
                logger.error("Document URL daily limit hit — skipping remaining URL lookups")
                daily_limit_hit = True
        time.sleep(0.02)
    out["url"] = out["rp_document_id"].astype(str).map(cache)
    return out


def unique_stories(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))
    work = df.copy()
    if "rp_document_id" in work.columns and "rp_entity_id" in work.columns:
        work = work.drop_duplicates(subset=["rp_document_id", "rp_entity_id"], keep="first")
    else:
        work = work.drop_duplicates(subset=["title", "entity_name", "timestamp_utc"], keep="first")
    out = pd.DataFrame(
        {
            "timestamp_utc": work["timestamp_utc"],
            "company_name": work["entity_name"],
            "title": work["title"],
            "source_name": work["source_name"],
            "url": work.get("url"),
            "entity_relevance": work.get("entity_relevance"),
            "entity_sentiment": work.get("entity_sentiment"),
            "title_similarity_days": work.get("title_similarity_days"),
            "rp_document_id": work.get("rp_document_id"),
        }
    )
    return out.reset_index(drop=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def normalize_title(text: str) -> str:
    cleaned = text.lower()
    cleaned = re.sub(r"\s*\|\s*.*$", "", cleaned)  # drop trailing "| Source"
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def title_similarity(a: str, b: str) -> float:
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.95
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def run_pull(
    *,
    output_dir: Path,
    min_entity_relevance: float | None,
    skip_urls: bool = False,
    universe: Path | None = None,
    tickers: str | None = None,
    entity_ids: str | None = None,
    missed_csv: Path | None = None,
    limit_entities: int = 0,
    start: str | None = None,
    end: str | None = None,
    window_end: str | None = None,
    window_minutes: int = 15,
    force_new_dataset: bool = True,
) -> dict[str, Any]:
    """Pull MRVR stories for a universe over a time window."""
    output_dir.mkdir(parents=True, exist_ok=True)
    api = make_api()
    mapping = resolve_entity_mapping(
        api,
        output_dir=output_dir,
        universe=universe,
        tickers=tickers,
        entity_ids=entity_ids,
        missed_csv=missed_csv,
        limit_entities=limit_entities,
        reuse_mapping=True,
    )
    entity_ids_list = sorted(set(mapping["rp_entity_id"].astype(str).tolist()))
    if not entity_ids_list:
        raise SystemExit("No entities resolved for universe")

    ds = ensure_dataset(
        api,
        output_dir / "dataset_id.txt",
        force_new=force_new_dataset,
    )
    start_dt, end_dt = resolve_time_window(
        start=start,
        end=end,
        window_end=window_end,
        window_minutes=window_minutes,
    )
    raw = query_window(
        ds,
        start=start_dt,
        end=end_dt,
        entity_ids=entity_ids_list,
        min_entity_relevance=min_entity_relevance,
    )
    if not skip_urls:
        raw = fill_missing_urls(api, raw)
    elif "url" not in raw.columns:
        raw["url"] = None
    raw.to_csv(output_dir / "raw_records.csv", index=False)
    stories = unique_stories(raw)
    stories.to_csv(output_dir / "stories_unique.csv", index=False)

    summary = {
        "mode": "pull",
        "window_start": start_dt.isoformat(),
        "window_end": end_dt.isoformat(),
        "mapped_entities": len(entity_ids_list),
        "raw_rows": int(len(raw)),
        "unique_stories": int(len(stories)),
        "url_non_null": int(stories["url"].fillna("").astype(str).str.len().gt(0).sum())
        if not stories.empty
        else 0,
        "min_entity_relevance": min_entity_relevance,
        "provider": PROVIDER_MRVR,
        "dataset_id": ds.id,
        "skip_urls": skip_urls,
        "fields": list(INTERNAL_FIELDS),
    }
    write_json(output_dir / "run_summary.json", summary)
    logger.info(
        "pull done — raw=%d unique=%d url_filled=%d window=%s→%s",
        summary["raw_rows"],
        summary["unique_stories"],
        summary["url_non_null"],
        start_dt.isoformat(),
        end_dt.isoformat(),
    )
    return summary


def run_last15(
    *,
    missed_csv: Path,
    output_dir: Path,
    min_entity_relevance: float | None,
    window_minutes: int,
    skip_urls: bool = False,
    universe: Path | None = None,
    tickers: str | None = None,
    entity_ids: str | None = None,
    limit_entities: int = 0,
    start: str | None = None,
    end: str | None = None,
    window_end: str | None = None,
) -> dict[str, Any]:
    """Alias for ``pull`` (backward compatible name)."""
    return run_pull(
        output_dir=output_dir,
        min_entity_relevance=min_entity_relevance,
        skip_urls=skip_urls,
        universe=universe,
        tickers=tickers,
        entity_ids=entity_ids,
        missed_csv=missed_csv if missed_csv.exists() else None,
        limit_entities=limit_entities,
        start=start,
        end=end,
        window_end=window_end,
        window_minutes=window_minutes,
        force_new_dataset=True,
    )

def query_range_via_datafile(
    ds: Dataset,
    *,
    start: datetime,
    end: datetime,
    output_csv: Path,
) -> pd.DataFrame:
    """Async datafile pull for ranges that exceed the JSON 10k cap."""
    start_s = start.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
    end_s = end.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
    logger.info("Edge datafile %s → %s (dataset=%s)", start_s, end_s, ds.id)
    job = ds.request_datafile(
        start_date=start_s,
        end_date=end_s,
        output_format="csv",
        compressed=False,
        fields=INTERNAL_FIELDS,
        allow_empty=True,
    )
    if job is None:
        logger.warning("Datafile empty for %s → %s", start_s, end_s)
        return pd.DataFrame(columns=[*INTERNAL_FIELDS, "url"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    job.save_to_file(str(output_csv))
    frame = pd.read_csv(output_csv)
    # Normalize column names to snake_case expected internally
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    for col in INTERNAL_FIELDS:
        if col not in frame.columns:
            frame[col] = None
    frame["url"] = None
    return frame[INTERNAL_FIELDS + ["url"]]


def query_range_chunked(
    ds: Dataset,
    *,
    start: datetime,
    end: datetime,
    entity_ids: list[str],
    min_entity_relevance: float | None,
    chunk_hours: int = 6,
) -> pd.DataFrame:
    """JSON pull with auto-shrink on 10k overflow."""
    frames: list[pd.DataFrame] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(hours=chunk_hours), end)
        try:
            frame = query_window(
                ds,
                start=cursor,
                end=chunk_end,
                entity_ids=entity_ids,
                min_entity_relevance=min_entity_relevance,
            )
            frames.append(frame)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "exceeds 10000" in msg and chunk_hours > 1:
                logger.warning(
                    "Chunk %s→%s exceeded 10k; splitting to %dh",
                    cursor,
                    chunk_end,
                    max(1, chunk_hours // 2),
                )
                mid = cursor + (chunk_end - cursor) / 2
                left = query_range_chunked(
                    ds,
                    start=cursor,
                    end=mid,
                    entity_ids=entity_ids,
                    min_entity_relevance=min_entity_relevance,
                    chunk_hours=max(1, chunk_hours // 2),
                )
                right = query_range_chunked(
                    ds,
                    start=mid,
                    end=chunk_end,
                    entity_ids=entity_ids,
                    min_entity_relevance=min_entity_relevance,
                    chunk_hours=max(1, chunk_hours // 2),
                )
                frames.extend([left, right])
            else:
                logger.error("Chunk failed %s→%s: %s", cursor, chunk_end, exc)
        cursor = chunk_end
        time.sleep(0.15)
    if not frames:
        return pd.DataFrame(columns=[*INTERNAL_FIELDS, "url"])
    return pd.concat(frames, ignore_index=True)


def run_recover(
    *,
    missed_csv: Path,
    output_dir: Path,
    min_entity_relevance: float | None,
    match_threshold: float,
    skip_urls: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    api = make_api()
    missed = read_missed_stories(missed_csv)
    tickers = unique_tickers(missed)
    mapping = map_tickers(api, tickers)
    mapping.to_csv(output_dir / "entity_mapping.csv", index=False)
    entity_ids = sorted(set(mapping["rp_entity_id"].astype(str).tolist()))
    ticker_to_entity = dict(zip(mapping["ticker"], mapping["rp_entity_id"], strict=True))

    dated = missed.dropna(subset=["Pub Date"])
    if dated.empty:
        raise SystemExit("MissedStories Pub Date column is empty / unparseable")

    start = dated["Pub Date"].min().to_pydatetime().replace(tzinfo=UTC) - timedelta(hours=6)
    end = dated["Pub Date"].max().to_pydatetime().replace(tzinfo=UTC) + timedelta(hours=6)

    # Bake universe into dataset so datafile respects entity filter (JSON 10k too small)
    ds = ensure_dataset(
        api,
        None,
        entity_ids=entity_ids,
        min_entity_relevance=min_entity_relevance,
        force_new=True,
    )
    (output_dir / "dataset_id.txt").write_text(str(ds.id), encoding="utf-8")

    raw = query_range_via_datafile(
        ds,
        start=start,
        end=end,
        output_csv=output_dir / "datafile_raw.csv",
    )
    if "url" not in raw.columns:
        raw["url"] = None
    raw.to_csv(output_dir / "raw_records.csv", index=False)
    stories = unique_stories(raw)
    stories.to_csv(output_dir / "stories_unique.csv", index=False)

    entity_to_ticker = {v: k for k, v in ticker_to_entity.items()}
    edge = raw.copy()
    if "rp_entity_id" in edge.columns:
        edge["ticker"] = edge["rp_entity_id"].astype(str).map(entity_to_ticker)
    else:
        edge["ticker"] = None

    by_ticker: dict[str, pd.DataFrame] = {
        str(t): g for t, g in edge.groupby("ticker") if pd.notna(t)
    }

    match_rows: list[dict[str, Any]] = []
    matched_doc_ids: set[str] = set()
    for _, story in missed.iterrows():
        ticker = str(story["Ticker"]).upper()
        headline = str(story["Headline"])
        candidates = by_ticker.get(ticker, pd.DataFrame())
        best_score = 0.0
        best_title = ""
        best_doc = ""
        best_ts = ""
        best_source = ""
        for _, cand in candidates.iterrows():
            score = title_similarity(headline, str(cand.get("title") or ""))
            if score > best_score:
                best_score = score
                best_title = str(cand.get("title") or "")
                best_doc = str(cand.get("rp_document_id") or "")
                best_ts = str(cand.get("timestamp_utc") or "")
                best_source = str(cand.get("source_name") or "")
        matched = best_score >= match_threshold
        if matched and best_doc:
            matched_doc_ids.add(best_doc)
        match_rows.append(
            {
                "ticker": ticker,
                "missed_headline": headline,
                "publisher": story.get("Publisher"),
                "pub_date": story.get("Pub Date"),
                "matched": matched,
                "match_score": round(best_score, 3),
                "edge_title": best_title if matched else "",
                "edge_source": best_source if matched else "",
                "edge_timestamp_utc": best_ts if matched else "",
                "rp_document_id": best_doc if matched else "",
                "edge_url": "",
            }
        )

    url_cache: dict[str, str] = {}
    if not skip_urls:
        daily_limit_hit = False
        for doc_id in sorted(matched_doc_ids):
            if daily_limit_hit:
                url_cache[doc_id] = ""
                continue
            try:
                url_cache[doc_id] = str(api.get_document_url(doc_id) or "")
            except Exception as exc:  # noqa: BLE001
                logger.warning("get_document_url failed for %s: %s", doc_id, exc)
                url_cache[doc_id] = ""
                if "Requests-per-day" in str(exc):
                    daily_limit_hit = True
            time.sleep(0.02)
    for row in match_rows:
        doc_id = str(row.get("rp_document_id") or "")
        if doc_id:
            row["edge_url"] = url_cache.get(doc_id, "")

    matches = pd.DataFrame(match_rows)
    matches.to_csv(output_dir / "missed_match_report.csv", index=False)
    matched_n = int(matches["matched"].sum()) if not matches.empty else 0
    summary = {
        "mode": "recover",
        "query_start": start.isoformat(),
        "query_end": end.isoformat(),
        "missed_stories": int(len(missed)),
        "missed_with_pub_date": int(len(dated)),
        "matched": matched_n,
        "match_rate": round(matched_n / len(missed), 4) if len(missed) else 0.0,
        "match_threshold": match_threshold,
        "raw_rows": int(len(raw)),
        "unique_stories": int(len(stories)),
        "mapped_entities": len(entity_ids),
        "min_entity_relevance": min_entity_relevance,
        "provider": PROVIDER_MRVR,
        "dataset_id": ds.id,
        "note": "MRVR-only web content; premium publishers in MissedStories may be out of scope",
    }
    write_json(output_dir / "run_summary.json", summary)
    logger.info(
        "recover done — matched %d / %d (%.1f%%)",
        matched_n,
        len(missed),
        100.0 * summary["match_rate"],
    )
    return summary


def run_feed(
    *,
    output_dir: Path,
    min_entity_relevance: float | None,
    interval_minutes: int,
    max_buckets: int,
    skip_urls: bool = False,
    universe: Path | None = None,
    tickers: str | None = None,
    entity_ids: str | None = None,
    missed_csv: Path | None = None,
    limit_entities: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    api = make_api()
    mapping = resolve_entity_mapping(
        api,
        output_dir=output_dir,
        universe=universe,
        tickers=tickers,
        entity_ids=entity_ids,
        missed_csv=missed_csv,
        limit_entities=limit_entities,
        reuse_mapping=True,
    )
    entity_ids_list = sorted(set(mapping["rp_entity_id"].astype(str).tolist()))
    ds = ensure_dataset(api, output_dir / "dataset_id.txt", force_new=True)

    buckets_done = 0
    while max_buckets <= 0 or buckets_done < max_buckets:
        end = datetime.now(UTC).replace(microsecond=0)
        start = end - timedelta(minutes=interval_minutes)
        stamp = end.strftime("%Y%m%d_%H%M%S")
        bucket_dir = output_dir / f"bucket_{stamp}"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        raw = query_window(
            ds,
            start=start,
            end=end,
            entity_ids=entity_ids_list,
            min_entity_relevance=min_entity_relevance,
        )
        if not skip_urls:
            raw = fill_missing_urls(api, raw)
        elif "url" not in raw.columns:
            raw["url"] = None
        stories = unique_stories(raw)
        stories.to_csv(bucket_dir / "stories_unique.csv", index=False)
        write_json(
            bucket_dir / "run_summary.json",
            {
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "raw_rows": int(len(raw)),
                "unique_stories": int(len(stories)),
                "dataset_id": ds.id,
                "skip_urls": skip_urls,
            },
        )
        buckets_done += 1
        logger.info(
            "feed bucket %s — unique=%d (bucket %d)",
            stamp,
            len(stories),
            buckets_done,
        )
        if max_buckets > 0 and buckets_done >= max_buckets:
            break
        sleep_s = interval_minutes * 60
        logger.info("Sleeping %d seconds until next bucket", sleep_s)
        time.sleep(sleep_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["pull", "last15", "recover", "feed"],
        help="pull (or last15 alias) | recover MissedStories | continuous feed",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=None,
        help="Universe CSV with RP_ENTITY_ID [, COMPANY_NAME] [, ticker] (e.g. us_sml.csv)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers to map via Edge (e.g. AAPL,MSFT,NVDA)",
    )
    parser.add_argument(
        "--entity-ids",
        type=str,
        default=None,
        help="Comma-separated RP entity ids (skip ticker mapping)",
    )
    parser.add_argument(
        "--missed-csv",
        type=Path,
        default=Path("MissedStories.csv"),
        help="MissedStories CSV (Ticker column) for recover, or fallback universe",
    )
    parser.add_argument(
        "--limit-entities",
        type=int,
        default=0,
        help="Optional cap on universe size after load (0 = all)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--min-entity-relevance",
        type=float,
        default=None,
        help="Optional entity_relevance floor (e.g. 90)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Window start UTC ISO (use with --end), e.g. 2026-07-28T12:00:00Z",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Window end UTC ISO (use with --start)",
    )
    parser.add_argument(
        "--window-end",
        type=str,
        default=None,
        help="Window end UTC ISO; default now. Used with --window-minutes",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=15,
        help="Minutes before --window-end (default 15). Ignored if --start/--end set",
    )
    parser.add_argument("--match-threshold", type=float, default=0.72)
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=15,
        help="Feed mode: bucket length in minutes",
    )
    parser.add_argument(
        "--max-buckets",
        type=int,
        default=1,
        help="For feed mode: number of buckets then exit (0 = forever)",
    )
    parser.add_argument(
        "--skip-urls",
        action="store_true",
        help="Do not call document URL endpoint (saves quota; leave url empty)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.output_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        mode_name = "pull" if args.mode == "last15" else args.mode
        args.output_dir = Path("runs") / f"edge_{mode_name}_{stamp}"

    if args.mode in {"pull", "last15"}:
        run_pull(
            output_dir=args.output_dir,
            min_entity_relevance=args.min_entity_relevance,
            skip_urls=args.skip_urls,
            universe=args.universe,
            tickers=args.tickers,
            entity_ids=args.entity_ids,
            missed_csv=args.missed_csv if args.missed_csv.exists() else None,
            limit_entities=args.limit_entities,
            start=args.start,
            end=args.end,
            window_end=args.window_end,
            window_minutes=args.window_minutes,
        )
    elif args.mode == "recover":
        run_recover(
            missed_csv=args.missed_csv,
            output_dir=args.output_dir,
            min_entity_relevance=args.min_entity_relevance,
            match_threshold=args.match_threshold,
            skip_urls=args.skip_urls,
        )
    else:
        run_feed(
            output_dir=args.output_dir,
            min_entity_relevance=args.min_entity_relevance,
            interval_minutes=args.interval_minutes,
            max_buckets=args.max_buckets,
            skip_urls=args.skip_urls,
            universe=args.universe,
            tickers=args.tickers,
            entity_ids=args.entity_ids,
            missed_csv=args.missed_csv if args.missed_csv.exists() else None,
            limit_entities=args.limit_entities,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())