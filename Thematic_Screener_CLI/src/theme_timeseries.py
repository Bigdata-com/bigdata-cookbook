"""Daily theme-attention and company-exposure time-series builders."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import date
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

BIGDATA_VOLUME_URL = (
    os.environ.get("BIGDATA_API_BASE_URL", "https://api.bigdata.com").rstrip("/")
    + "/v1/search/volume"
)
DEFAULT_CATEGORY: dict[str, Any] = {
    "mode": "INCLUDE",
    "values": ["news_premium", "transcripts", "filings"],
}
ATTENTION_WINDOW_DAYS = 30
ROLLING_EXPOSURE_DAYS = 7

THEME_VOLUME_COLUMNS: tuple[str, ...] = (
    "date",
    "documents",
    "chunks",
    "sentiment",
    "documents_30d_mean",
    "documents_30d_std",
    "attention_zscore",
)
COMPANY_EXPOSURE_COLUMNS: tuple[str, ...] = (
    "date",
    "theme",
    "company_name",
    "evidence_count",
    "unique_documents",
    "exposure_score",
    "sentiment",
)


def _iso_bound(value: date, *, end: bool) -> str:
    suffix = "T23:59:59Z" if end else "T00:00:00Z"
    return f"{value.isoformat()}{suffix}"


def build_volume_payload(
    search_queries: Sequence[str],
    *,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    """Build a Bigdata.com Search Volume request for one thematic definition."""
    queries = tuple(query.strip() for query in search_queries if query.strip())
    if not queries:
        raise ValueError("At least one non-empty search query is required")
    if len(queries) > 5:
        raise ValueError("Bigdata.com Search Volume accepts at most five query texts")
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")

    query: dict[str, Any] = {
        "filters": {
            "timestamp": {
                "start": _iso_bound(start_date, end=False),
                "end": _iso_bound(end_date, end=True),
            },
            "category": DEFAULT_CATEGORY,
        }
    }
    if len(queries) == 1:
        query["text"] = queries[0]
    else:
        query["texts"] = list(queries)
    return {"query": query}


def fetch_theme_volume(
    search_queries: Sequence[str],
    *,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Fetch and normalize daily volume and sentiment from Bigdata.com."""
    api_key = os.environ.get("BIGDATA_API_KEY")
    if not api_key:
        raise RuntimeError("BIGDATA_API_KEY is not set")
    payload = build_volume_payload(
        search_queries,
        start_date=start_date,
        end_date=end_date,
    )
    request = Request(
        BIGDATA_VOLUME_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-KEY": api_key},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Bigdata.com Search Volume failed ({exc.code}): {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Bigdata.com Search Volume connection error: {exc}") from exc
    return normalize_theme_volume(parsed, start_date=start_date, end_date=end_date)


def normalize_theme_volume(
    payload: dict[str, Any],
    *,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Normalize a Search Volume response and fill missing calendar dates."""
    results = payload.get("results")
    volume = results.get("volume", []) if isinstance(results, dict) else []
    raw = pd.DataFrame(volume)
    calendar = pd.DataFrame({"date": pd.date_range(start=start_date, end=end_date, freq="D")})
    if raw.empty:
        normalized = calendar.assign(
            documents=0,
            chunks=0,
            sentiment=pd.Series(pd.NA, index=calendar.index, dtype="Float64"),
        )
    else:
        required = {"date", "documents", "chunks", "sentiment"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"Search Volume response is missing columns: {sorted(missing)}")
        raw = raw.loc[:, list(required)].assign(
            date=lambda frame: pd.to_datetime(frame["date"], errors="coerce"),
            documents=lambda frame: pd.to_numeric(frame["documents"], errors="coerce").fillna(0),
            chunks=lambda frame: pd.to_numeric(frame["chunks"], errors="coerce").fillna(0),
            sentiment=lambda frame: pd.to_numeric(frame["sentiment"], errors="coerce"),
        )
        normalized = calendar.merge(raw, on="date", how="left").assign(
            documents=lambda frame: frame["documents"].fillna(0).astype(int),
            chunks=lambda frame: frame["chunks"].fillna(0).astype(int),
        )
    return add_attention_metrics(normalized)


def add_attention_metrics(
    volume_df: pd.DataFrame,
    *,
    window_days: int = ATTENTION_WINDOW_DAYS,
) -> pd.DataFrame:
    """Add rolling volume baseline and z-score columns."""
    if window_days < 2:
        raise ValueError("window_days must be at least 2")
    result = volume_df.sort_values("date").reset_index(drop=True).copy()
    shifted = result["documents"].shift(1)
    rolling = shifted.rolling(window=window_days, min_periods=7)
    result["documents_30d_mean"] = rolling.mean()
    result["documents_30d_std"] = rolling.std(ddof=0)
    denominator = result["documents_30d_std"].replace(0, pd.NA)
    result["attention_zscore"] = (result["documents"] - result["documents_30d_mean"]) / denominator
    return result.loc[:, list(THEME_VOLUME_COLUMNS)]


def _numeric_column(
    frame: pd.DataFrame,
    column: str,
    *,
    default: float,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def build_company_exposure_daily(
    evidence_df: pd.DataFrame,
    *,
    theme: str,
) -> pd.DataFrame:
    """Aggregate classified evidence to daily company-level exposure rows."""
    required = {"timestamp", "company_name"}
    missing = required - set(evidence_df.columns)
    if missing:
        raise ValueError(f"Evidence is missing columns: {sorted(missing)}")
    if evidence_df.empty:
        return pd.DataFrame(columns=COMPANY_EXPOSURE_COLUMNS)

    dated = evidence_df.copy().assign(
        date=lambda frame: pd.to_datetime(
            frame["timestamp"], errors="coerce", format="mixed"
        ).dt.normalize(),
        _relevance=lambda frame: _numeric_column(frame, "relevance", default=0.0),
        _sentiment=lambda frame: _numeric_column(frame, "sentiment", default=0.0),
        theme=theme,
    )
    dated["_evidence_score"] = dated["_relevance"] * dated["_sentiment"].abs()
    dated = dated.dropna(subset=["date", "company_name"])
    if dated.empty:
        return pd.DataFrame(columns=COMPANY_EXPOSURE_COLUMNS)

    document_column = "document_id" if "document_id" in dated.columns else "timestamp"
    daily = (
        dated.groupby(["date", "theme", "company_name"], as_index=False)
        .agg(
            evidence_count=("company_name", "size"),
            unique_documents=(document_column, "nunique"),
            exposure_score=("_evidence_score", "sum"),
            sentiment=("_sentiment", "mean"),
        )
        .sort_values(["date", "exposure_score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return daily.loc[:, list(COMPANY_EXPOSURE_COLUMNS)]


def build_exposure_aggregate_daily(
    company_daily_df: pd.DataFrame,
    *,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Build a complete daily aggregate series with seven-day rolling metrics."""
    calendar = pd.DataFrame({"date": pd.date_range(start=start_date, end=end_date, freq="D")})
    if company_daily_df.empty:
        grouped = pd.DataFrame(
            columns=[
                "date",
                "evidence_count",
                "unique_documents",
                "exposure_score",
                "company_breadth",
                "sentiment",
            ]
        )
    else:
        grouped = company_daily_df.groupby("date", as_index=False).agg(
            evidence_count=("evidence_count", "sum"),
            unique_documents=("unique_documents", "sum"),
            exposure_score=("exposure_score", "sum"),
            company_breadth=("company_name", "nunique"),
            sentiment=("sentiment", "mean"),
        )
    result = calendar.merge(grouped, on="date", how="left")
    zero_columns = (
        "evidence_count",
        "unique_documents",
        "exposure_score",
        "company_breadth",
    )
    result.loc[:, list(zero_columns)] = result.loc[:, list(zero_columns)].fillna(0)
    result["exposure_score_7d"] = (
        result["exposure_score"]
        .rolling(
            ROLLING_EXPOSURE_DAYS,
            min_periods=1,
        )
        .sum()
    )
    result["company_breadth_7d"] = (
        result["company_breadth"]
        .rolling(
            ROLLING_EXPOSURE_DAYS,
            min_periods=1,
        )
        .mean()
    )
    return result
