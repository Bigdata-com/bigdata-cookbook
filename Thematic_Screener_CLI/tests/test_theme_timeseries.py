"""Tests for thematic attention and exposure time series."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.theme_timeseries import (
    add_attention_metrics,
    build_company_exposure_daily,
    build_exposure_aggregate_daily,
    build_volume_payload,
    normalize_theme_volume,
)


def test_build_volume_payload_uses_texts_for_multiple_queries() -> None:
    payload = build_volume_payload(
        ["Rate cuts and refinancing.", "Lower yields and credit demand."],
        start_date=date(2025, 9, 1),
        end_date=date(2026, 9, 1),
    )

    query = payload["query"]
    assert query["texts"] == [
        "Rate cuts and refinancing.",
        "Lower yields and credit demand.",
    ]
    assert query["filters"]["timestamp"]["end"] == "2026-09-01T23:59:59Z"


def test_build_volume_payload_rejects_more_than_five_queries() -> None:
    with pytest.raises(ValueError, match="at most five"):
        build_volume_payload(
            [f"query {index}" for index in range(6)],
            start_date=date(2025, 9, 1),
            end_date=date(2026, 9, 1),
        )


def test_normalize_theme_volume_fills_missing_dates() -> None:
    payload = {
        "results": {
            "volume": [
                {
                    "date": "2026-08-30",
                    "documents": 10,
                    "chunks": 15,
                    "sentiment": -0.2,
                },
                {
                    "date": "2026-09-01",
                    "documents": 20,
                    "chunks": 31,
                    "sentiment": 0.3,
                },
            ]
        }
    }

    result = normalize_theme_volume(
        payload,
        start_date=date(2026, 8, 30),
        end_date=date(2026, 9, 1),
    )

    assert result["documents"].tolist() == [10, 0, 20]
    assert result["chunks"].tolist() == [15, 0, 31]


def test_attention_metrics_use_prior_observations() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-08-01", periods=9),
            "documents": [10] * 8 + [30],
            "chunks": [20] * 9,
            "sentiment": [0.1] * 9,
        }
    )

    result = add_attention_metrics(frame, window_days=8)

    assert pd.isna(result.loc[7, "attention_zscore"])
    assert pd.isna(result.loc[8, "attention_zscore"])
    assert result.loc[8, "documents_30d_mean"] == 10


def test_build_company_exposure_daily_uses_relevance_times_abs_sentiment() -> None:
    evidence = pd.DataFrame(
        {
            "timestamp": [
                "2026-09-01T09:00:00Z",
                "2026-09-01T11:00:00Z",
                "2026-09-02T09:00:00Z",
            ],
            "company_name": ["Bank A", "Bank A", "Retailer B"],
            "document_id": ["d1", "d2", "d3"],
            "relevance": [0.8, 0.5, 0.4],
            "sentiment": [-0.5, 0.2, 0.25],
        }
    )

    daily = build_company_exposure_daily(evidence, theme="Rate-cut repricing")

    bank = daily.loc[daily["company_name"] == "Bank A"].iloc[0]
    assert bank["evidence_count"] == 2
    assert bank["unique_documents"] == 2
    assert bank["exposure_score"] == pytest.approx(0.5)
    assert bank["sentiment"] == pytest.approx(-0.15)


def test_build_exposure_aggregate_daily_fills_calendar() -> None:
    company_daily = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-09-01")],
            "theme": ["Rate-cut repricing"],
            "company_name": ["Bank A"],
            "evidence_count": [2],
            "unique_documents": [2],
            "exposure_score": [0.5],
            "sentiment": [-0.15],
        }
    )

    result = build_exposure_aggregate_daily(
        company_daily,
        start_date=date(2026, 8, 31),
        end_date=date(2026, 9, 2),
    )

    assert result["exposure_score"].tolist() == [0, 0.5, 0]
    assert result["exposure_score_7d"].tolist() == [0, 0.5, 0.5]
