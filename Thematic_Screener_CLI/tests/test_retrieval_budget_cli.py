"""Tests for CLI retrieval budget helpers."""

from __future__ import annotations

import pandas as pd

from src.retrieval_budget import (
    build_retrieval_preset_rows,
    estimate_retrieval_cost_usd,
    format_retrieval_budget_report,
)


def test_build_retrieval_preset_rows_matches_mcp_pricing() -> None:
    rows = build_retrieval_preset_rows(184_000)
    by_name = {row["name"]: row for row in rows}
    assert by_name["quick_scan"]["selected_chunks"] == 920
    assert by_name["quick_scan"]["estimated_cost_usd"] == 1.38
    assert by_name["balanced"]["selected_chunks"] == 3_680
    assert by_name["balanced"]["estimated_cost_usd"] == 5.52
    assert by_name["deep_dive"]["selected_chunks"] == 9_200
    assert by_name["deep_dive"]["estimated_cost_usd"] == 13.80
    assert by_name["full"]["selected_chunks"] == 184_000
    assert by_name["full"]["estimated_cost_usd"] == 276.00


def test_estimate_retrieval_cost_usd_zero_for_non_positive_chunks() -> None:
    assert estimate_retrieval_cost_usd(0) == 0.0


def test_format_retrieval_budget_report_includes_presets_and_total() -> None:
    summary_df = pd.DataFrame(
        [
            {"plan_file": "plan_a.json", "theme": "query a", "chunks": 100},
            {"plan_file": "plan_b.json", "theme": "query b", "chunks": 50},
        ]
    )
    presets = build_retrieval_preset_rows(150)
    report = format_retrieval_budget_report(
        run_name="cost-preview-mag7",
        main_theme="US Government Shutdown",
        universe="mag7.csv",
        start_date="2025-06-01",
        end_date="2026-06-09",
        label_count=2,
        company_count=7,
        summary_df=summary_df,
        presets=presets,
    )
    assert "Total: 150" in report
    assert "quick_scan" in report
    assert "balanced" in report
    assert "deep_dive" in report
    assert "full" in report
    assert "100.0%" in report
    assert "$0.01" in report or "$0.02" in report
    assert "To retrieve all chunks" in report
