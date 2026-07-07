"""Bigdata retrieval chunk and dollar estimates shared by CLI and MCP."""

from __future__ import annotations

from typing import TypedDict

import pandas as pd

from src.screener import format_plans_summary

RETRIEVAL_CHUNKS_PER_COST_UNIT = 10
RETRIEVAL_COST_USD_PER_10_CHUNKS = 0.015

DEFAULT_RETRIEVAL_PRESETS: tuple[dict[str, float | str], ...] = (
    {"name": "quick_scan", "chunk_percentage": 0.005},
    {"name": "balanced", "chunk_percentage": 0.02},
    {"name": "deep_dive", "chunk_percentage": 0.05},
    {"name": "full", "chunk_percentage": 1.0},
)


class RetrievalPresetRow(TypedDict):
    name: str
    chunk_percentage: float
    selected_chunks: int
    estimated_cost_usd: float


def estimate_retrieval_cost_usd(
    selected_chunks: int,
    *,
    cost_usd_per_10_chunks: float = RETRIEVAL_COST_USD_PER_10_CHUNKS,
    chunks_per_cost_unit: int = RETRIEVAL_CHUNKS_PER_COST_UNIT,
) -> float:
    """Estimate retrieval spend ($0.015 per 10 chunks by default)."""
    if selected_chunks <= 0:
        return 0.0
    return round((selected_chunks / chunks_per_cost_unit) * cost_usd_per_10_chunks, 2)


def build_retrieval_preset_rows(
    total_expected_chunks: int,
    *,
    presets: tuple[dict[str, float | str], ...] | None = None,
    cost_usd_per_10_chunks: float = RETRIEVAL_COST_USD_PER_10_CHUNKS,
) -> list[RetrievalPresetRow]:
    """Build preset rows with selected chunk counts and dollar estimates."""
    selected_presets = presets or DEFAULT_RETRIEVAL_PRESETS
    rows: list[RetrievalPresetRow] = []
    for preset in selected_presets:
        chunk_percentage = float(preset["chunk_percentage"])
        selected_chunks = round(total_expected_chunks * chunk_percentage)
        rows.append(
            RetrievalPresetRow(
                name=str(preset["name"]),
                chunk_percentage=chunk_percentage,
                selected_chunks=selected_chunks,
                estimated_cost_usd=estimate_retrieval_cost_usd(
                    selected_chunks,
                    cost_usd_per_10_chunks=cost_usd_per_10_chunks,
                ),
            )
        )
    return rows


def format_retrieval_budget_report(
    *,
    run_name: str,
    main_theme: str,
    universe: str,
    start_date: str,
    end_date: str,
    label_count: int,
    company_count: int,
    summary_df: pd.DataFrame,
    presets: list[RetrievalPresetRow],
    cost_usd_per_10_chunks: float = RETRIEVAL_COST_USD_PER_10_CHUNKS,
) -> str:
    """Format a human-readable retrieval budget preview for stdout."""
    lines = [
        "Bigdata retrieval cost preview",
        f"Run: {run_name}",
        f"Theme: {main_theme}",
        f"Universe: {universe} ({company_count:,} companies)",
        f"Date range: {start_date} to {end_date}",
        f"Labels: {label_count}",
        "",
        format_plans_summary(summary_df),
        "",
        (
            "Pricing: "
            f"${cost_usd_per_10_chunks:.3f} per {RETRIEVAL_CHUNKS_PER_COST_UNIT} chunks "
            f"(estimated_cost_usd = selected_chunks / {RETRIEVAL_CHUNKS_PER_COST_UNIT} "
            f"× {cost_usd_per_10_chunks:.3f})"
        ),
        "",
        "Retrieval presets:",
        f"{'Preset':<12} {'Sample %':>10} {'Chunks':>12} {'Est. cost':>12}",
        f"{'-' * 12} {'-' * 10} {'-' * 12} {'-' * 12}",
    ]
    for row in presets:
        lines.append(
            f"{row['name']:<12} "
            f"{row['chunk_percentage'] * 100:>9.1f}% "
            f"{row['selected_chunks']:>12,} "
            f"${row['estimated_cost_usd']:>11,.2f}"
        )
    lines.extend(
        [
            "",
            (
                "To retrieve all chunks: use preset `full`, pass "
                "`--chunk-percentage 1.0` on search, or approve budget with "
                "`chunk_percentage: 1.0` in MCP."
            ),
            "Note: planning only — no documents retrieved. OpenAI taxonomy cost is separate.",
        ]
    )
    return "\n".join(lines)
