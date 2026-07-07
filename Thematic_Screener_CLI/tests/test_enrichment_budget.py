from __future__ import annotations

from pathlib import Path

from src.mcp_workflow import (
    DEFAULT_ENRICHMENT_MAX_CONCURRENT,
    DEFAULT_ENRICHMENT_REQUESTS_PER_MINUTE,
    ENRICHMENT_COST_SAFETY_MARGIN,
    MAX_LABELING_BATCH_SIZE,
    MCP_ENRICHMENT_SPLIT_THRESHOLD_SECONDS,
    MCP_LABELING_BATCH_SAFE_SECONDS,
    MIN_ESTIMATED_COST_USD,
    MIN_LABELING_BATCH_SIZE,
    SUMMARY_COMPANY_RETENTION_FACTOR,
    _apply_enrichment_cost,
    _approx_tokens,
    _compute_labeling_batch_size,
    _context,
    _enrichment_execution_mode,
    _enrichment_execution_next_actions,
    _enrichment_execution_note,
    _estimate_enrichment_plan,
    _estimate_mcp_labeling_batch_seconds,
    _estimate_openai_batch_seconds,
    _estimate_summary_batch_seconds,
    _format_duration,
    _format_enrichment_cost,
    estimate_enrichment_budget,
)


def test_format_duration_short_intervals() -> None:
    assert _format_duration(0) == "under 1 minute"
    assert _format_duration(45) == "about 45 seconds"
    assert _format_duration(90) == "about 2 minutes"
    assert _format_duration(7200) == "about 2.0 hours"


def test_estimate_openai_batch_seconds_uses_rate_and_concurrency() -> None:
    rpm_bound = _estimate_openai_batch_seconds(
        request_count=100,
        requests_per_minute=DEFAULT_ENRICHMENT_REQUESTS_PER_MINUTE,
        max_concurrent_requests=DEFAULT_ENRICHMENT_MAX_CONCURRENT,
    )
    assert rpm_bound == 12

    concurrency_bound = _estimate_openai_batch_seconds(
        request_count=100,
        requests_per_minute=10_000,
        max_concurrent_requests=10,
        seconds_per_wave=2.0,
    )
    assert concurrency_bound == 20


def test_summary_batch_seconds_scales_with_payload() -> None:
    small = _estimate_summary_batch_seconds(
        [{"input_chars": 1500}],
        requests_per_minute=10_000,
        max_concurrent_requests=20,
    )
    large = _estimate_summary_batch_seconds(
        [{"input_chars": 15000}],
        requests_per_minute=10_000,
        max_concurrent_requests=20,
    )
    assert large >= small


def test_apply_enrichment_cost_uses_token_pricing_and_margin() -> None:
    raw = _apply_enrichment_cost(
        input_tokens=1_000_000,
        output_tokens=0,
        input_usd_per_mtok=0.10,
        output_usd_per_mtok=0.40,
        safety_margin=1.0,
    )
    assert raw == 0.1
    adjusted = _apply_enrichment_cost(
        input_tokens=1_000_000,
        output_tokens=0,
        input_usd_per_mtok=0.10,
        output_usd_per_mtok=0.40,
        safety_margin=ENRICHMENT_COST_SAFETY_MARGIN,
    )
    assert adjusted == 0.15


def test_format_enrichment_cost_never_rounds_to_zero() -> None:
    assert _format_enrichment_cost(0.002) == f"${MIN_ESTIMATED_COST_USD:.3f}"


def test_estimate_enrichment_budget_for_smoke_run() -> None:
    run_name = "live_mcp_smoke_20260618_215813"
    context = _context(run_name)
    if not context.results_path.exists():
        import pytest

        pytest.skip(f"fixture run missing results at {context.results_path}")
    response = estimate_enrichment_budget(run_name)
    budget = response["enrichment_budget"]
    assert budget["sentence_count"] == 29
    assert budget["estimated_total_cost_usd"] >= MIN_ESTIMATED_COST_USD
    assert budget["estimation_method"] == "token_based_with_safety_margin"
    assert budget["estimated_summary_company_count"] >= 1


def test_fresh_summary_company_estimate_uses_retention_factor() -> None:
    run_name = "spacex_ipo_global_plan_20260618_223605"
    context = _context(run_name)
    if not context.results_path.exists():
        import pytest

        pytest.skip(f"fixture run missing results at {context.results_path}")
    labeled_path = Path("runs") / run_name / "labeled_sentences.csv"
    backup = labeled_path.read_text() if labeled_path.exists() else None
    try:
        if labeled_path.exists():
            labeled_path.unlink()
        plan = _estimate_enrichment_plan(_context(run_name))
        raw = int(plan["raw_company_count"])
        expected = max(1, min(raw, round(raw * SUMMARY_COMPANY_RETENTION_FACTOR)))
        assert int(plan["estimated_summary_company_count"]) == expected
        assert plan["uses_prior_labels"] is False
    finally:
        if backup is not None:
            labeled_path.write_text(backup)


def test_approx_tokens_uses_char_heuristic() -> None:
    assert _approx_tokens("abcd") == 1
    assert _approx_tokens("a" * 40) == 10


def test_enrichment_execution_mode_recommends_split_for_long_runs() -> None:
    threshold = MCP_ENRICHMENT_SPLIT_THRESHOLD_SECONDS
    assert _enrichment_execution_mode(threshold + 1, 10) == "split"
    assert _enrichment_execution_mode(10, threshold + 1) == "split"
    assert _enrichment_execution_mode(60, 60) == "combined"


def test_enrichment_execution_next_actions() -> None:
    assert _enrichment_execution_next_actions("split") == [
        "run_labeling",
        "run_company_summaries",
    ]
    assert "run_enrichment" in _enrichment_execution_next_actions("combined")
    assert "batches" in _enrichment_execution_note("split")


def test_compute_labeling_batch_size_stays_within_target() -> None:
    batch_size = _compute_labeling_batch_size(
        sentence_count=2129,
        requests_per_minute=DEFAULT_ENRICHMENT_REQUESTS_PER_MINUTE,
        max_concurrent_requests=DEFAULT_ENRICHMENT_MAX_CONCURRENT,
        target_seconds=MCP_LABELING_BATCH_SAFE_SECONDS,
    )
    assert MIN_LABELING_BATCH_SIZE <= batch_size <= MAX_LABELING_BATCH_SIZE
    seconds = _estimate_mcp_labeling_batch_seconds(
        batch_size,
        DEFAULT_ENRICHMENT_REQUESTS_PER_MINUTE,
        DEFAULT_ENRICHMENT_MAX_CONCURRENT,
    )
    assert seconds <= MCP_LABELING_BATCH_SAFE_SECONDS
    assert batch_size == 120

