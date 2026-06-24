from __future__ import annotations

from src.mcp_workflow import (
    DEFAULT_ENRICHMENT_MAX_CONCURRENT,
    MCP_LABELING_BATCH_SAFE_SECONDS,
    MCP_LABELING_WAVE_SECONDS,
    MCP_LABELING_WAVE_SECONDS_STRAGGLER,
    _compute_labeling_batch_size,
    _estimate_mcp_labeling_batch_seconds,
    _estimate_openai_batch_seconds,
    _resolve_labeling_straggler_seconds,
)
from src.screener import LabelingLatencyStats, _labeling_latency_stats


def test_labeling_latency_stats_empty() -> None:
    stats = _labeling_latency_stats([])
    assert stats == LabelingLatencyStats(p50=None, p95=None, p99=None, max_seconds=None)


def test_labeling_latency_stats_percentiles() -> None:
    stats = _labeling_latency_stats([1.0, 2.0, 3.0, 4.0, 20.0])
    assert stats.p50 == 3.0
    assert stats.p95 == 20.0
    assert stats.p99 == 20.0
    assert stats.max_seconds == 20.0


def test_estimate_mcp_labeling_batch_seconds_uses_straggler_wave_model() -> None:
    seconds = _estimate_mcp_labeling_batch_seconds(
        request_count=120,
        requests_per_minute=10_000,
        max_concurrent_requests=DEFAULT_ENRICHMENT_MAX_CONCURRENT,
    )
    assert seconds == int(
        _estimate_openai_batch_seconds(
            120,
            10_000,
            DEFAULT_ENRICHMENT_MAX_CONCURRENT,
            MCP_LABELING_WAVE_SECONDS_STRAGGLER,
        )
    )
    assert seconds == 90


def test_total_labeling_estimate_uses_average_wave_model() -> None:
    seconds = _estimate_openai_batch_seconds(
        request_count=2129,
        requests_per_minute=10_000,
        max_concurrent_requests=DEFAULT_ENRICHMENT_MAX_CONCURRENT,
        seconds_per_wave=MCP_LABELING_WAVE_SECONDS,
    )
    assert seconds == 216


def test_mcp_batch_estimate_stays_within_safe_target_for_recommended_size() -> None:
    batch_size = 120
    seconds = _estimate_mcp_labeling_batch_seconds(
        batch_size,
        requests_per_minute=10_000,
        max_concurrent_requests=DEFAULT_ENRICHMENT_MAX_CONCURRENT,
    )
    assert seconds <= MCP_LABELING_BATCH_SAFE_SECONDS


def test_resolve_labeling_straggler_seconds_uses_observed_tail() -> None:
    progress = {
        "last_batch_elapsed_seconds": 58.0,
        "batch_size": 150,
        "max_concurrent_requests": 40,
        "last_batch_latency_max_seconds": 58.0,
        "last_batch_latency_p99_seconds": 55.0,
    }
    straggler = _resolve_labeling_straggler_seconds(progress)
    assert straggler >= MCP_LABELING_WAVE_SECONDS_STRAGGLER
    assert straggler >= 58.0


def test_compute_labeling_batch_size_caps_at_safe_limit() -> None:
    batch_size = _compute_labeling_batch_size(
        sentence_count=2129,
        requests_per_minute=10_000,
        max_concurrent_requests=DEFAULT_ENRICHMENT_MAX_CONCURRENT,
    )
    assert batch_size == 120
