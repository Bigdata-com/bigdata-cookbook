from __future__ import annotations

from src.mcp_workflow import (
    RETRIEVAL_CHUNKS_PER_COST_UNIT,
    RETRIEVAL_COST_USD_PER_UNIT,
    _estimate_retrieval_cost_usd,
    _max_chunks_for_retrieval_budget,
    _resolve_budget_selection,
    _retrieval_pricing_from_budget,
)


def test_estimate_retrieval_cost_usd_charges_per_10_chunks() -> None:
    assert _estimate_retrieval_cost_usd(10) == 0.01
    assert _estimate_retrieval_cost_usd(920) == 1.38
    assert _estimate_retrieval_cost_usd(3680) == 5.52


def test_max_chunks_for_retrieval_budget_inverts_pricing() -> None:
    assert _max_chunks_for_retrieval_budget(0.015) == RETRIEVAL_CHUNKS_PER_COST_UNIT
    assert _max_chunks_for_retrieval_budget(5.52) == 3680


def test_resolve_budget_selection_uses_dollar_cap_with_per_10_chunk_pricing() -> None:
    budget_payload = {
        "presets": [],
        "retrieval_cost_usd_per_10_chunks": RETRIEVAL_COST_USD_PER_UNIT,
        "retrieval_chunks_per_cost_unit": RETRIEVAL_CHUNKS_PER_COST_UNIT,
    }
    resolved = _resolve_budget_selection(
        {"max_cost_usd": 5.52},
        budget_payload,
        total_expected_chunks=184_000,
    )
    assert resolved["selected_chunks"] == 3680


def test_retrieval_pricing_from_budget_handles_legacy_per_chunk_key() -> None:
    cost_usd_per_unit, chunks_per_unit = _retrieval_pricing_from_budget(
        {"cost_per_chunk_usd": 0.015}
    )
    assert cost_usd_per_unit == RETRIEVAL_COST_USD_PER_UNIT
    assert chunks_per_unit == RETRIEVAL_CHUNKS_PER_COST_UNIT
