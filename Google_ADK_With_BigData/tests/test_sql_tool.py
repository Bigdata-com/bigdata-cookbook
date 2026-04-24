"""Internal SQLite tools (no network)."""

from __future__ import annotations


def test_internal_query_database_select_nvda_pf002(agent_module) -> None:
    result = agent_module.internal_query_database(
        "SELECT ticker, portfolio_id FROM holdings WHERE portfolio_id = 'PF002' AND ticker = 'NVDA'"
    )
    assert "error" not in result
    assert result["row_count"] == 1
    assert result["results"][0]["ticker"] == "NVDA"


def test_internal_query_database_rejects_non_select(agent_module) -> None:
    result = agent_module.internal_query_database("DELETE FROM holdings WHERE 1=1")
    assert "error" in result


def test_internal_portfolio_summary_pf002(agent_module) -> None:
    result = agent_module.internal_portfolio_summary("PF002")
    assert "error" not in result
    assert result["portfolio"] is not None
    assert result["portfolio"]["portfolio_id"] == "PF002"
    tickers = {h["ticker"] for h in result["holdings"]}
    assert "NVDA" in tickers
