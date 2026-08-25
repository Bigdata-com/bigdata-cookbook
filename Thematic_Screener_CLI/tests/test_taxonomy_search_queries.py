"""Tests for label/search_query taxonomy split."""

# ruff: noqa: E501

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from src.helpers import get_leaf_labels, get_leaf_pairs, get_leaf_search_queries
from src.run_context import RunContext
from src.screener import (
    EXCEL_MAX_CELL_CHARS,
    EXCEL_TRUNCATION_NOTICE,
    Node,
    analyst_focus_with_leaf_cap,
    build_plans,
    export_excel,
    normalize_max_leaf_labels,
    write_taxonomy_artifacts,
)
from src.search_query import has_exposure_meta_language, normalize_summary_to_search_query


def _sample_tree() -> Node:
    return Node.model_validate(
        {
            "node": 1,
            "label": "Data center development exposure",
            "summary": "Company exposure to data center construction, operation, supply, or financing.",
            "search_query": "",
            "children": [
                {
                    "node": 2,
                    "label": "Cooling suppliers",
                    "summary": "Companies selling HVAC, chillers, and liquid cooling systems.",
                    "search_query": (
                        "The company sells HVAC, chillers, and liquid cooling systems to data "
                        "center operators."
                    ),
                    "children": [],
                }
            ],
        }
    )


def test_get_leaf_labels_and_search_queries() -> None:
    root = _sample_tree()
    assert get_leaf_labels(root) == ["Cooling suppliers"]
    assert get_leaf_search_queries(root) == [
        "The company sells HVAC, chillers, and liquid cooling systems to data center operators."
    ]
    assert get_leaf_pairs(root) == [
        (
            "Cooling suppliers",
            "The company sells HVAC, chillers, and liquid cooling systems to data center operators.",
        )
    ]


def test_search_query_fallback_from_summary() -> None:
    root = Node.model_validate(
        {
            "node": 1,
            "label": "Root",
            "summary": "Root summary",
            "search_query": "",
            "children": [
                {
                    "node": 2,
                    "label": "Cooling vendors",
                    "summary": "Companies profiting from HVAC and liquid cooling deployments.",
                    "search_query": "",
                    "children": [],
                }
            ],
        }
    )
    queries = get_leaf_search_queries(root)
    assert queries[0].startswith("The company")
    assert not has_exposure_meta_language(queries[0])


def test_normalize_summary_to_search_query_strips_meta_language() -> None:
    text = "Suppliers exposed to SpaceX IPO-driven production and capex scaling."
    rewritten = normalize_summary_to_search_query(text)
    assert "exposed to" not in rewritten.lower()
    assert rewritten.startswith("The company")


def test_write_taxonomy_artifacts_persists_label_and_query_files(tmp_path: Path) -> None:
    root = _sample_tree()
    themes_path = tmp_path / "themes.txt"
    search_queries_path = tmp_path / "search_queries.txt"
    taxonomy_tree_path = tmp_path / "taxonomy_tree.json"

    labels, queries = write_taxonomy_artifacts(
        root,
        themes_path=themes_path,
        search_queries_path=search_queries_path,
        taxonomy_tree_path=taxonomy_tree_path,
    )

    assert labels == ["Cooling suppliers"]
    assert themes_path.read_text(encoding="utf-8").strip() == "Cooling suppliers"
    assert "The company sells HVAC" in search_queries_path.read_text(encoding="utf-8")
    assert queries[0].startswith("The company")
    assert taxonomy_tree_path.exists()


def test_run_context_read_search_queries_legacy_fallback(tmp_path: Path) -> None:
    context = RunContext.create(tmp_path, "legacy_run")
    context.ensure_run_dir()
    context.write_themes(
        ["Companies profiting from HVAC and liquid cooling deployments for data centers."]
    )

    queries = context.read_search_queries()
    assert len(queries) == 1
    assert queries[0].startswith("The company")
    assert not has_exposure_meta_language(queries[0])


def test_build_plans_uses_search_query_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured_texts: list[str] = []

    def fake_plan_search(**kwargs: object) -> dict[str, list[dict[str, object]]]:
        captured_texts.append(str(kwargs["text"]))
        return {"baskets": [{"query": {"text": kwargs["text"]}, "expected_chunks": 1}]}

    monkeypatch.setattr("src.screener.plan_search", fake_plan_search)
    monkeypatch.setattr("src.screener.save_plan", lambda plan, path: None)

    build_plans(
        labels=["Cooling suppliers"],
        search_queries=["The company sells liquid cooling systems to data centers."],
        company_ids=["ABC123"],
        plans_dir=tmp_path,
    )

    assert captured_texts == ["The company sells liquid cooling systems to data centers."]


def test_build_plans_removes_plans_from_a_previous_taxonomy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_plan_search(**kwargs: object) -> dict[str, list[dict[str, object]]]:
        return {"baskets": [{"query": {"text": kwargs["text"]}, "expected_chunks": 1}]}

    monkeypatch.setattr("src.screener.plan_search", fake_plan_search)
    monkeypatch.setattr(
        "src.screener.save_plan",
        lambda plan, path: Path(path).write_text("{}", encoding="utf-8"),
    )
    stale_plan = tmp_path / "Retired_pathway.json"
    stale_plan.parent.mkdir(parents=True, exist_ok=True)
    stale_plan.write_text("{}", encoding="utf-8")

    saved_paths = build_plans(
        labels=["Cooling suppliers"],
        search_queries=["The company sells liquid cooling systems to data centers."],
        company_ids=["ABC123"],
        plans_dir=tmp_path,
    )

    assert not stale_plan.exists()
    assert sorted(path.name for path in tmp_path.glob("*.json")) == [
        path.name for path in saved_paths
    ]


def test_export_excel_marks_cells_over_the_excel_limit(tmp_path: Path) -> None:
    oversized_quote = "x" * (EXCEL_MAX_CELL_CHARS + 500)
    screener_df = pd.DataFrame(
        {
            "company_name": ["Acme Corp."],
            "label": ["Cooling suppliers"],
            "text": [oversized_quote],
        }
    )
    root = Node.model_validate(
        {
            "node": 1,
            "label": "Theme",
            "summary": "Root.",
            "search_query": "",
            "children": [],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        output_path = export_excel(
            screener_df=screener_df,
            company_summaries_df=pd.DataFrame(columns=["company_name", "summary"]),
            root=root,
            path=tmp_path / "report.xlsx",
        )

    written = pd.read_excel(output_path, sheet_name="Results")
    cell = str(written.loc[0, "text"])
    assert len(cell) <= EXCEL_MAX_CELL_CHARS
    assert cell.endswith(EXCEL_TRUNCATION_NOTICE)


def test_normalize_max_leaf_labels_treats_zero_as_uncapped() -> None:
    assert normalize_max_leaf_labels(None) is None
    assert normalize_max_leaf_labels(0) is None
    assert normalize_max_leaf_labels(15) == 15


def test_analyst_focus_with_leaf_cap_appends_limit() -> None:
    capped = analyst_focus_with_leaf_cap("Hidden hops.", 12)
    assert capped.endswith("Limit the final tree to at most 12 leaf nodes.")
    assert analyst_focus_with_leaf_cap("Hidden hops.", 0) == "Hidden hops."
