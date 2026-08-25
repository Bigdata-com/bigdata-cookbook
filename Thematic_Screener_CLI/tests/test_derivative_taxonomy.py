"""Tests for derivative-hop taxonomy validation and grounding briefs."""

from __future__ import annotations

from pathlib import Path

from src.derivative_grounding import collect_grounding, format_grounding_brief
from src.derivative_taxonomy import derivative_preview, validate_derivatives_taxonomy
from src.prompts import (
    SYSTEM_MESSAGE_LABELS,
    SYSTEM_MESSAGE_LABELS_DERIVATIVES,
    SYSTEM_PROMPT_LABELING,
    TAXONOMY_STYLE_DERIVATIVES,
)
from src.screener import Node, load_universe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TSX_UNIVERSE = PROJECT_ROOT / "tsx_top150_rp_entities.csv"


def _leaf(node: int, label: str, search_query: str) -> dict[str, object]:
    return {
        "node": node,
        "label": label,
        "summary": f"{label} exposure pathway.",
        "search_query": search_query,
        "children": [],
    }


def _valid_tree() -> Node:
    return Node.model_validate(
        {
            "node": 1,
            "label": "Oil price increase exposure",
            "summary": "Theme root.",
            "search_query": "",
            "children": [
                {
                    "node": 2,
                    "label": "1st derivative",
                    "summary": "Direct impact.",
                    "search_query": "",
                    "children": [
                        _leaf(
                            5,
                            "Airline fuel costs",
                            "The company reports jet fuel expense on passenger flights.",
                        )
                    ],
                },
                {
                    "node": 3,
                    "label": "2nd derivative",
                    "summary": "Impact of the impact.",
                    "search_query": "",
                    "children": [
                        _leaf(
                            6,
                            "Airline capacity discipline",
                            "The company reduced available seat miles to protect unit margins.",
                        )
                    ],
                },
                {
                    "node": 4,
                    "label": "3rd derivative",
                    "summary": "Knock-on effects.",
                    "search_query": "",
                    "children": [
                        _leaf(
                            7,
                            "Discount retailers",
                            "The company reports mix toward value and private-label merchandise.",
                        )
                    ],
                },
            ],
        }
    )


def test_validate_derivatives_taxonomy_accepts_three_hop_tree() -> None:
    assert validate_derivatives_taxonomy(_valid_tree()) == []


def test_validate_derivatives_taxonomy_requires_three_branches() -> None:
    root = _valid_tree()
    root.children = root.children[:2]
    findings = validate_derivatives_taxonomy(root)
    assert any(item["check"] == "derivative_branches" for item in findings)


def test_validate_derivatives_taxonomy_flags_first_order_query_overlap() -> None:
    root = _valid_tree()
    root.children[1].children[0].search_query = (
        "The company reports jet fuel expense on passenger flights."
    )
    findings = validate_derivatives_taxonomy(root)
    assert any(item["check"] == "derivative_query_overlap" for item in findings)


def test_derivative_preview_groups_leaves_by_branch() -> None:
    preview = derivative_preview(_valid_tree())
    assert preview["1st derivative"] == ["Airline fuel costs"]
    assert preview["2nd derivative"] == ["Airline capacity discipline"]
    assert preview["3rd derivative"] == ["Discount retailers"]


def test_derivatives_prompt_contains_oil_example_and_hop_rules() -> None:
    assert "1st derivative" in SYSTEM_MESSAGE_LABELS_DERIVATIVES
    assert "2nd derivative" in SYSTEM_MESSAGE_LABELS_DERIVATIVES
    assert "3rd derivative" in SYSTEM_MESSAGE_LABELS_DERIVATIVES
    assert "Oil price" in SYSTEM_MESSAGE_LABELS_DERIVATIVES
    assert "{grounding_brief}" in SYSTEM_MESSAGE_LABELS_DERIVATIVES
    assert "5-8 leaf labels" in SYSTEM_MESSAGE_LABELS
    assert TAXONOMY_STYLE_DERIVATIVES == "derivatives"


def test_labeling_prompt_requires_hop_evidence() -> None:
    assert "2nd or 3rd" in SYSTEM_PROMPT_LABELING


def test_format_grounding_brief_includes_citations() -> None:
    payload = {
        "hops": [
            {
                "branch": "1st derivative",
                "query": "direct impact",
                "chunks": [
                    {
                        "text": "Jet fuel costs rose 20 percent.",
                        "source_name": "Reuters",
                        "timestamp": "2026-01-15",
                        "url": "https://example.com/article",
                    }
                ],
            }
        ]
    }
    brief = format_grounding_brief(payload)
    assert "1st derivative" in brief
    assert "Reuters" in brief
    assert "2026-01-15" in brief
    assert "https://example.com/article" in brief


def test_collect_grounding_uses_injected_retriever() -> None:
    calls: list[str] = []

    def fake_retriever(
        query: str,
        start_date: str,
        end_date: str,
        max_chunks: int,
    ) -> list[dict[str, str]]:
        calls.append(query)
        return [
            {
                "text": f"Evidence for {query}",
                "source_name": "Example Source",
                "timestamp": "2026-02-01",
                "url": "https://example.com/1",
            }
        ]

    payload = collect_grounding(
        "Oil price increase",
        retriever=fake_retriever,
        max_chunks_per_hop=1,
    )
    assert len(calls) == 3
    assert len(payload["hops"]) == 3
    assert all(hop["chunks"] for hop in payload["hops"])


def test_load_universe_reads_tsx_top150_csv() -> None:
    universe = load_universe(TSX_UNIVERSE)
    assert len(universe) >= 150
    assert "RP_ENTITY_ID" in universe.columns
    assert "COMPANY_NAME" in universe.columns
    assert universe["RP_ENTITY_ID"].iloc[0]
    assert "Royal Bank" in str(universe["COMPANY_NAME"].iloc[0])
