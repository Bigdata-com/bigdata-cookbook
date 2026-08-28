"""Tests for derivative-hop chart builders and notebook helpers."""

from __future__ import annotations

import logging

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from src import viz  # noqa: E402
from src.derivative_taxonomy import leaf_branch_map  # noqa: E402
from src.notebook_support import quiet_output  # noqa: E402
from src.prompts import DERIVATIVE_BRANCH_LABELS  # noqa: E402
from src.screener import Node  # noqa: E402

CHART_BUILDERS = (
    viz.plot_hop_coverage,
    viz.plot_top_pathways,
    viz.plot_top_companies,
    viz.plot_company_hop_matrix,
    viz.plot_evidence_timeline,
)


def _leaf(node: int, label: str) -> dict[str, object]:
    return {
        "node": node,
        "label": label,
        "summary": f"{label} pathway.",
        "search_query": f"The company reports {label}.",
        "children": [],
    }


def _tree() -> Node:
    return Node.model_validate(
        {
            "node": 1,
            "label": "Oil price increase exposure",
            "summary": "Theme root.",
            "search_query": "",
            "children": [
                {
                    "node": 2,
                    "label": DERIVATIVE_BRANCH_LABELS[0],
                    "summary": "Direct impact.",
                    "search_query": "",
                    "children": [_leaf(5, "Upstream crude producers")],
                },
                {
                    "node": 3,
                    "label": DERIVATIVE_BRANCH_LABELS[1],
                    "summary": "Second hop.",
                    "search_query": "",
                    "children": [_leaf(6, "Airline capacity discipline")],
                },
                {
                    "node": 4,
                    "label": DERIVATIVE_BRANCH_LABELS[2],
                    "summary": "Third hop.",
                    "search_query": "",
                    "children": [_leaf(7, "Discount retail trade-down")],
                },
            ],
        }
    )


def _labeled_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sentence_id": ["0", "1", "2", "3"],
            "company_name": ["Suncor", "Suncor", "Air Canada", "Loblaw"],
            "label": [
                "Upstream crude producers",
                "Upstream crude producers",
                "Airline capacity discipline",
                "Discount retail trade-down",
            ],
            "timestamp": [
                "2025-07-01T00:00:00",
                "2025-08-11T00:00:00",
                "2026-01-05T00:00:00",
                "2026-02-02T00:00:00",
            ],
        }
    )


def test_leaf_branch_map_assigns_each_leaf_to_its_hop() -> None:
    branch_map = leaf_branch_map(_tree())

    assert branch_map["Upstream crude producers"] == DERIVATIVE_BRANCH_LABELS[0]
    assert branch_map["Airline capacity discipline"] == DERIVATIVE_BRANCH_LABELS[1]
    assert branch_map["Discount retail trade-down"] == DERIVATIVE_BRANCH_LABELS[2]


def test_attach_hop_column_maps_labels_to_hops() -> None:
    evidence_df = viz.attach_hop_column(_labeled_frame(), _tree())

    assert evidence_df[viz.HOP_COLUMN].tolist() == [
        DERIVATIVE_BRANCH_LABELS[0],
        DERIVATIVE_BRANCH_LABELS[0],
        DERIVATIVE_BRANCH_LABELS[1],
        DERIVATIVE_BRANCH_LABELS[2],
    ]


def test_attach_hop_column_handles_frame_without_labels() -> None:
    evidence_df = viz.attach_hop_column(pd.DataFrame({"company_name": ["Suncor"]}), _tree())

    assert evidence_df[viz.HOP_COLUMN].isna().all()


def test_mindmap_renders_every_leaf() -> None:
    figure = viz.plot_mindmap(_tree())

    rendered = {text.get_text() for text in figure.axes[0].texts}
    assert "Upstream crude producers" in rendered
    assert "Discount retail trade-down" in rendered


def test_exposure_mindmap_renders_branch_leaves() -> None:
    root = Node.model_validate(
        {
            "node": 1,
            "label": "Supply Chain Reshaping",
            "summary": "Theme root.",
            "search_query": "",
            "children": [
                {
                    "node": 2,
                    "label": "Reshoring",
                    "summary": "Manufacturing relocation.",
                    "search_query": "",
                    "children": [
                        _leaf(5, "Nearshoring to Mexico"),
                        _leaf(6, "US fab expansion"),
                    ],
                },
                {
                    "node": 3,
                    "label": "Supplier diversification",
                    "summary": "Multi-source procurement.",
                    "search_query": "",
                    "children": [_leaf(7, "Dual sourcing semiconductors")],
                },
            ],
        }
    )

    figure = viz.plot_mindmap(root)
    rendered = {text.get_text() for text in figure.axes[0].texts}

    assert "Nearshoring to Mexico" in rendered
    assert "Dual sourcing semiconductors" in rendered


def test_exposure_charts_render_without_hop_column() -> None:
    frame = pd.DataFrame(
        {
            "sentence_id": ["0", "1", "2"],
            "company_name": ["Apple", "Apple", "NVIDIA"],
            "label": ["Nearshoring", "Nearshoring", "Dual sourcing"],
            "timestamp": [
                "2025-07-01T00:00:00",
                "2025-08-11T00:00:00",
                "2026-01-05T00:00:00",
            ],
        }
    )

    for builder in (viz.plot_top_pathways, viz.plot_top_companies, viz.plot_evidence_timeline):
        assert builder(frame).axes, f"{builder.__name__} produced no axes"


def test_company_pathway_matrix_renders_scoring_grid() -> None:
    scoring_df = pd.DataFrame(
        {
            "company_name": ["Apple", "NVIDIA"],
            "Nearshoring": [3, 0],
            "Dual sourcing": [1, 4],
            "Composite Score": [4, 4],
        }
    )

    figure = viz.plot_company_pathway_matrix(scoring_df)
    rendered = {text.get_text() for text in figure.axes[0].texts}

    assert "3" in rendered
    assert "4" in rendered


def test_charts_render_from_labeled_evidence() -> None:
    evidence_df = viz.attach_hop_column(_labeled_frame(), _tree())

    for builder in CHART_BUILDERS:
        assert builder(evidence_df).axes, f"{builder.__name__} produced no axes"


def test_indirect_hops_excludes_the_direct_hop() -> None:
    assert DERIVATIVE_BRANCH_LABELS[0] not in viz.INDIRECT_HOPS
    assert set(viz.INDIRECT_HOPS) == set(DERIVATIVE_BRANCH_LABELS[1:])


def test_hop_filter_ranks_companies_on_indirect_evidence_only() -> None:
    """A high-volume direct name must not displace a thin indirect one."""
    frame = pd.DataFrame(
        {
            "sentence_id": [str(index) for index in range(7)],
            "company_name": ["BigOil"] * 5 + ["QuietRetailer", "QuietBank"],
            "label": ["Upstream crude producers"] * 5
            + ["Discount retail trade-down", "Airline capacity discipline"],
            "timestamp": ["2026-01-05T00:00:00"] * 7,
        }
    )
    evidence_df = viz.attach_hop_column(frame, _tree())

    figure = viz.plot_top_companies(evidence_df, top_n=2, hops=viz.INDIRECT_HOPS)

    plotted = {label.get_text() for label in figure.axes[0].get_yticklabels()}
    assert plotted == {"QuietRetailer", "QuietBank"}
    assert "BigOil" not in plotted


def test_hop_filter_limits_matrix_columns_to_the_requested_hops() -> None:
    evidence_df = viz.attach_hop_column(_labeled_frame(), _tree())

    figure = viz.plot_company_hop_matrix(evidence_df, hops=viz.INDIRECT_HOPS)

    columns = [label.get_text() for label in figure.axes[0].get_xticklabels()]
    assert columns == list(viz.INDIRECT_HOPS)


def test_charts_degrade_gracefully_on_empty_evidence() -> None:
    empty_df = pd.DataFrame(columns=["sentence_id", "company_name", "label", "timestamp"])
    evidence_df = viz.attach_hop_column(empty_df, _tree())

    for builder in CHART_BUILDERS:
        assert builder(evidence_df).axes, f"{builder.__name__} produced no axes"


def test_quiet_output_suppresses_stdout_and_restores_log_levels(capsys) -> None:
    logger = logging.getLogger(viz.__name__.split(".")[0] + ".screener")
    logger.setLevel(logging.INFO)

    with quiet_output():
        print("planner chatter")
        assert logger.level == logging.ERROR

    assert logger.level == logging.INFO
    assert capsys.readouterr().out == ""
