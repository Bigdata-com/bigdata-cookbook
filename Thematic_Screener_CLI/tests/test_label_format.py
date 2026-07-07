from __future__ import annotations

from src.helpers import humanize_taxonomy_label
from src.screener import (
    analyst_focus_with_leaf_cap,
    normalize_max_leaf_labels,
)


def test_analyst_focus_with_leaf_cap_appends_instruction() -> None:
    focus = analyst_focus_with_leaf_cap("Mag7 AI infra exposure", 15)
    assert focus.startswith("Mag7 AI infra exposure")
    assert focus.endswith("Limit the final tree to at most 15 leaf nodes.")


def test_analyst_focus_with_leaf_cap_skips_when_unlimited() -> None:
    assert analyst_focus_with_leaf_cap("Mag7 AI infra exposure", None) == "Mag7 AI infra exposure"
    assert analyst_focus_with_leaf_cap("Mag7 AI infra exposure", 0) == "Mag7 AI infra exposure"


def test_normalize_max_leaf_labels() -> None:
    assert normalize_max_leaf_labels(15) == 15
    assert normalize_max_leaf_labels(0) is None
    assert normalize_max_leaf_labels(None) is None



def test_humanize_taxonomy_label_splits_pascal_case() -> None:
    assert (
        humanize_taxonomy_label("SustainedBrentSpikeAboveKeyThreshold")
        == "Sustained Brent Spike Above Key Threshold"
    )
    assert (
        humanize_taxonomy_label("LNGSpotPriceDislocationInAsiaEurope")
        == "LNG Spot Price Dislocation In Asia Europe"
    )


def test_humanize_taxonomy_label_preserves_existing_spaces() -> None:
    assert humanize_taxonomy_label("Delayed Government Contract Payments") == (
        "Delayed Government Contract Payments"
    )


def test_humanize_taxonomy_label_strips_misformatted_label_suffix() -> None:
    assert humanize_taxonomy_label(
        "RedSeaOrSuezRouteInterruptionAffectsLNGAndOilFlows: Alternative routing delays"
    ) == "Red Sea Or Suez Route Interruption Affects LNG And Oil Flows"
