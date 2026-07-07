from __future__ import annotations

from src.helpers import humanize_taxonomy_label


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
