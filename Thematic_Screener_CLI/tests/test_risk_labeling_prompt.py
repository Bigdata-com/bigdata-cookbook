from __future__ import annotations

from src.modes import get_profile
from src.screener import Node, _labeling_prompt_labels, _labeling_system_prompt


def test_risk_labeling_prompt_uses_label_summary_pairs() -> None:
    root = Node.model_validate(
        {
            "node": 0,
            "label": "Government Shutdown",
            "summary": "Federal funding lapse risk",
            "search_query": "",
            "children": [
                {
                    "node": 1,
                    "label": "Contract payment delays",
                    "summary": (
                        "The company experiences delayed payment from federal customers "
                        "for delivered work."
                    ),
                    "search_query": "federal payment delays",
                    "children": [],
                }
            ],
        }
    )
    profile = get_profile("risk-analyzer")
    labels = ["Contract payment delays"]
    prompt_labels = _labeling_prompt_labels(profile, labels, root)
    assert prompt_labels == [
        "Contract payment delays: The company experiences delayed payment from federal "
        "customers for delivered work."
    ]
    prompt = _labeling_system_prompt(
        profile,
        main_theme="US Government Shutdown",
        labels=labels,
        analyst_focus="Mag7 exposure",
        prompt_labels=prompt_labels,
    )
    assert "Label: Summary" in prompt
    assert "Contract payment delays: The company experiences delayed payment" in prompt
