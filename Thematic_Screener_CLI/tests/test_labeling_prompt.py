from __future__ import annotations

from src.prompts import SYSTEM_PROMPT_LABELING
from src.screener import DEFAULT_ANALYST_FOCUS, DEFAULT_MAIN_THEME


def test_labeling_prompt_includes_analyst_focus() -> None:
    analyst_focus = "Spanish government entities in Spain only"
    prompt = SYSTEM_PROMPT_LABELING.format(
        main_theme="Commercial deals with the Spanish government",
        analyst_focus=analyst_focus,
        labels=["Concession operators"],
    )
    assert "Analyst focus (mandatory scope):" in prompt
    assert analyst_focus in prompt
    assert "mechanism matches a label but the scope does not" in prompt


def test_labeling_prompt_defaults_are_formattable() -> None:
    prompt = SYSTEM_PROMPT_LABELING.format(
        main_theme=DEFAULT_MAIN_THEME,
        analyst_focus=DEFAULT_ANALYST_FOCUS,
        labels=["Example label"],
    )
    assert DEFAULT_MAIN_THEME in prompt
    assert DEFAULT_ANALYST_FOCUS in prompt
