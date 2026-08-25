from __future__ import annotations

import pytest

from src.notebook_support import QUOTE_TRUNCATION_MARKER, preview_quote


def test_preview_quote_keeps_short_passages_intact() -> None:
    assert preview_quote("Freight costs rose sharply.") == "Freight costs rose sharply."


def test_preview_quote_truncates_long_passages_to_the_word_budget() -> None:
    passage = " ".join(f"word{index}" for index in range(40))

    result = preview_quote(passage, words=5)

    assert result == f"word0 word1 word2 word3 word4 {QUOTE_TRUNCATION_MARKER}"


def test_preview_quote_collapses_whitespace() -> None:
    assert preview_quote("  duty  free\n threshold\t ends ") == "duty free threshold ends"


def test_preview_quote_handles_missing_text() -> None:
    assert preview_quote("") == ""
    assert preview_quote(None) == ""


def test_preview_quote_rejects_non_positive_word_budget() -> None:
    with pytest.raises(ValueError, match="words must be at least 1"):
        preview_quote("any passage", words=0)
