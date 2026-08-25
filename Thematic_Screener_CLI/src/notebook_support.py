"""Helpers for running the screener pipeline inside a notebook.

The planning, retrieval, and batching libraries stream verbose progress to
stdout/stderr, which is useful on a terminal but noisy in a client-facing
report. Exceptions are unaffected and still propagate normally.
"""

from __future__ import annotations

import contextlib
import io
import logging
from collections.abc import Iterator

QUOTE_PREVIEW_WORDS = 14
QUOTE_TRUNCATION_MARKER = "…"

NOISY_LOGGERS: tuple[str, ...] = (
    "bigdata_smart_batching",
    "bigdata_client",
    "httpx",
    "matplotlib",
    "openai",
    "src.derivative_grounding",
    "src.screener",
)


@contextlib.contextmanager
def quiet_output(level: int = logging.ERROR) -> Iterator[None]:
    """Suppress progress chatter and raise noisy log levels for the block."""
    previous_levels = {name: logging.getLogger(name).level for name in NOISY_LOGGERS}
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            yield
    finally:
        for name, previous in previous_levels.items():
            logging.getLogger(name).setLevel(previous)


def preview_quote(text: object, words: int = QUOTE_PREVIEW_WORDS) -> str:
    """Shorten a source passage to a lead-in of at most ``words`` words.

    Executed notebooks are committed to a public repository, so they show only
    enough of a licensed document to make the attribution checkable. The full
    passage stays in the run directory and the exported workbook.
    """
    if words < 1:
        raise ValueError(f"words must be at least 1, got {words}")

    tokens = str(text).split() if text is not None else []
    if not tokens:
        return ""
    if len(tokens) <= words:
        return " ".join(tokens)
    return " ".join(tokens[:words]) + f" {QUOTE_TRUNCATION_MARKER}"
