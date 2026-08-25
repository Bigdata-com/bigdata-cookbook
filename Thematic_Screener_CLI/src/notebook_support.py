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
