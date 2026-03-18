from __future__ import annotations

import logging
import re
import time
from functools import wraps
from typing import Any, Callable, TypeVar

from rich.console import Console
from rich.logging import RichHandler

F = TypeVar("F", bound=Callable[..., Any])

console = Console()


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(console=console, rich_tracebacks=True, show_path=False)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def sanitize_filename(name: str) -> str:
    """Convert entity/topic names into safe filesystem names."""
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    return sanitized.strip("_").lower()


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> Callable[[F], F]:
    """Decorator that retries a function with exponential backoff."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = logging.getLogger(func.__module__)
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_retries - 1:
                        logger.error(
                            "Failed after %d attempts: %s", max_retries, exc
                        )
                        raise
                    delay = min(base_delay * (2**attempt), max_delay)
                    logger.warning(
                        "Attempt %d/%d failed (%s), retrying in %.1fs",
                        attempt + 1,
                        max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            return None  # unreachable but satisfies type checker

        return wrapper  # type: ignore[return-value]

    return decorator


def format_elapsed(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"
