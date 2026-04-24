"""Pytest fixtures: isolate agent data dir and reload ``financial_agent.agent`` cleanly."""

from __future__ import annotations

import sys
import warnings
from collections.abc import Generator
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pytest_sessionstart(session: pytest.Session) -> None:
    """Pre-load ADK so authlib runs, then prepend an ``ignore`` (authlib uses ``always`` first)."""
    import google.adk.agents  # noqa: F401
    from authlib.deprecate import AuthlibDeprecationWarning

    warnings.filterwarnings("ignore", category=AuthlibDeprecationWarning, append=False)


def _clear_financial_agent_modules() -> None:
    for name in list(sys.modules):
        if name == "financial_agent" or name.startswith("financial_agent."):
            del sys.modules[name]


@pytest.fixture
def agent_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator:
    """Fresh ``financial_agent.agent`` with temp data dir and no Gemini calls at import."""
    data_dir = tmp_path / "agent_data"
    monkeypatch.setenv("FINANCIAL_AGENT_DATA_DIR", str(data_dir))
    monkeypatch.setenv("FINANCIAL_AGENT_SKIP_EMBED_WARMUP", "1")
    monkeypatch.delenv("BIGDATA_API_KEY", raising=False)

    _clear_financial_agent_modules()
    import financial_agent.agent as agent_mod

    yield agent_mod
    _clear_financial_agent_modules()
