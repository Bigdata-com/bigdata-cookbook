from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

import src.tracing as tracing
from src.tracing import trace_cookbook_execution


@pytest.fixture(autouse=True)
def _tracing_enabled(monkeypatch):
    """conftest.py disables tracing repo-wide for the test session (other
    test modules import src.screener, which fires tracing at import time).
    These tests exercise trace_cookbook_execution itself, so they need it
    enabled by default; tests of the opt-out set the var back explicitly."""
    monkeypatch.delenv("BIGDATA_DISABLE_TRACING", raising=False)


class FakeResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code


class FakeSession:
    """Records POST calls; never touches the network."""

    def __init__(self, *, raises: bool = False, status_code: int = 200) -> None:
        self.calls: list[dict[str, Any]] = []
        self._raises = raises
        self._status_code = status_code

    def post(self, url: str, json: dict[str, Any], timeout: float) -> FakeResponse:
        if self._raises:
            raise ConnectionError("network is unreachable")
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse(self._status_code)


def setup_function() -> None:
    tracing._traced_cookbooks.clear()


def test_posts_to_track_events_path() -> None:
    session = FakeSession()
    trace_cookbook_execution("TSTestA", session=session)
    assert len(session.calls) == 1
    assert session.calls[0]["url"].endswith("/track-events")


def test_body_matches_expected_shape() -> None:
    session = FakeSession()
    with patch.object(
        tracing,
        "_package_version",
        side_effect=lambda pkg: "1.2.3" if pkg == "bigdata-research-tools" else None,
    ):
        trace_cookbook_execution("TSTestB", session=session)
    body = session.calls[0]["json"]
    assert body == {
        "event_name": "BigdataCookbookExecution",
        "properties": {
            "cookbook_name": "TSTestB",
            "bigdataResearchToolsVersion": "1.2.3",
        },
    }


def test_version_keys_omitted_when_package_not_installed() -> None:
    session = FakeSession()
    with patch.object(tracing, "_package_version", return_value=None):
        trace_cookbook_execution("TSTestC", session=session)
    assert session.calls[0]["json"]["properties"] == {"cookbook_name": "TSTestC"}


def test_fires_once_per_cookbook_name_per_process() -> None:
    session = FakeSession()
    trace_cookbook_execution("TSTestD", session=session)
    trace_cookbook_execution("TSTestD", session=session)
    assert len(session.calls) == 1


def test_session_post_raising_does_not_propagate() -> None:
    session = FakeSession(raises=True)
    trace_cookbook_execution("TSTestE", session=session)  # must not raise


def test_http_500_does_not_raise() -> None:
    session = FakeSession(status_code=500)
    trace_cookbook_execution("TSTestF", session=session)  # must not raise
    assert len(session.calls) == 1


def test_no_key_no_session_does_not_raise_and_does_not_post(monkeypatch) -> None:
    monkeypatch.delenv("BIGDATA_API_KEY", raising=False)
    trace_cookbook_execution("TSTestG")  # must not raise
    assert "TSTestG" in tracing._traced_cookbooks


def test_disable_tracing_opt_out(monkeypatch) -> None:
    session = FakeSession()
    monkeypatch.setenv("BIGDATA_DISABLE_TRACING", "1")
    trace_cookbook_execution("TSTestH", session=session)
    assert session.calls == []
    # Opt-out returns before the guard: the name is not marked as traced.
    assert "TSTestH" not in tracing._traced_cookbooks


def test_disable_tracing_case_insensitive_values(monkeypatch) -> None:
    session = FakeSession()
    for value in ("1", "true", "True", "TRUE", "yes", "Yes"):
        tracing._traced_cookbooks.clear()
        monkeypatch.setenv("BIGDATA_DISABLE_TRACING", value)
        trace_cookbook_execution("TSTestI", session=session)
        assert session.calls == [], f"value={value!r} should disable tracing"


def test_disable_tracing_falsy_value_does_not_disable(monkeypatch) -> None:
    session = FakeSession()
    monkeypatch.setenv("BIGDATA_DISABLE_TRACING", "0")
    trace_cookbook_execution("TSTestJ", session=session)
    assert len(session.calls) == 1
