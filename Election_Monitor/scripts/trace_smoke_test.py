"""Unit tests for BigdataRestClient.trace_cookbook_execution (ADS-400).

No network calls: the client's ``session`` is swapped for a fake object that
records POST calls (or raises / returns a bad status) instead of hitting
the network.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src import bigdata_rest
from src.bigdata_rest import BigdataRestClient, trace_cookbook_execution


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


def make_client(session: FakeSession) -> BigdataRestClient:
    client = BigdataRestClient(api_key="test-key")
    client.session = session
    return client


class TraceCookbookExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        bigdata_rest._traced_cookbooks.clear()

    def test_posts_to_track_events_path(self) -> None:
        session = FakeSession()
        client = make_client(session)
        client.trace_cookbook_execution("SmokeTestA")
        self.assertEqual(len(session.calls), 1)
        self.assertTrue(session.calls[0]["url"].endswith("/track-events"))

    def test_body_matches_expected_shape(self) -> None:
        session = FakeSession()
        client = make_client(session)
        with patch.object(
            bigdata_rest,
            "_package_version",
            side_effect=lambda pkg: "1.2.3" if pkg == "bigdata-research-tools" else None,
        ):
            client.trace_cookbook_execution("SmokeTestB")
        body = session.calls[0]["json"]
        self.assertEqual(
            body,
            {
                "event_name": "BigdataCookbookExecution",
                "properties": {
                    "cookbook_name": "SmokeTestB",
                    "bigdataResearchToolsVersion": "1.2.3",
                },
            },
        )

    def test_version_keys_omitted_when_package_not_installed(self) -> None:
        session = FakeSession()
        client = make_client(session)
        with patch.object(bigdata_rest, "_package_version", return_value=None):
            client.trace_cookbook_execution("SmokeTestC")
        body = session.calls[0]["json"]
        self.assertEqual(body["properties"], {"cookbook_name": "SmokeTestC"})

    def test_fires_once_per_cookbook_name_per_process(self) -> None:
        session = FakeSession()
        client = make_client(session)
        client.trace_cookbook_execution("SmokeTestD")
        client.trace_cookbook_execution("SmokeTestD")
        self.assertEqual(len(session.calls), 1)

    def test_session_post_raising_does_not_propagate(self) -> None:
        session = FakeSession(raises=True)
        client = make_client(session)
        client.trace_cookbook_execution("SmokeTestE")  # must not raise

    def test_http_500_does_not_raise(self) -> None:
        session = FakeSession(status_code=500)
        client = make_client(session)
        client.trace_cookbook_execution("SmokeTestF")  # must not raise
        self.assertEqual(len(session.calls), 1)


class ModuleLevelTraceCookbookExecutionTests(unittest.TestCase):
    """Tests for the module-level trace_cookbook_execution (ADS-400 refactor)."""

    def setUp(self) -> None:
        bigdata_rest._traced_cookbooks.clear()

    def test_posts_via_provided_session(self) -> None:
        session = FakeSession()
        trace_cookbook_execution("ModuleTestA", session=session)
        self.assertEqual(len(session.calls), 1)
        self.assertTrue(session.calls[0]["url"].endswith("/track-events"))
        self.assertEqual(
            session.calls[0]["json"]["properties"]["cookbook_name"], "ModuleTestA"
        )

    def test_guard_shared_with_method_fires_pair_once(self) -> None:
        """A method call and a module-level call for the same name post once total."""
        session = FakeSession()
        client = make_client(session)
        client.trace_cookbook_execution("ModuleTestB")
        trace_cookbook_execution("ModuleTestB", session=session)
        self.assertEqual(len(session.calls), 1)

    def test_module_level_fires_once_per_name(self) -> None:
        session = FakeSession()
        trace_cookbook_execution("ModuleTestC", session=session)
        trace_cookbook_execution("ModuleTestC", session=session)
        self.assertEqual(len(session.calls), 1)

    def test_no_key_no_session_does_not_raise_and_does_not_post(self) -> None:
        with patch.object(bigdata_rest.os, "getenv", return_value=None):
            trace_cookbook_execution("ModuleTestD")  # must not raise
        # Guard still consumes the name even though nothing was sent.
        self.assertIn("ModuleTestD", bigdata_rest._traced_cookbooks)

    def test_network_error_without_session_does_not_raise(self) -> None:
        from unittest.mock import MagicMock

        fake_session = MagicMock()
        fake_session.post.side_effect = ConnectionError("network is unreachable")
        with patch.object(bigdata_rest, "requests") as fake_requests_module:
            fake_requests_module.Session.return_value = fake_session
            trace_cookbook_execution("ModuleTestE", api_key="test-key")  # must not raise
        fake_session.post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
