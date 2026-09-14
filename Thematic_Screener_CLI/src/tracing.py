"""Standalone cookbook execution tracing for Thematic_Screener (ADS-400).

Thematic_Screener_CLI has no BigdataRestClient / bigdata_rest.py — search
auth and requests are handled internally by ``bigdata_smart_batching``. This
is a self-contained copy of the ``trace_cookbook_execution`` helper that
lives in every other cookbook's ``bigdata_rest.py``, kept in sync by hand:
same payload shape, same guard, same opt-out.
"""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_BASE_URL = os.getenv("BIGDATA_API_BASE_URL", "https://api.bigdata.com")

# --- cookbook execution tracing (ADS-400) ---------------------------------
# Payload kept identical to the SDK's BigdataCookbookExecution event so that
# Mixpanel history stays continuous. Do not rename the event or the
# cookbook_name values without a cutover agreed with the dashboard owners.
TRACK_EVENTS_PATH = "/track-events"  # not /v1/track-events, which 404s
TRACE_TIMEOUT = 5  # telemetry must never hold up a notebook
_traced_cookbooks: set[str] = set()


def _package_version(package: str) -> str | None:
    """Installed version of ``package``, or None when it is not installed."""
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return None


def trace_cookbook_execution(
    cookbook_name: str,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    session: requests.Session | None = None,
) -> None:
    """Report one cookbook execution to Mixpanel. Never raises.

    Fires at most once per cookbook name per process, mirroring the
    ``_initialization_sent`` guard of the retired SDK helper. Reads
    ``BIGDATA_API_KEY`` from the environment when ``api_key`` is None and no
    ``session`` is given; returns silently if no key is available anywhere.
    Pass ``session`` (an already-authenticated ``requests.Session``) to reuse
    an existing session instead of opening a new one. Set
    ``BIGDATA_DISABLE_TRACING=1`` (or "true"/"yes", case-insensitive) in the
    environment to opt out entirely — checked before the once-per-name guard
    and before any network call, so test entry points can safely import any
    module without emitting telemetry.
    """
    if os.getenv("BIGDATA_DISABLE_TRACING", "").strip().lower() in ("1", "true", "yes"):
        return
    if cookbook_name in _traced_cookbooks:
        return
    _traced_cookbooks.add(cookbook_name)

    if session is None:
        key = api_key or os.getenv("BIGDATA_API_KEY")
        if not key:
            return
        session = requests.Session()
        session.headers.update({"X-API-KEY": key, "Content-Type": "application/json"})

    properties: dict[str, Any] = {"cookbook_name": cookbook_name}
    for prop_key, package in (
        ("bigdataResearchToolsVersion", "bigdata-research-tools"),
        ("bigdataClientVersion", "bigdata-client"),
    ):
        found = _package_version(package)
        if found is not None:
            properties[prop_key] = found

    try:
        session.post(
            f"{base_url.rstrip('/')}{TRACK_EVENTS_PATH}",
            json={
                "event_name": "BigdataCookbookExecution",
                "properties": properties,
            },
            timeout=TRACE_TIMEOUT,
        )
    except Exception:
        pass  # telemetry must never break the notebook
