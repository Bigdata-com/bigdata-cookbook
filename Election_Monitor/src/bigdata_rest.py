"""Minimal Bigdata.com REST helpers (no bigdata-client / research-tools).

Copy this module into a project's ``src/`` (or import patterns from it) when
migrating off the deprecated SDK. Prefer ``bigdata-smart-batching`` for
large-universe search — see Thematic_Screener_CLI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

DEFAULT_BASE_URL = os.getenv("BIGDATA_API_BASE_URL", "https://api.bigdata.com")
UNIVERSE_ID_ALIASES = ("RP_ENTITY_ID", "RP_COMPANY_ID")
UNIVERSE_NAME_ALIASES = ("COMPANY_NAME", "NAME", "COMPANY")


class BigdataRestClient:
    """Thin REST client authenticated with ``BIGDATA_API_KEY``."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 120,
    ) -> None:
        key = api_key or os.getenv("BIGDATA_API_KEY")
        if not key:
            raise ValueError("BIGDATA_API_KEY is not set")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"X-API-KEY": key, "Content-Type": "application/json"}
        )

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        response = self.session.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def search(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        """POST /v1/search — ``query`` is the full search request body."""
        data = self.post("/v1/search", {"query": query} if "query" not in query else query)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.get("results") or data.get("documents") or [])
        return []

    def find_companies(self, name: str, limit: int = 5) -> list[dict[str, Any]]:
        data = self.post("/v1/knowledge-graph/companies", {"query": name, "limit": limit})
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.get("results") or data.get("companies") or [])
        return []

    def get_entities_by_id(self, entity_ids: list[str]) -> dict[str, dict[str, Any]]:
        """POST /v1/knowledge-graph/entities/id — returns ``{entity_id: entity_info}``."""
        data = self.post("/v1/knowledge-graph/entities/id", {"values": entity_ids})
        if isinstance(data, dict):
            results = data.get("results")
            if isinstance(results, dict):
                return results
            if isinstance(results, list):
                return {item.get("id"): item for item in results if isinstance(item, dict)}
        return {}

    def find_sources(self, name: str, limit: int = 10) -> list[dict[str, Any]]:
        data = self.post("/v1/knowledge-graph/sources", {"query": name, "limit": limit})
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.get("results") or data.get("sources") or [])
        return []


def load_universe(universe_path: str | Path) -> pd.DataFrame:
    """Load a company universe CSV with ``RP_ENTITY_ID`` + ``COMPANY_NAME``."""
    path = Path(universe_path)
    raw = pd.read_csv(path)
    id_col = next((c for c in UNIVERSE_ID_ALIASES if c in raw.columns), None)
    name_col = next((c for c in UNIVERSE_NAME_ALIASES if c in raw.columns), None)
    if id_col is None or name_col is None:
        raise ValueError(
            f"Universe CSV must include an ID column ({UNIVERSE_ID_ALIASES}) "
            f"and a name column ({UNIVERSE_NAME_ALIASES}); got {list(raw.columns)}"
        )
    return pd.DataFrame(
        {
            "RP_ENTITY_ID": raw[id_col].astype(str).str.strip(),
            "COMPANY_NAME": raw[name_col].astype(str).str.strip(),
        }
    )


def company_ids_from_universe(universe: pd.DataFrame) -> list[str]:
    return universe["RP_ENTITY_ID"].tolist()
