"""Thin async client for the Bigdata.com Remote MCP server.

Wraps the three tools used in the credit-narrative workflow
(``bigdata_screen_credit_factor``, ``bigdata_get_credit_factor``,
``bigdata_search``) plus ``find_securities`` for entity resolution, and
converts their structured responses into pandas DataFrames so the notebook
can stay focused on the analysis instead of MCP plumbing.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

BIGDATA_MCP_URL = "https://mcp.bigdata.com/"


class BigdataClient:
    """Async client for the Bigdata.com Remote MCP server.

    Use as an async context manager::

        async with BigdataClient.connect() as client:
            df = await client.screen_credit_factor(entity_universe=[...])
    """

    def __init__(self, session: ClientSession):
        self._session = session

    @classmethod
    def connect(cls, api_key: str | None = None) -> "_BigdataClientContext":
        api_key = api_key or os.environ.get("BIGDATA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set the BIGDATA_API_KEY environment variable to your Bigdata.com "
                "API key (Developer Platform > API Keys)."
            )
        return _BigdataClientContext(api_key)

    async def _call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        result = await self._session.call_tool(tool_name, arguments)
        if result.isError:
            message = result.content[0].text if result.content else "Unknown MCP error"
            raise RuntimeError(f"{tool_name} failed: {message}")
        if result.structuredContent is not None:
            return result.structuredContent
        # Some tools (e.g. find_securities) only return a text content block
        # with a JSON payload, occasionally wrapped in a single-item list.
        parsed = json.loads(result.content[0].text)
        return parsed[0] if isinstance(parsed, list) else parsed

    async def find_securities(
        self, query: str, security_types: list[str] | None = None
    ) -> pd.DataFrame:
        args: dict[str, Any] = {"query": query}
        if security_types:
            args["security_types"] = security_types
        payload = await self._call("find_securities", args)
        return pd.DataFrame(payload["results"])

    async def resolve_entity_id(self, company_name: str) -> str:
        """Resolve a company name to its RavenPack entity ID (best public match)."""
        matches = await self.find_securities(company_name, security_types=["COMPANY"])
        public = matches[matches["listing_type"] == "PUBLIC"]
        chosen = public.iloc[0] if len(public) else matches.iloc[0]
        return chosen["id"]

    async def screen_credit_factor(
        self,
        entity_universe: list[str] | None = None,
        horizon: str = "daily",
        screen_direction: str = "negative",
        company_limit: int = 20,
        scores_per_company: int = 5,
    ) -> pd.DataFrame:
        args = {
            "entity_universe": entity_universe,
            "horizon": horizon,
            "screen_direction": screen_direction,
            "company_limit": company_limit,
            "scores_per_company": scores_per_company,
        }
        payload = await self._call("bigdata_screen_credit_factor", args)
        return pd.DataFrame(payload["records"])

    async def get_credit_factor(
        self,
        rp_entity_id: str,
        horizon: str = "daily",
        negative_limit: int = 5,
        positive_limit: int = 5,
    ) -> pd.DataFrame:
        args = {
            "rp_entity_id": rp_entity_id,
            "horizon": horizon,
            "negative_limit": negative_limit,
            "positive_limit": positive_limit,
        }
        payload = await self._call("bigdata_get_credit_factor", args)
        return pd.DataFrame(payload["records"])

    async def search_news(
        self, text: str, context: str | None = None, max_chunks: int = 10
    ) -> list[dict[str, Any]]:
        """Smart-mode bigdata_search, flattened to one row per result document."""
        query: dict[str, Any] = {"text": text, "max_chunks": max_chunks}
        if context:
            query["context"] = context
        payload = await self._call(
            "bigdata_search", {"request": {"search_mode": "smart", "query": query}}
        )
        return payload.get("results", payload) if isinstance(payload, dict) else payload


class _BigdataClientContext:
    """Async context manager that owns the MCP transport + session lifecycle."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._http_client_cm = None
        self._transport_cm = None
        self._session_cm = None

    async def __aenter__(self) -> BigdataClient:
        self._http_client_cm = create_mcp_http_client(headers={"x-api-key": self._api_key})
        http_client = await self._http_client_cm.__aenter__()
        self._transport_cm = streamable_http_client(BIGDATA_MCP_URL, http_client=http_client)
        # mcp>=1.10 returns (read, write, get_session_id); older versions return (read, write).
        read, write, *_ = await self._transport_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        session = await self._session_cm.__aenter__()
        await session.initialize()
        return BigdataClient(session)

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(exc_type, exc, tb)
        if self._transport_cm is not None:
            await self._transport_cm.__aexit__(exc_type, exc, tb)
        if self._http_client_cm is not None:
            await self._http_client_cm.__aexit__(exc_type, exc, tb)
