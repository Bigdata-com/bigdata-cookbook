"""Thin async client for the Bigdata.com remote MCP server.

Deliberately avoids the Bigdata Python SDKs (`bigdata-client` /
`bigdata-research-tools`). It speaks the Model Context Protocol directly against
`https://mcp.bigdata.com/` over streamable HTTP, authenticated with the `x-api-key`
header, using only the standard `mcp` client library.

Exposes just what the FX Forecast Report needs:
    - country_tearsheet(country)  -> markdown string, or None if unsupported
    - search(text, ...)           -> normalized list of evidence documents
    - search_many(specs)          -> parallel searches over one connection
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from config.drivers import is_tearsheet_supported

DEFAULT_MCP_URL = "https://mcp.bigdata.com/"
COUNTRY_TEARSHEET_TOOL = "bigdata_country_tearsheet"
SEARCH_TOOL = "bigdata_search"


@dataclass
class SearchSpec:
    """A single parameterized search request."""

    text: str
    context: str | None = "search news and research"
    mode: str = "smart"
    max_chunks: int = 12
    # Free-form label so callers can group results back to a driver/topic.
    label: str = ""


@dataclass
class EvidenceDoc:
    """One normalized document returned by a search (chunks joined)."""

    doc_id: str
    headline: str
    source: str
    url: str
    timestamp: str
    text: str
    label: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "doc_id": self.doc_id,
            "headline": self.headline,
            "source": self.source,
            "url": self.url,
            "timestamp": self.timestamp,
            "text": self.text,
            "label": self.label,
        }


@dataclass
class BigdataMCPClient:
    """Minimal async wrapper over the Bigdata.com remote MCP server."""

    api_key: str
    url: str = DEFAULT_MCP_URL
    timeout_seconds: float = 120.0
    _errors: list[str] = field(default_factory=list)

    # -- low-level -----------------------------------------------------------
    @asynccontextmanager
    async def _session(self) -> AsyncIterator[ClientSession]:
        """Open an initialized MCP session (one streamable-HTTP connection)."""
        headers = {"x-api-key": self.api_key}
        async with streamablehttp_client(self.url, headers=headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    async def _call_tool(
        self, session: ClientSession, name: str, arguments: dict[str, Any]
    ) -> str:
        """Call a tool on an open session and return its text payload."""
        result = await session.call_tool(name, arguments)
        text = _extract_text(result)
        if getattr(result, "isError", False):
            self._errors.append(f"{name}: {text[:300]}")
        return text

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Open a fresh connection, call one tool, and return its text payload."""
        async with self._session() as session:
            return await self._call_tool(session, name, arguments)

    # -- country tearsheet ---------------------------------------------------
    async def country_tearsheet(self, country: str) -> str | None:
        """Return the markdown tearsheet for a country, or None if unsupported.

        The Bigdata.com country tearsheet only covers a fixed set of countries; for
        anything outside that set we return None so callers can fall back to search.
        """
        if not is_tearsheet_supported(country):
            return None
        text = await self.call_tool(COUNTRY_TEARSHEET_TOOL, {"country": country.upper()})
        return text or None

    # -- search --------------------------------------------------------------
    @staticmethod
    def _build_request(spec: SearchSpec) -> dict[str, Any]:
        query: dict[str, Any] = {"text": spec.text, "max_chunks": spec.max_chunks}
        if spec.mode == "smart" and spec.context:
            query["context"] = spec.context
        return {"search_mode": spec.mode, "query": query}

    async def _run_search(self, session: ClientSession, spec: SearchSpec) -> list[EvidenceDoc]:
        raw = await self._call_tool(
            session, SEARCH_TOOL, {"request": self._build_request(spec)}
        )
        return _normalize_search(raw, label=spec.label)

    async def search(
        self,
        text: str,
        context: str | None = "search news and research",
        mode: str = "smart",
        max_chunks: int = 12,
        label: str = "",
    ) -> list[EvidenceDoc]:
        """Run a single search and return normalized evidence documents."""
        spec = SearchSpec(
            text=text, context=context, mode=mode, max_chunks=max_chunks, label=label
        )
        async with self._session() as session:
            return await self._run_search(session, spec)

    async def search_many(
        self, specs: list[SearchSpec], max_concurrency: int = 6
    ) -> list[list[EvidenceDoc]]:
        """Run several searches concurrently over a single connection.

        Results are returned in the same order as `specs`.
        """
        if not specs:
            return []
        semaphore = asyncio.Semaphore(max_concurrency)
        async with self._session() as session:

            async def _one(spec: SearchSpec) -> list[EvidenceDoc]:
                async with semaphore:
                    try:
                        return await self._run_search(session, spec)
                    except Exception as exc:  # noqa: BLE001 - keep one failure isolated
                        self._errors.append(f"search '{spec.label or spec.text[:40]}': {exc}")
                        return []

            return await asyncio.gather(*(_one(spec) for spec in specs))

    @property
    def errors(self) -> list[str]:
        """Non-fatal errors accumulated during calls (for diagnostics)."""
        return list(self._errors)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_text(result: Any) -> str:
    """Join the text payload of an MCP CallToolResult."""
    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        piece = getattr(item, "text", None)
        if piece:
            parts.append(piece)
    if parts:
        return "\n".join(parts)
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return json.dumps(structured)
    return ""


def _normalize_search(raw: str, label: str = "") -> list[EvidenceDoc]:
    """Parse a bigdata_search text payload into EvidenceDoc records (one per document)."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, list):
        # Some tools wrap the payload in a single-element list.
        parsed = parsed[0] if parsed else {}
    if not isinstance(parsed, dict):
        return []

    docs: list[EvidenceDoc] = []
    for item in parsed.get("results", []) or []:
        chunks = item.get("chunks", []) or []
        text = "\n".join(c.get("text", "") for c in chunks if c.get("text")).strip()
        if not text:
            continue
        source = item.get("source") or {}
        source_name = source.get("name", "") if isinstance(source, dict) else str(source)
        docs.append(
            EvidenceDoc(
                doc_id=str(item.get("id", "")),
                headline=item.get("headline", "") or "",
                source=source_name,
                url=item.get("url", "") or "",
                timestamp=item.get("timestamp", "") or "",
                text=text,
                label=label,
            )
        )
    return docs
