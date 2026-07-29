"""
Research Agent API - Synchronous Client
========================================

A robust Python client for the Bigdata.com Research Agent API that consumes the
Server-Sent Events (SSE) stream and returns a single, complete result object with
correctly attributed citations.

The implementation follows the protocol described in the Bigdata.com concept guides:

- https://docs.bigdata.com/how-to-guides/agents/concepts/streaming-responses
- https://docs.bigdata.com/how-to-guides/agents/concepts/grounding-and-citations
- https://docs.bigdata.com/how-to-guides/agents/concepts/error-handling
- https://docs.bigdata.com/how-to-guides/agents/concepts/code-execution-and-charts
- https://docs.bigdata.com/how-to-guides/agents/concepts/conversation-continuity

Features
--------
- **Synchronous interface**: one blocking call returns the finished answer.
- **Complete message coverage**: every public SSE message type is handled, and
  unknown types are ignored so new API versions do not break the client.
- **Correct citations**: document citations and tool-level citations are handled
  separately, so a reference produced by a non-search tool never renders as an
  empty "N/A" entry.
- **Typed errors**: HTTP status codes and in-stream ``ERROR`` events map to
  specific exception classes.
- **Retries**: exponential backoff with jitter for ``429``, ``5xx``, and transient
  network failures.
- **Charts**: ``CHART`` events are collected as Vega-Lite specs anchored to answer
  offsets.

Quick start
-----------
    from research_client import ResearchClient, setup_logging

    setup_logging(log_file="research.log", console=True)

    client = ResearchClient()
    result = client.research("How is the S&P 500 performing?")

    print(result.get_markdown_with_citations())

Citations
---------
``GROUNDING`` references carry ``start``/``end`` character offsets into the
*cumulative* answer text, so they can only be applied once the stream has been
fully consumed. This client buffers the answer verbatim and resolves offsets at
render time.

A reference's ``source`` is populated only for the search tool. Every other tool
(market tearsheet, earnings calendar, Python code execution, ...) omits it and
grounds the span at the whole-tool level via ``audit_id``. Those become
:class:`ToolCitation` entries labelled with the tool's audit title rather than
blank document cards.

Classes
-------
- ResearchClient: executes research queries
- ResearchResult: answer, citations, charts, and run metadata
- Citation: a cited document (``BIGDATA``) or web result (``EXTERNAL``)
- ToolCitation: a whole-tool attribution for a non-search tool
- Source / Chunk: nested citation detail
- GroundingReference: an answer span attributed to a source
- Chart: a Vega-Lite chart anchored to an answer span
- AuditTrace: a tool execution trace

Exceptions
----------
All errors derive from :class:`ResearchAgentError`.

For more information, see https://docs.bigdata.com/how-to-guides/agents
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import requests
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError,
    ReadTimeout,
    Timeout,
)

__all__ = [
    "ResearchClient",
    "ResearchResult",
    "Citation",
    "ToolCitation",
    "Chunk",
    "Source",
    "GroundingReference",
    "Chart",
    "AuditTrace",
    "ResearchAgentError",
    "AuthenticationError",
    "EntitlementError",
    "InvalidRequestError",
    "ResourceNotFoundError",
    "RateLimitError",
    "ServerError",
    "StreamError",
    "StreamTimeoutError",
    "TruncatedStreamError",
    "format_source_date",
    "setup_logging",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ResearchAgentError(Exception):
    """Base class for every Research Agent failure."""


class AuthenticationError(ResearchAgentError):
    """HTTP 401 - the API key is missing or invalid."""


class EntitlementError(ResearchAgentError):
    """HTTP 403 - the key is valid but not entitled to this resource."""


class InvalidRequestError(ResearchAgentError):
    """HTTP 400 / 422 - the request payload is malformed or failed validation."""


class ResourceNotFoundError(ResearchAgentError):
    """HTTP 404 - an unknown ``chat_id``, ``from_checkpoint_id``, or template."""


class RateLimitError(ResearchAgentError):
    """HTTP 429 - rate limit exceeded. Retryable with backoff."""


class ServerError(ResearchAgentError):
    """HTTP 5xx - transient server-side failure. Retryable with backoff."""


class StreamError(ResearchAgentError):
    """An in-stream ``ERROR`` event. The stream terminates; do not retry."""


class StreamTimeoutError(ResearchAgentError):
    """No data was received within ``stream_timeout``. Retryable."""


class TruncatedStreamError(ResearchAgentError):
    """The stream ended without a ``COMPLETE`` event, so the answer is incomplete."""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def setup_logging(
    log_file: str = "research_client.log",
    level: int = logging.INFO,
    console: bool = True,
    file_mode: str = "a",
) -> logging.Logger:
    """
    Configure logging for the ``research_client`` module.

    Call this once at application startup, before creating a
    :class:`ResearchClient`, to see retry attempts, tool errors, and progress.

    Args:
        log_file: Path to the log file. Pass an empty string to skip file logging.
        level: Logging level.
        console: Also log to stdout.
        file_mode: ``"a"`` to append, ``"w"`` to overwrite.

    Returns:
        The configured logger.
    """
    logger.handlers.clear()
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_file:
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Avoid duplicate records if the host application also configures the root logger.
    logger.propagate = False

    logger.info(
        "Logging configured: file=%s, console=%s, level=%s",
        log_file,
        console,
        logging.getLevelName(level),
    )
    _flush_handlers()
    return logger


def _flush_handlers() -> None:
    """Flush handlers so log lines survive a hard failure mid-stream."""
    for handler in logger.handlers:
        handler.flush()


# ---------------------------------------------------------------------------
# Source field helpers
#
# A GroundingReference's ``source`` is one of two shapes discriminated on
# ``type``. BIGDATA documents carry ``src_name`` / ``ts`` / ``url`` at the top
# level; EXTERNAL results nest the display name and URL under ``action``.
# Reading only the BIGDATA field names is why external results used to render
# without a source name.
# ---------------------------------------------------------------------------

_MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def _action_of(source: dict[str, Any]) -> dict[str, Any]:
    action = source.get("action")
    return action if isinstance(action, dict) else {}


def _source_id(source: dict[str, Any]) -> str | None:
    return source.get("id")


def _source_headline(source: dict[str, Any]) -> str | None:
    return source.get("hd") or _action_of(source).get("hd")


def _source_name(source: dict[str, Any]) -> str | None:
    return source.get("src_name") or _action_of(source).get("name")


def _source_url(source: dict[str, Any]) -> str | None:
    return source.get("url") or _action_of(source).get("url")


def _source_timestamp(source: dict[str, Any]) -> str | None:
    return source.get("ts")


def _dedup_key(source: dict[str, Any]) -> str | None:
    """
    Build a deduplication key for a source.

    Preference order per the grounding guide: ``id`` (most reliable), then
    ``url``, then ``hd``. Never ``audit_id`` -- one tool call can return many
    documents.
    """
    return _source_id(source) or _source_url(source) or _source_headline(source)


# A marker slot the API reserved but never grounded: a space wedged between a
# word and its closing punctuation. Matched only at a clause boundary so
# Markdown tables and ordinary prose are left alone.
_UNFILLED_SLOT = re.compile(r"(?<=\S) (?=[.,;:](?:\s|$))")


def _close_unfilled_slots(text: str) -> str:
    """Remove citation slots the API reserved but did not fill."""
    return _UNFILLED_SLOT.sub("", text)


def format_source_date(timestamp: str | None) -> str:
    """
    Format an ISO timestamp as the brand-standard ``MMM DD, YYYY``.

    Returns an empty string when the timestamp is missing or unparseable.
    """
    if not timestamp or len(timestamp) < 10:
        return ""
    try:
        year, month, day = int(timestamp[0:4]), int(timestamp[5:7]), int(timestamp[8:10])
    except ValueError:
        return ""
    if not 1 <= month <= 12:
        return ""
    return f"{_MONTHS[month - 1]} {day:02d}, {year}"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Source:
    """Publisher information for a citation."""

    id: str | None = None
    name: str | None = None
    rank: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in (("id", self.id), ("name", self.name), ("rank", self.rank))
            if value is not None
        }


@dataclass
class Chunk:
    """A matched text excerpt from a cited document."""

    cnum: int | None = None
    text: str | None = None
    relevance: float | None = None
    sentiment: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in (
                ("cnum", self.cnum),
                ("text", self.text),
                ("relevance", self.relevance),
                ("sentiment", self.sentiment),
            )
            if value is not None
        }


@dataclass
class Citation:
    """
    A cited document, in Bigdata.com citation format.

    Built from either a ``BigdataDocument`` (``type == "BIGDATA"``) or an
    ``ExternalResult`` (``type == "EXTERNAL"``). Both shapes are normalised to
    the same fields.
    """

    id: str | None = None
    headline: str | None = None
    timestamp: str | None = None
    source: Source | None = None
    url: str | None = None
    source_type: str | None = None
    chunks: list[Chunk] = field(default_factory=list)

    @classmethod
    def from_source(cls, data: dict[str, Any]) -> Citation:
        """Create a Citation from a grounding ``source`` or an audit result value."""
        rank = data.get("source_rank")
        source = Source(
            id=data.get("src_key"),
            name=_source_name(data),
            rank=f"RANK_{rank}" if rank is not None else None,
        )

        chunks: list[Chunk] = []
        # EXTERNAL results carry a single flat excerpt.
        if data.get("text"):
            chunks.append(
                Chunk(
                    text=data.get("text"),
                    relevance=data.get("relevance"),
                    sentiment=data.get("sentiment"),
                )
            )
        # BIGDATA documents carry an array of matched chunks.
        for chunk in data.get("chunks") or []:
            chunks.append(
                Chunk(
                    cnum=chunk.get("cnum"),
                    text=chunk.get("text"),
                    relevance=chunk.get("relevance"),
                    sentiment=chunk.get("sentiment"),
                )
            )

        return cls(
            id=_source_id(data),
            headline=_source_headline(data),
            timestamp=_source_timestamp(data),
            source=source if source.to_dict() else None,
            url=_source_url(data),
            source_type=data.get("type"),
            chunks=chunks,
        )

    # Retained under the previous name so existing integrations keep working.
    from_audit_result = from_source

    @property
    def source_name(self) -> str | None:
        return self.source.name if self.source else None

    def label(self) -> str:
        """Brand-standard attribution label: ``Source name - MMM DD, YYYY``."""
        name = self.source_name or self.headline or "Unknown source"
        date = format_source_date(self.timestamp)
        return f"{name} - {date}" if date else name

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.id is not None:
            result["id"] = self.id
        if self.headline is not None:
            result["headline"] = self.headline
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp
        if self.source is not None:
            source_dict = self.source.to_dict()
            if source_dict:
                result["source"] = source_dict
        if self.url is not None:
            result["url"] = self.url
        if self.source_type is not None:
            result["source_type"] = self.source_type
        if self.chunks:
            result["chunks"] = [c.to_dict() for c in self.chunks if c.to_dict()]
        return result


@dataclass
class ToolCitation:
    """
    A whole-tool attribution.

    Emitted when a ``GROUNDING`` reference has no ``source``, which is normal and
    expected for every non-search tool. The claim is grounded in that tool's
    result as a whole rather than in one document, so it is rendered from the
    tool name and its matching ``AUDIT`` trace instead of document metadata.
    """

    tool_name: str
    audit_id: str | None = None
    title: str | None = None

    def label(self) -> str:
        return self.title or self.tool_name.replace("_", " ").title()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"tool_name": self.tool_name, "type": "TOOL"}
        if self.audit_id:
            result["audit_id"] = self.audit_id
        if self.title:
            result["title"] = self.title
        return result


@dataclass
class GroundingReference:
    """
    An answer span attributed to a source.

    ``start`` and ``end`` are character offsets into the *cumulative* answer text,
    so they are only valid once every ``ANSWER`` chunk has been concatenated.
    """

    start: int
    end: int
    tool_name: str = ""
    audit_id: str | None = None
    source: dict[str, Any] | None = None

    @classmethod
    def from_grounding(cls, ref: dict[str, Any]) -> GroundingReference:
        source = ref.get("source")
        return cls(
            start=ref.get("start", 0),
            end=ref.get("end", 0),
            tool_name=ref.get("tool_name", ""),
            audit_id=ref.get("audit_id") or ref.get("action_audits_id"),
            source=source if isinstance(source, dict) else None,
        )

    @property
    def is_tool_level(self) -> bool:
        """True when the reference grounds a whole tool result, not a document."""
        return self.source is None

    @property
    def citation_id(self) -> str | None:
        return _source_id(self.source) if self.source else None

    @property
    def headline(self) -> str | None:
        return _source_headline(self.source) if self.source else None

    @property
    def source_name(self) -> str | None:
        return _source_name(self.source) if self.source else None

    @property
    def timestamp(self) -> str | None:
        return _source_timestamp(self.source) if self.source else None

    @property
    def url(self) -> str | None:
        return _source_url(self.source) if self.source else None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"start": self.start, "end": self.end}
        if self.tool_name:
            result["tool_name"] = self.tool_name
        if self.audit_id:
            result["audit_id"] = self.audit_id
        for key, value in (
            ("citation_id", self.citation_id),
            ("headline", self.headline),
            ("source_name", self.source_name),
            ("timestamp", self.timestamp),
            ("url", self.url),
        ):
            if value:
                result[key] = value
        return result


@dataclass
class Chart:
    """
    A chart produced by the Python code execution tool.

    ``vega_lite_spec`` is a standard Vega-Lite specification. ``start`` and
    ``end`` point at the answer span the chart illustrates, using the same
    offset model as grounding references.
    """

    chart_id: str
    title: str
    chart_type: str
    vega_lite_spec: dict[str, Any]
    caption: str | None = None
    data_points: int | None = None
    start: int = 0
    end: int = 0

    @classmethod
    def from_message(cls, msg: dict[str, Any]) -> Chart:
        return cls(
            chart_id=msg.get("chart_id", ""),
            title=msg.get("title", ""),
            chart_type=msg.get("chart_type", ""),
            vega_lite_spec=msg.get("vega_lite_spec", {}),
            caption=msg.get("caption"),
            data_points=msg.get("data_points"),
            start=msg.get("start", 0),
            end=msg.get("end", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chart_id": self.chart_id,
            "title": self.title,
            "caption": self.caption,
            "chart_type": self.chart_type,
            "data_points": self.data_points,
            "start": self.start,
            "end": self.end,
            "vega_lite_spec": self.vega_lite_spec,
        }


@dataclass
class AuditTrace:
    """A tool execution trace, linked to grounding references by ``tool_id``."""

    audit_type: str
    tool_id: str
    title: str | None = None
    query: str | None = None
    content: str | None = None

    @classmethod
    def from_trace(cls, trace: dict[str, Any]) -> AuditTrace:
        query = trace.get("query")
        return cls(
            audit_type=trace.get("audit_type", ""),
            tool_id=trace.get("tool_id", ""),
            title=trace.get("title"),
            query=query.get("text") if isinstance(query, dict) else None,
            content=trace.get("content"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in (
                ("audit_type", self.audit_type),
                ("tool_id", self.tool_id),
                ("title", self.title),
                ("query", self.query),
            )
            if value
        }


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ResearchResult:
    """
    A completed research run.

    Attributes:
        answer: The answer in Markdown, concatenated verbatim from ``ANSWER``
            chunks so grounding offsets stay valid.
        citations: Every document returned by the search tool.
        grounding_refs: Answer spans attributed to a source or a tool.
        charts: Vega-Lite charts anchored to answer spans.
        structured_output: Extracted JSON, when a schema was requested.
        chat_id: Conversation ID for follow-up questions.
        checkpoint_id: Checkpoint for resuming or branching this conversation.
        consumption: Per-tier resource usage from the ``COMPLETE`` event.
        audit_traces: Tool execution traces keyed by ``tool_id``.
        tool_errors: Count of ``TOOL_ERROR`` events per tool name.
        plan_steps: The final research plan, as ``(description, status)`` pairs.
        processing_time_ms: Wall-clock time for the run.
    """

    answer: str
    citations: list[Citation] = field(default_factory=list)
    grounding_refs: list[GroundingReference] = field(default_factory=list)
    charts: list[Chart] = field(default_factory=list)
    structured_output: Any | None = None
    chat_id: str | None = None
    checkpoint_id: str | None = None
    consumption: list[dict[str, Any]] = field(default_factory=list)
    audit_traces: dict[str, AuditTrace] = field(default_factory=dict)
    tool_errors: dict[str, int] = field(default_factory=dict)
    plan_steps: list[tuple[str, str]] = field(default_factory=list)
    processing_time_ms: int | None = None

    # -- basic accessors ----------------------------------------------------

    def get_answer(self) -> str:
        """The answer text, without inline citation markers."""
        return self.answer

    def get_citations(self) -> list[dict[str, Any]]:
        """Every collected document citation, as dictionaries."""
        return [c.to_dict() for c in self.citations]

    def get_citations_json(self, indent: int = 2) -> str:
        return json.dumps(self.get_citations(), indent=indent)

    # -- citation numbering -------------------------------------------------

    def _resolve_tool_citation(self, ref: GroundingReference) -> ToolCitation:
        """
        Build a tool-level citation, enriched from its audit trace.

        ``audit_id`` is documented to match a trace's ``tool_id``; in practice the
        API also returns opaque action ids, so fall back to matching on the tool
        name, which the trace uses as its ``tool_id``.
        """
        trace = None
        if ref.audit_id:
            trace = self.audit_traces.get(ref.audit_id)
        if trace is None:
            trace = self.audit_traces.get(ref.tool_name)
        return ToolCitation(
            tool_name=ref.tool_name or "unknown_tool",
            audit_id=ref.audit_id,
            title=trace.title if trace else None,
        )

    def _build_numbering(
        self, include_tool_citations: bool = True
    ) -> tuple[list[dict[str, Any]], dict[int, set[int]]]:
        """
        Assign a citation number to every unique source and map answer offsets to
        the numbers that belong there.

        Numbers are assigned in the order the markers appear in the answer, so a
        reader encounters ``[1]`` before ``[2]``.

        Returns:
            ``(entries, markers)`` where ``entries`` is the numbered reference
            list and ``markers`` maps an answer offset to the numbers to insert
            at it.
        """
        entries: list[dict[str, Any]] = []
        markers: dict[int, set[int]] = {}
        numbering: dict[str, int] = {}

        # Index documents from AUDIT so a grounding source can be enriched with
        # the full chunk list the audit trace carried.
        by_id = {c.id: c for c in self.citations if c.id}

        ordered_refs = sorted(self.grounding_refs, key=lambda r: (r.end, r.start))

        for ref in ordered_refs:
            source = ref.source
            if source is None:
                if not include_tool_citations:
                    continue
                tool_citation = self._resolve_tool_citation(ref)
                key = f"tool::{tool_citation.tool_name}::{tool_citation.audit_id or ''}"
                entry_data = tool_citation.to_dict()
            else:
                source_key = _dedup_key(source)
                if not source_key:
                    continue
                key = f"doc::{source_key}"
                citation = by_id.get(ref.citation_id) or Citation.from_source(source)
                entry_data = citation.to_dict()

            number = numbering.get(key)
            if number is None:
                number = len(numbering) + 1
                numbering[key] = number
                entries.append({"number": number, **entry_data})

            markers.setdefault(ref.end, set()).add(number)

        return entries, markers

    def get_numbered_citations(self, include_tool_citations: bool = True) -> list[dict[str, Any]]:
        """
        Citations numbered to match the inline markers in
        :meth:`get_answer_with_citations`.

        Args:
            include_tool_citations: Include whole-tool attributions (references
                with no document source). Set to ``False`` for a document-only
                reference list.

        Falls back to the full AUDIT citation list when the run produced no
        grounding references.
        """
        if not self.grounding_refs:
            return [{"number": i + 1, **c.to_dict()} for i, c in enumerate(self.citations)]

        entries, _ = self._build_numbering(include_tool_citations)
        return entries

    def get_answer_with_citations(
        self,
        include_tool_citations: bool = True,
        marker_format: str = "[{numbers}]",
        tidy_unfilled_slots: bool = True,
    ) -> str:
        """
        The answer with inline citation markers such as ``[1]`` or ``[2, 3]``.

        Markers are inserted at each reference's ``end`` offset, working right to
        left so earlier offsets stay valid. The API reserves a space before the
        closing punctuation of a citable sentence, so markers need no padding.

        Args:
            include_tool_citations: Also mark spans grounded at the tool level.
            marker_format: Template for the marker; ``{numbers}`` is replaced with
                the comma-separated citation numbers.
            tidy_unfilled_slots: Close up reserved slots that received no marker.
                The API reserves them for every citable sentence but only grounds
                some, which would otherwise leave stray spaces such as ``capex .``
        """
        if not self.grounding_refs:
            return _close_unfilled_slots(self.answer) if tidy_unfilled_slots else self.answer

        _, markers = self._build_numbering(include_tool_citations)

        # Work right to left so offsets that have not been visited stay valid.
        annotated = self.answer
        for offset in sorted(markers, reverse=True):
            if not 0 <= offset <= len(annotated):
                continue
            numbers = ", ".join(str(n) for n in sorted(markers[offset]))
            annotated = (
                annotated[:offset] + marker_format.format(numbers=numbers) + annotated[offset:]
            )

        return _close_unfilled_slots(annotated) if tidy_unfilled_slots else annotated

    def get_markdown_with_citations(
        self,
        include_tool_citations: bool = True,
        sources_heading: str = "## Sources",
    ) -> str:
        """
        A self-contained Markdown document: the annotated answer followed by a
        deduplicated "Sources" section.

        Each entry uses the brand-standard ``Source name - MMM DD, YYYY`` format,
        linked to the source URL when one is available.
        """
        annotated = self.get_answer_with_citations(include_tool_citations)
        entries = self.get_numbered_citations(include_tool_citations)
        if not entries:
            return annotated

        lines = [annotated, "", sources_heading, ""]
        for entry in entries:
            lines.append(f"{entry['number']}. {self.format_reference(entry)}")
        return "\n".join(lines)

    @staticmethod
    def format_reference(entry: dict[str, Any]) -> str:
        """Render one numbered citation entry as a Markdown attribution string."""
        if entry.get("type") == "TOOL":
            title = entry.get("title") or entry["tool_name"].replace("_", " ").title()
            return f"{title} - Bigdata.com `{entry['tool_name']}`"

        source = entry.get("source") or {}
        name = source.get("name") or entry.get("headline") or "Unknown source"
        date = format_source_date(entry.get("timestamp"))
        label = f"{name} - {date}" if date else name
        headline = entry.get("headline")
        url = entry.get("url")

        text = f"[{label}]({url})" if url else label
        return f"{text} - {headline}" if headline else text

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The full result as a dictionary."""
        result: dict[str, Any] = {"answer": self.answer}
        if self.citations:
            result["citations"] = self.get_citations()
        if self.charts:
            result["charts"] = [c.to_dict() for c in self.charts]
        if self.structured_output is not None:
            result["structured_output"] = self.structured_output
        if self.chat_id:
            result["chat_id"] = self.chat_id
        if self.checkpoint_id:
            result["checkpoint_id"] = self.checkpoint_id
        if self.consumption:
            result["consumption"] = self.consumption
        if self.tool_errors:
            result["tool_errors"] = self.tool_errors
        if self.processing_time_ms:
            result["processing_time_ms"] = self.processing_time_ms
        return result

    def to_dict_with_inline_citations(self, include_tool_citations: bool = True) -> dict[str, Any]:
        """The full result with inline markers in the answer and numbered citations."""
        result = self.to_dict()
        result["answer"] = self.get_answer_with_citations(include_tool_citations)
        result["citations"] = self.get_numbered_citations(include_tool_citations)
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_json_with_inline_citations(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict_with_inline_citations(), indent=indent)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class ResearchClient:
    """
    Synchronous client for the Bigdata.com Research Agent API.

    Thread-safe: instance attributes are read-only during a request, so a single
    client may be shared across threads.

    Example:
        Basic usage::

            client = ResearchClient()
            result = client.research("How is the S&P 500 performing?")
            print(result.get_markdown_with_citations())

        With charts and a resilient retry profile::

            client = ResearchClient(max_retries=5, stream_timeout=120.0)
            result = client.research(
                "Chart Tesla's quarterly revenue for the last six quarters",
                chart_generation=True,
            )

        Follow-up questions::

            first = client.research("What is NVIDIA's market position?")
            second = client.follow_up("Compare that to AMD", first)
    """

    #: Network failures worth retrying.
    RETRYABLE_EXCEPTIONS = (
        ConnectionError,
        Timeout,
        ReadTimeout,
        ChunkedEncodingError,
        StreamTimeoutError,
        TruncatedStreamError,
        RateLimitError,
        ServerError,
    )

    VALID_RESEARCH_EFFORTS = ("lite", "standard")
    VALID_MODEL_NAMES = ("base", "pro")
    VALID_PERSISTENCE_MODES = ("enabled", "disabled")

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://agents.bigdata.com/v1",
        timeout: int = 300,
        stream_timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        retry_max_delay: float = 60.0,
        persistence_mode: str = "enabled",
        code_execution: bool | None = None,
        chart_generation: bool | None = None,
    ) -> None:
        """
        Args:
            api_key: Bigdata.com API key. Falls back to ``BIGDATA_API_KEY``.
            base_url: API base URL.
            timeout: Connection timeout in seconds.
            stream_timeout: Maximum seconds to wait between SSE chunks before
                treating the connection as stalled. Set to ``0`` or ``None`` to
                disable.
            max_retries: Retry attempts for transient failures.
            retry_delay: Initial backoff delay in seconds.
            retry_backoff: Exponential backoff multiplier.
            retry_max_delay: Upper bound on the backoff delay.
            persistence_mode: ``"enabled"`` saves conversation history so
                ``chat_id`` follow-ups work. The API itself defaults to
                ``"disabled"``; this client opts in because ``follow_up()`` and
                mid-stream resume both depend on it.
            code_execution: Default for the Python code execution tool. ``None``
                leaves the server-side default (enabled) in place.
            chart_generation: Default for chart generation. ``None`` leaves the
                server-side default (disabled) in place.

        Raises:
            ValueError: If no API key is available or ``persistence_mode`` is invalid.
        """
        self.api_key = api_key or os.getenv("BIGDATA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set the BIGDATA_API_KEY environment variable or pass api_key."
            )
        if persistence_mode not in self.VALID_PERSISTENCE_MODES:
            raise ValueError(
                f"persistence_mode must be one of {self.VALID_PERSISTENCE_MODES}, "
                f"got '{persistence_mode}'"
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.stream_timeout = stream_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.retry_max_delay = retry_max_delay
        self.persistence_mode = persistence_mode
        self.code_execution = code_execution
        self.chart_generation = chart_generation

    # -- retry helpers ------------------------------------------------------

    def _should_retry(self, exception: Exception) -> bool:
        """Whether a failure is transient enough to retry."""
        if isinstance(exception, self.RETRYABLE_EXCEPTIONS):
            return True
        # A raw HTTPError can still surface from raise_for_status() on an
        # unmapped status code.
        if isinstance(exception, requests.HTTPError) and exception.response is not None:
            return exception.response.status_code >= 500
        return False

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Exponential backoff with full jitter, as recommended for 429 and 5xx."""
        delay = min(self.retry_delay * (self.retry_backoff**attempt), self.retry_max_delay)
        return delay + random.uniform(0, delay)

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        """
        Convert an HTTP error into a typed exception before touching the stream.

        Consuming an SSE stream from a non-2xx response silently yields zero
        events, so this must run first.
        """
        status = response.status_code
        if status < 400:
            return

        try:
            detail = json.dumps(response.json())
        except ValueError:
            detail = response.text[:500]

        if status == 401:
            raise AuthenticationError("API key is missing or invalid (401)")
        if status == 403:
            raise EntitlementError(f"Not entitled to this resource (403): {detail}")
        if status == 404:
            raise ResourceNotFoundError(
                f"Resource not found (404) - check chat_id or checkpoint id: {detail}"
            )
        if status == 429:
            raise RateLimitError("Rate limit exceeded (429)")
        if status in (400, 422):
            raise InvalidRequestError(f"Request rejected ({status}): {detail}")
        if status >= 500:
            raise ServerError(f"Upstream failure ({status}): {detail}")
        response.raise_for_status()

    # -- payload ------------------------------------------------------------

    def _build_payload(
        self,
        message: str,
        research_effort: str,
        chat_id: str | None,
        model_name: str,
        from_checkpoint_id: str | None,
        expected_output: str | None,
        structured_output_schema: dict[str, Any] | None,
        code_execution: bool | None,
        chart_generation: bool | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "message": message,
            "research_effort": research_effort,
            "model_name": model_name,
            "persistence_mode": self.persistence_mode,
        }
        if chat_id:
            payload["chat_id"] = chat_id
        if from_checkpoint_id:
            payload["from_checkpoint_id"] = from_checkpoint_id
        if expected_output:
            payload["expected_output"] = expected_output
        if structured_output_schema:
            payload["structured_output_schema"] = structured_output_schema

        code_execution = self.code_execution if code_execution is None else code_execution
        chart_generation = self.chart_generation if chart_generation is None else chart_generation
        python_config: dict[str, Any] = {}
        if code_execution is not None:
            python_config["enabled"] = code_execution
        if chart_generation is not None:
            python_config["chart_generation_enabled"] = chart_generation
        if python_config:
            payload["tools_configs"] = {"python_code_execution": python_config}

        return payload

    # -- streaming ----------------------------------------------------------

    def _consume_stream(
        self,
        response: requests.Response,
        result: ResearchResult,
        on_event: Callable[[str, dict[str, Any]], None] | None,
    ) -> str | None:
        """
        Read the SSE stream into ``result`` and return the conversation ``chat_id``.

        Raises:
            StreamError: On an in-stream ``ERROR`` event.
            StreamTimeoutError: When no data arrives within ``stream_timeout``.
            TruncatedStreamError: When the stream ends without ``COMPLETE``.
        """
        chat_id: str | None = None
        completed = False
        last_data_time = time.time()
        answer_parts: list[str] = []
        seen_citation_ids: set[str] = set()

        for raw_line in response.iter_lines():
            now = time.time()
            if self.stream_timeout and (now - last_data_time) > self.stream_timeout:
                raise StreamTimeoutError(
                    f"No data received for {now - last_data_time:.1f}s "
                    f"(stream_timeout={self.stream_timeout}s)"
                )
            # SSE is always UTF-8, so decode explicitly rather than relying on a
            # charset in the response Content-Type.
            line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else raw_line
            # Lines that are not SSE data frames are heartbeats or comments.
            if not line or not line.startswith("data: "):
                continue
            last_data_time = now

            body = line[6:].strip()
            if body == "[DONE]":
                break
            try:
                event = json.loads(body)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed SSE line")
                continue

            chat_id = chat_id or event.get("chat_id")
            msg = event.get("message") or {}
            msg_type = msg.get("type", "")
            if on_event:
                on_event(msg_type, msg)

            if msg_type == "ANSWER":
                answer_parts.append(msg.get("content", ""))

            elif msg_type == "GROUNDING":
                for ref in msg.get("references", []):
                    result.grounding_refs.append(GroundingReference.from_grounding(ref))

            elif msg_type == "AUDIT":
                for trace in msg.get("audit_traces", []):
                    audit = AuditTrace.from_trace(trace)
                    if audit.tool_id:
                        result.audit_traces[audit.tool_id] = audit
                    for group in trace.get("results") or []:
                        for value in group.get("values", []):
                            citation = Citation.from_source(value)
                            if citation.id and citation.id not in seen_citation_ids:
                                seen_citation_ids.add(citation.id)
                                result.citations.append(citation)

            elif msg_type == "CHART":
                result.charts.append(Chart.from_message(msg))
                logger.info("CHART: %s (%s)", msg.get("title"), msg.get("chart_type"))

            elif msg_type == "PLANNING":
                steps = (msg.get("plan") or {}).get("steps", [])
                result.plan_steps = [(s.get("description", ""), s.get("status", "")) for s in steps]

            elif msg_type == "ACTION":
                logger.info("ACTION: %s", msg.get("tool_name"))

            elif msg_type == "STRUCTURED_OUTPUT":
                result.structured_output = msg.get("content")

            elif msg_type == "LLM_RETRY":
                # Informational: the agent recovers on its own.
                logger.info("Agent retrying upstream LLM call: %s", msg.get("message"))

            elif msg_type == "TOOL_ERROR":
                tool = msg.get("tool_name") or "unknown"
                result.tool_errors[tool] = result.tool_errors.get(tool, 0) + 1
                logger.warning("Tool error from %s: %s", tool, msg.get("error"))

            elif msg_type == "ERROR":
                raise StreamError(msg.get("error", "Unknown research agent error"))

            elif msg_type == "COMPLETE":
                completed = True
                result.consumption = msg.get("consumption", [])
                result.checkpoint_id = msg.get("checkpoint_id")
            # Unknown types are ignored so new API versions stay compatible.

        result.answer = "".join(answer_parts)

        if not completed:
            raise TruncatedStreamError(
                "Stream ended without a COMPLETE event; the answer is incomplete"
            )
        return chat_id

    # -- public API ---------------------------------------------------------

    def research(
        self,
        message: str,
        research_effort: str = "standard",
        chat_id: str | None = None,
        *,
        model_name: str = "base",
        from_checkpoint_id: str | None = None,
        expected_output: str | None = None,
        structured_output_schema: dict[str, Any] | None = None,
        code_execution: bool | None = None,
        chart_generation: bool | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> ResearchResult:
        """
        Execute a research query and return the complete result.

        Args:
            message: The research question. Natural-language time references such
                as "last 24 hours" are understood.
            research_effort: ``"lite"`` for a quick answer (~10-20s) or
                ``"standard"`` for multi-step research (~20-60s).
            chat_id: Continue an existing conversation.
            model_name: ``"base"`` for default routing, ``"pro"`` for the most
                capable available model.
            from_checkpoint_id: Resume or branch from a specific checkpoint,
                obtained from a previous result's ``checkpoint_id``.
            expected_output: Guidance for the answer's tone, structure, and format.
            structured_output_schema: JSON Schema for structured extraction. The
                extracted object arrives on ``result.structured_output``.
            code_execution: Override whether the agent may run sandboxed Python.
            chart_generation: Override whether the agent may emit ``CHART`` events.
                Charts are off by default server-side.
            on_event: Called as ``on_event(msg_type, message)`` for every streamed
                event, for progress display.

        Returns:
            The completed :class:`ResearchResult`.

        Raises:
            ValueError: On an invalid ``research_effort`` or ``model_name``.
            AuthenticationError, EntitlementError, InvalidRequestError,
            ResourceNotFoundError: Non-retryable HTTP failures.
            RateLimitError, ServerError: Raised only after retries are exhausted.
            StreamError: The agent reported an unrecoverable error mid-stream.
            TruncatedStreamError: The stream ended without completing.
        """
        if research_effort not in self.VALID_RESEARCH_EFFORTS:
            raise ValueError(
                f"research_effort must be one of {self.VALID_RESEARCH_EFFORTS}, "
                f"got '{research_effort}'"
            )
        if model_name not in self.VALID_MODEL_NAMES:
            raise ValueError(
                f"model_name must be one of {self.VALID_MODEL_NAMES}, got '{model_name}'"
            )

        logger.info(
            "Starting research (effort=%s, model=%s, chat_id=%s)",
            research_effort,
            model_name,
            chat_id or "new",
        )
        start_time = time.time()

        payload = self._build_payload(
            message=message,
            research_effort=research_effort,
            chat_id=chat_id,
            model_name=model_name,
            from_checkpoint_id=from_checkpoint_id,
            expected_output=expected_output,
            structured_output_schema=structured_output_schema,
            code_execution=code_execution,
            chart_generation=chart_generation,
        )
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        endpoint = f"{self.base_url}/research-agent"
        request_timeout = (
            (self.timeout, self.stream_timeout) if self.stream_timeout else self.timeout
        )

        session_chat_id = chat_id

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                delay = self._calculate_retry_delay(attempt - 1)
                # Resuming with the chat_id keeps prior turns in context; the
                # answer itself is regenerated from scratch, which is why per
                # attempt state is discarded below.
                if session_chat_id:
                    payload["chat_id"] = session_chat_id
                logger.warning(
                    "Retry %d/%d in %.1fs%s",
                    attempt,
                    self.max_retries,
                    delay,
                    f" (resuming chat_id={session_chat_id})" if session_chat_id else "",
                )
                _flush_handlers()
                time.sleep(delay)

            # Grounding offsets index the answer produced by a single response,
            # so every attempt starts from an empty result.
            result = ResearchResult(answer="")

            try:
                logger.info("Request attempt %d/%d", attempt + 1, self.max_retries + 1)
                _flush_handlers()

                with requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=request_timeout,
                ) as response:
                    self._raise_for_status(response)
                    session_chat_id = (
                        self._consume_stream(response, result, on_event) or session_chat_id
                    )

            except Exception as exc:
                if self._should_retry(exc) and attempt < self.max_retries:
                    logger.warning(
                        "Retryable error on attempt %d/%d: %s: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        type(exc).__name__,
                        exc,
                    )
                    _flush_handlers()
                    continue
                logger.error("Request failed: %s: %s", type(exc).__name__, exc)
                _flush_handlers()
                raise

            result.chat_id = session_chat_id
            result.processing_time_ms = int((time.time() - start_time) * 1000)

            if result.tool_errors:
                logger.warning(
                    "Completed with tool errors: %s",
                    ", ".join(f"{k}={v}" for k, v in result.tool_errors.items()),
                )
            logger.info(
                "Research complete: %d citations, %d grounding refs, %d charts, %dms",
                len(result.citations),
                len(result.grounding_refs),
                len(result.charts),
                result.processing_time_ms,
            )
            _flush_handlers()
            return result

        raise RuntimeError("Unreachable: retry loop exited without a result")

    def follow_up(
        self,
        message: str,
        previous_result: ResearchResult,
        research_effort: str = "standard",
        **kwargs: Any,
    ) -> ResearchResult:
        """
        Ask a follow-up question in an existing conversation.

        Requires the client to run with ``persistence_mode="enabled"`` (the
        default), otherwise the server retains no history to continue from.

        Args:
            message: The follow-up question.
            previous_result: A result from a prior ``research()`` or ``follow_up()``.
            research_effort: ``"lite"`` or ``"standard"``.
            **kwargs: Any additional keyword argument accepted by :meth:`research`.

        Raises:
            ValueError: If the previous result carries no ``chat_id``, or the
                client has persistence disabled.
        """
        if self.persistence_mode != "enabled":
            raise ValueError(
                "follow_up() requires persistence_mode='enabled' on the client; "
                "the conversation was not saved."
            )
        if not previous_result.chat_id:
            raise ValueError("Previous result has no chat_id to follow up on")

        return self.research(
            message=message,
            research_effort=research_effort,
            chat_id=previous_result.chat_id,
            **kwargs,
        )
