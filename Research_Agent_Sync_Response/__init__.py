"""
Research Agent API - Synchronous Client

A wrapper for the Bigdata.com Research Agent API that consumes the SSE stream and
returns a finished answer with correctly attributed citations.

Usage:
    from research_client import ResearchClient

    client = ResearchClient()
    result = client.research("How is the S&P 500 performing?")

    print(result.get_answer())                    # plain answer
    print(result.get_answer_with_citations())     # answer with [1], [2] markers
    print(result.get_markdown_with_citations())   # answer plus a Sources section
    print(result.get_citations_json())            # citations as JSON
"""

from .research_client import (
    AuditTrace,
    AuthenticationError,
    Chart,
    Chunk,
    Citation,
    EntitlementError,
    GroundingReference,
    InvalidRequestError,
    RateLimitError,
    ResearchAgentError,
    ResearchClient,
    ResearchResult,
    ResourceNotFoundError,
    ServerError,
    Source,
    StreamError,
    StreamTimeoutError,
    ToolCitation,
    TruncatedStreamError,
    format_source_date,
    setup_logging,
)

__all__ = [
    "AuditTrace",
    "AuthenticationError",
    "Chart",
    "Chunk",
    "Citation",
    "EntitlementError",
    "GroundingReference",
    "InvalidRequestError",
    "RateLimitError",
    "ResearchAgentError",
    "ResearchClient",
    "ResearchResult",
    "ResourceNotFoundError",
    "ServerError",
    "Source",
    "StreamError",
    "StreamTimeoutError",
    "ToolCitation",
    "TruncatedStreamError",
    "format_source_date",
    "setup_logging",
]
