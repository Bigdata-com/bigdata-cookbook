"""
Research Agent API - Synchronous Client

Simple wrapper for the Research Agent API with citations in Bigdata.com format.

Usage:
    from dist import ResearchClient
    
    client = ResearchClient()
    result = client.research("What are the key risks facing NVIDIA?")
    
    # Just the answer
    print(result.answer)
    
    # Just the citations (as JSON)
    print(result.get_citations_json())
    
    # Full result with answer and citations
    print(result.to_json())
"""

from .research_client import (
    ResearchClient,
    ResearchResult,
    Citation,
    Chunk,
    Source,
    GroundingReference,
)

__all__ = [
    "ResearchClient",
    "ResearchResult",
    "Citation",
    "Chunk",
    "Source",
    "GroundingReference",
]

