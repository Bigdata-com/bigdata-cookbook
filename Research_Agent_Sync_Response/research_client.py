"""
Research Agent API - Synchronous Client

A simple Python wrapper that provides synchronous-like responses from the
Research Agent streaming API with complete citations in Bigdata.com format.

Usage:
    from research_client import ResearchClient
    
    client = ResearchClient()
    result = client.research("What are the key risks facing NVIDIA?")
    
    print(result.answer)
    print(result.citations)
"""

import os
import json
import time
import logging
import requests
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class Source:
    """Source information for a citation."""
    id: Optional[str] = None
    name: Optional[str] = None
    rank: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.id is not None:
            result["id"] = self.id
        if self.name is not None:
            result["name"] = self.name
        if self.rank is not None:
            result["rank"] = self.rank
        return result


@dataclass
class Chunk:
    """Text chunk from a citation."""
    cnum: Optional[int] = None
    text: Optional[str] = None
    relevance: Optional[float] = None
    sentiment: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.cnum is not None:
            result["cnum"] = self.cnum
        if self.text is not None:
            result["text"] = self.text
        if self.relevance is not None:
            result["relevance"] = self.relevance
        if self.sentiment is not None:
            result["sentiment"] = self.sentiment
        return result


@dataclass
class Citation:
    """
    Citation in Bigdata.com format.
    
    Fields match the standard Bigdata.com citation structure:
    - id: Document ID
    - headline: Document headline/title
    - timestamp: Publication timestamp
    - source: Source information (id, name, rank)
    - url: Document URL
    - chunks: Array of text chunks with relevance and sentiment
    """
    id: Optional[str] = None
    headline: Optional[str] = None
    timestamp: Optional[str] = None
    source: Optional[Source] = None
    url: Optional[str] = None
    chunks: List[Chunk] = field(default_factory=list)
    
    @classmethod
    def from_audit_result(cls, data: Dict[str, Any]) -> "Citation":
        """Create Citation from AUDIT event result."""
        # Extract source info
        source = Source(
            id=data.get("src_key"),
            name=data.get("src_name") or data.get("name") or (
                data.get("action", {}).get("name") if data.get("action") else None
            ),
            rank=f"RANK_{data.get('source_rank')}" if data.get("source_rank") else None
        )
        
        # Extract chunks
        chunks = []
        
        # Direct text field (EXTERNAL results)
        if data.get("text"):
            chunks.append(Chunk(
                text=data.get("text"),
                relevance=data.get("relevance"),
                sentiment=data.get("sentiment")
            ))
        
        # Chunks array (BIGDATA results)
        if data.get("chunks"):
            for chunk_data in data.get("chunks", []):
                chunks.append(Chunk(
                    cnum=chunk_data.get("cnum"),
                    text=chunk_data.get("text"),
                    relevance=chunk_data.get("relevance"),
                    sentiment=chunk_data.get("sentiment")
                ))
        
        return cls(
            id=data.get("id"),
            headline=data.get("hd"),
            timestamp=data.get("ts"),
            source=source if any([source.id, source.name, source.rank]) else None,
            url=data.get("url") or (data.get("action", {}).get("url") if data.get("action") else None),
            chunks=chunks
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
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
        if self.chunks:
            result["chunks"] = [c.to_dict() for c in self.chunks if c.to_dict()]
        return result


@dataclass
class GroundingReference:
    """
    Reference mapping a citation to a specific position in the answer.
    
    Used to insert inline citation numbers like [1], [2] at the correct positions.
    """
    start: int  # Character position in answer
    end: int
    citation_id: Optional[str] = None
    headline: Optional[str] = None
    source_name: Optional[str] = None
    timestamp: Optional[str] = None
    url: Optional[str] = None
    
    @classmethod
    def from_grounding(cls, ref: Dict[str, Any]) -> "GroundingReference":
        """Create from GROUNDING event reference."""
        source = ref.get("source", {})
        return cls(
            start=ref.get("start", 0),
            end=ref.get("end", 0),
            citation_id=source.get("id"),
            headline=source.get("hd"),
            source_name=source.get("src_name"),
            timestamp=source.get("ts"),
            url=source.get("url")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"start": self.start, "end": self.end}
        if self.citation_id:
            result["citation_id"] = self.citation_id
        if self.headline:
            result["headline"] = self.headline
        if self.source_name:
            result["source_name"] = self.source_name
        if self.timestamp:
            result["timestamp"] = self.timestamp
        if self.url:
            result["url"] = self.url
        return result


@dataclass
class ResearchResult:
    """
    Complete research result with answer and citations.
    
    Attributes:
        answer: The complete research answer in Markdown format
        citations: List of citations in Bigdata.com format
        grounding_refs: List of grounding references (positions for inline citations)
        chat_id: ID for follow-up conversations
        processing_time_ms: Time taken to complete research
    """
    answer: str
    citations: List[Citation] = field(default_factory=list)
    grounding_refs: List[GroundingReference] = field(default_factory=list)
    chat_id: Optional[str] = None
    processing_time_ms: Optional[int] = None
    
    def get_answer(self) -> str:
        """Get just the answer (without inline citations)."""
        return self.answer
    
    def get_answer_with_citations(self) -> str:
        """
        Get answer with inline citation numbers [1, 2, 3], etc.
        
        Returns the answer text with citation markers inserted at the
        positions indicated by grounding references.
        
        Improvements:
        - Groups nearby citations into single consolidated markers
        - Deduplicates repeated citation numbers at same position
        - Displays citation numbers in ascending order for readability
        """
        if not self.grounding_refs:
            return self.answer
        
        # First, assign citation numbers in order of appearance (by position ascending)
        # This ensures consistent numbering with get_numbered_citations()
        citation_nums = {}
        num_counter = 1
        refs_sorted_asc = sorted(self.grounding_refs, key=lambda r: r.start)
        
        for ref in refs_sorted_asc:
            key = ref.headline or ref.citation_id or f"{ref.source_name}_{ref.start}"
            if key not in citation_nums:
                citation_nums[key] = num_counter
                num_counter += 1
        
        # Group citations by position (within threshold of 5 chars)
        # This consolidates multiple citations at the same logical point
        POSITION_THRESHOLD = 5
        position_groups = {}  # position -> set of citation numbers
        
        for ref in self.grounding_refs:
            key = ref.headline or ref.citation_id or f"{ref.source_name}_{ref.start}"
            cite_num = citation_nums[key]
            pos = ref.end if ref.end > 0 else ref.start
            
            # Find existing group within threshold or create new one
            found_group = None
            for group_pos in position_groups:
                if abs(group_pos - pos) <= POSITION_THRESHOLD:
                    found_group = group_pos
                    break
            
            if found_group is not None:
                position_groups[found_group].add(cite_num)
            else:
                position_groups[pos] = {cite_num}
        
        # Insert consolidated markers from end to start (to preserve positions)
        result = self.answer
        for pos in sorted(position_groups.keys(), reverse=True):
            nums = sorted(position_groups[pos])  # Sort ascending for readability
            if 0 <= pos <= len(result):
                # Format as [1, 2, 3] for multiple, or [1] for single
                if len(nums) == 1:
                    marker = f" [{nums[0]}]"
                else:
                    marker = f" [{', '.join(str(n) for n in nums)}]"
                result = result[:pos] + marker + result[pos:]
        
        return result
    
    def get_numbered_citations(self) -> List[Dict[str, Any]]:
        """
        Get citations with their assigned numbers.
        
        Returns list of dicts with 'number' and citation details,
        matching the inline citation numbers in get_answer_with_citations().
        """
        if not self.grounding_refs:
            # Fall back to regular citations
            return [{"number": i+1, **c.to_dict()} for i, c in enumerate(self.citations)]
        
        # Build numbered citations from grounding refs
        # Sort by position (ascending) to match get_answer_with_citations() numbering
        citation_nums = {}
        numbered = []
        num_counter = 1
        refs_sorted_asc = sorted(self.grounding_refs, key=lambda r: r.start)
        
        for ref in refs_sorted_asc:
            key = ref.headline or ref.citation_id or f"{ref.source_name}_{ref.start}"
            if key not in citation_nums:
                citation_nums[key] = num_counter
                
                # Find matching full citation
                matching_citation = None
                for c in self.citations:
                    if c.id == ref.citation_id or c.headline == ref.headline:
                        matching_citation = c
                        break
                
                entry = {"number": num_counter}
                if matching_citation:
                    entry.update(matching_citation.to_dict())
                else:
                    # Use grounding ref data
                    if ref.headline:
                        entry["headline"] = ref.headline
                    if ref.source_name:
                        entry["source"] = {"name": ref.source_name}
                    if ref.timestamp:
                        entry["timestamp"] = ref.timestamp
                    if ref.url:
                        entry["url"] = ref.url
                
                numbered.append(entry)
                num_counter += 1
        
        return numbered
    
    def get_citations(self) -> List[Dict[str, Any]]:
        """Get citations as list of dictionaries."""
        return [c.to_dict() for c in self.citations]
    
    def get_citations_json(self, indent: int = 2) -> str:
        """Get citations as formatted JSON string."""
        return json.dumps(self.get_citations(), indent=indent)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire result to dictionary."""
        result = {"answer": self.answer}
        if self.citations:
            result["citations"] = self.get_citations()
        if self.chat_id:
            result["chat_id"] = self.chat_id
        if self.processing_time_ms:
            result["processing_time_ms"] = self.processing_time_ms
        return result
    
    def to_dict_with_inline_citations(self) -> Dict[str, Any]:
        """Convert to dictionary with answer containing inline citation numbers."""
        result = {"answer": self.get_answer_with_citations()}
        if self.grounding_refs:
            result["citations"] = self.get_numbered_citations()
        elif self.citations:
            result["citations"] = self.get_citations()
        if self.chat_id:
            result["chat_id"] = self.chat_id
        if self.processing_time_ms:
            result["processing_time_ms"] = self.processing_time_ms
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert entire result to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_json_with_inline_citations(self, indent: int = 2) -> str:
        """Convert to JSON with inline citation numbers in answer."""
        return json.dumps(self.to_dict_with_inline_citations(), indent=indent)


class ResearchClient:
    """
    Simple synchronous client for the Research Agent API.
    
    Example:
        >>> client = ResearchClient()
        >>> result = client.research("What are NVIDIA's key risks?")
        >>> print(result.answer)
        >>> print(result.get_citations_json())
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://agents.bigdata.com/v1",
        timeout: int = 300
    ):
        """
        Initialize the client.
        
        Args:
            api_key: Bigdata.com API key (or set BIGDATA_API_KEY env var)
            base_url: API base URL
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("BIGDATA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set BIGDATA_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.base_url = base_url
        self.timeout = timeout
    
    # NOTE: Additional parameters can be added to the research function based on the requirements.
    def research(
        self,
        message: str,
        research_effort: str = "standard",
        chat_id: Optional[str] = None
    ) -> ResearchResult:
        """
        Execute a research query and return complete result with citations.
        
        This method sends a request to the Bigdata.com Research Agent API,
        which performs multi-step research using retrieval-augmented generation (RAG)
        across web, premium sources, and your own content.

        Additional parameters can be added to the research function based on the requirements.
        
        Args:
            message: Your research question or prompt. Can include instructions
                for formatting, tone, and structure. Supports natural language
                including time references like "last 24 hours" or "this month".
                
            research_effort: Controls the depth and speed of research.
                - "lite": Quick response (~10-20 seconds). Equivalent to the
                  former Chat Service. Best for simple, factual queries.
                - "standard": Deep research (~20-60 seconds). The agent performs
                  multiple reasoning steps until it has enough data. Recommended
                  for complex analysis and detailed prompts.
                  
            chat_id: Optional conversation ID from a previous response. Use this
                to ask follow-up questions that maintain context from earlier
                exchanges. Get this value from result.chat_id.
        
        Returns:
            ResearchResult: Complete result containing:
                - answer: The synthesized research response
                - citations: List of source citations in Bigdata.com format
                - grounding_refs: Inline citation position references
                - chat_id: Conversation ID for follow-up questions
                - processing_time_ms: API processing time
        
        Raises:
            ValueError: If research_effort is not "lite" or "standard"
            requests.HTTPError: If the API request fails
        
        Example:
            >>> client = ResearchClient()
            >>> result = client.research(
            ...     message="What are the key risks facing NVIDIA?",
            ...     research_effort="standard"
            ... )
            >>> print(result.get_answer())
            >>> print(f"Found {len(result.citations)} citations")
            
            # Follow-up question
            >>> result2 = client.research(
            ...     message="How does this compare to AMD?",
            ...     chat_id=result.chat_id
            ... )
        """
        # Validate research_effort parameter
        valid_efforts = ("lite", "standard")
        if research_effort not in valid_efforts:
            raise ValueError(
                f"research_effort must be one of {valid_efforts}, got '{research_effort}'"
            )
        
        logger.info(f"Starting research query (effort={research_effort}, chat_id={chat_id or 'new'})")
        start_time = time.time()
        
        # Build request payload per Bigdata.com API spec
        payload = {
            "message": message,
            "research_effort": research_effort,
        }
        if chat_id:
            payload["chat_id"] = chat_id
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        
        # Collect response parts
        answer_parts = []
        citations_map = {}  # Use dict to deduplicate by id
        grounding_refs = []  # Inline citation positions
        response_chat_id = None
        
        # Stream and collect
        endpoint = f"{self.base_url}/research-agent"
        
        with requests.post(
            endpoint,
            headers=headers,
            json=payload,
            stream=True,
            timeout=self.timeout
        ) as response:
            response.raise_for_status()
            
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                
                # Capture chat_id
                if not response_chat_id:
                    response_chat_id = event.get("chat_id")
                
                msg = event.get("message", {})
                msg_type = msg.get("type", "")
                
                # Log message type received for client visibility
                if msg_type:
                    logger.info(f"Received message type: {msg_type}")
                
                # Collect answer chunks
                if msg_type == "ANSWER":
                    content = msg.get("content", "")
                    answer_parts.append(content)
                    logger.debug(f"Received answer chunk ({len(content)} chars)")
                
                # Collect citations from AUDIT events
                elif msg_type == "AUDIT":
                    citations_added = 0
                    for trace in msg.get("audit_traces", []):
                        for result_group in trace.get("results", []):
                            for val in result_group.get("values", []):
                                try:
                                    citation = Citation.from_audit_result(val)
                                    # Deduplicate by id or headline
                                    key = citation.id or citation.headline
                                    if key and key not in citations_map:
                                        citations_map[key] = citation
                                        citations_added += 1
                                except Exception:
                                    pass
                    if citations_added > 0:
                        logger.info(f"AUDIT: Added {citations_added} citations (total: {len(citations_map)})")
                
                # Collect grounding references (inline citation positions)
                elif msg_type == "GROUNDING":
                    refs_added = 0
                    for ref in msg.get("references", []):
                        try:
                            grounding_refs.append(GroundingReference.from_grounding(ref))
                            refs_added += 1
                        except Exception:
                            pass
                    if refs_added > 0:
                        logger.info(f"GROUNDING: Added {refs_added} inline citation references")
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"Research complete: {len(citations_map)} citations, "
            f"{len(grounding_refs)} grounding refs, {processing_time}ms"
        )
        
        return ResearchResult(
            answer="".join(answer_parts),
            citations=list(citations_map.values()),
            grounding_refs=grounding_refs,
            chat_id=response_chat_id,
            processing_time_ms=processing_time
        )
    
    def follow_up(
        self,
        message: str,
        previous_result: ResearchResult,
        research_effort: str = "standard"
    ) -> ResearchResult:
        """
        Ask a follow-up question in an existing conversation.
        
        The Research Agent supports multi-turn dialogue, allowing you to
        refine queries or ask follow-up questions while maintaining context
        from previous exchanges.
        
        Args:
            message: Your follow-up question. The agent will use context
                from the previous conversation to provide a relevant response.
                
            previous_result: The ResearchResult from a previous research()
                or follow_up() call. Must have a valid chat_id.
                
            research_effort: Controls research depth.
                - "lite": Quick response
                - "standard": Deep research (default)
        
        Returns:
            ResearchResult: New result with answer, citations, and the same
                chat_id for continued conversation.
        
        Raises:
            ValueError: If previous_result has no chat_id
        
        Example:
            >>> result1 = client.research("What is NVIDIA's market position?")
            >>> result2 = client.follow_up(
            ...     message="What about their AI chip competitors?",
            ...     previous_result=result1
            ... )
            >>> result3 = client.follow_up(
            ...     message="Compare their valuations",
            ...     previous_result=result2
            ... )
        """
        if not previous_result.chat_id:
            raise ValueError("Previous result has no chat_id for follow-up")
        
        return self.research(
            message=message,
            research_effort=research_effort,
            chat_id=previous_result.chat_id
        )

