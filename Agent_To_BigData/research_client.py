"""
Research Agent API - Synchronous Client (FIXED CITATION HANDLING)

A simple Python wrapper that provides synchronous-like responses from the
Research Agent streaming API with complete citations in Bigdata.com format.

FIXED: Citation numbers now properly match between answer text and source list.

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
import re
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set

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
    
    FIXED: Citation numbers now properly align between answer text and source list.
    
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
    
    # =========================================================================
    # NEW METHODS FOR CITATION DETECTION AND HANDLING
    # =========================================================================
    
    def has_existing_citations(self) -> bool:
        """
        Check if the answer already contains citation markers like [1], [2], etc.
        
        The Research Agent often embeds citations directly in its answer.
        Returns True if any [N] patterns are found.
        """
        return bool(re.search(r'\[\d+\]', self.answer))
    
    def extract_citation_numbers_from_answer(self) -> Set[int]:
        """
        Extract all citation numbers that appear in the answer text.
        
        Returns a set of integers representing all [N] citations found.
        Example: "See [1] and [13][14]" returns {1, 13, 14}
        """
        return set(int(m) for m in re.findall(r'\[(\d+)\]', self.answer))
    
    # =========================================================================
    # ANSWER RETRIEVAL METHODS
    # =========================================================================
    
    def get_answer(self) -> str:
        """Get the raw answer (may already include citation markers)."""
        return self.answer
    
    def get_answer_with_citations(self) -> str:
        """
        Get answer with inline citation numbers [1], [2], [3], etc.
        
        FIXED BEHAVIOR:
        - If the answer already contains citations from the Research Agent,
          returns the answer as-is to preserve the original grounding.
        - Otherwise, inserts citations based on GROUNDING reference positions.
        
        This ensures citation numbers in the text match the source list.
        """
        # If answer already has citation markers from Research Agent, return as-is
        if self.has_existing_citations():
            logger.debug("Answer already has citations, returning as-is")
            return self.answer
        
        # If no grounding refs available, return plain answer
        if not self.grounding_refs:
            logger.debug("No grounding refs, returning plain answer")
            return self.answer
        
        # Insert citations based on GROUNDING positions
        # Assign numbers in order of first appearance (by position)
        citation_nums = {}
        num_counter = 1
        refs_sorted_asc = sorted(self.grounding_refs, key=lambda r: r.start)
        
        for ref in refs_sorted_asc:
            key = ref.headline or ref.citation_id or f"{ref.source_name}_{ref.start}"
            if key not in citation_nums:
                citation_nums[key] = num_counter
                num_counter += 1
        
        # Group citations by position (within threshold of 5 chars)
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
            nums = sorted(position_groups[pos])
            if 0 <= pos <= len(result):
                if len(nums) == 1:
                    marker = f" [{nums[0]}]"
                else:
                    marker = f" [{', '.join(str(n) for n in nums)}]"
                result = result[:pos] + marker + result[pos:]
        
        return result
    
    # =========================================================================
    # CITATION RETRIEVAL METHODS (FIXED)
    # =========================================================================
    
    def get_numbered_citations(self) -> List[Dict[str, Any]]:
        """
        Get citations with numbers that MATCH the inline citations in the answer.
        
        FIXED BEHAVIOR:
        - If answer has existing citations [1], [13], etc., returns citations
          numbered to match those exact references.
        - Uses GROUNDING refs to correlate citation numbers with source details.
        - Ensures source #13 in the list corresponds to [13] in the text.
        """
        # Case 1: Answer has existing citation markers from Research Agent
        if self.has_existing_citations():
            return self._get_citations_for_existing_markers()
        
        # Case 2: No grounding refs - fall back to sequential numbering
        if not self.grounding_refs:
            return [{"number": i+1, **c.to_dict()} for i, c in enumerate(self.citations)]
        
        # Case 3: Build numbers from grounding refs (original logic)
        citation_nums = {}
        numbered = []
        num_counter = 1
        refs_sorted_asc = sorted(self.grounding_refs, key=lambda r: r.start)
        
        for ref in refs_sorted_asc:
            key = ref.headline or ref.citation_id or f"{ref.source_name}_{ref.start}"
            if key not in citation_nums:
                citation_nums[key] = num_counter
                
                # Find matching full citation from AUDIT data
                matching_citation = None
                for c in self.citations:
                    if c.id == ref.citation_id or c.headline == ref.headline:
                        matching_citation = c
                        break
                
                entry = {"number": num_counter}
                if matching_citation:
                    entry.update(matching_citation.to_dict())
                else:
                    # Use grounding ref data as fallback
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
    
    def _get_citations_for_existing_markers(self) -> List[Dict[str, Any]]:
        """
        Build citations list matching the numbers already in the answer text.
        
        Uses GROUNDING refs to correlate citation numbers with source details.
        Returns citations sorted by number so [1] is first, [13] is at position 13, etc.
        """
        used_numbers = self.extract_citation_numbers_from_answer()
        logger.debug(f"Found citation numbers in answer: {sorted(used_numbers)}")
        
        number_to_citation: Dict[int, Dict[str, Any]] = {}
        
        # Strategy 1: Use GROUNDING refs to map numbers to sources
        for ref in self.grounding_refs:
            pos = ref.end if ref.end > 0 else ref.start
            
            # Look for [N] near this position in the answer
            search_start = max(0, pos - 5)
            search_end = min(len(self.answer), pos + 20)
            nearby_text = self.answer[search_start:search_end]
            
            # Find citation numbers in nearby text
            matches = re.findall(r'\[(\d+)\]', nearby_text)
            for match in matches:
                num = int(match)
                if num in used_numbers and num not in number_to_citation:
                    entry = self._build_citation_entry(num, ref)
                    number_to_citation[num] = entry
                    logger.debug(f"Mapped citation [{num}] to: {entry.get('headline', 'Unknown')[:50]}")
        
        # Strategy 2: For any unmapped numbers, try matching by index in AUDIT citations
        for num in sorted(used_numbers):
            if num not in number_to_citation:
                # Try to find by searching all grounding refs for this number
                found = False
                for ref in self.grounding_refs:
                    # Check if this ref might correspond to citation number 'num'
                    # by looking at context
                    entry = self._try_match_citation(num, ref)
                    if entry:
                        number_to_citation[num] = entry
                        found = True
                        break
                
                # Fallback: Use citation at index (num-1) if available
                if not found and 0 < num <= len(self.citations):
                    entry = {"number": num}
                    entry.update(self.citations[num-1].to_dict())
                    number_to_citation[num] = entry
                    logger.debug(f"Fallback mapping [{num}] to AUDIT citation #{num}")
        
        # Return sorted by number
        result = [number_to_citation[n] for n in sorted(number_to_citation.keys()) if n in number_to_citation]
        logger.info(f"Built {len(result)} numbered citations for {len(used_numbers)} markers")
        return result
    
    def _build_citation_entry(self, num: int, ref: GroundingReference) -> Dict[str, Any]:
        """Build a citation entry from a grounding reference."""
        entry = {"number": num}
        
        # Try to find matching full citation from AUDIT data
        matching_citation = None
        for c in self.citations:
            if c.id == ref.citation_id:
                matching_citation = c
                break
            if c.headline and ref.headline and c.headline == ref.headline:
                matching_citation = c
                break
        
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
            if ref.citation_id:
                entry["id"] = ref.citation_id
        
        return entry
    
    def _try_match_citation(self, num: int, ref: GroundingReference) -> Optional[Dict[str, Any]]:
        """Try to match a citation number to a grounding reference."""
        # This is a heuristic - in practice, the Research Agent's numbering
        # should be consistent with GROUNDING refs
        return None  # Return None to fall back to index-based matching
    
    # =========================================================================
    # LEGACY METHODS (preserved for compatibility)
    # =========================================================================
    
    def get_citations(self) -> List[Dict[str, Any]]:
        """Get citations as list of dictionaries (without numbers)."""
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
        if self.grounding_refs or self.has_existing_citations():
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
    
    def research(
        self,
        message: str,
        research_effort: str = "standard",
        chat_id: Optional[str] = None,
        days_back: int = 90,
        source_categories: Optional[List[str]] = None
    ) -> ResearchResult:
        """
        Execute a research query and return structured results with citations.
        
        Args:
            message: Your research question or analysis request
            research_effort: "lite" (10-20s) or "standard" (20-60s deep analysis)
            chat_id: Previous chat_id for follow-up questions
            days_back: Number of days to search back (default: 90)
            source_categories: List of source categories (default: ["news_public"])
        
        Returns:
            ResearchResult: Complete result with answer and properly numbered citations
        """
        # Validate research_effort parameter
        valid_efforts = ("lite", "standard")
        if research_effort not in valid_efforts:
            raise ValueError(
                f"research_effort must be one of {valid_efforts}, got '{research_effort}'"
            )
        
        # Default to news_public if not specified
        categories = source_categories or ["news_public"]
        
        logger.info(f"Starting research query (effort={research_effort}, chat_id={chat_id or 'new'}, categories={categories})")
        start_time = time.time()
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        # Build request payload per Bigdata.com API spec
        payload = {
            "message": message,
            "research_effort": research_effort,
            "tools_configs": {
                "search": {
                    "query_filters": {
                        "period": {
                            "start": start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                            "end": end_date.strftime("%Y-%m-%dT%H:%M:%S.999Z")
                        },
                        "content": {
                            "any_of": [
                                {
                                    "type": "SOURCE_CATEGORY_ID",
                                    "source_category_ids": categories
                                }
                            ]
                        }
                    }
                }
            }
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
        
        Args:
            message: Your follow-up question
            previous_result: The ResearchResult from a previous call
            research_effort: "lite" or "standard"
        
        Returns:
            ResearchResult: New result with answer and citations
        """
        if not previous_result.chat_id:
            raise ValueError("Previous result has no chat_id for follow-up")
        
        return self.research(
            message=message,
            research_effort=research_effort,
            chat_id=previous_result.chat_id
        )
