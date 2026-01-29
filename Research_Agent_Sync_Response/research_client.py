"""
Research Agent API - Synchronous Client
========================================

A robust Python client for the Bigdata.com Research Agent API that provides
synchronous responses with complete citations, automatic retry handling,
and network resilience.

Features
--------
- **Synchronous Interface**: Simple blocking API for research queries
- **Automatic Retries**: Exponential backoff for transient failures (connection errors, timeouts, 5xx errors)
- **Stream Timeout Detection**: Detects stalled connections and triggers retries
- **Conversation Continuity**: Automatically resumes interrupted conversations using chat_id
- **Citations in Bigdata.com Format**: Structured citations with source info, timestamps, and text chunks
- **Inline Citation Support**: Answer text with [1], [2] markers linked to numbered references

Installation
------------
No additional dependencies beyond `requests` are required:

    pip install requests

Quick Start
-----------
    from research_client import ResearchClient, setup_logging
    
    # Optional: Enable logging to see retry attempts and progress
    setup_logging(log_file="research.log", console=True)
    
    # Create client (uses BIGDATA_API_KEY environment variable)
    client = ResearchClient()
    
    # Execute research query
    result = client.research("What are the key risks facing NVIDIA?")
    
    # Access the results
    print(result.answer)                      # Plain answer text
    print(result.get_answer_with_citations()) # Answer with [1], [2] markers
    print(result.get_citations_json())        # Citations as JSON

Configuration
-------------
    client = ResearchClient(
        api_key="your-api-key",       # Or set BIGDATA_API_KEY env var
        timeout=300,                   # Connection timeout (seconds)
        stream_timeout=30.0,           # Max wait for data during streaming
        max_retries=3,                 # Retry attempts for transient failures
        retry_delay=1.0,               # Initial retry delay (seconds)
        retry_backoff=2.0,             # Exponential backoff multiplier
        retry_max_delay=60.0           # Maximum retry delay cap
    )

Retry Behavior
--------------
The client automatically retries on these transient errors:
- ConnectionError: Network connectivity issues
- Timeout/ReadTimeout: Request or read timeouts
- StreamTimeoutError: No data received within stream_timeout
- ChunkedEncodingError: Connection broken during streaming
- HTTP 408, 429, 500, 502, 503, 504: Server errors and rate limiting

Non-retryable errors (raised immediately):
- HTTP 400, 401, 403, 404: Client errors
- ValueError: Invalid parameters

Network Resilience
------------------
If the connection is interrupted mid-stream:
1. The client captures any partial data and the conversation chat_id
2. On retry, it sends the original message with the chat_id to resume
3. Partial responses are accumulated across retries

Example with Follow-up Questions
--------------------------------
    result1 = client.research("What is NVIDIA's market position?")
    
    # Follow-up uses the same conversation context
    result2 = client.follow_up(
        "How does this compare to AMD?",
        previous_result=result1
    )

Classes
-------
- ResearchClient: Main client for executing research queries
- ResearchResult: Result object containing answer, citations, and metadata
- Citation: Individual citation in Bigdata.com format
- Source: Source information (id, name, rank)
- Chunk: Text chunk with relevance and sentiment scores
- GroundingReference: Position reference for inline citations

Exceptions
----------
- StreamTimeoutError: Raised when no data is received within stream_timeout
- requests.HTTPError: Raised for non-retryable HTTP errors
- ValueError: Raised for invalid parameters

Thread Safety
-------------
ResearchClient is thread-safe. A single instance may be shared across threads
(e.g. one client serving 100+ requests per second). Each research() or
follow_up() call uses only local state; instance attributes are read-only
during requests. The requests library is used without a shared Session, and
Python's logging is thread-safe. Call setup_logging() once at application
startup, not from request handlers.

For more information, see: https://docs.bigdata.com/research-agent
"""

import os
import json
import time
import logging
import requests
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from requests.exceptions import (
    ConnectionError,
    Timeout,
    ReadTimeout,
    ChunkedEncodingError,
    RequestException
)


class StreamTimeoutError(Exception):
    """Raised when no data is received from the stream within the timeout period."""
    pass

# Configure module logger
logger = logging.getLogger(__name__)


def setup_logging(
    log_file: str = "research_client.log",
    level: int = logging.INFO,
    console: bool = True,
    file_mode: str = "a"
) -> logging.Logger:
    """
    Configure logging for the research_client module.
    
    Call this function before creating a ResearchClient to enable logging
    of retry attempts, connection errors, and API responses.
    
    Args:
        log_file: Path to log file (default: "research_client.log")
        level: Logging level (default: logging.INFO)
        console: If True, also log to console/stdout (default: True)
        file_mode: File mode - "a" for append, "w" for overwrite (default: "a")
    
    Returns:
        The configured logger instance
    
    Example:
        >>> from research_client import setup_logging, ResearchClient
        >>> setup_logging("output/research_client.log", console=True)
        >>> client = ResearchClient()
    """
    # Clear any existing handlers
    logger.handlers.clear()
    logger.setLevel(level)
    
    # Create formatter with timestamp
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler with immediate flush
    if log_file:
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Prevent propagation to root logger (avoid duplicate logs)
    logger.propagate = False
    
    logger.info(f"Logging configured: file={log_file}, console={console}, level={logging.getLevelName(level)}")
    _flush_handlers()
    
    return logger


def _flush_handlers():
    """Flush all handlers to ensure logs are written immediately."""
    for handler in logger.handlers:
        handler.flush()


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
    Synchronous client for the Bigdata.com Research Agent API.
    
    This client provides a simple interface for executing research queries
    with built-in retry logic, network resilience, and conversation continuity.
    
    Thread-safe: a single instance may be shared across threads (e.g. in a
    server handling many concurrent requests).
    
    Attributes:
        api_key (str): Bigdata.com API key
        base_url (str): API base URL
        timeout (int): Connection timeout in seconds
        stream_timeout (float): Maximum seconds to wait for streaming data
        max_retries (int): Maximum retry attempts for transient failures
        retry_delay (float): Initial delay between retries
        retry_backoff (float): Exponential backoff multiplier
        retry_max_delay (float): Maximum delay cap between retries
    
    Example:
        Basic usage::
        
            client = ResearchClient()
            result = client.research("What are NVIDIA's key risks?")
            print(result.answer)
            print(result.get_citations_json())
        
        With custom retry configuration::
        
            client = ResearchClient(
                max_retries=5,
                retry_delay=2.0,
                stream_timeout=60.0
            )
            result = client.research("Analyze market trends", research_effort="standard")
        
        Follow-up questions::
        
            result1 = client.research("What is Apple's market position?")
            result2 = client.follow_up("Compare to Microsoft", previous_result=result1)
    """
    
    # Exceptions that should trigger a retry
    RETRYABLE_EXCEPTIONS = (
        ConnectionError,      # Network connectivity issues
        Timeout,              # Request timeout (includes ConnectTimeout)
        ReadTimeout,          # No data received within read timeout
        ChunkedEncodingError, # Connection broken during streaming
        StreamTimeoutError,   # No data received within stream_timeout
    )
    
    # HTTP status codes that should trigger a retry
    RETRYABLE_STATUS_CODES = {
        408,  # Request Timeout
        429,  # Too Many Requests (rate limiting)
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://agents.bigdata.com/v1",
        timeout: int = 300,
        stream_timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        retry_max_delay: float = 60.0
    ):
        """
        Initialize the client.
        
        Args:
            api_key: Bigdata.com API key (or set BIGDATA_API_KEY env var)
            base_url: API base URL
            timeout: Request timeout in seconds (for initial connection)
            stream_timeout: Maximum seconds to wait for data during streaming (default: 30.0).
                If no data is received within this period, triggers a retry.
                Set to 0 or None to disable stream timeout checking.
            max_retries: Maximum number of retry attempts for transient failures (default: 3)
            retry_delay: Initial delay between retries in seconds (default: 1.0)
            retry_backoff: Multiplier for exponential backoff (default: 2.0)
            retry_max_delay: Maximum delay between retries in seconds (default: 60.0)
        """
        self.api_key = api_key or os.getenv("BIGDATA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set BIGDATA_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.base_url = base_url
        self.timeout = timeout
        self.stream_timeout = stream_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.retry_max_delay = retry_max_delay
    
    def _should_retry(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.
        
        Args:
            exception: The exception that was raised
            
        Returns:
            True if the request should be retried, False otherwise
        """
        # Check for retryable exception types
        if isinstance(exception, self.RETRYABLE_EXCEPTIONS):
            return True
        
        # Check for HTTP errors with retryable status codes
        if isinstance(exception, requests.HTTPError):
            response = exception.response
            if response is not None and response.status_code in self.RETRYABLE_STATUS_CODES:
                return True
        
        return False
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate the delay before the next retry using exponential backoff.
        
        Args:
            attempt: The current attempt number (0-indexed)
            
        Returns:
            Delay in seconds before the next retry
        """
        delay = self.retry_delay * (self.retry_backoff ** attempt)
        return min(delay, self.retry_max_delay)
    
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
        
        The method includes automatic retry logic with exponential backoff for
        transient failures such as:
        - Network connectivity issues (ConnectionError)
        - Request timeouts (Timeout)
        - Server errors (HTTP 500, 502, 503, 504)
        - Rate limiting (HTTP 429)
        
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
            requests.HTTPError: If the API request fails after all retries
            requests.ConnectionError: If connection fails after all retries
            requests.Timeout: If request times out after all retries
        
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
        
        endpoint = f"{self.base_url}/research-agent"
        last_exception = None
        
        # Track chat_id across retry attempts (for conversation continuity)
        session_chat_id = chat_id  # Start with user-provided chat_id (if any)
        
        # Collect response parts (accumulated across retries if needed)
        answer_parts = []
        citations_map = {}  # Use dict to deduplicate by id
        grounding_refs = []  # Inline citation positions
        
        # Retry loop with exponential backoff
        for attempt in range(self.max_retries + 1):
            try:
                # Reset response collectors for this attempt (but keep accumulated data)
                attempt_answer_parts = []
                response_chat_id = None
                
                if attempt > 0:
                    retry_delay = self._calculate_retry_delay(attempt - 1)
                    
                    # If we received a chat_id before the failure, use it to continue
                    if session_chat_id:
                        payload["chat_id"] = session_chat_id
                        payload["message"] = message
                        logger.warning(
                            f"Retry attempt {attempt}/{self.max_retries} after {retry_delay:.1f}s delay "
                            f"(resuming chat_id={session_chat_id})"
                        )
                    else:
                        logger.warning(
                            f"Retry attempt {attempt}/{self.max_retries} after {retry_delay:.1f}s delay"
                        )
                    _flush_handlers()  # Ensure retry message is written to disk
                    time.sleep(retry_delay)
                
                logger.info(f"Starting request attempt {attempt + 1}/{self.max_retries + 1}")
                _flush_handlers()
                
                # Stream and collect
                # Use tuple timeout: (connect_timeout, read_timeout)
                # stream_timeout controls how long to wait between data chunks
                request_timeout = (self.timeout, self.stream_timeout) if self.stream_timeout else self.timeout
                
                with requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=request_timeout
                ) as response:
                    response.raise_for_status()
                    
                    # Track time for stream timeout detection
                    last_data_time = time.time()
                    
                    for line in response.iter_lines(decode_unicode=True):
                        current_time = time.time()
                        
                        # Check stream timeout (time since last meaningful data)
                        if self.stream_timeout and (current_time - last_data_time) > self.stream_timeout:
                            raise StreamTimeoutError(
                                f"No data received for {current_time - last_data_time:.1f}s "
                                f"(stream_timeout={self.stream_timeout}s)"
                            )
                        
                        if not line or not line.startswith("data: "):
                            continue
                        
                        # Update last data time when we receive actual SSE data
                        last_data_time = current_time
                        
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        
                        # Capture chat_id and update session_chat_id for retry continuity
                        if not response_chat_id:
                            response_chat_id = event.get("chat_id")
                            if response_chat_id and not session_chat_id:
                                session_chat_id = response_chat_id
                                logger.info(f"Received chat_id: {session_chat_id}")
                                _flush_handlers()
                        
                        msg = event.get("message", {})
                        msg_type = msg.get("type", "")
                        
                        # Log message type received for client visibility
                        if msg_type:
                            logger.info(f"Received message type: {msg_type}")
                            _flush_handlers()  # Flush immediately so logs are visible
                        
                        # Collect answer chunks
                        if msg_type == "ANSWER":
                            content = msg.get("content", "")
                            attempt_answer_parts.append(content)
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
                
                # Success - merge this attempt's answer parts with accumulated parts
                answer_parts.extend(attempt_answer_parts)
                
                # Calculate processing time and return result
                processing_time = int((time.time() - start_time) * 1000)
                
                if attempt > 0:
                    logger.info(f"Request succeeded after {attempt} retry attempt(s)")
                
                logger.info(
                    f"Research complete: {len(citations_map)} citations, "
                    f"{len(grounding_refs)} grounding refs, {processing_time}ms"
                )
                _flush_handlers()
                
                return ResearchResult(
                    answer="".join(answer_parts),
                    citations=list(citations_map.values()),
                    grounding_refs=grounding_refs,
                    chat_id=session_chat_id or response_chat_id,
                    processing_time_ms=processing_time
                )
                
            except Exception as e:
                last_exception = e
                
                # Preserve any partial answer data collected before the failure
                if attempt_answer_parts:
                    answer_parts.extend(attempt_answer_parts)
                    logger.info(f"Preserved {len(attempt_answer_parts)} answer chunks from failed attempt")
                
                # Check if we should retry
                if self._should_retry(e) and attempt < self.max_retries:
                    logger.warning(
                        f"Retryable error on attempt {attempt + 1}/{self.max_retries + 1}: "
                        f"{type(e).__name__}: {str(e)}"
                    )
                    if session_chat_id:
                        logger.info(f"Will resume conversation with chat_id={session_chat_id} on next attempt")
                    _flush_handlers()  # Ensure error is logged before retry
                    continue
                else:
                    # Non-retryable error or max retries exceeded
                    if attempt >= self.max_retries:
                        logger.error(
                            f"Max retries ({self.max_retries}) exceeded. "
                            f"Last error: {type(e).__name__}: {str(e)}"
                        )
                    else:
                        logger.error(
                            f"Non-retryable error: {type(e).__name__}: {str(e)}"
                        )
                    _flush_handlers()  # Ensure error is logged before raising
                    raise
        
        # This should not be reached, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected error in retry loop")
    
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

