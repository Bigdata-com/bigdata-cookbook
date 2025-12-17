"""
Topic-based news search service using Bigdata.com API.

This module provides asynchronous news search capabilities with separate methods
for baseline (entity-only) and topic (sentiment-filtered) searches.

KEY CLASSES:
    TopicSearchService: Main service class for news searches

KEY PUBLIC METHODS:
    search_ticker(): Topic-based search for a single ticker (parallel topics)
    search_multiple_tickers(): Topic searches for multiple tickers in parallel
    search_baseline(): Entity-filtered search (no topics, no sentiment filter)
    search_single_topic(): Single topic search with sentiment filter
    get_company_data(): Resolve ticker to entity ID via Knowledge Graph API
    generate_topic_variations(): Generate 3 query variations using Gemini AI
    close(): Clean up HTTP session

FEATURES:
    - Parallel execution of multiple topic searches (async/await)
    - Query reformulation using Gemini AI (optional, generates 3 variations per topic)
    - Smart rate limiting (200 RPM across all requests)
    - Company data caching (entity IDs, 24hr TTL)
    - Multi-level deduplication (within searches and across topics)
    - Shared HTTP session for connection pooling (faster performance)
    - Configurable chunk budgets (default: 100 total chunks per search)
    - Relevance-based filtering and best-chunk selection

USAGE EXAMPLES:
    service = TopicSearchService(api_key="your-key")
    
    # Baseline search (entity filter only, no sentiment)
    baseline_results = await service.search_baseline("AAPL", entity_id, days=7)
    
    # Topic search (sentiment-filtered parallel topic searches)
    topic_results = await service.search_ticker("AAPL", days=7)
    
    # Topic search with query reformulation (4x searches per topic: original + 3 variations)
    topic_results = await service.search_ticker("AAPL", days=7, query_reformulation=True)
    
    # Multiple tickers (topic search)
    all_results = await service.search_multiple_tickers(["AAPL", "GOOGL"], days=7)
    
    await service.close()
"""

import asyncio
import aiohttp
import os
import random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
import logging
from pydantic import BaseModel

from .rate_limiter import RateLimiter
from .company_cache import CompanyDataCache, CompanyData
from .llm_service import LLMService
from .llm_factory import LLMServiceFactory
from config.topics import STANDARD_TOPICS

logger = logging.getLogger(__name__)


class TopicVariations(BaseModel):
    """Pydantic model for topic variations from Gemini."""
    variation_1: str
    variation_2: str
    variation_3: str


class TopicSearchService:
    """
    Service for performing topic-based news searches.
    
    Features:
    - Always runs baseline company search (no sentiment filter)
    - Runs 27 topic searches in parallel (with sentiment filter)
    - Rate limiting to respect 200 RPM API limit
    - Caching of company data (entity IDs and names)
    - Shared HTTP session for connection pooling (much faster!)
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.bigdata.com/v1",
        rate_limiter: Optional[RateLimiter] = None,
        company_cache: Optional[CompanyDataCache] = None,
            gemini_service: Optional[LLMService] = None,
    ):
        """
        Initialize topic search service.
        
        Args:
            api_key: Bigdata.com API key
            base_url: Base URL for Bigdata API
            rate_limiter: Rate limiter instance (creates new if None)
            company_cache: Company cache instance (creates new if None)
            gemini_service: LLM service instance for query reformulation (creates new if None, backward compatible)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limiter = rate_limiter or RateLimiter(max_tokens=200, refill_period=60)
        self.company_cache = company_cache or CompanyDataCache(ttl_hours=24)
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Initialize LLM service for query reformulation (lazy init)
        # Accept either LLMService instance or legacy GeminiService (for backward compatibility)
        self._llm_service = gemini_service
        
        logger.info("TopicSearchService initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create shared aiohttp session.
        Reuses connection pool for much better performance.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    def _get_llm_service(self) -> LLMService:
        """
        Get or create LLM service instance (lazy initialization).
        Only creates if query reformulation is used.
        """
        if self._llm_service is None:
            try:
                # Try to initialize using factory with auto-detection
                self._llm_service = LLMServiceFactory.create(provider='auto')
                logger.info(
                    f"LLM service initialized for query reformulation "
                    f"(provider: {self._llm_service.provider_name}, model: {self._llm_service.model})"
                )
            except ValueError as e:
                logger.error(f"Failed to initialize LLM service: {e}")
                raise ValueError(
                    "Query reformulation requires LLM provider credentials:\n"
                    "  - OPENAI_API_KEY for OpenAI\n"
                    "  - GEMINI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS for Gemini\n"
                    "Or set LLM_PROVIDER env var to 'openai' or 'gemini'.\n"
                    "Please set one of these or disable query reformulation."
                ) from e
        return self._llm_service
    
    async def generate_topic_variations(
        self,
        topic,  # Can be str or dict with topic_text
        company_name: str
    ) -> List[str]:
        """
        Generate 3 variations of a topic query using Gemini AI.
        
        Args:
            topic: Original topic query template (may contain {company} placeholder)
                  Can be either a string or a dict with 'topic_text' key
            company_name: Company name to substitute in the template
            
        Returns:
            List of 3 topic variations
        """
        # Handle both dict and string topic formats
        if isinstance(topic, dict):
            topic_text = topic.get("topic_text", "")
        else:
            topic_text = topic
        
        # Format the topic with company name first
        formatted_topic = topic_text.format(company=company_name) if "{company}" in topic_text else topic_text
        
        prompt = f"""You are a financial news search expert. Given this search query about a company:

"{formatted_topic}"

Generate 3 alternative search queries that would find similar or related news articles. The variations should:
- Maintain the core intent and meaning
- Use different wording or phrasing
- Cover slightly different angles of the same topic
- Be suitable for semantic search (not keyword search)
- Stay focused on financial/business news context

Return exactly 3 variations, each as a complete search query."""

        try:
            llm_service = self._get_llm_service()
            variations = await llm_service.generate_content(
                prompt=prompt,
                response_schema=TopicVariations
            )
            
            result = [
                variations.variation_1,
                variations.variation_2,
                variations.variation_3
            ]
            
            # Log original and variations (truncated for readability)
            logger.info(f"Query Reformulation Generated:")
            logger.info(f"  Original: {formatted_topic[:80]}{'...' if len(formatted_topic) > 80 else ''}")
            for i, var in enumerate(result, 1):
                logger.info(f"  Variation {i}: {var[:80]}{'...' if len(var) > 80 else ''}")
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to generate topic variations: {e}. Using original topic only.")
            # Return empty list on failure - caller will use original topic only
            return []
    
    async def close(self):
        """Close the shared HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("HTTP session closed")
    
    def _format_timestamp(self, dt: datetime) -> str:
        """Format datetime to ISO 8601 with milliseconds and Z suffix."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        
        iso_str = dt.strftime('%Y-%m-%dT%H:%M:%S')
        milliseconds = dt.microsecond // 1000
        return f"{iso_str}.{milliseconds:03d}Z"
    
    def _get_time_ago(self, timestamp_str: str) -> str:
        """Convert timestamp string to human-readable time ago."""
        try:
            # Parse ISO format timestamp (handle both with and without Z)
            if timestamp_str.endswith('Z'):
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            elif '+' in timestamp_str or timestamp_str.count('-') > 2:
                timestamp = datetime.fromisoformat(timestamp_str)
            else:
                # Assume UTC if no timezone
                timestamp = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
            
            now = datetime.now(timezone.utc)
            diff = now - timestamp
            
            if diff.days > 0:
                return f"{diff.days}d ago"
            elif diff.seconds >= 3600:
                hours = diff.seconds // 3600
                return f"{hours}h ago"
            elif diff.seconds >= 60:
                minutes = diff.seconds // 60
                return f"{minutes}m ago"
            else:
                return "Just now"
        except Exception as e:
            logger.warning(f"Error parsing timestamp '{timestamp_str}': {e}")
            return "Unknown time"
    
    def _deduplicate_by_document_id(self, results: List[Dict]) -> List[Dict]:
        """
        Deduplicate results by document ID, keeping the chunk with highest relevance.
        
        The API can return the same document multiple times with different chunks.
        We keep only one instance per document ID - the one with the best chunk.
        
        Args:
            results: List of raw API results (each has id and chunks)
            
        Returns:
            Deduplicated list with one entry per document ID
        """
        # Group by document ID
        doc_groups = {}
        
        for article in results:
            doc_id = article.get("id")
            if not doc_id:
                continue
            
            chunks = article.get("chunks", [])
            if not chunks:
                continue
            
            # Find the best chunk for this article instance
            sorted_chunks = sorted(chunks, key=lambda x: x.get("relevance", 0), reverse=True)
            best_chunk = sorted_chunks[0]
            
            # If we haven't seen this document, or this chunk is better, keep it
            if doc_id not in doc_groups or best_chunk.get("relevance", 0) > doc_groups[doc_id]["best_relevance"]:
                doc_groups[doc_id] = {
                    "article": article,
                    "best_chunk": best_chunk,
                    "best_relevance": best_chunk.get("relevance", 0)
                }
        
        # Return deduplicated list
        deduplicated = [
            {
                "article": data["article"],
                "best_chunk": data["best_chunk"],
                "relevance": data["best_relevance"]
            }
            for data in doc_groups.values()
        ]
        
        logger.debug(
            f"Deduplication: {len(results)} raw results -> {len(deduplicated)} unique documents"
        )
        
        return deduplicated
    
    def _deduplicate_across_topics(self, topic_results: List[Dict]) -> List[Dict]:
        """
        Deduplicate topic results when same article appears in multiple topics.
        Keep the version with highest relevance score.
        
        Args:
            topic_results: List of formatted article dictionaries from multiple topics
            
        Returns:
            Deduplicated list with one entry per document ID
        """
        doc_map = {}
        
        for article in topic_results:
            doc_id = article.get("id")
            if not doc_id:
                continue
            
            relevance = article.get("relevance", 0)
            
            # Keep the version with highest relevance
            if doc_id not in doc_map or relevance > doc_map[doc_id].get("relevance", 0):
                doc_map[doc_id] = article
        
        return list(doc_map.values())
    
    async def get_company_data(self, ticker: str) -> Optional[CompanyData]:
        """
        Get company entity ID and name from Knowledge Graph API.
        Uses cache when available.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            CompanyData with entity_id and company_name, or None if not found
        """
        ticker = ticker.upper()
        
        # Check cache first
        cached = self.company_cache.get(ticker)
        if cached:
            return cached
        
        logger.info(f"Fetching company data for {ticker} from Knowledge Graph API")
        
        # Acquire rate limit token
        await self.rate_limiter.acquire()
        
        try:
            session = await self._get_session()
            async with session.post(
                    f"{self.base_url}/knowledge-graph/companies",
                    headers={
                        "X-API-KEY": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "query": ticker,
                        "types": ["PUBLIC"]
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        companies = data.get("results", [])
                        
                        if companies:
                            entity_id = companies[0]["id"]
                            company_name = companies[0]["name"]
                            
                            # Cache the result
                            company_data = self.company_cache.set(
                                ticker, entity_id, company_name
                            )
                            
                            logger.info(
                                f"Found company: {company_name} ({ticker}) -> {entity_id}"
                            )
                            return company_data
                        else:
                            logger.warning(f"No companies found for ticker: {ticker}")
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Knowledge Graph API error: {response.status} - {error_text}"
                        )
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching company data for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching company data for {ticker}: {str(e)}")
        
        return None
    
    async def search_baseline(
        self,
        ticker: str,
        entity_id: str,
        days: int = 7,
        max_chunks: int = 100
    ) -> List[Dict]:
        """
        Perform baseline company search (no sentiment filter).
        This replicates the original simple search behavior.
        
        Args:
            ticker: Stock ticker symbol
            entity_id: Bigdata entity ID
            days: Number of days to look back
            max_chunks: Maximum chunks to return
            
        Returns:
            List of article dictionaries
        """
        logger.info(f"Running baseline search for {ticker}")
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        # Retry logic for rate limit and CloudFront blocking errors
        max_retries = 3
        base_retry_delay = 5.0  # Base delay in seconds
        
        def calculate_wait_time(attempt: int, is_403: bool = False) -> float:
            """
            Calculate wait time with exponential backoff and jitter.
            
            Args:
                attempt: Current attempt number (0-indexed)
                is_403: If True, use longer delays for CloudFront blocks
            
            Returns:
                Wait time in seconds
            """
            # For 403 errors, use longer base delay (CloudFront blocks need more time)
            base = base_retry_delay * (3 if is_403 else 1)
            # Exponential backoff: base * 2^attempt
            exponential_delay = base * (2 ** attempt)
            # Add jitter: random 0-30% of the delay
            jitter = exponential_delay * random.uniform(0, 0.3)
            return exponential_delay + jitter
        
        for attempt in range(max_retries):
            # Acquire rate limit token before each attempt
            await self.rate_limiter.acquire()
            
            try:
                session = await self._get_session()
                async with session.post(
                    f"{self.base_url}/search",
                    headers={
                        "X-API-KEY": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "query": {
                            "text": "earnings financial results stock news",
                            "filters": {
                                "timestamp": {
                                    "start": self._format_timestamp(start_time),
                                    "end": self._format_timestamp(end_time)
                                },
                                "entity": {
                                    "all_of": [entity_id]
                                },
                                "category": {
                                    "mode": "INCLUDE",
                                    "values": [
                                        "expert_interviews",
                                        "filings",
                                        "news",
                                        "podcasts",
                                        "research",
                                        "transcripts"
                                    ]
                                }
                            },
                            "max_chunks": max_chunks
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    # Handle rate limit errors (429) with retry
                    if response.status == 429:
                        error_text = await response.text()
                        if attempt < max_retries - 1:
                            wait_time = calculate_wait_time(attempt, is_403=False)
                            logger.warning(
                                f"Rate limit error (429) for baseline search ({ticker}). "
                                f"Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                            continue  # Retry the loop
                        else:
                            logger.error(
                                f"Rate limit error (429) after {max_retries} attempts: {error_text}"
                            )
                            raise Exception(f"Rate limit exceeded after {max_retries} retries: {error_text}")
                    
                    # Handle CloudFront blocking errors (403) with retry
                    if response.status == 403:
                        error_text = await response.text()
                        if attempt < max_retries - 1:
                            wait_time = calculate_wait_time(attempt, is_403=True)
                            logger.warning(
                                f"CloudFront blocking error (403) for baseline search ({ticker}). "
                                f"Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                            continue  # Retry the loop
                        else:
                            logger.error(
                                f"CloudFront blocking error (403) after {max_retries} attempts"
                            )
                            raise Exception(f"CloudFront blocking after {max_retries} retries")
                    
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", [])
                        
                        logger.debug(f"Baseline search raw results: {len(results)} articles")
                        
                        # Deduplicate by document ID first
                        deduplicated = self._deduplicate_by_document_id(results)
                        
                        articles = []
                        for i, item in enumerate(deduplicated):
                            article = item["article"]
                            best_chunk = item["best_chunk"]
                            relevance = item["relevance"]
                            
                            chunk_text = best_chunk.get("text", "")
                            summary = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                            
                            # Parse timestamp for time_ago
                            timestamp = article.get("timestamp", "")
                            time_ago = self._get_time_ago(timestamp)
                            
                            # Get detections from the chunk
                            detections = best_chunk.get("detections", [])
                            
                            # Try to get category from article, or infer from search query
                            # The API filters by category but may not return it in response
                            doc_type = article.get("category") or article.get("document_type") or article.get("type") or ""
                            # If not found, check if we can infer from the article structure
                            # We search multiple categories, so try to infer from other fields
                            if not doc_type:
                                # Check if it's a transcript based on headline or other indicators
                                headline_lower = article.get("headline", "").lower()
                                if "transcript" in headline_lower or "earnings call" in headline_lower:
                                    doc_type = "transcripts"
                                elif "filing" in headline_lower or "sec" in headline_lower:
                                    doc_type = "filings"
                                elif "podcast" in headline_lower:
                                    doc_type = "podcasts"
                                elif "research" in headline_lower or "report" in headline_lower:
                                    doc_type = "research"
                                else:
                                    # Default to news if we can't determine
                                    doc_type = "news"
                            
                            # Extract and clean source field
                            source_value = article.get("source", {}).get("name", "Unknown") if isinstance(article.get("source"), dict) else str(article.get("source", "Unknown"))
                            # Remove "Unknown," prefix if present
                            if source_value.startswith("Unknown,"):
                                source_value = source_value.replace("Unknown,", "", 1).strip()
                            elif source_value == "Unknown":
                                source_value = ""
                            
                            articles.append({
                                "id": article.get("id", f"baseline_{i}"),
                                "headline": article.get("headline", "No headline"),
                                "timestamp": timestamp,
                                "time_ago": time_ago,
                                "source": source_value,
                                "summary": summary,
                                "full_text": chunk_text,  # Full chunk text for expanded view
                                "document_url": article.get("url"),  # URL to original article (from "url" field)
                                "relevance": relevance,
                                "document_type": doc_type,
                                "search_type": "baseline",
                                "topic": None,
                                "ticker": ticker,
                                "detections": detections,  # Entity detections from the chunk
                            })
                        
                        logger.info(f"Baseline search for {ticker}: {len(articles)} unique articles (from {len(results)} raw results)")
                        return articles
                    
                    else:
                        # Don't log every 404/empty result as error
                        if response.status != 404:
                            error_text = await response.text()
                            # 429 and 403 should have been caught above, but handle them here too as fallback
                            if response.status == 429:
                                if attempt < max_retries - 1:
                                    wait_time = calculate_wait_time(attempt, is_403=False)
                                    logger.warning(
                                        f"Rate limit error (429) for baseline search ({ticker}). "
                                        f"Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                                    )
                                    await asyncio.sleep(wait_time)
                                    continue
                                else:
                                    raise Exception(f"Rate limit exceeded after {max_retries} retries: {error_text}")
                            elif response.status == 403:
                                if attempt < max_retries - 1:
                                    wait_time = calculate_wait_time(attempt, is_403=True)
                                    logger.warning(
                                        f"CloudFront blocking error (403) for baseline search ({ticker}). "
                                        f"Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                                    )
                                    await asyncio.sleep(wait_time)
                                    continue
                                else:
                                    raise Exception(f"CloudFront blocking after {max_retries} retries")
                            logger.error(
                                f"Baseline search error for {ticker}: "
                                f"{response.status} - {error_text[:200]}"  # Truncate long error messages
                            )
                        return []
            
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = calculate_wait_time(attempt, is_403=False)
                    logger.warning(f"Timeout in baseline search for {ticker}, retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                    continue
                logger.error(f"Timeout in baseline search for {ticker} after {max_retries} attempts")
                return []
            except Exception as e:
                error_str = str(e)
                if ("Rate limit" in error_str or "CloudFront" in error_str) and attempt < max_retries - 1:
                    # Re-raise retryable errors to be handled by retry logic
                    raise
                logger.error(f"Error in baseline search for {ticker}: {error_str}")
                return []
        
        # If we get here, all retries failed
        logger.error(f"Failed to complete baseline search for {ticker} after {max_retries} attempts")
        return []
    
    async def search_single_topic(
        self,
        ticker: str,
        entity_id: str,
        company_name: str,
        topic: str,
        topic_index: int,
        days: int = 7,
        max_chunks: int = 10
    ) -> List[Dict]:
        """
        Perform topic-based search with sentiment filtering.
        
        Args:
            ticker: Stock ticker symbol
            entity_id: Bigdata entity ID
            company_name: Company name to format topic
            topic: Topic template string or dict with {topic_name, topic_text}
            topic_index: Index of topic in STANDARD_TOPICS list
            days: Number of days to look back
            max_chunks: Maximum chunks to return
            
        Returns:
            List of article dictionaries tagged with topic
        """
        # Handle both string and dictionary topic formats
        if isinstance(topic, dict):
            topic_name = topic.get("topic_name", f"Topic {topic_index}")
            topic_text = topic.get("topic_text", "")
        else:
            # Legacy string format
            topic_name = f"Topic {topic_index}"
            topic_text = topic
        
        # Format topic with company name
        formatted_topic = topic_text.format(company=company_name)
        
        # Truncate for log readability
        truncated_topic = formatted_topic[:60] + "..." if len(formatted_topic) > 60 else formatted_topic
        logger.debug(f"Searching topic {topic_index} for {ticker}: {truncated_topic}")
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        # Retry logic for rate limit and CloudFront blocking errors
        max_retries = 3
        base_retry_delay = 5.0  # Base delay in seconds
        
        def calculate_wait_time(attempt: int, is_403: bool = False) -> float:
            """
            Calculate wait time with exponential backoff and jitter.
            
            Args:
                attempt: Current attempt number (0-indexed)
                is_403: If True, use longer delays for CloudFront blocks
            
            Returns:
                Wait time in seconds
            """
            # For 403 errors, use longer base delay (CloudFront blocks need more time)
            base = base_retry_delay * (3 if is_403 else 1)
            # Exponential backoff: base * 2^attempt
            exponential_delay = base * (2 ** attempt)
            # Add jitter: random 0-30% of the delay
            jitter = exponential_delay * random.uniform(0, 0.3)
            return exponential_delay + jitter
        
        for attempt in range(max_retries):
            # Acquire rate limit token before each attempt
            await self.rate_limiter.acquire()
            
            try:
                session = await self._get_session()
                async with session.post(
                    f"{self.base_url}/search",
                    headers={
                        "X-API-KEY": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "query": {
                            "text": formatted_topic,
                            "filters": {
                                "timestamp": {
                                    "start": self._format_timestamp(start_time),
                                    "end": self._format_timestamp(end_time)
                                },
                                "entity": {
                                    "all_of": [entity_id]
                                },
                                "category": {
                                    "mode": "INCLUDE",
                                    "values": [
                                        "expert_interviews",
                                        "filings",
                                        "news",
                                        "podcasts",
                                        "research",
                                        "transcripts"
                                    ]
                                },
                                "sentiment": {
                                    "values": ["positive", "negative"]
                                }
                            },
                            "max_chunks": max_chunks
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    # Handle rate limit errors (429) with retry
                    if response.status == 429:
                        error_text = await response.text()
                        if attempt < max_retries - 1:
                            wait_time = calculate_wait_time(attempt, is_403=False)
                            logger.warning(
                                f"Rate limit error (429) for topic search ({ticker}). "
                                f"Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                            continue  # Retry the loop
                        else:
                            logger.error(
                                f"Rate limit error (429) after {max_retries} attempts: {error_text}"
                            )
                            raise Exception(f"Rate limit exceeded after {max_retries} retries: {error_text}")
                    
                    # Handle CloudFront blocking errors (403) with retry
                    if response.status == 403:
                        error_text = await response.text()
                        if attempt < max_retries - 1:
                            wait_time = calculate_wait_time(attempt, is_403=True)
                            logger.warning(
                                f"CloudFront blocking error (403) for topic search ({ticker}). "
                                f"Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                            continue  # Retry the loop
                        else:
                            logger.error(
                                f"CloudFront blocking error (403) after {max_retries} attempts"
                            )
                            raise Exception(f"CloudFront blocking after {max_retries} retries")
                    
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", [])
                        
                        # Deduplicate by document ID first
                        deduplicated = self._deduplicate_by_document_id(results)
                        
                        articles = []
                        for i, item in enumerate(deduplicated):
                            article = item["article"]
                            best_chunk = item["best_chunk"]
                            relevance = item["relevance"]
                            
                            chunk_text = best_chunk.get("text", "")
                            summary = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                            
                            # Parse timestamp for time_ago
                            timestamp = article.get("timestamp", "")
                            time_ago = self._get_time_ago(timestamp)
                            
                            # Get detections from the chunk
                            detections = best_chunk.get("detections", [])
                            
                            # Try to get category from article, or infer from search query
                            # The API filters by category but may not return it in response
                            doc_type = article.get("category") or article.get("document_type") or article.get("type") or ""
                            # If not found, check if we can infer from the article structure
                            # We search multiple categories, so try to infer from other fields
                            if not doc_type:
                                # Check if it's a transcript based on headline or other indicators
                                headline_lower = article.get("headline", "").lower()
                                if "transcript" in headline_lower or "earnings call" in headline_lower:
                                    doc_type = "transcripts"
                                elif "filing" in headline_lower or "sec" in headline_lower:
                                    doc_type = "filings"
                                elif "podcast" in headline_lower:
                                    doc_type = "podcasts"
                                elif "research" in headline_lower or "report" in headline_lower:
                                    doc_type = "research"
                                else:
                                    # Default to news if we can't determine
                                    doc_type = "news"
                            
                            # Extract and clean source field
                            source_value = article.get("source", {}).get("name", "Unknown") if isinstance(article.get("source"), dict) else str(article.get("source", "Unknown"))
                            # Remove "Unknown," prefix if present
                            if source_value.startswith("Unknown,"):
                                source_value = source_value.replace("Unknown,", "", 1).strip()
                            elif source_value == "Unknown":
                                source_value = ""
                            
                            articles.append({
                                "id": article.get("id", f"topic_{topic_index}_{i}"),
                                "headline": article.get("headline", "No headline"),
                                "timestamp": timestamp,
                                "time_ago": time_ago,
                                "source": source_value,
                                "summary": summary,
                                "full_text": chunk_text,  # Full chunk text for expanded view
                                "document_url": article.get("url"),  # URL to original article (from "url" field)
                                "relevance": relevance,
                                "document_type": doc_type,
                                "search_type": "topic",
                                "topic": formatted_topic,
                                "topic_name": topic_name,
                                "topic_index": topic_index,
                                "ticker": ticker,
                                "detections": detections,  # Entity detections from the chunk
                            })
                        
                        if articles:
                            logger.debug(
                                f"Topic {topic_index} for {ticker}: {len(articles)} unique articles (from {len(results)} raw results)"
                            )
                        
                        return articles
                    
                    else:
                        # Don't log every 404/empty result as error
                        if response.status != 404:
                            error_text = await response.text()
                            # 429 and 403 should have been caught above, but handle them here too as fallback
                            if response.status == 429:
                                if attempt < max_retries - 1:
                                    wait_time = calculate_wait_time(attempt, is_403=False)
                                    logger.warning(
                                        f"Rate limit error (429) for topic search ({ticker}). "
                                        f"Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                                    )
                                    await asyncio.sleep(wait_time)
                                    continue
                                else:
                                    raise Exception(f"Rate limit exceeded after {max_retries} retries: {error_text}")
                            elif response.status == 403:
                                if attempt < max_retries - 1:
                                    wait_time = calculate_wait_time(attempt, is_403=True)
                                    logger.warning(
                                        f"CloudFront blocking error (403) for topic search ({ticker}). "
                                        f"Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                                    )
                                    await asyncio.sleep(wait_time)
                                    continue
                                else:
                                    raise Exception(f"CloudFront blocking after {max_retries} retries")
                            logger.warning(
                                f"Topic search error for {ticker} topic {topic_index}: "
                                f"{response.status} - {error_text[:200]}"  # Truncate long error messages
                            )
                        return []
            
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = calculate_wait_time(attempt, is_403=False)
                    logger.warning(f"Timeout in topic {topic_index} search for {ticker}, retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                    continue
                logger.warning(f"Timeout in topic {topic_index} search for {ticker} after {max_retries} attempts")
                return []
            except Exception as e:
                error_str = str(e)
                if ("Rate limit" in error_str or "CloudFront" in error_str) and attempt < max_retries - 1:
                    # Re-raise retryable errors to be handled by retry logic
                    raise
                logger.warning(f"Error in topic {topic_index} search for {ticker}: {error_str}")
                return []
        
        # If we get here, all retries failed
        logger.error(f"Failed to complete topic {topic_index} search for {ticker} after {max_retries} attempts")
        return []
    
    async def search_ticker(
        self,
        ticker: str,
        days: int = 7,
        topics: Optional[List[str]] = None,
        batch_size: int = 50,
        custom_topics: Optional[List[str]] = None,
        min_relevance: float = 0.0,
        query_reformulation: bool = False
    ) -> Dict:
        """
        Search topics for a ticker (sentiment-filtered parallel topic searches).
        
        For baseline/entity-only search, use search_baseline() directly instead.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            topics: List of topic templates (uses STANDARD_TOPICS if None) - DEPRECATED
            batch_size: Number of concurrent requests per batch
            custom_topics: Custom list of topic templates (overrides topics parameter)
            min_relevance: Minimum relevance threshold to filter results
            query_reformulation: If True, use Gemini to generate 3 variations of each topic
            
        Returns:
            Dictionary with topic results and metadata
        """
        start_time = datetime.now()
        ticker = ticker.upper()
        
        mode_str = "TOPIC search with query reformulation" if query_reformulation else "TOPIC search"
        logger.info(f"Starting {mode_str} for {ticker}")
        
        # Get company data
        company_data = await self.get_company_data(ticker)
        if not company_data:
            logger.error(f"Could not resolve company data for {ticker}")
            return {
                "ticker": ticker,
                "error": "Company not found",
                "topic_results": [],
                "total_results": 0,
            }
        
        entity_id = company_data.entity_id
        company_name = company_data.company_name
        
        # Use custom_topics if provided, otherwise fallback to topics or STANDARD_TOPICS
        base_topics = custom_topics or topics or STANDARD_TOPICS
        
        # Generate topic variations if query reformulation is enabled
        search_topics_with_info = []  # List of (topic_text, is_variation, base_topic_index)
        
        if query_reformulation:
            logger.info(f"Generating query variations for {len(base_topics)} topics...")
            variation_start = datetime.now()
            
            # Generate variations for each topic in parallel
            variation_tasks = [
                self.generate_topic_variations(topic, company_name) 
                for topic in base_topics
            ]
            all_variations = await asyncio.gather(*variation_tasks, return_exceptions=True)
            
            # Build search topics list: original + variations
            for i, (base_topic, variations) in enumerate(zip(base_topics, all_variations)):
                # Extract topic_name from base topic (if it's a dict)
                if isinstance(base_topic, dict):
                    topic_name = base_topic.get("topic_name", f"Topic {i}")
                else:
                    topic_name = f"Topic {i}"
                
                # Add original topic
                search_topics_with_info.append((base_topic, False, i))
                
                # Add variations (if generation succeeded)
                # Wrap variations in dict with same topic_name as base topic
                if isinstance(variations, list) and variations:
                    for variation in variations:
                        # Create dict with same topic_name but variation text
                        variation_dict = {
                            "topic_name": topic_name,
                            "topic_text": variation
                        }
                        search_topics_with_info.append((variation_dict, True, i))
                elif isinstance(variations, Exception):
                    logger.warning(f"Failed to generate variations for topic {i}: {variations}")
            
            variation_elapsed = (datetime.now() - variation_start).total_seconds()
            logger.info(
                f"Generated variations in {variation_elapsed:.2f}s: "
                f"{len(base_topics)} original topics → {len(search_topics_with_info)} total queries"
            )
        else:
            # No reformulation: just use original topics
            # Handle both dict and string topic formats
            search_topics_with_info = []
            for i, topic in enumerate(base_topics):
                # If topic is a dict, use it as-is; if string, wrap it
                if isinstance(topic, dict):
                    search_topics_with_info.append((topic, False, i))
                else:
                    # Legacy string format - create a dict with generic topic name
                    search_topics_with_info.append(({"topic_name": f"Topic {i}", "topic_text": topic}, False, i))
        
        total_queries = len(search_topics_with_info)
        
        # Chunk allocation: 100 chunk budget divided across ALL queries (originals + variations)
        TOTAL_CHUNK_BUDGET = 300
        chunks_per_query = max(1, TOTAL_CHUNK_BUDGET // total_queries)  # Min 1 per query
        
        logger.info(
            f"Chunk allocation for {ticker}: "
            f"queries={total_queries} × {chunks_per_query} chunks "
            f"(total={chunks_per_query * total_queries}, budget={TOTAL_CHUNK_BUDGET})"
        )
        
        # Create all topic search tasks with metadata
        topic_tasks_with_info = [
            (
                self.search_single_topic(
                    ticker, entity_id, company_name, topic_text, query_idx, days,
                    max_chunks=chunks_per_query
                ),
                topic_text,
                is_variation,
                base_idx
            )
            for query_idx, (topic_text, is_variation, base_idx) in enumerate(search_topics_with_info)
        ]
        
        # Execute in batches to avoid overwhelming the system
        topic_results = []
        query_result_counts = []  # Track results per query for logging
        
        for i in range(0, len(topic_tasks_with_info), batch_size):
            batch = topic_tasks_with_info[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(topic_tasks_with_info) + batch_size - 1) // batch_size
            
            logger.info(
                f"Executing batch {batch_num}/{total_batches} "
                f"({len(batch)} searches) for {ticker}"
            )
            
            # Extract just the tasks for execution
            batch_tasks = [task for task, _, _, _ in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results and log counts
            for (task, topic_text, is_variation, base_idx), result in zip(batch, batch_results):
                if isinstance(result, list):
                    result_count = len(result)
                    topic_results.extend(result)
                    
                    # Log query and result count
                    query_type = "Variation" if is_variation else "Original"
                    
                    # Handle both dict and string topic formats
                    if isinstance(topic_text, dict):
                        topic_str = topic_text.get("topic_text", str(topic_text))
                    else:
                        topic_str = topic_text
                    
                    # Format with company name if needed
                    formatted_topic = topic_str.format(company=company_name) if "{company}" in topic_str else topic_str
                    
                    # Truncate for log readability
                    truncated_query = formatted_topic[:70] + "..." if len(formatted_topic) > 70 else formatted_topic
                    logger.info(
                        f"  [{query_type}] {truncated_query} → {result_count} results"
                    )
                    
                    query_result_counts.append({
                        'query': formatted_topic,
                        'type': query_type,
                        'base_topic_idx': base_idx,
                        'count': result_count
                    })
                elif isinstance(result, Exception):
                    logger.error(f"Topic search failed: {result}")
        
        # Deduplicate across all topics (same article may appear in multiple topics/variations)
        raw_topic_count = len(topic_results)
        topic_results = self._deduplicate_across_topics(topic_results)
        
        if raw_topic_count > len(topic_results):
            logger.info(
                f"Cross-topic deduplication for {ticker}: "
                f"{raw_topic_count} raw -> {len(topic_results)} unique"
            )
        
        # Filter by relevance if needed
        if min_relevance > 0:
            topic_results = [r for r in topic_results if r.get("relevance", 0) >= min_relevance]
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Log query reformulation effectiveness summary
        if query_reformulation and query_result_counts:
            original_counts = [q['count'] for q in query_result_counts if q['type'] == 'Original']
            variation_counts = [q['count'] for q in query_result_counts if q['type'] == 'Variation']
            
            total_original = sum(original_counts)
            total_variations = sum(variation_counts)
            
            logger.info(
                f"Query Reformulation Summary for {ticker}:"
            )
            logger.info(
                f"  Original queries: {len(original_counts)} queries → {total_original} raw results"
            )
            logger.info(
                f"  Variation queries: {len(variation_counts)} queries → {total_variations} raw results"
            )
            logger.info(
                f"  Total raw results: {total_original + total_variations} → {len(topic_results)} after deduplication"
            )
            logger.info(
                f"  Variations contributed: {total_variations} additional results ({total_variations/(total_original+total_variations)*100:.1f}% of raw results)"
            )
        
        logger.info(
            f"Completed topic search for {ticker}: "
            f"{len(topic_results)} results in {elapsed:.2f}s"
        )
        
        return {
            "ticker": ticker,
            "company_name": company_name,
            "entity_id": entity_id,
            "topic_results": topic_results,
            "total_results": len(topic_results),
            "search_stats": {
                "topics_searched": len(base_topics),
                "total_queries": total_queries,
                "query_reformulation_enabled": query_reformulation,
                "elapsed_seconds": round(elapsed, 2),
                "rate_limiter_stats": self.rate_limiter.get_metrics(),
            }
        }
    
    async def search_multiple_tickers(
        self,
        tickers: List[str],
        days: int = 7,
        topics: Optional[List[str]] = None,
        custom_topics: Optional[List[str]] = None,
        min_relevance: float = 0.0,
        query_reformulation: bool = False
    ) -> List[Dict]:
        """
        Search multiple tickers in parallel (topic searches only).
        
        For baseline searches, call search_baseline() for each ticker instead.
        
        Args:
            tickers: List of stock ticker symbols
            days: Number of days to look back
            topics: List of topic templates (uses STANDARD_TOPICS if None) - DEPRECATED
            custom_topics: Custom list of topic templates (overrides topics parameter)
            min_relevance: Minimum relevance threshold to filter results
            query_reformulation: If True, use Gemini to generate 3 variations of each topic
            
        Returns:
            List of results dictionaries, one per ticker
        """
        mode_str = "with query reformulation" if query_reformulation else "standard"
        logger.info(f"Searching {len(tickers)} tickers (TOPIC mode, {mode_str}): {', '.join(tickers)}")
        
        tasks = [
            self.search_ticker(
                ticker, 
                days, 
                topics=topics,
                custom_topics=custom_topics,
                min_relevance=min_relevance,
                query_reformulation=query_reformulation
            )
            for ticker in tickers
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, dict):
                valid_results.append(result)
            else:
                logger.error(f"Failed to search ticker {tickers[i]}: {result}")
                valid_results.append({
                    "ticker": tickers[i],
                    "error": str(result),
                    "topic_results": [],
                    "total_results": 0,
                })
        
        return valid_results

