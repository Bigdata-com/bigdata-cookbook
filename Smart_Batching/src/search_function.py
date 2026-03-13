"""
Smart Batching Search Function

This module provides a two-step system for efficient semantic search:
1. Planning: Organize search using smart batching and return total expected chunks
2. Execution: Perform search with proportional sampling to preserve distribution
"""

import csv
import json
import logging
import os
import sys
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Configure logging
# Disable logging exceptions to prevent ZMQ socket errors in Jupyter notebooks
# when multiple threads log simultaneously
logging.raiseExceptions = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# Enable debug logging for entity extraction issues
# Set to DEBUG to see response structure details
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_API_BASE_URL = os.getenv("BIGDATA_API_BASE_URL", "https://api.stg.bigdata.com")
SEARCH_ENDPOINT = "/v1/search"
COMENTION_ENDPOINT = "/v1/search/co-mentions/entities"
MAX_ENTITIES_IN_ANY_OF = 500
MAX_CHUNKS_PER_BASKET = 1000
DEFAULT_REQUESTS_PER_MINUTE = 100
DEFAULT_MAX_WORKERS = 40
DEFAULT_WINDOW_SIZE_SECONDS = 5


class SlidingWindowRateLimiter:
    """
    Thread-safe sliding window rate limiter with burst prevention.
    Uses 5-second windows to prevent request bursts.
    """
    
    def __init__(self, max_requests: int = DEFAULT_REQUESTS_PER_MINUTE, period_seconds: int = 60, window_size: int = DEFAULT_WINDOW_SIZE_SECONDS):
        """
        Initialize sliding window rate limiter.
        
        Args:
            max_requests: Maximum requests allowed per period (default: 100)
            period_seconds: Time period in seconds (default: 60 for per-minute)
            window_size: Size of sliding window in seconds (default: 5 for burst prevention)
        """
        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self.window_size = window_size
        self.max_per_window = int(max_requests * window_size / period_seconds)
        
        # Track requests in sliding windows
        self.request_times = deque()
        self._lock = threading.Lock()
        
        # Metrics tracking
        self.total_requests = 0
        self.total_wait_time = 0.0
        self.throttle_events = 0
        self.rate_limit_warnings = 0
    
    def _clean_old_requests(self, current_time: float) -> None:
        """Remove requests outside the sliding window."""
        cutoff_time = current_time - self.period_seconds
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()
    
    def _requests_in_window(self, current_time: float) -> int:
        """Count requests in the current window."""
        window_start = current_time - self.window_size
        return sum(1 for t in self.request_times if t >= window_start)
    
    def acquire(self, timeout: float = 60.0) -> float:
        """
        Acquire permission to make a request, blocking if necessary.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            Wait time in seconds (0 if no wait was needed)
        """
        start_time = time.time()
        total_wait = 0.0
        
        while True:
            with self._lock:
                current_time = time.time()
                self._clean_old_requests(current_time)
                
                # Check if we can make a request
                requests_in_period = len(self.request_times)
                requests_in_window = self._requests_in_window(current_time)
                
                if requests_in_period < self.max_requests and requests_in_window < self.max_per_window:
                    # Permission granted
                    self.request_times.append(current_time)
                    self.total_requests += 1
                    self.total_wait_time += total_wait
                    return total_wait
                
                # Need to wait
                self.throttle_events += 1
                if requests_in_window >= self.max_per_window:
                    # Wait for window to clear
                    wait_time = self.window_size / 10  # Small incremental wait
                else:
                    # Wait for period to clear
                    oldest_request = self.request_times[0]
                    wait_time = (oldest_request + self.period_seconds - current_time) + 0.1
            
            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError("Rate limiter timeout exceeded")
            
            # Wait outside the lock
            time.sleep(min(wait_time, 1.0))
            total_wait += wait_time
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "total_wait_time_seconds": round(self.total_wait_time, 2),
                "throttle_events": self.throttle_events,
                "rate_limit_warnings": self.rate_limit_warnings,
                "current_requests_in_period": len(self.request_times),
                "max_requests_per_period": self.max_requests,
                "max_requests_per_window": self.max_per_window,
            }


class ConcurrencySemaphore:
    """
    Semaphore to limit simultaneous connections.
    Prevents connection spikes that can trigger security filters.
    """
    
    def __init__(self, max_concurrent: int = DEFAULT_MAX_WORKERS):
        """
        Initialize concurrency semaphore.
        
        Args:
            max_concurrent: Maximum simultaneous connections (default: 40)
        """
        self.semaphore = threading.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.active_count = 0
        self._lock = threading.Lock()
        
        # Metrics
        self.total_acquisitions = 0
        self.peak_concurrent = 0
    
    def __enter__(self):
        """Acquire semaphore."""
        self.semaphore.acquire()
        with self._lock:
            self.active_count += 1
            self.total_acquisitions += 1
            self.peak_concurrent = max(self.peak_concurrent, self.active_count)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release semaphore."""
        with self._lock:
            self.active_count -= 1
        self.semaphore.release()
        return False
    
    def get_stats(self) -> dict:
        """Get semaphore statistics."""
        with self._lock:
            return {
                "active_connections": self.active_count,
                "max_concurrent": self.max_concurrent,
                "total_acquisitions": self.total_acquisitions,
                "peak_concurrent": self.peak_concurrent,
            }


def load_universe_from_csv(csv_path: str, id_column: str = 'id') -> List[str]:
    """
    Load entity IDs from CSV file.
    
    Supports two formats:
    1. CSV with header row containing 'id' column (e.g., id,name)
    2. Simple CSV with one entity ID per line (no header)
    
    Args:
        csv_path: Path to CSV file
        id_column: Name of the column containing entity IDs (default: 'id')
        
    Returns:
        List of entity IDs
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV file is empty or invalid
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    companies = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
        
        if first_row is None:
            raise ValueError(f"CSV file is empty: {csv_path}")
        
        # Check if first row is a header containing the id_column
        first_row_lower = [col.strip().lower() for col in first_row]
        
        if id_column.lower() in first_row_lower:
            # CSV has header - find the index of the id column
            id_idx = first_row_lower.index(id_column.lower())
            for row in reader:
                if row and len(row) > id_idx and row[id_idx].strip():
                    companies.append(row[id_idx].strip())
        else:
            # No header - treat first row as data (first column contains IDs)
            if first_row and first_row[0].strip():
                companies.append(first_row[0].strip())
            for row in reader:
                if row and row[0].strip():
                    companies.append(row[0].strip())
    
    if not companies:
        raise ValueError(f"CSV file contains no valid entity IDs: {csv_path}")
    
    logger.info(f"Loaded {len(companies)} entity IDs from {csv_path}")
    return companies


def validate_date_format(date_str: str) -> bool:
    """Validate date format is YYYY-MM-DD."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def validate_date_range(start_date: str, end_date: str) -> None:
    """
    Validate date range.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Raises:
        ValueError: If dates are invalid or start > end
    """
    if not validate_date_format(start_date):
        raise ValueError(f"Invalid start date format: {start_date}. Expected YYYY-MM-DD")
    
    if not validate_date_format(end_date):
        raise ValueError(f"Invalid end date format: {end_date}. Expected YYYY-MM-DD")
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    if start > end:
        raise ValueError(f"Start date ({start_date}) must be before or equal to end date ({end_date})")


def validate_chunk_percentage(chunk_percentage: float) -> None:
    """
    Validate chunk percentage is between 0.0 and 1.0.
    
    Args:
        chunk_percentage: Percentage value to validate
        
    Raises:
        ValueError: If percentage is out of bounds
    """
    if not isinstance(chunk_percentage, (int, float)):
        raise ValueError(f"chunk_percentage must be a number, got {type(chunk_percentage)}")
    
    if chunk_percentage < 0.0 or chunk_percentage > 1.0:
        raise ValueError(f"chunk_percentage must be between 0.0 and 1.0, got {chunk_percentage}")


def date_to_iso(date_str: str, is_start: bool = True) -> str:
    """
    Convert YYYY-MM-DD date to ISO format with timezone.
    
    Args:
        date_str: Date in YYYY-MM-DD format
        is_start: True for start date (00:00:00), False for end date (23:59:59)
        
    Returns:
        ISO format date string
    """
    if is_start:
        return f"{date_str}T00:00:00Z"
    else:
        return f"{date_str}T23:59:59Z"


def get_smart_batching_planner(api_key: Optional[str] = None, api_base_url: Optional[str] = None):
    """
    Get or create SmartBatchingPlanner instance.
    Tries to import from existing module, otherwise creates a simplified version.
    
    Args:
        api_key: API key for Bigdata API
        api_base_url: API base URL (if provided, will set in environment for config)
        
    Returns:
        SmartBatchingPlanner instance
    """
    try:
        # Try to import from existing smart_batching module
        sys.path.insert(0, str(Path(__file__).parent))
        
        # If api_base_url is provided, set it in environment so config picks it up
        if api_base_url:
            original_url = os.environ.get("BIGDATA_API_BASE_URL")
            os.environ["BIGDATA_API_BASE_URL"] = api_base_url
        
        try:
            from .smart_batching import SmartBatchingPlanner
            planner = SmartBatchingPlanner(api_key=api_key, api_base_url=api_base_url)
            return planner
        finally:
            # Restore original environment variable if we changed it
            if api_base_url:
                if original_url:
                    os.environ["BIGDATA_API_BASE_URL"] = original_url
                else:
                    os.environ.pop("BIGDATA_API_BASE_URL", None)
    except ImportError as e:
        # If not available, we'll need to implement a simplified version inline
        logger.warning(f"smart_batching module not found ({e}), using simplified planner")
        return None
    except Exception as e:
        logger.warning(f"Error initializing SmartBatchingPlanner ({e}), using simplified planner")
        return None


def plan_search(
    text: str,
    universe_csv_path: str,
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    volume_query_mode: str = "three_pass",
    max_iterations_per_batch: int = 10,
    apply_volume_splits: bool = True
) -> Dict:
    """
    Plan a search using smart batching approach.
    
    This function organizes the search by:
    1. Loading universe of companies from CSV
    2. Getting comention volumes for all companies
    3. Creating optimized baskets
    4. Building complete query structures with text embedded
    
    Args:
        text: Search query text
        universe_csv_path: Path to CSV file with entity IDs (one per line)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        api_key: API key (defaults to BIGDATA_API_KEY env var)
        api_base_url: API base URL (defaults to env var or DEFAULT_API_BASE_URL)
        volume_query_mode: Method for querying volumes. Options:
            - "three_pass": Original 3-pass approach (query all, then verify twice)
            - "iterative": Per-batch iterative approach (query batch, remove found, repeat until empty)
        max_iterations_per_batch: Max iterations per batch when using "iterative" mode (default 10)
        apply_volume_splits: If True (default), use volume time series to split periods per company.
                If False, use time-based granularity and estimated sub-period volumes only.
    Returns:
        Dict with planning results:
        - total_expected_chunks: Total chunks expected
        - baskets: List of basket configs with complete query structures
        - planning_metadata: Additional metadata
    """
    # Input validation
    if not text or not text.strip():
        raise ValueError("text cannot be empty")
    
    validate_date_range(start_date, end_date)
    
    api_key = api_key or os.getenv("BIGDATA_API_KEY")
    if not api_key:
        raise ValueError("API key must be provided or set in BIGDATA_API_KEY environment variable")
    
    api_base_url = api_base_url or os.getenv("BIGDATA_API_BASE_URL", DEFAULT_API_BASE_URL)
    
    logger.info(f"Planning search for text: '{text}'")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Load universe
    companies = load_universe_from_csv(universe_csv_path)
    logger.info(f"Loaded {len(companies)} companies from universe")
    
    # Try to use SmartBatchingPlanner if available
    planner = get_smart_batching_planner(api_key=api_key, api_base_url=api_base_url)
    
    if planner is None:
        # Simplified implementation without SmartBatchingPlanner
        # This creates a basic plan with all companies in one basket per period
        logger.warning("Using simplified planning (SmartBatchingPlanner not available)")
        
        # Create a single basket for the entire period
        start_iso = date_to_iso(start_date, is_start=True)
        end_iso = date_to_iso(end_date, is_start=False)
        
        # Build query structure matching benchmark format
        query = {
            "auto_enrich_filters": False,
            "text": text,
            "filters": {
                "timestamp": {
                    "start": start_iso,
                    "end": end_iso
                },
                "entity": {
                    "any_of": companies[:MAX_ENTITIES_IN_ANY_OF],# Limit to API max
                "search_in": "BODY"  
                }
            },
            "ranking_params": {
                "source_boost": 1,
                "freshness_boost": 0,
                "reranker": {
                    "enabled": False
                }
            },
            "max_chunks": MAX_CHUNKS_PER_BASKET
        }
        
        # Estimate expected chunks (simplified - would need comention API for accurate)
        # For now, we'll use a placeholder that can be updated
        estimated_chunks = len(companies) * 10  # Rough estimate
        
        basket = {
            "basket_id": "basket_0",
            "companies": companies[:MAX_ENTITIES_IN_ANY_OF],
            "expected_chunks": estimated_chunks,
            "period_start": start_date,
            "period_end": end_date,
            "query": query
        }
        
        plan = {
            "total_expected_chunks": estimated_chunks,
            "baskets": [basket],
            "planning_metadata": {
                "total_companies": len(companies),
                "companies_processed": len(companies[:MAX_ENTITIES_IN_ANY_OF]),
                "periods": 1,
                "uses_smart_batching": False
            }
        }
    else:
        # Use SmartBatchingPlanner with adaptive time splitting (plan_all_periods)
        # so each query stays <= 1000 chunks by splitting the time window when needed
        logger.info("Using SmartBatchingPlanner with adaptive time-window splitting")
        
        report = planner.plan_all_periods(
            topic=text,
            start_date=start_date,
            end_date=end_date,
            volume_query_mode=volume_query_mode,
            max_iterations_per_batch=max_iterations_per_batch,
            universe_csv_path=universe_csv_path,
            apply_volume_splits=apply_volume_splits,
        )
        
        smart = report.get("configurations", {}).get("smart", {})
        period_details = smart.get("period_details", [])
        
        baskets = []
        all_companies_in_plan = set()
        total_expected = 0
        
        for period_detail in period_details:
            period_start = period_detail.get("start_date", start_date)
            period_end = period_detail.get("end_date", end_date)
            start_iso = date_to_iso(period_start, is_start=True)
            end_iso = date_to_iso(period_end, is_start=False)
            
            for basket_raw in period_detail.get("baskets", []):
                expected_chunks = basket_raw.get("total_chunks", 0)
                total_expected += expected_chunks
                all_companies_in_plan.update(basket_raw.get("companies", []))
                
                # Cap at API limit: each sub-period query can request at most 1000 chunks
                max_chunks_for_query = (
                    min(expected_chunks, MAX_CHUNKS_PER_BASKET) if expected_chunks > 0 else 100
                )
                
                query = {
                    "auto_enrich_filters": False,
                    "text": text,
                    "filters": {
                        "timestamp": {"start": start_iso, "end": end_iso},
                        "entity": {
                            "any_of": basket_raw["companies"],
                            "search_in": "BODY"
                        }
                    },
                    "ranking_params": {
                        "source_boost": 1,
                        "freshness_boost": 0,
                        "reranker": {"enabled": False}
                    },
                    "max_chunks": max_chunks_for_query
                }
                basket = {
                    "basket_id": basket_raw["basket_id"],
                    "companies": basket_raw["companies"],
                    "expected_chunks": expected_chunks,
                    "period_start": period_start,
                    "period_end": period_end,
                    "query": query
                }
                baskets.append(basket)
        
        # Add very_low baskets (companies with 0 chunks in full period) for full window
        # Continue global basket counter from where plan_all_periods left off
        very_low_companies = [c for c in companies if c not in all_companies_in_plan]
        if very_low_companies:
            global_basket_counter = len(baskets)  # Continue numbering
            very_low_baskets_raw = planner.create_baskets(
                {}, max_chunks=MAX_CHUNKS_PER_BASKET, very_low_companies=very_low_companies
            )
            start_iso = date_to_iso(start_date, is_start=True)
            end_iso = date_to_iso(end_date, is_start=False)
            for basket_raw in very_low_baskets_raw:
                query = {
                    "auto_enrich_filters": False,
                    "text": text,
                    "filters": {
                        "timestamp": {"start": start_iso, "end": end_iso},
                        "entity": {
                            "any_of": basket_raw["companies"],
                            "search_in": "BODY"
                        }
                    },
                    "ranking_params": {
                        "source_boost": 1,
                        "freshness_boost": 0,
                        "reranker": {"enabled": False}
                    },
                    "max_chunks": 100
                }
                basket = {
                    "basket_id": f"very_low_basket_{global_basket_counter}",
                    "companies": basket_raw["companies"],
                    "expected_chunks": 0,
                    "period_start": start_date,
                    "period_end": end_date,
                    "query": query
                }
                global_basket_counter += 1
                baskets.append(basket)
        
        plan = {
            "total_expected_chunks": total_expected,
            "baskets": baskets,
            "planning_metadata": {
                "total_companies": len(companies),
                "companies_with_chunks": len(all_companies_in_plan),
                "companies_very_low": len(very_low_companies),
                "comention_queries": smart.get("comention_queries", 0),
                "baskets_created": len(baskets),
                "uses_smart_batching": True,
                "adaptive_time_splitting": True,
            }
        }
    
    logger.info(f"Planning complete: {plan['total_expected_chunks']:,} expected chunks in {len(plan['baskets'])} baskets")
    return plan


def make_search_request(
    query: Dict,
    api_key: str,
    api_base_url: str,
    rate_limiter: SlidingWindowRateLimiter,
    concurrency_limiter: ConcurrencySemaphore,
    max_retries: int = 5
) -> Optional[Dict]:
    """
    Make a search API request with rate limiting and retry logic.
    
    Args:
        query: Complete query structure
        api_key: API key
        api_base_url: API base URL
        rate_limiter: Rate limiter instance
        concurrency_limiter: Concurrency limiter instance
        max_retries: Maximum retry attempts
        
    Returns:
        API response JSON or None if failed
    """
    url = f"{api_base_url}{SEARCH_ENDPOINT}"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {"query": query}
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            with concurrency_limiter:
                rate_limiter.acquire(timeout=120.0)
                
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(1.0)
                        continue
                elif response.status_code == 403:
                    retry_count += 1
                    if retry_count < max_retries:
                        delay = min(1.0 * (1.5 ** (retry_count // 10)), 5.0)
                        time.sleep(delay)
                        continue
                else:
                    logger.error(f"API error {response.status_code}: {response.text[:200]}")
                    return None
                    
        except requests.exceptions.Timeout:
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(1.0)
                continue
            return None
        except Exception as e:
            logger.error(f"Request error: {e}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(1.0)
                continue
            return None
    
    logger.warning(f"Failed after {max_retries} retries")
    return None


def deduplicate_documents(documents: List[Dict]) -> List[Dict]:
    """
    Deduplicate documents, merging chunks from duplicate documents.
    
    When the same document appears multiple times (e.g., from different baskets
    searching for different entities), this function merges their chunks to
    ensure no chunk is lost.
    
    Args:
        documents: List of document dictionaries (each containing a chunks array)
        
    Returns:
        List of unique documents with merged chunks
    """
    doc_map = {}  # doc_id -> document with merged chunks
    docs_without_id = []
    
    for doc in documents:
        doc_id = doc.get("id", "")
        if not doc_id:
            # Keep documents without ID as-is
            docs_without_id.append(doc)
        elif doc_id not in doc_map:
            # First occurrence - store a copy with chunks as a list
            doc_copy = doc.copy()
            doc_copy["chunks"] = list(doc.get("chunks", []))
            doc_map[doc_id] = doc_copy
        else:
            # Duplicate document - merge chunks, avoiding duplicates
            existing_doc = doc_map[doc_id]
            existing_chunks = existing_doc.get("chunks", [])
            
            # Build set of existing chunk identifiers (using chunk_index or text hash as fallback)
            existing_chunk_ids = set()
            for chunk in existing_chunks:
                chunk_id = chunk.get("chunk_index")
                if chunk_id is None:
                    # Fallback: use hash of text content
                    chunk_id = hash(chunk.get("text", ""))
                existing_chunk_ids.add(chunk_id)
            
            # Add new chunks that don't already exist
            for chunk in doc.get("chunks", []):
                chunk_id = chunk.get("chunk_index")
                if chunk_id is None:
                    chunk_id = hash(chunk.get("text", ""))
                if chunk_id not in existing_chunk_ids:
                    existing_chunks.append(chunk)
                    existing_chunk_ids.add(chunk_id)
    
    original_count = len(documents)
    unique_documents = list(doc_map.values()) + docs_without_id
    logger.info(f"Deduplicated: {len(unique_documents)} unique documents from {original_count} total (chunks merged)")
    
    return unique_documents


def execute_search(
    search_plan: Dict,
    chunk_percentage: float,
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    max_workers: int = DEFAULT_MAX_WORKERS
) -> List[Dict]:
    """
    Execute search with proportional sampling.

    Args:
        search_plan: Planning result from plan_search()
        chunk_percentage: Percentage of total chunks to retrieve (0.0 to 1.0)
        requests_per_minute: Rate limit for API requests
        api_key: API key (defaults to BIGDATA_API_KEY env var)
        api_base_url: API base URL
        max_workers: Number of parallel workers

    Returns:
        List of document dictionaries (each containing a chunks array).
        Note: Use deduplicate_documents() after this to merge duplicate documents.
    """
    # Input validation
    validate_chunk_percentage(chunk_percentage)
    
    if "baskets" not in search_plan or not search_plan["baskets"]:
        raise ValueError("Invalid search plan: missing or empty baskets")
    
    api_key = api_key or os.getenv("BIGDATA_API_KEY")
    if not api_key:
        raise ValueError("API key must be provided or set in BIGDATA_API_KEY environment variable")
    
    api_base_url = api_base_url or os.getenv("BIGDATA_API_BASE_URL", DEFAULT_API_BASE_URL)
    
    total_expected = search_plan.get('total_expected_chunks', 0)
    target_chunks = int(total_expected * chunk_percentage)
    logger.info(f"Executing search with {chunk_percentage*100:.1f}% of chunks")
    logger.info(f"Total maximum expected chunks: {target_chunks:,}")
    
    # Initialize rate limiter and concurrency limiter
    rate_limiter = SlidingWindowRateLimiter(max_requests=requests_per_minute)
    concurrency_limiter = ConcurrencySemaphore(max_concurrent=max_workers)
    
    # Calculate proportional chunks per basket
    baskets_to_search = []
    for basket in search_plan["baskets"]:
        expected = basket.get("expected_chunks", 0)
        
        if expected > 0:
            # Calculate proportional chunks, capped at API limit (1000)
            proportional_chunks = min(max(1, int(expected * chunk_percentage)), MAX_CHUNKS_PER_BASKET)
        else:
            # For very_low baskets (expected_chunks == 0), use the max_chunks from query or default
            # These baskets might still return some results from the semantic search
            proportional_chunks = basket.get("query", {}).get("max_chunks", 100)
        
        basket_query = basket["query"].copy()
        basket_query["max_chunks"] = proportional_chunks
        baskets_to_search.append({
            "basket_id": basket["basket_id"],
            "query": basket_query,
            "expected_chunks": expected,
            "proportional_chunks": proportional_chunks,
            "companies": basket.get("companies", [])  # Include companies for entity extraction
        })
    
    logger.info(f"Searching {len(baskets_to_search)} baskets")

    # Execute searches in parallel
    all_documents = []
    failed_baskets = []
    start_time = time.time()
    
    def search_basket(basket_info):
        """Search a single basket and return documents with enriched chunks."""
        try:
            # Get entity IDs from the basket (companies we're searching for)
            # This is the most reliable source since we know which companies are in each basket
            basket_entity_ids = basket_info.get("companies", [])
            if not basket_entity_ids:
                logger.warning(f"Basket {basket_info.get('basket_id')} has no companies list")
            
            response = make_search_request(
                query=basket_info["query"],
                api_key=api_key,
                api_base_url=api_base_url,
                rate_limiter=rate_limiter,
                concurrency_limiter=concurrency_limiter
            )
            
            if response and "results" in response:
                # results is an array of documents, each with a chunks array
                documents = response["results"]
                total_chunks = 0
                
                # Also try to get entity IDs from the query filters as backup
                query_entity_ids = []
                query_filters = basket_info.get("query", {}).get("filters", {})
                entity_filter = query_filters.get("entity", {})
                if entity_filter:
                    # Get entities from any_of, all_of, or none_of
                    query_entity_ids.extend(entity_filter.get("any_of", []))
                    query_entity_ids.extend(entity_filter.get("all_of", []))
                
                # Use basket companies as primary source, query filters as backup
                primary_entity_ids = basket_entity_ids if basket_entity_ids else query_entity_ids
                
                # Debug: Log structure of first response to understand format
                first_doc_logged = False
                
                for document in documents:
                    document_chunks = document.get("chunks", [])
                    total_chunks += len(document_chunks)
                    # Get document-level entity information (try both snake_case and camelCase)
                    reporting_entities = document.get("reporting_entities") or document.get("reportingEntities") or []
                    
                    # Debug: Log first document and chunk structure
                    if not first_doc_logged and document_chunks:
                        logger.debug(f"Document keys: {list(document.keys())}")
                        logger.debug(f"First chunk keys: {list(document_chunks[0].keys())}")
                        logger.debug(f"First chunk detections: {document_chunks[0].get('detections')}")
                        logger.debug(f"First chunk entities: {document_chunks[0].get('entities')}")
                        first_doc_logged = True
                    
                    # Enrich each chunk with entity information
                    for chunk in document_chunks:
                        # Extract entity information from chunk
                        # Try multiple possible field names and structures
                        entity_ids = []
                        
                        # Try 'detections' array (most common in REST API)
                        detections = chunk.get("detections", [])
                        if detections:
                            for d in detections:
                                # Detection can be: {"id": "...", ...} or just a string ID
                                entity_id = d.get("id") if isinstance(d, dict) else (d if isinstance(d, str) else None)
                                if entity_id:
                                    entity_ids.append(entity_id)
                        
                        # Try 'entities' array (alternative name)
                        if not entity_ids:
                            entities = chunk.get("entities", [])
                            if entities:
                                for e in entities:
                                    # Entity can be: {"id": "...", "key": "...", ...} or just a string
                                    entity_id = e.get("id") or e.get("key") if isinstance(e, dict) else (e if isinstance(e, str) else None)
                                    if entity_id:
                                        entity_ids.append(entity_id)
                        
                        # Try document-level reporting_entities
                        if not entity_ids and reporting_entities:
                            if isinstance(reporting_entities, list):
                                entity_ids = [e for e in reporting_entities if e]
                            else:
                                entity_ids = [reporting_entities] if reporting_entities else []
                        
                        # ALWAYS use entities from basket/query as fallback (and primary source if nothing else found)
                        # Since we're filtering by these entities, ALL chunks returned are guaranteed to be related to them
                        # This is the most reliable source when entity detections aren't in the API response
                        if not entity_ids:
                            if primary_entity_ids:
                                entity_ids = primary_entity_ids.copy()
                            else:
                                # This shouldn't happen - we're filtering by entities, so we should have them
                                logger.warning(f"No entities available for chunk from basket {basket_info.get('basket_id')}. "
                                             f"Basket companies: {basket_entity_ids}, Query entities: {query_entity_ids}")
                        
                        # Store entity information (always set, even if empty)
                        chunk["entity_ids"] = entity_ids
                        chunk["primary_entity_id"] = entity_ids[0] if entity_ids else None
                
                logger.info(f"Basket {basket_info['basket_id']}: Retrieved {len(documents)} documents with {total_chunks} chunks")
                return documents, basket_info["basket_id"]
            else:
                logger.warning(f"Basket {basket_info['basket_id']}: No results or error")
                return [], basket_info["basket_id"]
        except Exception as e:
            logger.error(f"Basket {basket_info['basket_id']}: Error - {e}")
            return [], basket_info["basket_id"]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(search_basket, basket_info): basket_info
            for basket_info in baskets_to_search
        }
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            basket_info = futures[future]
            try:
                documents, basket_id = future.result()
                if documents:
                    all_documents.extend(documents)
                else:
                    failed_baskets.append(basket_id)
            except Exception as e:
                logger.error(f"Error processing basket {basket_info['basket_id']}: {e}")
                failed_baskets.append(basket_info["basket_id"])
            
            if completed % 10 == 0:
                logger.info(f"Progress: {completed}/{len(baskets_to_search)} baskets completed")
    
    elapsed = time.time() - start_time
    
    # Count total chunks across all documents
    total_chunks = sum(len(doc.get("chunks", [])) for doc in all_documents)
    
    logger.info(f"Search complete: {len(all_documents)} documents with {total_chunks} chunks retrieved in {elapsed:.2f}s")
    if failed_baskets:
        logger.warning(f"Failed baskets: {len(failed_baskets)}")
    
    return all_documents


def save_plan(plan: Dict, file_path: str) -> None:
    """Save search plan to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2)
    logger.info(f"Plan saved to {file_path}")


def load_plan(file_path: str) -> Dict:
    """Load search plan from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        plan = json.load(f)
    logger.info(f"Plan loaded from {file_path}")
    return plan
