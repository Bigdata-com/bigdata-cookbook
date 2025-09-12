"""
Module for executing concurrent and rate-limited searches via
the Bigdata client.

This module defines a `RateLimitedSearchManager` class to manage multiple
search requests efficiently while respecting request-per-minute (RPM) limits
of the Bigdata API.

Copyright (C) 2024, RavenPack | Bigdata.com. All rights reserved.
Author: Shawn Azdam (sazdam@ravenpack.com)
"""
import logging
import threading
import time
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union
from tqdm import tqdm
from bigdata_client import Bigdata
from bigdata_client.daterange import AbsoluteDateRange
from bigdata_client.document import Document
from bigdata_client.models.advanced_search_query import QueryComponent
from bigdata_client.models.search import DocumentType, SortBy


REQUESTS_PER_MINUTE_LIMIT = 100
MAX_WORKERS = 4


class RateLimitedSearchManager:
    """
    Rate-limited search executor for managing concurrent searches via
    the Bigdata SDK.

    This class implements a token bucket algorithm for rate limiting and
    provides thread-safe access to the search functionality.
    """

    def __init__(self,
                 bigdata: Bigdata,
                 rpm: int = REQUESTS_PER_MINUTE_LIMIT,
                 bucket_size: int = None):
        """
        Initialize the rate-limited search manager.

        :param bigdata:
            The Bigdata SDK client instance used for executing searches.
        :param rpm:
            Queries per minute limit. Defaults to 100.
        :param bucket_size:
            Size of the token bucket. Defaults to the value of `rpm`.
        """
        self.bigdata = bigdata
        self.rpm = rpm
        self.bucket_size = bucket_size or rpm
        self.tokens = self.bucket_size
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def _refill_tokens(self):
        """
        Refill tokens based on elapsed time since the last refill.
        Tokens are replenished at a rate proportional to the RPM limit.
        """
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = int(elapsed * (self.rpm / 60.0))

        if new_tokens > 0:
            with self._lock:
                self.tokens = min(self.bucket_size, self.tokens + new_tokens)
                self.last_refill = now

    def _acquire_token(self, timeout: float = None) -> bool:
        """
        Attempt to acquire a token for executing a search request.

        :param timeout:
            Maximum time (in seconds) to wait for a token.
            Defaults to no timeout.
        :return:
            True if a token is acquired, False if timed out.
        """
        start = time.time()
        while True:
            self._refill_tokens()

            with self._lock:
                if self.tokens > 0:
                    self.tokens -= 1
                    return True

            if timeout and (time.time() - start) > timeout:
                return False

            time.sleep(0.1)  # Prevent tight looping

    def _search(
            self,
            query: QueryComponent,
            date_range: Optional[AbsoluteDateRange] = None,
            sortby: SortBy = SortBy.RELEVANCE,
            scope: DocumentType = DocumentType.ALL,
            limit: int = 10,
            timeout: float = None
    ) -> Optional[List[Document]]:
        """
        Execute a single search with rate limiting.

        :param query:
            The search query to execute.
        :param date_range:
            A date range filter for the search results.
        :param sortby:
            The sorting criterion for the search results.
            Defaults to SortBy.RELEVANCE.
        :param scope:
            The scope of the documents to include.
            Defaults to DocumentType.ALL.
        :param limit:
            The maximum number of documents to return.
            Defaults to 10.
        :param timeout:
            The maximum time (in seconds) to wait for a token.
        :return:
            A list of search results, or None if a rate limit timeout occurred.
        """
        if not self._acquire_token(timeout):
            logging.warning('Timed out attempting to acquire rate limit token')
            return None

        try:
            results = self.bigdata.search.new(
                query=query,
                date_range=date_range,
                sortby=sortby,
                scope=scope
            ).run(limit=limit)
            return results
        except Exception as e:
            logging.error(f'Search error: {e}')
            return None

    def concurrent_search(
            self,
            queries: List[QueryComponent],
            date_range: Union[AbsoluteDateRange,
                              List[AbsoluteDateRange], None] = None,
            sortby: SortBy = SortBy.RELEVANCE,
            scope: DocumentType = DocumentType.ALL,
            limit: int = 10,
            max_workers: int = MAX_WORKERS,
            timeout: float = None
    ) -> List[List[Document]]:
        """
        Execute multiple searches concurrently while respecting rate limits.
        The order of results is preserved based on the input queries.

        :param queries:
            A list of search queries to execute.
        :param date_range:
            A date range filter for all searches.
        :param sortby:
            The sorting criterion for the search results.
            Defaults to SortBy.RELEVANCE.
        :param scope:
            The scope of the documents to include.
            Defaults to DocumentType.ALL.
        :param limit:
            The maximum number of documents to return per query.
            Defaults to 10.
        :param max_workers:
            The maximum number of concurrent threads.
            Defaults to MAX_WORKERS.
        :param timeout:
            The maximum time (in seconds) to wait for a token
            per request.
        :return:
            A list of lists, where each inner list contains results
            for the corresponding request.
        """
        if not isinstance(date_range, list):
            date_range = [date_range]

        tasks = list(itertools.product(queries, date_range))
        results = [[] for _ in range(len(tasks))]  # Preserve order

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._search,
                    query=query,
                    date_range=date_range,
                    sortby=sortby,
                    scope=scope,
                    limit=limit,
                    timeout=timeout
                ): idx
                for idx, (query, date_range) in enumerate(tasks)
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Querying Bigdata..."):
                idx = futures[future]
                try:
                    if search_results := future.result():
                        results[idx] = search_results
                except Exception as e:
                    logging.error(f'Error in search {idx}: {e}')

        return results


def run_search(
        bigdata: Bigdata,
        queries: List[QueryComponent],
        date_range: Union[AbsoluteDateRange,
                          List[AbsoluteDateRange], None] = None,
        sortby: SortBy = SortBy.RELEVANCE,
        scope: DocumentType = DocumentType.ALL,
        limit: int = 10,
) -> List[List[Document]]:
    """
    Convenience function to execute multiple searches concurrently
    with rate limiting.
    This function creates an instance of `RateLimitedSearchManager`
    and utilizes it to run searches for all provided queries.

    :param bigdata:
        An instance of the Bigdata client used to execute the searches.
    :param queries:
        A list of QueryComponent objects, each representing a query to execute.
    :param date_range:
        A date range filter for the search results.
    :param sortby:
        The sorting criterion for the search results.
        Defaults to SortBy.RELEVANCE.
    :param scope:
        The scope of the documents to include.
        Defaults to DocumentType.ALL.
    :param limit:
        The maximum number of documents to return per query.
        Defaults to 10.
    :return:
        A list of lists, where each inner list contains results
        for the corresponding query.
    """
    manager = RateLimitedSearchManager(bigdata)
    return manager.concurrent_search(
        queries=queries,
        date_range=date_range,
        sortby=sortby,
        scope=scope,
        limit=limit,
    )
