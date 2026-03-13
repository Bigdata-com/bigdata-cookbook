"""
Smart Batching for Semantic Search

This module implements a planning system that uses the comention endpoint to determine
chunk volumes per company, then creates optimized baskets of companies for semantic
search queries across multiple time periods with adaptive granularity.

The system uses a two-phase approach:
1. Phase 1: Query the full time period once to get total chunk volumes for all companies
2. Phase 2: Automatically determine optimal time granularity for each company based on
   volume and create baskets that minimize the total number of semantic search queries.
"""

import csv
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, TypedDict
from collections import defaultdict

import requests

from .smart_batching_config import (
    API_BASE_URL,
    COMENTION_ENDPOINT,
    VOLUME_ENDPOINT,
    MAX_ENTITIES_PER_QUERY,
    MAX_ENTITIES_IN_ANY_OF,
    MAX_CHUNKS_PER_BASKET,
    START_DATE,
    END_DATE,
    VOLUME_BUCKETS,
    PERIOD_CONFIGS,
    UNIVERSE_CSV_PATH,
)


class VolumePoint(TypedDict, total=False):
    """One point in a volume time series from the volume endpoint."""

    date: str
    documents: int
    chunks: int
    sentiment: float


class SmartBatchingPlanner:
    """
    Main orchestrator class for smart batching planning.
    
    This class handles the complete workflow of:
    - Loading company universes from CSV
    - Querying comention volumes via the Bigdata API
    - Creating optimized baskets of companies for semantic search
    - Determining optimal time granularity per company
    - Exporting planning results to CSV files
    """

    def __init__(self, api_key: Optional[str] = None, api_base_url: Optional[str] = None):
        """
        Initialize the planner.

        Args:
            api_key: BigData API key. If None, will try to get from environment variable BIGDATA_API_KEY.
            api_base_url: API base URL. If None, will use API_BASE_URL from config or environment variable.
        """
        self.api_key = api_key or os.getenv("BIGDATA_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in BIGDATA_API_KEY environment variable")
        
        # Use provided api_base_url, or fall back to config/environment
        self.api_base_url = api_base_url or os.getenv("BIGDATA_API_BASE_URL") or API_BASE_URL
        self.api_url = f"{self.api_base_url}{COMENTION_ENDPOINT}"
        self.volume_url = f"{self.api_base_url}{VOLUME_ENDPOINT}"
        self.headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

    def load_universe(self, csv_path: str = UNIVERSE_CSV_PATH, id_column: str = 'id') -> List[str]:
        """
        Read companies from CSV file.
        
        Supports two formats:
        1. CSV with header row containing 'id' column (e.g., id,name)
        2. Simple CSV with one entity ID per line (no header)

        Args:
            csv_path: Path to CSV file containing company IDs
            id_column: Name of the column containing entity IDs (default: 'id')

        Returns:
            List of company IDs
        """
        companies = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            first_row = next(reader, None)
            
            if first_row is None:
                return companies
            
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
        return companies

    def get_comention_volumes(
        self,
        companies: List[str],
        topic: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[Dict[str, int], int]:
        """
        Iteratively query comention endpoint for all companies to get chunk volumes.
        
        Uses a three-pass approach for maximum accuracy:
        1. First pass: Query all companies in batches
        2. Second pass: Verify companies that didn't appear in first pass results
                       (they may have low volume that was pushed out by other high-volume entities)
        3. Third pass: Final verification for companies that still didn't appear in second pass
                      (ensures we catch all companies with any volume, even very low volume)

        Args:
            companies: List of company IDs
            topic: Topic string for the comention query
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)

        Returns:
            Tuple of (company_volumes_dict, query_count) where:
            - company_volumes_dict: Dict mapping company_id -> total_chunks_count
              Companies with 0 chunks will not appear in the dict (confirmed 0 after 3 passes)
            - query_count: Number of API queries made
        """
        company_volumes = {}
        remaining_companies = companies.copy()
        query_count = 0
        total_queries_needed = (len(companies) + MAX_ENTITIES_IN_ANY_OF - 1) // MAX_ENTITIES_IN_ANY_OF

        # Convert dates to ISO format with timezone
        start_iso = f"{start_date}T00:00:00Z"
        end_iso = f"{end_date}T23:59:59Z"

        print(f"    Querying {len(companies)} companies in batches of {MAX_ENTITIES_IN_ANY_OF} (estimated {total_queries_needed} queries)...")

        # Track companies that were queried but didn't appear in results
        unverified_companies = []

        # FIRST PASS: Query all companies in batches
        while remaining_companies:
            # Take up to MAX_ENTITIES_IN_ANY_OF companies per query (API complexity limit)
            batch = remaining_companies[:MAX_ENTITIES_IN_ANY_OF]
            
            if not batch:
                break
            
            payload = {
                "query": {
                    "text": topic,
                    "filters": {
                        "timestamp": {
                            "start": start_iso,
                            "end": end_iso,
                        },
                        "entity": {
                            "all_of": [],
                            "any_of": batch,
                            "none_of": [],
                            "search_in": "ALL",
                        },
                    },
                    "limit": MAX_ENTITIES_PER_QUERY,
                }
            }

            try:
                response = requests.post(self.api_url, json=payload, headers=self.headers)
                response.raise_for_status()
                data = response.json()
                query_count += 1
                
                # Extract company volumes from response
                results = data.get("results", {})
                companies_data = results.get("companies", [])
                
                # Track which companies from our batch appeared in the response with chunk data
                found_company_ids = set()
                for company_data in companies_data:
                    company_id = company_data.get("id")
                    # Only consider companies that have total_chunks_count field
                    # Companies with only total_headlines_count are treated as not found
                    if "total_chunks_count" not in company_data:
                        continue
                    chunks_count = company_data["total_chunks_count"]
                    if company_id and company_id in batch and chunks_count > 0:
                        company_volumes[company_id] = chunks_count
                        found_company_ids.add(company_id)
                
                # Identify companies from our batch that didn't appear in results
                # These might have low volume that was pushed out by other high-volume entities
                for company_id in batch:
                    if company_id not in found_company_ids:
                        unverified_companies.append(company_id)
                
                # Show progress - count companies found from our universe batch
                found_count = len(found_company_ids)
                found_from_universe = [cid for cid in found_company_ids if cid in batch]
                found_from_universe_count = len(found_from_universe)
                print(f"      Query {query_count}/{total_queries_needed}: Found {found_from_universe_count} companies from universe batch (out of {len(batch)} input)")

                # Remove processed companies from remaining list
                remaining_companies = remaining_companies[MAX_ENTITIES_IN_ANY_OF:]
                
            except requests.exceptions.HTTPError as e:
                # Try to get error details from response
                error_msg = str(e)
                try:
                    error_details = response.json()
                    error_msg = f"{error_msg}\nResponse: {json.dumps(error_details, indent=2)}"
                except:
                    try:
                        error_text = response.text
                        error_msg = f"{error_msg}\nResponse: {error_text[:500]}"
                    except:
                        pass
                raise RuntimeError(f"Error querying comention endpoint: {error_msg}")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Error querying comention endpoint: {e}")

        # SECOND PASS: Verify companies that didn't appear in first pass
        still_unverified = []
        if unverified_companies:
            print(f"\n    Verification pass 1: Re-checking {len(unverified_companies)} companies from universe that didn't appear in first pass...")
            verification_queries_needed = (len(unverified_companies) + MAX_ENTITIES_IN_ANY_OF - 1) // MAX_ENTITIES_IN_ANY_OF
            
            remaining_unverified = unverified_companies.copy()
            verification_count = 0
            
            while remaining_unverified:
                # Query smaller batches for verification (can use same size or smaller)
                batch = remaining_unverified[:MAX_ENTITIES_IN_ANY_OF]
                
                if not batch:
                    break
                
                payload = {
                    "query": {
                        "text": topic,
                        "filters": {
                            "timestamp": {
                                "start": start_iso,
                                "end": end_iso,
                            },
                            "entity": {
                                "all_of": [],
                                "any_of": batch,
                                "none_of": [],
                                "search_in": "ALL",
                            },
                        },
                        "limit": MAX_ENTITIES_PER_QUERY,
                    }
                }

                try:
                    response = requests.post(self.api_url, json=payload, headers=self.headers)
                    response.raise_for_status()
                    data = response.json()
                    query_count += 1
                    verification_count += 1
                    
                    # Extract company volumes from response
                    results = data.get("results", {})
                    companies_data = results.get("companies", [])
                    
                    # Track which companies from this batch appeared with chunk data
                    found_in_verification = set()
                    verified_count = 0
                    for company_data in companies_data:
                        company_id = company_data.get("id")
                        # Only consider companies that have total_chunks_count field
                        if "total_chunks_count" not in company_data:
                            continue
                        chunks_count = company_data["total_chunks_count"]
                        if company_id and company_id in batch and chunks_count > 0:
                            # Only update if this company was in our verification batch
                            company_volumes[company_id] = chunks_count
                            found_in_verification.add(company_id)
                            verified_count += 1
                    
                    # Track companies that still didn't appear
                    for company_id in batch:
                        if company_id not in found_in_verification:
                            still_unverified.append(company_id)
                    
                    # Count companies from universe found in this verification pass
                    found_in_this_pass = [cid for cid in found_in_verification if cid in batch]
                    found_from_universe_count = len(found_in_this_pass)
                    print(f"      Verification query {verification_count}/{verification_queries_needed}: Found {found_from_universe_count} companies from universe (out of {len(batch)} input)")
                    
                    # Remove processed companies from remaining list
                    remaining_unverified = remaining_unverified[MAX_ENTITIES_IN_ANY_OF:]
                    
                except requests.exceptions.HTTPError as e:
                    error_msg = str(e)
                    try:
                        error_details = response.json()
                        error_msg = f"{error_msg}\nResponse: {json.dumps(error_details, indent=2)}"
                    except:
                        try:
                            error_text = response.text
                            error_msg = f"{error_msg}\nResponse: {error_text[:500]}"
                        except:
                            pass
                    raise RuntimeError(f"Error in verification query: {error_msg}")
                except requests.exceptions.RequestException as e:
                    raise RuntimeError(f"Error in verification query: {e}")
        
        # THIRD PASS: Final verification for companies that still didn't appear
        if still_unverified:
            print(f"\n    Verification pass 2: Final check for {len(still_unverified)} companies from universe that still need verification...")
            final_verification_queries_needed = (len(still_unverified) + MAX_ENTITIES_IN_ANY_OF - 1) // MAX_ENTITIES_IN_ANY_OF
            
            remaining_final_unverified = still_unverified.copy()
            final_verification_count = 0
            
            while remaining_final_unverified:
                batch = remaining_final_unverified[:MAX_ENTITIES_IN_ANY_OF]
                
                if not batch:
                    break
                
                payload = {
                    "query": {
                        "text": topic,
                        "filters": {
                            "timestamp": {
                                "start": start_iso,
                                "end": end_iso,
                            },
                            "entity": {
                                "all_of": [],
                                "any_of": batch,
                                "none_of": [],
                                "search_in": "ALL",
                            },
                        },
                        "limit": MAX_ENTITIES_PER_QUERY,
                    }
                }

                try:
                    response = requests.post(self.api_url, json=payload, headers=self.headers)
                    response.raise_for_status()
                    data = response.json()
                    query_count += 1
                    final_verification_count += 1
                    
                    # Extract company volumes from response
                    results = data.get("results", {})
                    companies_data = results.get("companies", [])
                    
                    final_verified_count = 0
                    found_in_final_pass = []
                    for company_data in companies_data:
                        company_id = company_data.get("id")
                        # Only consider companies that have total_chunks_count field
                        if "total_chunks_count" not in company_data:
                            continue
                        chunks_count = company_data["total_chunks_count"]
                        if company_id and company_id in batch and chunks_count > 0:
                            # Only update if this company was in our final verification batch
                            company_volumes[company_id] = chunks_count
                            found_in_final_pass.append(company_id)
                            final_verified_count += 1
                    
                    # Count companies from universe found in final verification
                    found_from_universe_count = len(found_in_final_pass)
                    print(f"      Final verification query {final_verification_count}/{final_verification_queries_needed}: Found {found_from_universe_count} companies from universe (out of {len(batch)} input)")
                    
                    # Remove processed companies from remaining list
                    remaining_final_unverified = remaining_final_unverified[MAX_ENTITIES_IN_ANY_OF:]
                    
                except requests.exceptions.HTTPError as e:
                    error_msg = str(e)
                    try:
                        error_details = response.json()
                        error_msg = f"{error_msg}\nResponse: {json.dumps(error_details, indent=2)}"
                    except:
                        try:
                            error_text = response.text
                            error_msg = f"{error_msg}\nResponse: {error_text[:500]}"
                        except:
                            pass
                    raise RuntimeError(f"Error in final verification query: {error_msg}")
                except requests.exceptions.RequestException as e:
                    raise RuntimeError(f"Error in final verification query: {e}")
            
            # Companies that still don't appear after third pass are confirmed to have 0 volume
            confirmed_zero = len(still_unverified) - len([c for c in still_unverified if c in company_volumes])
            if confirmed_zero > 0:
                print(f"    Confirmed {confirmed_zero} companies with zero volume after final verification")
        elif unverified_companies:
            # If there were unverified companies but none remain after second pass, all were found
            confirmed_zero = len(unverified_companies) - len([c for c in unverified_companies if c in company_volumes])
            if confirmed_zero > 0:
                print(f"    Confirmed {confirmed_zero} companies with zero volume after first verification pass")

        print(f"    Completed {query_count} queries. Found {len(company_volumes)} companies with chunks > 0")
        return company_volumes, query_count

    def get_comention_volumes_iterative(
        self,
        companies: List[str],
        topic: str,
        start_date: str,
        end_date: str,
        max_iterations_per_batch: int = 10,
    ) -> Tuple[Dict[str, int], int, List[str]]:
        """
        Query comention endpoint using iterative per-batch approach.
        
        For each batch of companies:
        1. Query the batch
        2. Remove found companies from the batch
        3. Re-query remaining companies in the same batch
        4. Stop when an iteration returns 0 new companies
        5. Move to the next batch
        
        This is more efficient than the 3-pass approach because it:
        - Handles each batch independently
        - Stops as soon as no new companies are found
        - Never re-queries already found companies

        Args:
            companies: List of company IDs
            topic: Topic string for the comention query
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)
            max_iterations_per_batch: Maximum iterations per batch to prevent infinite loops (default 10)

        Returns:
            Tuple of (company_volumes_dict, query_count, very_low_companies) where:
            - company_volumes_dict: Dict mapping company_id -> total_chunks_count (chunks > 0)
            - query_count: Number of API queries made
            - very_low_companies: List of company IDs that have no chunks (0 or not found)
        """
        company_volumes = {}
        very_low_companies = []  # Companies with no chunks (to be added to very_low baskets)
        query_count = 0
        total_batches = (len(companies) + MAX_ENTITIES_IN_ANY_OF - 1) // MAX_ENTITIES_IN_ANY_OF

        # Convert dates to ISO format with timezone
        start_iso = f"{start_date}T00:00:00Z"
        end_iso = f"{end_date}T23:59:59Z"

        print(f"    [ITERATIVE MODE] Querying {len(companies)} companies in {total_batches} batches of {MAX_ENTITIES_IN_ANY_OF}")
        print(f"    Each batch iterates until no new companies are found (max {max_iterations_per_batch} iterations)")

        # Process each batch independently
        for batch_idx in range(total_batches):
            batch_start = batch_idx * MAX_ENTITIES_IN_ANY_OF
            batch_end = min(batch_start + MAX_ENTITIES_IN_ANY_OF, len(companies))
            batch_original = companies[batch_start:batch_end]
            batch_original_size = len(batch_original)
            
            # Use set for O(1) lookups and removals
            batch_remaining_set = set(batch_original)
            batch_found_total = 0
            
            iteration = 0
            while batch_remaining_set and iteration < max_iterations_per_batch:
                iteration += 1
                
                # Convert set to list for API call
                batch_remaining_list = list(batch_remaining_set)
                
                payload = {
                    "query": {
                        "text": topic,
                        "filters": {
                            "timestamp": {
                                "start": start_iso,
                                "end": end_iso,
                            },
                            "entity": {
                                "all_of": [],
                                "any_of": batch_remaining_list,
                                "none_of": [],
                                "search_in": "ALL",
                            },
                        },
                        "limit": MAX_ENTITIES_PER_QUERY,
                    }
                }

                try:
                    response = requests.post(self.api_url, json=payload, headers=self.headers)
                    response.raise_for_status()
                    data = response.json()
                    query_count += 1
                    
                    # Extract company volumes from response
                    results = data.get("results", {})
                    companies_data = results.get("companies", [])
                    
                    # Find companies from our batch that appeared in results
                    # Use set to handle potential duplicates in API response
                    found_in_iteration = set()
                    for company_data in companies_data:
                        company_id = company_data.get("id")
                        if "total_chunks_count" not in company_data:
                            continue
                        chunks_count = company_data["total_chunks_count"]
                        # O(1) lookup with set
                        if company_id and company_id in batch_remaining_set and chunks_count > 0:
                            company_volumes[company_id] = chunks_count
                            found_in_iteration.add(company_id)
                    
                    found_count = len(found_in_iteration)
                    batch_found_total += found_count
                    
                    # Remove found companies from batch (O(1) set operations)
                    batch_remaining_set -= found_in_iteration
                    
                    print(f"      Batch {batch_idx + 1}/{total_batches}, Iter {iteration}: "
                          f"Found {found_count} new companies, {len(batch_remaining_set)} remaining")
                    
                    # Stop if no new companies found in this iteration
                    if found_count == 0:
                        break
                        
                except requests.exceptions.HTTPError as e:
                    error_msg = str(e)
                    try:
                        error_details = response.json()
                        error_msg = f"{error_msg}\nResponse: {json.dumps(error_details, indent=2)}"
                    except:
                        try:
                            error_text = response.text
                            error_msg = f"{error_msg}\nResponse: {error_text[:500]}"
                        except:
                            pass
                    raise RuntimeError(f"Error querying comention endpoint: {error_msg}")
                except requests.exceptions.RequestException as e:
                    raise RuntimeError(f"Error querying comention endpoint: {e}")
            
            # Companies remaining after all iterations are "very_low" (no chunks found)
            very_low_companies.extend(batch_remaining_set)
            
            # Summary for this batch
            zero_count = batch_original_size - batch_found_total
            print(f"      Batch {batch_idx + 1} complete: {batch_found_total} found, {zero_count} very_low, {iteration} iterations")

        print(f"    Completed {query_count} queries. Found {len(company_volumes)} companies with chunks > 0, {len(very_low_companies)} very_low")
        return company_volumes, query_count, very_low_companies

    def get_volume_timeseries(
        self,
        companies: List[str],
        topic: str,
        start_date: str,
        end_date: str,
    ) -> List[VolumePoint]:
        """
        Query the volume endpoint for a set of companies and return a time series.

        Batches companies by MAX_ENTITIES_IN_ANY_OF; aggregates results by date
        so that chunks (and optionally documents, sentiment) are summed per date
        across all batches.

        Args:
            companies: List of company IDs
            topic: Topic string for the query
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of VolumePoint dicts with keys date, documents, chunks, sentiment,
            sorted by date. Empty list on API error or empty response.
        """
        start_iso = f"{start_date}T00:00:00Z"
        end_iso = f"{end_date}T23:59:59Z"
        total_batches = (len(companies) + MAX_ENTITIES_IN_ANY_OF - 1) // MAX_ENTITIES_IN_ANY_OF

        for batch_idx in range(total_batches):
            batch_start = batch_idx * MAX_ENTITIES_IN_ANY_OF
            batch_end = min(batch_start + MAX_ENTITIES_IN_ANY_OF, len(companies))
            batch = companies[batch_start:batch_end]

            payload = {
                "query": {
                    "text": topic,
                    "filters": {
                        "timestamp": {"start": start_iso, "end": end_iso},
                        "entity": {
                            "all_of": [],
                            "any_of": batch,
                            "none_of": [],
                        },
                    },
                }
            }

            try:
                response = requests.post(
                    self.volume_url, json=payload, headers=self.headers
                )
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.HTTPError as e:
                error_msg = str(e)
                try:
                    error_details = response.json()
                    error_msg = f"{error_msg}\nResponse: {json.dumps(error_details, indent=2)}"
                except Exception:
                    try:
                        error_msg = f"{error_msg}\nResponse: {response.text[:500]}"
                    except Exception:
                        pass
                raise RuntimeError(f"Error querying volume endpoint: {error_msg}")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Error querying volume endpoint: {e}")

            # Parse response: accept list at top level or under "results" / "volume"
            raw_list = data.get('results', {}).get('volume', [])
            aggregated_by_date = []
            for point in raw_list:
                # Initialize VolumePoint
                volume_point: VolumePoint = {
                    "date": point.get("date"),
                    "documents": point.get("documents"),
                    "chunks": point.get("chunks"),
                    "sentiment": point.get("sentiment"),
                }
                aggregated_by_date.append(volume_point)


        out = aggregated_by_date
        out.sort(key=lambda p: p.get("date", ""))
        return out

    def filter_zero_volume(self, company_volumes: Dict[str, int]) -> Dict[str, int]:
        """
        Filter out companies with 0 chunks (no search needed).

        Args:
            company_volumes: Dict mapping company_id -> chunks

        Returns:
            Dict with only companies that have chunks > 0
        """
        return {cid: chunks for cid, chunks in company_volumes.items() if chunks > 0}

    def group_by_volume(self, company_volumes: Dict[str, int]) -> Dict[str, List[Tuple[str, int]]]:
        """
        Group companies into volume buckets.

        Args:
            company_volumes: Dict mapping company_id -> chunks

        Returns:
            Dict mapping bucket name -> list of (company_id, chunks) tuples, sorted by chunks descending
        """
        buckets = defaultdict(list)
        
        for company_id, chunks in company_volumes.items():
            if chunks >= VOLUME_BUCKETS["high"][0]:
                buckets["high"].append((company_id, chunks))
            elif chunks >= VOLUME_BUCKETS["medium"][0]:
                buckets["medium"].append((company_id, chunks))
            elif chunks >= VOLUME_BUCKETS["low"][0]:
                buckets["low"].append((company_id, chunks))

        # Sort each bucket by chunks (descending)
        for bucket_name in buckets:
            buckets[bucket_name].sort(key=lambda x: x[1], reverse=True)

        return dict(buckets)

    def create_baskets(
        self,
        company_volumes: Dict[str, int],
        max_chunks: int = MAX_CHUNKS_PER_BASKET,
        very_low_companies: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Create baskets of companies with total chunks < max_chunks.

        Args:
            company_volumes: Dict mapping company_id -> chunks (already filtered to exclude 0-chunk companies)
            max_chunks: Maximum total chunks per basket
            very_low_companies: Optional list of company IDs with 0 chunks (will be added to very_low baskets)

        Returns:
            List of basket dictionaries, each containing:
            - basket_id: Unique identifier
            - companies: List of company IDs
            - total_chunks: Sum of chunks for all companies in basket
            - company_count: Number of companies in basket
            - volume_range: Volume range category
            - company_chunks: Dict mapping company_id -> chunks for this basket
        """
        # Filter out zero-chunk companies
        filtered_volumes = self.filter_zero_volume(company_volumes)
        
        baskets = []
        basket_counter = 0

        if filtered_volumes:
            # Group by volume
            volume_groups = self.group_by_volume(filtered_volumes)

            # Process each volume group
            for volume_range, companies_list in volume_groups.items():
                current_basket = {
                    "companies": [],
                    "company_chunks": {},
                    "total_chunks": 0,
                }

                for company_id, chunks in companies_list:
                    # Check if adding this company would exceed the chunk limit OR entity limit
                    if (current_basket["total_chunks"] + chunks > max_chunks or
                        len(current_basket["companies"]) >= MAX_ENTITIES_IN_ANY_OF):
                        # Save current basket and start a new one
                        if current_basket["companies"]:
                            baskets.append({
                                "basket_id": f"{volume_range}_basket_{basket_counter}",
                                "companies": current_basket["companies"],
                                "company_chunks": current_basket["company_chunks"].copy(),
                                "total_chunks": current_basket["total_chunks"],
                                "company_count": len(current_basket["companies"]),
                                "volume_range": volume_range,
                            })
                            basket_counter += 1
                        
                        # Start new basket with this company
                        current_basket = {
                            "companies": [company_id],
                            "company_chunks": {company_id: chunks},
                            "total_chunks": chunks,
                        }
                    else:
                        # Add company to current basket
                        current_basket["companies"].append(company_id)
                        current_basket["company_chunks"][company_id] = chunks
                        current_basket["total_chunks"] += chunks

                # Don't forget the last basket in this volume range
                if current_basket["companies"]:
                    baskets.append({
                        "basket_id": f"{volume_range}_basket_{basket_counter}",
                        "companies": current_basket["companies"],
                        "company_chunks": current_basket["company_chunks"].copy(),
                        "total_chunks": current_basket["total_chunks"],
                        "company_count": len(current_basket["companies"]),
                        "volume_range": volume_range,
                    })
                    basket_counter += 1

        # Process very_low companies (0 chunks) - max 500 companies per basket
        if very_low_companies:
            very_low_basket_counter = 0
            for i in range(0, len(very_low_companies), MAX_ENTITIES_IN_ANY_OF):
                batch = very_low_companies[i:i + MAX_ENTITIES_IN_ANY_OF]
                baskets.append({
                    "basket_id": f"very_low_basket_{very_low_basket_counter}",
                    "companies": batch,
                    "company_chunks": {cid: 0 for cid in batch},
                    "total_chunks": 0,
                    "company_count": len(batch),
                    "volume_range": "very_low",
                })
                very_low_basket_counter += 1

        return baskets

    def split_period(
        self,
        start_date: str,
        end_date: str,
        period_type: str,
    ) -> List[Tuple[str, str]]:
        """
        Split a date range into sub-periods based on period type.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period_type: One of 'biyearly', 'yearly', 'quarterly', 'bimonthly', 'monthly', 'weekly'

        Returns:
            List of (start_date, end_date) tuples for each sub-period
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        periods = []

        if period_type == "biyearly":
            periods = [(start_date, end_date)]
        
        elif period_type == "yearly":
            # Split into years
            current = start
            while current < end:
                period_end = min(
                    datetime(current.year + 1, 1, 1) - timedelta(days=1),
                    end
                )
                periods.append((
                    current.strftime("%Y-%m-%d"),
                    period_end.strftime("%Y-%m-%d"),
                ))
                current = period_end + timedelta(days=1)
        
        elif period_type == "quarterly":
            # Split into quarters
            current = start
            while current < end:
                # Calculate quarter end
                quarter = (current.month - 1) // 3 + 1
                if quarter == 1:
                    quarter_end = datetime(current.year, 3, 31)
                elif quarter == 2:
                    quarter_end = datetime(current.year, 6, 30)
                elif quarter == 3:
                    quarter_end = datetime(current.year, 9, 30)
                else:
                    quarter_end = datetime(current.year, 12, 31)
                
                period_end = min(quarter_end, end)
                periods.append((
                    current.strftime("%Y-%m-%d"),
                    period_end.strftime("%Y-%m-%d"),
                ))
                current = period_end + timedelta(days=1)
        
        elif period_type == "bimonthly":
            # Split into 2-month blocks (Jan-Feb, Mar-Apr, May-Jun, Jul-Aug, Sep-Oct, Nov-Dec)
            current = start
            while current < end:
                # Second month of block: current.month + 1 (wrap to next year if Dec)
                second_month = current.month + 1
                block_year = current.year
                if second_month > 12:
                    second_month = 1
                    block_year = current.year + 1
                # Last day of second month
                if second_month == 12:
                    block_end = datetime(block_year, 12, 31)
                else:
                    block_end = datetime(block_year, second_month + 1, 1) - timedelta(days=1)
                period_end = min(block_end, end)
                periods.append((
                    current.strftime("%Y-%m-%d"),
                    period_end.strftime("%Y-%m-%d"),
                ))
                current = period_end + timedelta(days=1)

        elif period_type == "monthly":
            # Split into months
            current = start
            while current < end:
                # Calculate month end
                if current.month == 12:
                    month_end = datetime(current.year + 1, 1, 1) - timedelta(days=1)
                else:
                    month_end = datetime(current.year, current.month + 1, 1) - timedelta(days=1)
                
                period_end = min(month_end, end)
                periods.append((
                    current.strftime("%Y-%m-%d"),
                    period_end.strftime("%Y-%m-%d"),
                ))
                current = period_end + timedelta(days=1)
        
        elif period_type == "weekly":
            # Split into weeks
            current = start
            while current < end:
                week_end = min(current + timedelta(days=6), end)
                periods.append((
                    current.strftime("%Y-%m-%d"),
                    week_end.strftime("%Y-%m-%d"),
                ))
                current = week_end + timedelta(days=1)
        
        else:
            raise ValueError(f"Unknown period type: {period_type}")

        return periods

    def estimate_subperiod_volumes(
        self,
        company_total_chunks: int,
        sub_period_start: str,
        sub_period_end: str,
        full_period_start: str,
        full_period_end: str,
    ) -> int:
        """
        Estimate chunk volume for a sub-period using uniform distribution.

        Args:
            company_total_chunks: Total chunks for the company in the full period
            sub_period_start: Start date of sub-period (YYYY-MM-DD)
            sub_period_end: End date of sub-period (YYYY-MM-DD)
            full_period_start: Start date of full period (YYYY-MM-DD)
            full_period_end: End date of full period (YYYY-MM-DD)

        Returns:
            Estimated chunks for the sub-period (rounded to nearest integer)
        """
        sub_start = datetime.strptime(sub_period_start, "%Y-%m-%d")
        sub_end = datetime.strptime(sub_period_end, "%Y-%m-%d")
        full_start = datetime.strptime(full_period_start, "%Y-%m-%d")
        full_end = datetime.strptime(full_period_end, "%Y-%m-%d")

        sub_period_days = (sub_end - sub_start).days + 1  # Inclusive
        total_period_days = (full_end - full_start).days + 1  # Inclusive

        if total_period_days == 0:
            return 0

        estimated = (company_total_chunks * sub_period_days) / total_period_days
        return max(0, int(round(estimated)))

    def calculate_periods_needed(self, total_chunks: int) -> int:
        """
        Calculate how many periods are needed for a company based on its total chunks.

        Args:
            total_chunks: Total chunks for the company

        Returns:
            Number of periods needed (ceil(total_chunks / 1000))
        """
        return max(1, math.ceil(total_chunks / MAX_CHUNKS_PER_BASKET))

    def determine_split_granularity(
        self,
        periods_needed: int,
        target_period_type: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Split the date range into exactly periods_needed equal parts.

        Args:
            periods_needed: Number of periods needed (ceil(total_chunks / 1000))
            target_period_type: Unused (kept for backward compatibility)
            start_date: Start date of full period (YYYY-MM-DD)
            end_date: End date of full period (YYYY-MM-DD)

        Returns:
            Tuple of (period_type_label, list of (start, end) date tuples for periods)
        """
        if periods_needed <= 1:
            return ("full_range", [(start_date, end_date)])

        # Split into exactly periods_needed equal parts
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end - start).days + 1  # inclusive

        periods = []
        for i in range(periods_needed):
            # Calculate start and end day for this period (integer division for even split)
            period_start_day = i * total_days // periods_needed
            period_end_day = (i + 1) * total_days // periods_needed - 1

            period_start_dt = start + timedelta(days=period_start_day)
            period_end_dt = start + timedelta(days=period_end_day)

            periods.append((
                period_start_dt.strftime("%Y-%m-%d"),
                period_end_dt.strftime("%Y-%m-%d")
            ))

        return (f"split_{periods_needed}", periods)

    def determine_splits_from_volume(
        self,
        volume_series: List[VolumePoint],
        periods_needed: int,
        start_date: str,
        end_date: str,
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Split the date range into sub-periods so that chunk volume is balanced
        across periods using the volume time series. Falls back to equal-length
        splits if the series is empty or has zero total chunks.

        Args:
            volume_series: List of VolumePoint with at least "date" and "chunks"
            periods_needed: Number of periods to produce
            start_date: Full period start (YYYY-MM-DD)
            end_date: Full period end (YYYY-MM-DD)

        Returns:
            Tuple of (period_type_label, list of (start, end) date tuples)
        """
        if periods_needed <= 1:
            return ("full_range", [(start_date, end_date)])

        sorted_series = sorted(
            (p for p in volume_series if p.get("date")),
            key=lambda p: p.get("date", ""),
        )
        total_chunks = sum(p.get("chunks", 0) or 0 for p in sorted_series)

        if not sorted_series or total_chunks <= 0:
            return self.determine_split_granularity(
                periods_needed, "biyearly", start_date, end_date
            )

        # Cumulative chunks and find boundary dates (one date per boundary to avoid zero-length periods)
        cumulative = 0
        boundaries: List[str] = []
        target_step = total_chunks / periods_needed
        for point in sorted_series:
            cumulative += point.get("chunks", 0) or 0
            while len(boundaries) < periods_needed - 1 and cumulative >= (
                (len(boundaries) + 1) * target_step
            ):
                date_val = point.get("date", "")
                if not boundaries or date_val != boundaries[-1]:
                    boundaries.append(date_val)
                else:
                    break

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        periods: List[Tuple[str, str]] = []
        period_starts = [start_date] + [self._next_day(d) for d in boundaries]
        period_ends = list(boundaries) + [end_date]

        for s, e in zip(period_starts, period_ends):
            if s <= e:
                periods.append((s, e))
            else:
                periods.append((s, s))

        if len(periods) != periods_needed:
            return self.determine_split_granularity(
                periods_needed, "biyearly", start_date, end_date
            )
        return (f"split_{periods_needed}_volume", periods)

    def _next_day(self, date_str: str) -> str:
        """Return the day after date_str (YYYY-MM-DD) as YYYY-MM-DD."""
        d = datetime.strptime(date_str, "%Y-%m-%d")
        next_d = d + timedelta(days=1)
        return next_d.strftime("%Y-%m-%d")

    def sub_period_volumes_from_series(
        self,
        volume_series: List[VolumePoint],
        periods: List[Tuple[str, str]],
        company_group: Dict[str, int],
    ) -> List[Dict[str, int]]:
        """
        For each sub-period, compute chunk volume per company from the volume series.

        When the series is aggregated for the group, distributes each period's
        total chunks across companies proportionally to their full-period totals.

        Args:
            volume_series: Time series with date and chunks
            periods: List of (start_date, end_date) for each sub-period
            company_group: Dict company_id -> total_chunks (full period)

        Returns:
            List of dicts, one per period; each dict maps company_id -> chunks for that period
        """
        result: List[Dict[str, int]] = []
        group_total = sum(company_group.values()) or 1
        by_date: Dict[str, int] = {}
        for p in volume_series:
            d = p.get("date")
            if d:
                by_date[d] = by_date.get(d, 0) + (p.get("chunks") or 0)

        for period_start, period_end in periods:
            period_chunks = 0
            start_dt = datetime.strptime(period_start, "%Y-%m-%d")
            end_dt = datetime.strptime(period_end, "%Y-%m-%d")
            current = start_dt
            while current <= end_dt:
                key = current.strftime("%Y-%m-%d")
                period_chunks += by_date.get(key, 0)
                current += timedelta(days=1)
            sub_period_volumes: Dict[str, int] = {}
            for company_id, total in company_group.items():
                if group_total <= 0:
                    sub_period_volumes[company_id] = 0
                else:
                    proportion = total / group_total
                    sub_period_volumes[company_id] = max(
                        0, int(round(period_chunks * proportion))
                    )
            result.append(sub_period_volumes)
        return result

    def plan_all_periods(
        self,
        topic: str,
        start_date: str = START_DATE,
        end_date: str = END_DATE,
        volume_query_mode: str = "three_pass",
        max_iterations_per_batch: int = 10,
        universe_csv_path: Optional[str] = None,
        apply_volume_splits: bool = True,
    ) -> Dict:
        """
        Generate SMART batching plan with optimal granularity per company.

        Phase 1: Query full period once to get total volumes
        Phase 2: Automatically determine optimal granularity for each company based on volume
                and create baskets using adaptive splitting (sub-periods so each query <= 1000 chunks)

        Args:
            topic: Topic string for comention and semantic search queries
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            volume_query_mode: Method for querying volumes. Options:
                - "three_pass": Original 3-pass approach (query all, then verify twice)
                - "iterative": Per-batch iterative approach (query batch, remove found, repeat until empty)
            max_iterations_per_batch: Max iterations per batch when using "iterative" mode (default 10)
            universe_csv_path: Optional path to universe CSV; if not set, uses planner default.
            apply_volume_splits: If True (default), use volume time series to split periods per company.
                If False, use time-based granularity and estimated sub-period volumes only.

        Returns:
            Planning report with single SMART configuration
        """
        # Load universe
        companies = self.load_universe(universe_csv_path) if universe_csv_path else self.load_universe()
        total_companies = len(companies)

        report = {
            "topic": topic,
            "period_range": {
                "start": start_date,
                "end": end_date,
            },
            "total_companies": total_companies,
            "configurations": {},
        }

        # PHASE 1: Query full period once to get total volumes
        print("=" * 80)
        print(f"PHASE 1: Querying full period for all companies ({start_date} to {end_date})")
        print(f"         Mode: {volume_query_mode}")
        print("=" * 80)
        
        if volume_query_mode == "iterative":
            full_period_volumes, total_comention_queries, _ = self.get_comention_volumes_iterative(
                companies, topic, start_date, end_date, max_iterations_per_batch=max_iterations_per_batch
            )
        else:
            # Default: three_pass mode
            full_period_volumes, total_comention_queries = self.get_comention_volumes(
                companies, topic, start_date, end_date
            )
        print(f"\nPhase 1 complete: {total_comention_queries} comention queries")
        print(f"Found {len(full_period_volumes)} companies with chunks > 0\n")

        # Calculate periods needed for each company
        company_periods_needed = {}
        for company_id, chunks in full_period_volumes.items():
            company_periods_needed[company_id] = self.calculate_periods_needed(chunks)

        # Group companies by periods_needed
        companies_by_periods_needed = defaultdict(dict)
        for company_id, chunks in full_period_volumes.items():
            periods_needed = company_periods_needed[company_id]
            companies_by_periods_needed[periods_needed][company_id] = chunks

        print(f"Company categorization by periods needed:")
        for periods_needed in sorted(companies_by_periods_needed.keys()):
            count = len(companies_by_periods_needed[periods_needed])
            print(f"  {periods_needed} period(s) needed: {count} companies")
        print(f"  Zero chunks: {total_companies - len(full_period_volumes)} companies\n")

        # PHASE 2: Fetch volume time series per company group (concurrent) for groups needing splits
        volume_by_group: Dict[Tuple[int, Tuple[str, ...]], List[VolumePoint]] = {}
        groups_needing_volume = [
            (p, g)
            for p, g in sorted(companies_by_periods_needed.items())
            if p > 1 and apply_volume_splits
        ]
        if groups_needing_volume:
            print("Fetching volume time series for company groups (concurrent)...")
            with ThreadPoolExecutor(max_workers=min(8, len(groups_needing_volume))) as executor:
                future_to_key = {
                    executor.submit(
                        self.get_volume_timeseries,
                        list(g.keys()),
                        topic,
                        start_date,
                        end_date,
                    ): (p, tuple(sorted(g.keys())))
                    for p, g in groups_needing_volume
                }
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        series = future.result()
                        volume_by_group[key] = series
                    except Exception as e:
                        print(f"    Volume fetch failed for group: {e}")
                        volume_by_group[key] = []
            print(f"    Fetched volume for {len(volume_by_group)} groups.\n")

        # PHASE 2: Plan baskets using SMART configuration (single optimal plan)
        print("=" * 80)
        print("PHASE 2: Planning SMART configuration (optimal granularity per company)")
        print("=" * 80)

        config_report = {
            "comention_queries": total_comention_queries,
            "semantic_queries": 0,
            "companies_with_chunks": len(full_period_volumes),
            "companies_with_zero_chunks": total_companies - len(full_period_volumes),
            "baskets": [],
            "period_details": [],
            "uses_estimates": False,
            "adaptive_splitting": True,
            "granularity_groups": {},
        }

        total_semantic_queries = 0
        all_period_details = []
        uses_estimates = False
        granularity_groups = defaultdict(lambda: {"companies": 0, "periods": 0, "baskets": 0})
        global_basket_counter = 0

        # Process each group of companies by periods_needed
        for periods_needed, company_group in sorted(companies_by_periods_needed.items()):
            group_key = (periods_needed, tuple(sorted(company_group.keys())))
            volume_series = volume_by_group.get(group_key) if periods_needed > 1 else None
            use_volume_splits = (
                apply_volume_splits
                and periods_needed > 1
                and volume_series is not None
                and len(volume_series) > 0
                and sum(p.get("chunks", 0) or 0 for p in volume_series) > 0
            )

            if use_volume_splits:
                actual_period_type, actual_periods = self.determine_splits_from_volume(
                    volume_series, periods_needed, start_date, end_date
                )
                sub_period_volumes_list = self.sub_period_volumes_from_series(
                    volume_series, actual_periods, company_group
                )
            else:
                actual_period_type, actual_periods = self.determine_split_granularity(
                    periods_needed, "biyearly", start_date, end_date
                )
                sub_period_volumes_list = []

            n_per = len(actual_periods)
            period_word = "period" if n_per == 1 else "periods"
            print(f"  Companies needing {periods_needed} period(s): {len(company_group)} companies")
            print(f"    Using {actual_period_type} granularity ({n_per} {period_word})")

            granularity_groups[actual_period_type]["companies"] += len(company_group)
            granularity_groups[actual_period_type]["periods"] = len(actual_periods)

            # Process each period for this group
            for period_idx, (period_start, period_end) in enumerate(actual_periods):
                if use_volume_splits and period_idx < len(sub_period_volumes_list):
                    sub_period_volumes = sub_period_volumes_list[period_idx]
                else:
                    sub_period_volumes = {}
                    for company_id, total_chunks in company_group.items():
                        if periods_needed == 1:
                            sub_period_volumes[company_id] = total_chunks
                        else:
                            estimated_chunks = self.estimate_subperiod_volumes(
                                total_chunks,
                                period_start,
                                period_end,
                                start_date,
                                end_date,
                            )
                            if estimated_chunks > 0:
                                sub_period_volumes[company_id] = estimated_chunks
                                uses_estimates = True

                # Create baskets for this sub-period
                baskets = self.create_baskets(sub_period_volumes)
                total_semantic_queries += len(baskets)
                granularity_groups[actual_period_type]["baskets"] += len(baskets)
                
                # Track companies with chunks in this sub-period
                companies_with_chunks = len(self.filter_zero_volume(sub_period_volumes))
                
                # Update basket IDs: global counter, dates only when split, subdivision index (e.g. 1of6)
                period_start_short = period_start.replace("-", "")
                period_end_short = period_end.replace("-", "")
                for basket in baskets:
                    volume_range = basket.get("volume_range", "basket")
                    if n_per == 1:
                        # No split: no date, no subdivision index
                        basket["basket_id"] = f"{volume_range}_basket_{global_basket_counter}"
                    else:
                        # Split: basket number, subdivision index (1of6), then date range
                        basket["basket_id"] = (
                            f"{volume_range}_basket_{global_basket_counter}_"
                            f"{period_idx + 1}of{n_per}_{period_start_short}_{period_end_short}"
                        )
                    global_basket_counter += 1
                    basket["actual_granularity"] = actual_period_type
                    basket["periods_needed"] = periods_needed
                    basket["period_start"] = period_start
                    basket["period_end"] = period_end
                    basket["contains_estimates"] = periods_needed > 1 and not use_volume_splits

                period_detail = {
                    "period_index": period_idx,
                    "start_date": period_start,
                    "end_date": period_end,
                    "actual_granularity": actual_period_type,
                    "periods_needed": periods_needed,
                    "companies_in_group": len(company_group),
                    "comention_queries": 0,  # No additional queries needed (using Phase 1 data)
                    "semantic_queries": len(baskets),
                    "companies_with_chunks": companies_with_chunks,
                    "companies_with_zero_chunks": total_companies - companies_with_chunks,
                    "baskets": baskets,
                    "uses_estimates": periods_needed > 1 and not use_volume_splits,
                }
                all_period_details.append(period_detail)

        config_report["sub_periods"] = len(all_period_details)
        config_report["period_details"] = all_period_details
        config_report["uses_estimates"] = uses_estimates
        config_report["granularity_groups"] = dict(granularity_groups)

        config_report["semantic_queries"] = total_semantic_queries

        # Calculate efficiency metrics
        total_chunks = sum(full_period_volumes.values())
        avg_chunks_per_query = total_chunks / total_semantic_queries if total_semantic_queries > 0 else 0
        utilization = (avg_chunks_per_query / MAX_CHUNKS_PER_BASKET) * 100 if total_semantic_queries > 0 else 0

        config_report["efficiency_metrics"] = {
            "total_chunks": total_chunks,
            "semantic_search_queries": total_semantic_queries,
            "avg_chunks_per_query": round(avg_chunks_per_query, 2),
            "utilization_percent": round(utilization, 2),
        }

        # Basket distribution by volume range
        basket_distribution = defaultdict(int)
        for period_detail in config_report["period_details"]:
            for basket in period_detail["baskets"]:
                basket_distribution[basket["volume_range"]] += 1
        
        config_report["basket_distribution"] = dict(basket_distribution)

        report["configurations"]["smart"] = config_report

        return report

    def generate_report(self, report: Dict, output_path: Optional[str] = None) -> str:
        """
        Generate a human-readable report from the planning results.

        Args:
            report: Planning report dictionary
            output_path: Optional path to save report as JSON

        Returns:
            Human-readable report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("SMART BATCHING PLANNING REPORT")
        lines.append("=" * 80)
        lines.append(f"\nTopic: {report['topic']}")
        lines.append(f"Period: {report['period_range']['start']} to {report['period_range']['end']}")
        lines.append(f"Total Companies: {report['total_companies']}")
        lines.append("\n" + "=" * 80)
        lines.append("OPTIMIZATION: Using smart batching with adaptive splitting")
        lines.append("  - Full period queried once (~10 comention queries total)")
        lines.append("  - Adaptive splitting: Each company gets appropriate granularity based on volume")
        lines.append("  - Companies automatically split into finer periods if needed (yearly, quarterly, bimonthly, monthly, weekly)")
        lines.append("  - Estimated volumes use uniform distribution assumption")
        lines.append("=" * 80)

        # Show SMART configuration
        smart_config = report["configurations"].get("smart")
        if smart_config:
            lines.append(f"\nTotal Comention Queries (Phase 1): {smart_config['comention_queries']}")
            lines.append(f"\nSMART CONFIGURATION")
            lines.append("=" * 80)
            lines.append(f"Sub-periods: {smart_config['sub_periods']}")
            lines.append(f"Semantic Search Queries: {smart_config['semantic_queries']}")
            lines.append(f"Companies with chunks > 0: {smart_config['companies_with_chunks']}")
            lines.append(f"Companies with 0 chunks: {smart_config['companies_with_zero_chunks']}")
            
            if smart_config.get("granularity_groups"):
                lines.append(f"\nGranularity Groups:")
                for granularity, group_info in sorted(smart_config["granularity_groups"].items()):
                    lines.append(f"  {granularity}: {group_info['companies']} companies, {group_info['periods']} periods, {group_info['baskets']} baskets")
            
            if smart_config.get("adaptive_splitting"):
                lines.append(f"\nAdaptive Splitting:")
                lines.append(f"  - Each company automatically gets optimal granularity")
                lines.append(f"  - Based on periods_needed = ceil(total_chunks / 1000)")
                lines.append(f"  - Companies grouped by granularity for efficient querying")
            
            if smart_config.get("uses_estimates"):
                lines.append(f"\nNote: Uses estimated volumes for companies needing multiple periods")
                lines.append(f"      (based on uniform distribution assumption)")
            
            if smart_config.get("efficiency_metrics"):
                metrics = smart_config["efficiency_metrics"]
                lines.append(f"\nEfficiency Metrics:")
                lines.append(f"  Total chunks: {metrics['total_chunks']:,}")
                lines.append(f"  Semantic Search Queries: {metrics['semantic_search_queries']:,}")
                lines.append(f"  Avg chunks per query: {metrics['avg_chunks_per_query']:.2f}")
                lines.append(f"  Utilization: {metrics['utilization_percent']:.2f}%")
            
            if smart_config.get("basket_distribution"):
                lines.append(f"\nBasket Distribution:")
                for volume_range, count in smart_config["basket_distribution"].items():
                    lines.append(f"  {volume_range}: {count} baskets")

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            lines.append(f"\n\nFull report saved to: {output_path}")

        return "\n".join(lines)

    def export_to_csvs(
        self,
        report: Dict,
        entities_csv_path: str = "output/entities_baskets.csv",
        baskets_csv_path: str = "output/baskets_details.csv",
    ) -> Tuple[str, str]:
        """
        Export planning results to two CSV files:
        1. Entities CSV: entity_id, chunks, total_chunks, basket_id, period_start, period_end
        2. Baskets CSV: basket_id, start_date, end_date, entities (comma-separated), total_chunks, company_count

        Args:
            report: Planning report dictionary
            entities_csv_path: Path for entities CSV file
            baskets_csv_path: Path for baskets CSV file

        Returns:
            Tuple of (entities_csv_path, baskets_csv_path)
        """
        smart_config = report["configurations"].get("smart")
        if not smart_config:
            raise ValueError("No 'smart' configuration found in report")

        # Collect all entities and their basket assignments with chunk volumes
        entity_to_baskets = []  # List of (entity_id, chunks, basket_id, period_start, period_end)
        baskets_info = []  # List of basket info dicts
        company_total_chunks = defaultdict(int)  # Track total chunks per company
        
        baskets_seen = set()
        
        for period_detail in smart_config["period_details"]:
            period_start = period_detail["start_date"]
            period_end = period_detail["end_date"]
            
            for basket in period_detail["baskets"]:
                # Use period dates from basket if available, otherwise from period_detail
                basket_period_start = basket.get("period_start", period_start)
                basket_period_end = basket.get("period_end", period_end)
                basket_id = basket["basket_id"]
                total_chunks = basket["total_chunks"]
                company_count = basket["company_count"]
                companies = basket["companies"]
                
                # Get individual company chunks if available, otherwise estimate
                company_chunks = basket.get("company_chunks", {})
                
                # Add entity entries
                for company_id in companies:
                    # Use actual chunks if available, otherwise estimate
                    if company_id in company_chunks:
                        chunks = company_chunks[company_id]
                    else:
                        # Fallback: estimate chunks per company
                        chunks = int(round(total_chunks / company_count)) if company_count > 0 else 0
                    
                    # Track total chunks per company
                    company_total_chunks[company_id] += chunks
                    
                    entity_to_baskets.append({
                        "entity_id": company_id,
                        "chunks": chunks,
                        "basket_id": basket_id,
                        "period_start": basket_period_start,
                        "period_end": basket_period_end,
                    })
                
                # Add basket info (avoid duplicates)
                if basket_id not in baskets_seen:
                    baskets_seen.add(basket_id)
                    baskets_info.append({
                        "basket_id": basket_id,
                        "start_date": basket_period_start,
                        "end_date": basket_period_end,
                        "entities": ",".join(companies),
                        "total_chunks": total_chunks,
                        "company_count": company_count,
                    })

        # Write entities CSV
        with open(entities_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["entity_id", "chunks", "total_chunks", "basket_id", "period_start", "period_end"])
            
            # Sort by total chunks descending (largest to smallest), then by entity_id for consistency
            for entry in sorted(entity_to_baskets, key=lambda x: (-company_total_chunks[x["entity_id"]], x["entity_id"])):
                writer.writerow([
                    entry["entity_id"],
                    entry["chunks"],
                    company_total_chunks[entry["entity_id"]],
                    entry["basket_id"],
                    entry["period_start"],
                    entry["period_end"],
                ])

        # Write baskets CSV
        with open(baskets_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["basket_id", "start_date", "end_date", "entities", "total_chunks", "company_count"])
            
            # Sort by basket_id for consistency
            for basket_info in sorted(baskets_info, key=lambda x: x["basket_id"]):
                writer.writerow([
                    basket_info["basket_id"],
                    basket_info["start_date"],
                    basket_info["end_date"],
                    basket_info["entities"],
                    basket_info["total_chunks"],
                    basket_info["company_count"],
                ])

        return entities_csv_path, baskets_csv_path
