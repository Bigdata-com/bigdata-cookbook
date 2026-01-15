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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import requests

from smart_batching_config import (
    API_BASE_URL,
    COMENTION_ENDPOINT,
    MAX_ENTITIES_PER_QUERY,
    MAX_ENTITIES_IN_ANY_OF,
    MAX_CHUNKS_PER_BASKET,
    START_DATE,
    END_DATE,
    VOLUME_BUCKETS,
    PERIOD_CONFIGS,
    UNIVERSE_CSV_PATH,
)


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

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the planner.

        Args:
            api_key: BigData API key. If None, will try to get from environment variable BIGDATA_API_KEY.
        """
        self.api_key = api_key or os.getenv("BIGDATA_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in BIGDATA_API_KEY environment variable")
        
        self.api_url = f"{API_BASE_URL}{COMENTION_ENDPOINT}"
        self.headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

    def load_universe(self, csv_path: str = UNIVERSE_CSV_PATH) -> List[str]:
        """
        Read companies from CSV file.

        Args:
            csv_path: Path to CSV file containing company IDs (one per line)

        Returns:
            List of company IDs
        """
        companies = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
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
                
                # Track which companies from our batch appeared in the response
                found_company_ids = set()
                for company_data in companies_data:
                    company_id = company_data.get("id")
                    chunks_count = company_data.get("total_chunks_count", 0)
                    if company_id:
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
                    
                    # Track which companies from this batch appeared
                    found_in_verification = set()
                    verified_count = 0
                    for company_data in companies_data:
                        company_id = company_data.get("id")
                        chunks_count = company_data.get("total_chunks_count", 0)
                        if company_id and company_id in batch:
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
                        chunks_count = company_data.get("total_chunks_count", 0)
                        if company_id and company_id in batch:
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
    ) -> List[Dict]:
        """
        Create baskets of companies with total chunks < max_chunks.

        Args:
            company_volumes: Dict mapping company_id -> chunks (already filtered to exclude 0-chunk companies)
            max_chunks: Maximum total chunks per basket

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
        
        if not filtered_volumes:
            return []

        # Group by volume
        volume_groups = self.group_by_volume(filtered_volumes)
        
        baskets = []
        basket_counter = 0

        # Process each volume group
        for volume_range, companies_list in volume_groups.items():
            current_basket = {
                "companies": [],
                "company_chunks": {},
                "total_chunks": 0,
            }

            for company_id, chunks in companies_list:
                # Check if adding this company would exceed the limit
                if current_basket["total_chunks"] + chunks > max_chunks:
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
            period_type: One of 'biyearly', 'yearly', 'quarterly', 'monthly', 'weekly'

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
        Determine the actual period granularity to use based on periods needed.
        Always uses ALL periods from the selected granularity to ensure full coverage.

        Args:
            periods_needed: Number of periods needed for the company
            target_period_type: Target period configuration (biyearly, yearly, quarterly, monthly, weekly)
            start_date: Start date of full period (YYYY-MM-DD)
            end_date: End date of full period (YYYY-MM-DD)

        Returns:
            Tuple of (actual_period_type, list of (start, end) date tuples for periods)
        """
        # Get available periods for each granularity
        yearly_periods = self.split_period(start_date, end_date, "yearly")
        quarterly_periods = self.split_period(start_date, end_date, "quarterly")
        monthly_periods = self.split_period(start_date, end_date, "monthly")
        weekly_periods = self.split_period(start_date, end_date, "weekly")

        # Determine which granularity to use (round up to next available granularity)
        # Available granularities: 1 (biyearly), 2 (yearly), 8 (quarterly), 24 (monthly), 104 (weekly)
        if target_period_type == "biyearly":
            if periods_needed <= 1:
                return ("biyearly", [(start_date, end_date)])
            elif periods_needed <= 2:
                return ("yearly", yearly_periods)  # Use all 2 years
            elif periods_needed <= 8:
                return ("quarterly", quarterly_periods)  # Use all 8 quarters
            elif periods_needed <= 24:
                return ("monthly", monthly_periods)  # Use all 24 months
            else:
                return ("weekly", weekly_periods)  # Use all available weeks
        
        elif target_period_type == "yearly":
            if periods_needed <= 2:
                return ("yearly", yearly_periods)
            elif periods_needed <= 8:
                return ("quarterly", quarterly_periods)  # Use all 8 quarters
            elif periods_needed <= 24:
                return ("monthly", monthly_periods)  # Use all 24 months
            else:
                return ("weekly", weekly_periods)
        
        elif target_period_type == "quarterly":
            if periods_needed <= 8:
                return ("quarterly", quarterly_periods)
            elif periods_needed <= 24:
                return ("monthly", monthly_periods)  # Use all 24 months
            else:
                return ("weekly", weekly_periods)
        
        elif target_period_type == "monthly":
            if periods_needed <= 24:
                return ("monthly", monthly_periods)
            else:
                return ("weekly", weekly_periods)
        
        elif target_period_type == "weekly":
            return ("weekly", weekly_periods)
        
        else:
            # Default: use target period type
            periods = self.split_period(start_date, end_date, target_period_type)
            return (target_period_type, periods)

    def plan_all_periods(
        self,
        topic: str,
        start_date: str = START_DATE,
        end_date: str = END_DATE,
    ) -> Dict:
        """
        Generate SMART batching plan with optimal granularity per company.

        Phase 1: Query full period once to get total volumes
        Phase 2: Automatically determine optimal granularity for each company based on volume
                and create baskets using adaptive splitting

        Args:
            topic: Topic string for comention and semantic search queries
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Planning report with single SMART configuration
        """
        # Load universe
        companies = self.load_universe()
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
        print("PHASE 1: Querying full 2-year period for all companies")
        print("=" * 80)
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

        # Process each group of companies by periods_needed
        for periods_needed, company_group in sorted(companies_by_periods_needed.items()):
            # Determine optimal granularity for this group (using "biyearly" as base, but will adapt)
            actual_period_type, actual_periods = self.determine_split_granularity(
                periods_needed, "biyearly", start_date, end_date
            )
            
            print(f"  Companies needing {periods_needed} period(s): {len(company_group)} companies")
            print(f"    Using {actual_period_type} granularity ({len(actual_periods)} periods)")

            granularity_groups[actual_period_type]["companies"] += len(company_group)
            granularity_groups[actual_period_type]["periods"] = len(actual_periods)

            # Process each period for this group
            for period_idx, (period_start, period_end) in enumerate(actual_periods):
                # Build volumes for this sub-period
                sub_period_volumes = {}
                
                for company_id, total_chunks in company_group.items():
                    if periods_needed == 1:
                        # Single period: use full-period volume
                        sub_period_volumes[company_id] = total_chunks
                    else:
                        # Multiple periods: estimate sub-period volume
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
                
                # Update basket IDs to include period index, granularity, and date range for clarity
                for basket in baskets:
                    # Format dates for basket ID (YYYYMMDD format)
                    period_start_short = period_start.replace("-", "")
                    period_end_short = period_end.replace("-", "")
                    basket["basket_id"] = f"{actual_period_type}_{period_start_short}_{period_end_short}_{basket['basket_id']}"
                    basket["actual_granularity"] = actual_period_type
                    basket["periods_needed"] = periods_needed
                    basket["period_start"] = period_start
                    basket["period_end"] = period_end
                    # Mark if this basket contains estimated volumes
                    basket["contains_estimates"] = periods_needed > 1
                
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
                    "uses_estimates": periods_needed > 1,
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
        lines.append("  - Companies automatically split into finer periods if needed (yearly, quarterly, monthly, weekly)")
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
