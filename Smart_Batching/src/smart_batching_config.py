"""
Configuration constants for smart batching.

This module contains all configuration parameters for the smart batching system,
including API endpoints, limits, time periods, and file paths.
"""

import os

# API Configuration
# Use environment variable or default to production API
API_BASE_URL = os.getenv("BIGDATA_API_BASE_URL", "https://api.bigdata.com")
COMENTION_ENDPOINT = "/v1/search/co-mentions/entities"
VOLUME_ENDPOINT = "/v1/search/volume"
MAX_ENTITIES_PER_QUERY = 1000  # Max entities the API can return in a single response
MAX_ENTITIES_IN_ANY_OF = 500   # Max entities we can send in any_of filter (API complexity limit)
MAX_CHUNKS_PER_BASKET = 1000   # Maximum chunks per basket to stay within query limits

# Authentication
# Note: API_KEY is not required at import time - it's passed to SmartBatchingPlanner
# This allows the module to be imported without errors
API_KEY = os.environ.get("BIGDATA_API_KEY")  # Optional at import time

# Time Period Configuration
# Default date range: 2 years from January 2021 to December 2022
START_DATE = "2021-01-01"
END_DATE = "2022-12-31"

# Volume Buckets for basket creation
# Used to group companies by their chunk volume for efficient batching
VOLUME_BUCKETS = {
    "high": (500, float("inf")),      # 500-1000+ chunks
    "medium": (100, 500),              # 100-500 chunks
    "low": (1, 100),                   # 1-100 chunks
    "very_low": (0, 1),               # 0 chunks (headlines only, no semantic content)
}

# Period Configurations (total expected sub-periods over 2 years)
# Used for reference when determining optimal granularity
PERIOD_CONFIGS = {
    "biyearly": 1,      # 1 period (entire 2 years)
    "yearly": 2,        # 2 periods (1 year each)
    "quarterly": 8,     # 8 periods (4 quarters × 2 years)
    "monthly": 24,      # 24 periods (12 months × 2 years)
    "weekly": 104,      # 104 periods (52 weeks × 2 years)
}

# File Paths
UNIVERSE_CSV_PATH = "us_top3000.csv"  # Default universe file (example: US top 3000 companies)
