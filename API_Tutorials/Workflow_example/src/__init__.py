"""
Smart Batching module for BigData search operations.

This module provides:
- BigDataSession: Authentication wrapper for Bigdata.com API
- SmartBatchingPlanner: Intelligent search planning
- plan_search, execute_search: High-level search functions
- Helper functions for result processing and visualization
"""

from .bigdata_session import AUTH_MODE_API_KEY, BigDataSession
from .helper import (
    explode_to_dataframe,
    prepare_sentiment_dataframe,
    get_top_entities_by_volume,
    display_top_entities_dashboard,
    entity_statistics,
)
from .processing_results import (
    to_list_if_multiple,
    aggregate_results_by_chunk,
    extract_all_entities_from_df_columns,
    get_only_unique_entities_from_list,
    get_unknown_entities_from_list,
    get_unknown_entities_from_df_column,
    extract_company_ids,
    process_entities_id_search,
    process_batch,
    extract_companies_from_entity_list,
    map_create_only_companies_column,
    keep_only_companies_in_detections,
    process_entities_and_filter_companies,
)

__all__ = [
    # Authentication
    'BigDataSession',
    'AUTH_MODE_API_KEY',
    # Helper functions
    'explode_to_dataframe',
    'prepare_sentiment_dataframe',
    'get_top_entities_by_volume',
    'display_top_entities_dashboard',
    'entity_statistics',
    # Processing results
    'to_list_if_multiple',
    'aggregate_results_by_chunk',
    'extract_all_entities_from_df_columns',
    'get_only_unique_entities_from_list',
    'get_unknown_entities_from_list',
    'get_unknown_entities_from_df_column',
    'extract_company_ids',
    'process_entities_id_search',
    'process_batch',
    'extract_companies_from_entity_list',
    'map_create_only_companies_column',
    'keep_only_companies_in_detections',
    'process_entities_and_filter_companies',
]
