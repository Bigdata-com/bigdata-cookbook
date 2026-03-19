"""
Smart Batching module for BigData search operations.

This module provides:
- SmartBatchingPlanner: Intelligent search planning
- plan_search, execute_search: High-level search functions
- Helper functions for result processing
"""

from .search_function import (
    plan_search,
    execute_search,
    deduplicate_documents,
    save_plan,
    load_plan,
    load_universe_from_csv,
    execute_full_grid_search,
)
from .output_converter import convert_to_dataframe

__all__ = [
    'plan_search',
    'execute_search',
    'deduplicate_documents',
    'save_plan',
    'load_plan',
    'load_universe_from_csv',
    'convert_to_dataframe',
    'execute_full_grid_search',
]
