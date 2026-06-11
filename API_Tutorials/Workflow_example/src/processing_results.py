"""
Processing Results Module

Functions for processing and filtering search results, extracting entities,
and managing company data from API responses.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
from tqdm import tqdm


# =============================================================================
# AGGREGATION HELPERS
# =============================================================================

def to_list_if_multiple(series: pd.Series) -> list:
    """
    Convert a pandas Series to a list if it has multiple unique values,
    otherwise return a single-element list.
    
    Args:
        series: A pandas Series to process
        
    Returns:
        A list containing unique values from the series
    """
    unique_vals = series.unique()
    if len(unique_vals) > 1:
        return list(unique_vals)
    else:
        return [unique_vals[0]]


def aggregate_results_by_chunk(
    df: pd.DataFrame,
    group_cols: list[str] = None,
    list_cols: list[str] = None
) -> pd.DataFrame:
    """
    Aggregate a DataFrame by chunk_text, converting specified columns
    to lists when multiple values exist.
    
    Args:
        df: Input DataFrame with potentially duplicate rows per chunk
        group_cols: Columns to group by (default: ['chunk_text'])
        list_cols: Columns to aggregate as lists (default: ['label', 'theme'])
        
    Returns:
        Aggregated DataFrame with unique rows per chunk (one per distinct chunk_text)
    """
    if group_cols is None:
        group_cols = ['chunk_text']
    if list_cols is None:
        list_cols = ['label', 'theme']
    
    agg_dict = {col: to_list_if_multiple for col in list_cols}
    
    other_cols = [col for col in df.columns if col not in group_cols + list_cols]
    for col in other_cols:
        agg_dict[col] = 'first'
    
    return df.groupby(group_cols, as_index=False).agg(agg_dict)


# =============================================================================
# ENTITY EXTRACTION
# =============================================================================

def extract_all_entities_from_df_columns(df: pd.DataFrame, column_name: str) -> list:
    """
    Extract all entities from a DataFrame column containing lists.
    
    Args:
        df: Input DataFrame
        column_name: Name of the column containing entity lists
        
    Returns:
        Flattened list of all entities
    """
    list_of_lists = df[column_name].to_list()
    single_list = [item for sublist in list_of_lists for item in sublist]
    return single_list


def get_only_unique_entities_from_list(entity_list: list) -> list:
    """
    Get unique entities from a list.
    
    Args:
        entity_list: List of entities (may contain duplicates)
        
    Returns:
        List of unique entities
    """
    return list(set(entity_list))


def get_unknown_entities_from_list(
    entity_list_from_df: list,
    universe_csv: str,
) -> list:
    """
    Find entities that are not in the known universe.

    Args:
        entity_list_from_df: List of entity IDs from the DataFrame
        universe_csv: Path to CSV file containing known entity IDs

    Returns:
        List of entity IDs not found in the universe
    """
    known_entities = pd.read_csv(universe_csv)["id"].tolist()

    unknown_entities = [
        entity for entity in entity_list_from_df
        if entity not in known_entities
    ]

    return unknown_entities


def get_unknown_entities_from_df_column(
    df: pd.DataFrame,
    column_name: str,
    universe_csv: str,
) -> list:
    """
    Extract unknown entities from a DataFrame column.

    Combines extraction, deduplication, and filtering against known universe.

    Args:
        df: Input DataFrame
        column_name: Name of the column containing entity lists
        universe_csv: Path to CSV file containing known entity IDs

    Returns:
        List of unique entity IDs not found in the universe
    """
    df_entities_list = extract_all_entities_from_df_columns(df, column_name)
    df_entities_list_unique = get_only_unique_entities_from_list(df_entities_list)
    unknown_entities_list = get_unknown_entities_from_list(
        df_entities_list_unique, universe_csv
    )
    return unknown_entities_list


# =============================================================================
# COMPANY EXTRACTION FROM API
# =============================================================================

def extract_company_ids(api_response: dict) -> list[dict]:
    """
    Extract entity info for entities with category='companies' from API response.
    
    Args:
        api_response: Response dictionary from the entities API
        
    Returns:
        List of entity info dictionaries for companies only
    """
    return [
        entity_info 
        for entity_id, entity_info in api_response.get('results', {}).items()
        if entity_info.get('category') == 'companies'
    ]


def process_entities_id_search(entity_ids: list, session: Any) -> dict:
    """
    Query the API for entity information by IDs.
    
    Args:
        entity_ids: List of entity IDs to look up
        session: BigDataSession session object
        
    Returns:
        API response as dictionary
    """
    response = session.get_entities_by_ids(entity_ids)
    data = response.json()
    return data


def process_batch(batch: list, session: Any) -> list[dict]:
    """
    Process a single batch of entity IDs and return company information.
    
    Args:
        batch: List of entity IDs to process
        session: BigDataSession session object
        
    Returns:
        List of company info dictionaries
    """
    data = process_entities_id_search(batch, session)
    return extract_company_ids(data)


def extract_companies_from_entity_list(
    entity_list: list,
    session: Any,
    max_workers: int = 1,
    batch_size: int = 100,
) -> list[dict]:
    """
    Extract company entities from a list of entity IDs using parallel processing.

    Args:
        entity_list: List of entity IDs to process
        session: BigDataSession session object
        max_workers: Number of parallel workers
        batch_size: Number of entities per batch

    Returns:
        List of company info dictionaries
    """
    companies_ids_list = []

    batches = [
        entity_list[i:i+batch_size]
        for i in range(0, len(entity_list), batch_size)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(process_batch, batch, session): idx
            for idx, batch in enumerate(batches)
        }

        for future in tqdm(
            as_completed(future_to_batch),
            total=len(batches),
            desc="Processing batches...",
        ):
            batch_idx = future_to_batch[future]
            try:
                company_ids = future.result()
                companies_ids_list.extend(company_ids)
            except Exception as e:
                print(f"Batch {batch_idx} failed: {e}")

    return companies_ids_list


# =============================================================================
# COMPANY FILTERING
# =============================================================================

def map_create_only_companies_column(
    df: pd.DataFrame,
    target_universe_csv: str,
    other_companies: list[dict],
) -> pd.DataFrame:
    """
    Create a new column containing only company entity IDs.

    Filters entity_ids to keep only those that are companies (either from
    the target universe or from the other_companies list).

    Args:
        df: Input DataFrame with 'entity_ids' column
        target_universe_csv: Path to CSV file with target company IDs
        other_companies: List of company info dicts from API

    Returns:
        DataFrame with new 'entity_ids_companies' column
    """
    target_companies = pd.read_csv(target_universe_csv)["id"].tolist()
    other_companies_id = [company["id"] for company in other_companies]

    all_companies = target_companies + other_companies_id
    set_companies = set(str(x) for x in all_companies)
    
    df_temp = df.copy()
    df_temp['entity_ids_companies'] = df_temp['entity_ids'].apply(
        lambda x: [elem for elem in x if str(elem) in set_companies]
    )
    return df_temp


def keep_only_companies_in_detections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter detections to keep only company entities.
    
    Creates a new 'companies_detection' column containing only detections
    for entities that are in the 'entity_ids_companies' column.
    
    Args:
        df: Input DataFrame with 'entity_ids_companies' and 'detections' columns
        
    Returns:
        DataFrame with new 'companies_detection' column
    """
    df_temp = df.copy()
    
    def filter_detections(row):
        valid_ids = set(row["entity_ids_companies"]) 
        return [elem for elem in row["detections"] if elem["id"] in valid_ids]
    
    df_temp["companies_detection"] = df_temp.apply(filter_detections, axis=1)
    return df_temp


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def process_entities_and_filter_companies(
    df: pd.DataFrame,
    entity_column: str,
    universe_csv: str,
    session: Any,
    max_workers: int = 1,
) -> pd.DataFrame:
    """
    Complete pipeline to extract unknown entities and filter to companies only.

    This is a convenience function that combines:
    1. Getting unknown entities from the DataFrame
    2. Extracting company information from the API
    3. Creating a companies-only column
    4. Filtering detections to companies only

    Args:
        df: Input DataFrame with entity data
        entity_column: Name of the column containing entity IDs
        universe_csv: Path to CSV file with known entity IDs
        session: BigDataSession session object
        max_workers: Number of parallel workers for API calls

    Returns:
        DataFrame with filtered company data
    """
    # Get unknown entities
    entities = get_unknown_entities_from_df_column(
        df, entity_column, universe_csv
    )

    # Extract company information from API
    other_companies = extract_companies_from_entity_list(
        entities, session, max_workers=max_workers
    )

    # Create companies-only column
    df_with_companies = map_create_only_companies_column(
        df, universe_csv, other_companies
    )
    
    # Filter detections to companies only
    df_filtered = keep_only_companies_in_detections(df_with_companies)
    
    return df_filtered
