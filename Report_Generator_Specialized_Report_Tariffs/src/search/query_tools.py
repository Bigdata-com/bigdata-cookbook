"""
Module that includes simple, flexible functions to build 
hybrid query objects and generate date ranges at any desired frequency.

Copyright (C) 2024, RavenPack | Bigdata.com. All rights reserved.
Author: Jelena Starovic (jstarovic@ravenpack.com)
"""

from typing import List, Optional, Tuple
import pandas as pd
from bigdata_client.daterange import AbsoluteDateRange
from bigdata_client.models.advanced_search_query import QueryComponent
from bigdata_client.models.search import DocumentType
from bigdata_client import Organization
from bigdata_client.query import (
    Keyword,
    Entity,
    Any,
    Similarity,
    FiscalYear,
    ReportingEntity,
    Source,
    Topic,
)


def build_similarity_queries(sentences: List[str]) -> List[Similarity]:
    """
    Function to remove any duplicate sentences from the input list and
    convert their type to Similarity datatype.

    :param sentences:
        A list of strings, each representing a sentence to run a similarity search with
    :return:
        A list of sentences of Type Similarity
    """

    sentences = list(set(sentences))  # De-duplicate
    queries = [Similarity(sentence) for sentence in sentences]
    return queries


def build_batched_query(
    sentences: Optional[List[str]],
    keywords: Optional[List[str]],
    entity_keys: List[str],
    control_entities: Optional[List[str]],
    batch_size: int = 10,
    fiscal_year: int = None,
    scope: DocumentType = DocumentType.ALL,
) -> List[QueryComponent]:
    """
    Convenience function to build a list of queries objects of
    type QueryComponent. There is one query for each
    similarity query, batch pair.

    :param sentences:
        A list of sentence strings for similarity search querying.
    :param keywords:
        A list of keyword strings for keyword search querying.
    :param entity_keys:
        A list of entity key strings.
    :param control_entities:
        An entity we want to be included with every batch.       
    :param batch_size:
        Integer controlling the number of entities per batch query.
    :param fiscal_year:
        The fiscal year used when querying transcript documents.
    :param scope:
        The scope of the documents to include.
        Defaults to DocumentType.ALL.
    :return:
        A list of queries of type QueryComponent.
    """

    queries = []

    # Build similarity queries if sentences are provided
    if sentences:
        queries = build_similarity_queries(sentences)
    else:
        # If sentences are not provided, initialize a default query
        queries = []  # Default base query

    if keywords:
        keyword_query = Any([Keyword(word) for word in keywords])
    else:
        # If sentences are not provided, initialize a default query
        keyword_query = None

    if control_entities:
        control_query = Any([Entity(entity_id) for entity_id in control_entities])
    else:
        # If sentences are not provided, initialize a default query
        control_query = None

    # Batch entity keys
    entity_keys_batched = [
        entity_keys[i : i + batch_size] for i in range(0, len(entity_keys), batch_size)
    ]

    queries_expanded = []
    for batch in entity_keys_batched:
        entity_batch = [None]

        if batch:
            entity_type = (
                ReportingEntity
                if scope in (DocumentType.TRANSCRIPTS, DocumentType.FILINGS)
                else Entity
            )
            entity_batch = Any([entity_type(entity_key) for entity_key in batch])

        # If there are no base queries, start with a default empty query
        base_queries = queries if queries else [None]
        for base_query in base_queries:
            expanded_query = base_query if base_query else None

            if expanded_query:
                expanded_query &= entity_batch
            else:
                expanded_query = entity_batch

            if keyword_query:
                expanded_query &= keyword_query

            if control_query:
                expanded_query &= control_query
            
            # Add fiscal year filter if provided
            if fiscal_year:
                expanded_query &= FiscalYear(fiscal_year)

            # Append the expanded query to the final list
            queries_expanded.append(expanded_query)

    return queries_expanded

def create_date_intervals(
    sd: str, ed: str, freq: str
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Create date intervals for a given frequency.

    :param sd:
        The start date in 'YYYY-MM-DD' format.
    :param ed:
        The end date in 'YYYY-MM-DD' format.
    :param freq:
        Frequency string ('Y' for yearly, 'M' for monthly, 'W' for weekly, 'D' for daily).
    :return:
        List[Tuple[pd.Timestamp, pd.Timestamp]]: List of start and end date tuples.
    """
    # Convert start and end dates to pandas Timestamps
    start_date = pd.Timestamp(sd)
    end_date = pd.Timestamp(ed)

    # Adjust frequency for yearly and monthly to use appropriate start markers
    # 'AS' for year start, 'MS' for month start
    adjusted_freq = {"Y": "AS", "M": "MS"}.get(freq, freq)

    # Generate date range based on the adjusted frequency
    try:
        date_range = pd.date_range(start=start_date, end=end_date, freq=adjusted_freq)
    except ValueError:
        raise ValueError("Invalid frequency. Use 'Y', 'M', 'W', or 'D'.")
        
    try:
        # Create intervals
        intervals = []
        for i in range(len(date_range) - 1):
            intervals.append(
                (
                    date_range[i].replace(hour=0, minute=0, second=0),
                    (date_range[i + 1] - pd.Timedelta(seconds=1)).replace(
                        hour=23, minute=59, second=59
                    ),
                )
            )

        # Handle the last range to include the full end_date
        intervals.append(
            (
                date_range[-1].replace(hour=0, minute=0, second=0),
                end_date.replace(hour=23, minute=59, second=59),
            )
        )

        return intervals
    except: 
        raise ValueError(f"Frequency {adjusted_freq} is longer than date range ({sd}, {ed}). Please provide a longer date range, or a shorter frequency")

def create_date_ranges(
    start_date: str, end_date: str, freq: str
) -> List[AbsoluteDateRange]:
    """
    Create a list of AbsoluteDateRange objects for the given frequency.

    :param start_date:
        The start date in 'YYYY-MM-DD' format.
    :param end_date:
        The end date in 'YYYY-MM-DD' format.
    :param freq:
        Frequency string ('Y' for yearly, 'M' for monthly, 'W' for weekly, 'D' for daily).
    :return:
    - List[AbsoluteDateRange]: List of AbsoluteDateRange objects.
    """
    intervals = create_date_intervals(start_date, end_date, freq=freq)
    return [AbsoluteDateRange(start, end) for start, end in intervals]
