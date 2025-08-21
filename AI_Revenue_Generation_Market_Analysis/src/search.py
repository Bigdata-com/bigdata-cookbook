from logging import Logger, getLogger
from typing import List, Optional, Dict

from bigdata_client.document import Document
from bigdata_client.models.advanced_search_query import ListQueryComponent
from bigdata_client.models.entities import Company
from bigdata_client.models.search import DocumentType, SortBy
from pandas import DataFrame
from tqdm import tqdm

from bigdata_research_tools.client import bigdata_connection
from bigdata_research_tools.prompts.labeler import (
    get_other_entity_placeholder,
    get_target_entity_placeholder,
)
from bigdata_research_tools.search.query_builder import (
    build_batched_query,
    EntitiesToSearch,
    create_date_ranges,
)
from bigdata_research_tools.search.search import run_search
from bigdata_research_tools.tracing import Trace, TraceEventNames, send_trace
from bigdata_research_tools.search.search_utils import filter_search_results


logger: Logger = getLogger(__name__)


def search_cooling(
    sentences: List[str],
    start_date: str,
    end_date: str,
    scope: DocumentType = DocumentType.ALL,
    companies: Optional[List[Company]] = None,
    fiscal_year: Optional[int] = None,
    sources: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    control_entities: Optional[Dict] = None,
    freq: str = "M",
    sort_by: SortBy = SortBy.RELEVANCE,
    rerank_threshold: Optional[float] = None,
    document_limit: int = 50,
    batch_size: int = 10,
    **kwargs,
) -> DataFrame:
    """
    Screen for documents based on the input sentences and other filters.

    Args:
        sentences (List[str]): The list of sentences to screen for.
        start_date (str): The start date for the search.
        end_date (str): The end date for the search.
        scope (DocumentType): The document type scope
            (e.g., `DocumentType.ALL`, `DocumentType.TRANSCRIPTS`).
        companies (Optional[List[Company]]): The list of companies to use. 
            If None, all companies found in search results will be processed, 
            with each company treated as a target entity in separate rows.
        fiscal_year (int): The fiscal year to filter queries.
            If None, no fiscal year filter is applied.
        sources (Optional[List[str]]): List of sources to filter on. If none, we search across all sources.
        keywords (List[str]): A list of keywords for constructing keyword queries.
            If None, no keyword queries are created.
        control_entities (Dict): A dictionary of control entities of different types for creating co-mentions queries.
        freq (str): The frequency of the date ranges. Defaults to '3M'.
        sort_by (SortBy): The sorting criterion for the search results.
            Defaults to SortBy.RELEVANCE.
        rerank_threshold (Optional[float]): The threshold for reranking the search results.
            See https://sdk.bigdata.com/en/latest/how_to_guides/rerank_search.html
        document_limit (int): The maximum number of documents to return per Bigdata query.
        batch_size (int): The number of entities to include in each batched query.

    Returns:
        DataFrame: The DataFrame with the screening results.
        - Index: int
        - Columns:
            - timestamp_utc: datetime64
            - document_id: str
            - sentence_id: str
            - headline: str
            - entity_id: str
            - document_type: str
            - is_reporting_entity: bool
            - entity_name: str
            - entity_sector: str
            - entity_industry: str
            - entity_country: str
            - entity_ticker: str
            - text: str
            - other_entities: str
            - entities: List[Dict[str, Any]]
                - key: str
                - name: str
                - ticker: str
                - start: int
                - end: int
            - masked_text: str
            - other_entities_map: List[Tuple[int, str]]
    """

    if not kwargs.get("current_trace"):
        current_trace = Trace(
            event_name=TraceEventNames.COMPANY_SEARCH,
            document_type=scope,
            start_date=start_date,
            end_date=end_date,
            rerank_threshold=rerank_threshold,
            llm_model=None,
            frequency=freq,
            workflow_start_date=Trace.get_time_now(),
        )

        kwargs["current_trace"] = current_trace

    try:
        # Extract entities for search querying
        entity_keys = [entity.id for entity in companies] if companies else []

        # Create entity configs
        entities_config = EntitiesToSearch(companies=entity_keys) if entity_keys else None
        
        # If control_entities are provided, create a control EntityConfig
        # For this example, assuming control_entities are all company entities
        control_entities_config = None
        if control_entities:
            control_entities_config = EntitiesToSearch(**control_entities)

        # Build batched queries
        batched_query = build_batched_query(
            sentences=sentences,
            keywords=keywords,
            entities=entities_config,
            control_entities=control_entities_config,
            custom_batches=None,
            sources=sources,
            batch_size=batch_size,
            fiscal_year=fiscal_year,
            scope=scope,
        )

        # Create list of date ranges
        date_ranges = create_date_ranges(start_date, end_date, freq)

        no_queries = len(batched_query)
        no_dates = len(date_ranges)
        total_no = no_dates * no_queries

        logger.info(f"About to run {total_no} queries")
        logger.debug("Example Query:", batched_query[0])
        # Run concurrent search
        results = run_search(
            batched_query,
            date_ranges=date_ranges,
            limit=document_limit,
            scope=scope,
            sortby=sort_by,
            rerank_threshold=rerank_threshold,
            **kwargs,
        )

        results, entities = filter_search_results(results)
        # Filter entities to only include COMPANY entities
        entities = filter_company_entities(entities)
        
        # Determine whether to filter by companies based on document type
        # For filings and transcripts, we don't need to filter as we use reporting entities
        # For news, we need to check against our original universe of companies as a news article
        # may mention other companies we're not interested in
        needs_company_filtering = scope not in (
            DocumentType.FILINGS,
            DocumentType.TRANSCRIPTS,
        )

        df_sentences = process_screener_search_results(
            results=results,
            entities=entities,
            companies=companies if needs_company_filtering else None,
            document_type=scope,
            duplicate_for_all_entities=companies is None,
        )
    except Exception:
        execution_result = "error"
        raise
    else:
        execution_result = "success"
    finally:
        current_trace = kwargs.get("current_trace")
        if current_trace and current_trace.event_name == TraceEventNames.COMPANY_SEARCH:
            current_trace.workflow_end_date = Trace.get_time_now()
            current_trace.result = execution_result  # noqa
            send_trace(bigdata_connection(), current_trace)

    return df_sentences


def search_by_any(
    sentences: List[str],
    start_date: str,
    end_date: str,
    scope: DocumentType = DocumentType.ALL,
    companies: Optional[List[Company]] = None,
    fiscal_year: Optional[int] = None,
    sources: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    control_entities: Optional[Dict] = None,
    freq: str = "M",
    sort_by: SortBy = SortBy.RELEVANCE,
    rerank_threshold: Optional[float] = None,
    document_limit: int = 50,
    batch_size: int = 10,
    **kwargs,
) -> DataFrame:
    """
    Screen for documents based on input sentences and other filters.

    Args:
        sentences (List[str]): The list of sentences to screen for.
        start_date (str): The start date for the search.
        end_date (str): The end date for the search.
        scope (DocumentType): The document type scope
            (e.g., `DocumentType.ALL`, `DocumentType.TRANSCRIPTS`).
        companies (Optional[List[Company]]): The list of companies to use. 
            If None, all companies found in search results will be processed, 
            with each company treated as a target entity in separate rows.
        fiscal_year (int): The fiscal year to filter queries.
            If None, no fiscal year filter is applied.
        sources (Optional[List[str]]): List of sources to filter on. If none, we search across all sources.
        keywords (List[str]): A list of keywords for constructing keyword queries.
            If None, no keyword queries are created.
        control_entities (Dict): A dictionary of control entities of different types for creating co-mentions queries.
        freq (str): The frequency of the date ranges. Defaults to 'M'.
        sort_by (SortBy): The sorting criterion for the search results.
            Defaults to SortBy.RELEVANCE.
        rerank_threshold (Optional[float]): The threshold for reranking the search results.
            See https://sdk.bigdata.com/en/latest/how_to_guides/rerank_search.html
        document_limit (int): The maximum number of documents to return per Bigdata query.
        batch_size (int): The number of entities to include in each batched query.

    Returns:
        DataFrame: The DataFrame with the screening results.
        - Index: int
        - Columns:
            - timestamp_utc: datetime64
            - document_id: str
            - sentence_id: str
            - headline: str
            - entity_id: str
            - document_type: str
            - is_reporting_entity: bool
            - entity_name: str
            - entity_sector: str
            - entity_industry: str
            - entity_country: str
            - entity_ticker: str
            - text: str
            - other_entities: str
            - entities: List[Dict[str, Any]]
                - key: str
                - name: str
                - ticker: str
                - start: int
                - end: int
            - masked_text: str
            - other_entities_map: List[Tuple[int, str]]
    """

    if not kwargs.get("current_trace"):
        current_trace = Trace(
            event_name=TraceEventNames.COMPANY_SEARCH,
            document_type=scope,
            start_date=start_date,
            end_date=end_date,
            rerank_threshold=rerank_threshold,
            llm_model=None,
            frequency=freq,
            workflow_start_date=Trace.get_time_now(),
        )

        kwargs["current_trace"] = current_trace

    try:
        # Extract entities for search querying
        entity_keys = [entity.id for entity in companies] if companies else []

        # Create entity configs
        entities_config = EntitiesToSearch(companies=entity_keys) if entity_keys else None
        
        # If control_entities are provided, create a control EntityConfig
        # For this example, assuming control_entities are all company entities
        control_entities_config = None
        if control_entities:
            control_entities_config = EntitiesToSearch(**control_entities)

        # Build batched queries
        batched_query = build_batched_query(
            sentences=sentences,
            keywords=keywords,
            entities=entities_config,
            control_entities=control_entities_config,
            custom_batches=None,
            sources=sources,
            batch_size=batch_size,
            fiscal_year=fiscal_year,
            scope=scope,
        )

        # Create list of date ranges
        date_ranges = create_date_ranges(start_date, end_date, freq)

        no_queries = len(batched_query)
        no_dates = len(date_ranges)
        total_no = no_dates * no_queries

        logger.info(f"About to run {total_no} queries")
        logger.debug("Example Query:", batched_query[0])
        # Run concurrent search
        results = run_search(
            batched_query,
            date_ranges=date_ranges,
            limit=document_limit,
            scope=scope,
            sortby=sort_by,
            rerank_threshold=rerank_threshold,
            **kwargs,
        )

        results, entities = filter_search_results(results)
        # Filter entities to only include COMPANY entities
        entities = filter_company_entities(entities)
        
        # Determine whether to filter by companies based on document type
        # For filings and transcripts, we don't need to filter as we use reporting entities
        # For news, we need to check against our original universe of companies as a news article
        # may mention other companies we're not interested in
        needs_company_filtering = scope not in (
            DocumentType.FILINGS,
            DocumentType.TRANSCRIPTS,
        )

        df_sentences = process_screener_search_results(
            results=results,
            entities=entities,
            companies=companies if needs_company_filtering else None,
            document_type=scope,
            duplicate_for_all_entities=companies is None,
        )
    except Exception:
        execution_result = "error"
        raise
    else:
        execution_result = "success"
    finally:
        current_trace = kwargs.get("current_trace")
        if current_trace and current_trace.event_name == TraceEventNames.COMPANY_SEARCH:
            current_trace.workflow_end_date = Trace.get_time_now()
            current_trace.result = execution_result  # noqa
            send_trace(bigdata_connection(), current_trace)

    return df_sentences


# Alias for backward compatibility
search_ai_cost_cutting = search_by_any


def filter_company_entities(
    entities: List[ListQueryComponent],
) -> List[ListQueryComponent]:
    """
    Filter only COMPANY entities from the list of entities.

    Args:
        entities (List[ListQueryComponent]): A list of entities to filter.
    Returns:
        List[ListQueryComponent]: A list of COMPANY entities.
    """
    return [
        entity
        for entity in entities
        if hasattr(entity, "entity_type") and getattr(entity, "entity_type") == "COMP"
    ]


def process_screener_search_results(
    results: List[Document],
    entities: List[ListQueryComponent],
    companies: Optional[List[Company]] = None,
    document_type: DocumentType = DocumentType.NEWS,
    duplicate_for_all_entities: bool = False,
) -> DataFrame:
    """
    Build a unified DataFrame from search results for any document type.

    Args:
        results (List[Document]): A list of Bigdata search results.
        entities (List[ListQueryComponent]): A list of entities.
        companies (Optional[List[Company]]): A list of companies to filter for.
            Only used for non-reporting entity documents.
        document_type (DocumentType): The type of documents being processed.
        duplicate_for_all_entities (bool): If True, creates duplicate rows for each
            entity found in chunks, with each entity treated as target in turn.

    Returns:
        DataFrame: Standardized screening DataFrame with consistent schema:
        - Index: int
        - Columns:
            - timestamp_utc: datetime64
            - document_id: str
            - sentence_id: str
            - headline: str
            - entity_id: str
            - document_type: str (metadata field showing the document type)
            - is_reporting_entity: bool (True if entity is a reporting entity)
            - entity_name: str
            - entity_sector: str
            - entity_industry: str
            - entity_country: str
            - entity_ticker: str
            - text: str
            - other_entities: str
            - entities: List[Dict[str, Any]]
            - masked_text: str
            - other_entities_map: List[Tuple[int, str]]
    """
    entity_key_map = {entity.id: entity for entity in entities}

    rows = []
    for result in tqdm(results, desc=f"Processing {document_type} results..."):
        for chunk in result.chunks:
            # Build a list of entities present in the chunk
            chunk_entities = [
                {
                    "key": entity.key,
                    "name": (
                        entity_key_map[entity.key].name
                        if entity.key in entity_key_map
                        else None
                    ),
                    "ticker": (
                        entity_key_map[entity.key].ticker
                        if entity.key in entity_key_map
                        else None
                    ),
                    "start": entity.start,
                    "end": entity.end,
                }
                for entity in chunk.entities
                if entity.key in entity_key_map
            ]

            if not chunk_entities:
                continue  # Skip if no entities are mapped

            # Handle differently based on document type
            if document_type in (DocumentType.FILINGS, DocumentType.TRANSCRIPTS):
                # Process reporting entities
                for re_key in result.reporting_entities:
                    reporting_entity = entity_key_map.get(re_key)

                    if not reporting_entity:
                        continue  # Skip if reporting entity is not found

                    # Exclude the reporting entity from other entities
                    other_entities = [
                        e for e in chunk_entities if e["name"] != reporting_entity.name
                    ]

                    # Collect information in standard format
                    rows.append(
                        {
                            "timestamp_utc": result.timestamp,
                            "document_id": result.id,
                            "sentence_id": f"{result.id}-{chunk.chunk}",
                            "headline": result.headline,
                            "entity_id": re_key,
                            "document_type": document_type.value,
                            "is_reporting_entity": True,
                            "entity_name": reporting_entity.name,
                            "entity_sector": reporting_entity.sector,
                            "entity_industry": reporting_entity.industry,
                            "entity_country": reporting_entity.country,
                            "entity_ticker": reporting_entity.ticker,
                            "text": chunk.text,
                            "other_entities": ", ".join(
                                e["name"] for e in other_entities
                            ),
                            "entities": chunk_entities,
                        }
                    )
            else:
                # Process standard entities
                if duplicate_for_all_entities:
                    # Create a row for each entity in the chunk, treating each as target
                    for chunk_entity in chunk_entities:
                        entity_key = entity_key_map.get(chunk_entity["key"])

                        if not entity_key:
                            continue  # Skip if entity is not found

                        # Exclude the current entity from other entities
                        other_entities = [
                            e for e in chunk_entities if e["key"] != chunk_entity["key"]
                        ]

                        # Collect information in standard format
                        rows.append(
                            {
                                "timestamp_utc": result.timestamp,
                                "document_id": result.id,
                                "sentence_id": f"{result.id}-{chunk.chunk}",
                                "headline": result.headline,
                                "entity_id": chunk_entity["key"],
                                "document_type": document_type.value,
                                "is_reporting_entity": False,
                                "entity_name": entity_key.name,
                                "entity_sector": entity_key.sector,
                                "entity_industry": entity_key.industry,
                                "entity_country": entity_key.country,
                                "entity_ticker": entity_key.ticker,
                                "text": chunk.text,
                                "other_entities": ", ".join(
                                    e["name"] for e in other_entities if e["name"]
                                ),
                                "entities": chunk_entities,
                            }
                        )
                else:
                    # Original logic: filter by companies if provided
                    for chunk_entity in chunk_entities:
                        entity_key = entity_key_map.get(chunk_entity["key"])

                        if not entity_key:
                            continue  # Skip if entity is not found
                        
                        # if entity isn't in our original watchlist, skip
                        if companies and entity_key not in companies:
                            continue

                        # Exclude the entity from other entities
                        other_entities = [
                            e for e in chunk_entities if e["name"] != chunk_entity["name"]
                        ]

                        # Collect information in standard format
                        rows.append(
                            {
                                "timestamp_utc": result.timestamp,
                                "document_id": result.id,
                                "sentence_id": f"{result.id}-{chunk.chunk}",
                                "headline": result.headline,
                                "entity_id": chunk_entity["key"],
                                "document_type": document_type.value,
                                "is_reporting_entity": False,
                                "entity_name": entity_key.name,
                                "entity_sector": entity_key.sector,
                                "entity_industry": entity_key.industry,
                                "entity_country": entity_key.country,
                                "entity_ticker": entity_key.ticker,
                                "text": chunk.text,
                                "other_entities": ", ".join(
                                    e["name"] for e in other_entities
                                ),
                                "entities": chunk_entities,
                            }
                        )

    if not rows:
        raise ValueError("No rows to process")

    df = DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)

    # Deduplicate by quote text as well
    df = df.drop_duplicates(
        subset=["timestamp_utc", "document_id", "text", "entity_id"]
    )

    df = mask_sentences(df)
    return df.reset_index(drop=True)


def mask_sentences(
    df: DataFrame
) -> DataFrame:
    """
    Mask the target entity and other entities in the text.

    Args:
        df (DataFrame): The input DataFrame. Columns required:
            - text
            - masked_text
        document_type (DocumentType): The document type (for logging only)
    Returns:
        DataFrame: masked DataFrame. Will add/transform the columns:
            - text
            - masked_text
            - other_entities_map
    """
    df["text"] = df["text"].str.replace("{", "", regex=False)
    df["text"] = df["text"].str.replace("}", "", regex=False)

    df = mask_entity_coordinates(df=df)

    df["masked_text"] = df["masked_text"].apply(
        lambda x: x.replace("{", "").replace("}", "")
    )
    df = df[df["masked_text"] != "to_remove"]
    df["text"] = df["text"].apply(lambda x: x.replace("{", "").replace("}", ""))
    df = df[df.text != "to_remove"]
    return df


def mask_entity_coordinates(
    df: DataFrame,
) -> DataFrame:
    """
    Mask the target entity and other entities in the text.

    Args:
        df (DataFrame): The input DataFrame
    Returns:
        DataFrame: The masked DataFrame
    """
    i = 1
    entity_counter = {}
    df["masked_text"] = None
    df["other_entities_map"] = None

    # Ensure columns are compatible with string/object assignments
    df["masked_text"] = df["masked_text"].astype("object")
    df["other_entities_map"] = df["other_entities_map"].astype("object")

    # Process each row
    for idx, row in df.iterrows():
        text = row["text"]
        entities = sorted(row["entities"], key=lambda x: x["start"], reverse=True)
        masked_text = text

        # Get target entity coordinates
        target_start = []
        target_end = []
        for entity in entities:
            if entity["key"] == row["entity_id"]:
                target_start.append(entity["start"])
                target_end.append(entity["end"])

        # Apply masking
        other_entity_map = []
        for entity in entities:
            start, end = entity["start"], entity["end"]

            if entity["key"] == row["entity_id"]:
                # Mask target entity
                masked_text = f"{masked_text[:start]}{get_target_entity_placeholder()}{masked_text[end:]}"

            elif start not in target_start and end not in target_end:
                # Mask other entities
                if entity["key"] not in entity_counter:
                    entity_counter[entity["key"]] = i
                    mask = f"{get_other_entity_placeholder()}_{entity_counter[entity['key']]}"
                    masked_text = f"{masked_text[:start]}{mask}{masked_text[end:]}"
                    other_entity_map.append(
                        (entity_counter[entity["key"]], entity["name"])
                    )
                    i += 1
                else:
                    mask = f"{get_other_entity_placeholder()}_{entity_counter[entity['key']]}"
                    masked_text = f"{masked_text[:start]}{mask}{masked_text[end:]}"
                    other_entity_map.append(
                        (entity_counter[entity["key"]], entity["name"])
                    )

        # Update DataFrame
        df.at[idx, "masked_text"] = masked_text
        df.at[idx, "other_entities_map"] = (
            other_entity_map if other_entity_map else None
        )

    return df