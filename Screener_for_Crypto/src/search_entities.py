from bigdata_research_tools.search.query_builder import (
    build_batched_query,
    EntitiesToSearch,
    create_date_ranges,
)
from itertools import chain
from bigdata_research_tools.search.search import run_search
from bigdata_client.models.search import DocumentType, SortBy
from bigdata_research_tools.search.search_utils import filter_search_results
from typing import List, Optional, Dict

from bigdata_client.document import Document
from bigdata_client.models.advanced_search_query import ListQueryComponent
from pandas import DataFrame
from tqdm import tqdm
from bigdata_research_tools.search.screener_search import mask_sentences
from bigdata_research_tools.labeler.screener_labeler import (
    replace_company_placeholders,
)

def entity_type_checker(entities):
    unique_types = set(type(entity).__name__ for entity in entities)
    type_field_map = {
            'Person':'people',
            'Product': 'products',
            'Organization':'org',
            'Place':'place',
            'Topic':'topic',
            'Concept':'concepts', 
            'Entity':'companies',
            'Company':'companies'
        }
    if len(unique_types) == 1:
        return type_field_map[unique_types.pop()]
    else:
        raise ValueError("Multiple entity types found in the provided watchlist.")

def process_search_results(
    results: List[Document],
    chunks_entities: List[ListQueryComponent],
    watchlist: list,
    document_type: DocumentType = DocumentType.NEWS,
) -> DataFrame:
    """
    Build a unified DataFrame from search results for any document type.

    Args:
        results (List[Document]): A list of Bigdata search results.
        entities (List[ListQueryComponent]): A list of entities.
        watchlist (list): A list of entities to filter results and create rows for (your watchlist).
        document_type (DocumentType): The type of documents being processed.

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
            - entity_name: str
            - text: str
            - other_entities: str
            - entities: List[Dict[str, Any]]
            - masked_text: str
            - other_entities_map: List[Tuple[int, str]]
            - reporting_entity_name: str (if applicable)
            - reporting_entity_sector: str (if applicable)
            - reporting_entity_industry: str (if applicable)
            - reporting_entity_country: str (if applicable)
            - reporting_entity_ticker: str (if applicable)
    """
    chunks_entity_key_map = {entity.id: entity for entity in chunks_entities}

    rows = []
    for result in tqdm(results, desc=f"Processing {document_type} results..."):
        for chunk in result.chunks:
            # Build a list of entities present in the chunk
            chunk_entities = [
                {
                    "key": entity.key,
                    "name": (
                        chunks_entity_key_map[entity.key].name
                        if entity.key in chunks_entity_key_map
                        else None
                    ),
                    "start": entity.start,
                    "end": entity.end,
                }
                for entity in chunk.entities
                if entity.key in chunks_entity_key_map and chunks_entity_key_map[entity.key].entity_type in ['COMP'] or entity.key in [entity.id for entity in watchlist]
            ]
            #Other entities to be masked are either Companies found in the chunks or entities in our watchlist.
            ##TODO: Make this more generic to handle other entity types or entity groups within entity types (i.e. Crypto within Currencies) as well.

            if not chunk_entities:
                continue  # Skip if no entities are mapped

            # Process standard entities
            for chunk_entity in chunk_entities:
                entity_key = chunks_entity_key_map.get(chunk_entity["key"])

                if not entity_key:
                    continue  # Skip if entity is not found
                    
                # # if entity isn't in our original watchlist, skip
                if watchlist and entity_key not in watchlist:
                    continue

                # Exclude the entity from other entities
                other_entities = [
                    e for e in chunk_entities if e["name"] != chunk_entity["name"]
                ]

                # Collect information in standard format
                row_dict = {"timestamp_utc": result.timestamp,
                            "document_id": result.id,
                            "sentence_id": f"{result.id}-{chunk.chunk}",
                            "headline": result.headline,
                            "entity_id": chunk_entity["key"],
                            "document_type": document_type.value,
                            "entity_name": entity_key.name,
                            "text": chunk.text,
                            "other_entities": ", ".join(
                                e["name"] for e in other_entities
                            ),
                            "entities": chunk_entities,
                        }
                    
                # Handle differently based on document type
                if document_type in (DocumentType.FILINGS, DocumentType.TRANSCRIPTS):
                    # Process reporting entities
                    if result.reporting_entities:
                        for re_key in result.reporting_entities:
                            reporting_entity = chunks_entity_key_map.get(re_key)
                            # Collect information in standard format
                            if reporting_entity:
                                row_dict_copy = row_dict.copy()
                                row_dict_copy.update({
                                    "reporting_entity_name": reporting_entity.name,
                                    "reporting_entity_sector": reporting_entity.sector if reporting_entity.sector else None,
                                    "reporting_entity_industry": reporting_entity.industry if reporting_entity.industry else None,
                                    "reporting_entity_country": reporting_entity.country if reporting_entity.country else None,
                                    "reporting_entity_ticker": reporting_entity.ticker if reporting_entity.ticker else None,
                                })
                                rows.append(row_dict_copy)
                else:
                    rows.append(row_dict)

    if not rows:
        raise ValueError("No rows to process")

    df = DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)

    # Deduplicate by quote text as well
    df = df.drop_duplicates(
        subset=["timestamp_utc", "document_id", "text", "entity_id"]
    )

    df = mask_sentences(df)
    return df.reset_index(drop=True)

def search_by_entities(entities: list,
    sentences: List[str],
    start_date: str,
    end_date: str,
    scope: DocumentType = DocumentType.ALL,
    fiscal_year: Optional[int] = None,
    sources: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    control_entities: Optional[Dict] = None,
    freq: str = "3M",
    sort_by: SortBy = SortBy.RELEVANCE,
    rerank_threshold: Optional[float] = None,
    document_limit: int = 50,
    batch_size: int = 10,
    **kwargs,
) -> DataFrame:
    """
    Screen for documents based on the input sentences and other filters.

    Args:
        entities (list): The list of entities to use. All entities must be of the same type (i.e. Currencies, People, etc).
        sentences (List[str]): The list of sentences to screen for.
        start_date (str): The start date for the search.
        end_date (str): The end date for the search.
        scope (DocumentType): The document type scope
            (e.g., `DocumentType.ALL`, `DocumentType.TRANSCRIPTS`).
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
    # Extract entities for search querying
    entity_keys = [entity.id for entity in entities]

    field_entity_type = entity_type_checker(entities)

    # Create entity configs
    entities_config = EntitiesToSearch(**{field_entity_type:entity_keys})

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

    print(f"Running {total_no} searches ({no_queries} queries over {no_dates} date ranges)")
    print(f"Example query:\n{batched_query[0]}\n")

    # Run concurrent search
    results = run_search(
        batched_query,
        date_ranges=date_ranges,
        limit=document_limit,
        scope=scope,
        sortby=sort_by,
        rerank_threshold=rerank_threshold,
    )

    if list(chain.from_iterable(results)) is None:
        print("No results found for the given queries and date ranges.")
        return DataFrame()  # Return empty DataFrame if no results

    else:
        results, chunks_entities = filter_search_results(results)

        df = process_search_results(
            results=results,
            chunks_entities=chunks_entities,
            watchlist=entities,
            document_type=scope)

        return df

def post_process_dataframe(df: DataFrame) -> DataFrame:
        """
        Post-process the labeled DataFrame.

        Args:
            df: DataFrame to process. Schema:
                - Index: int
                - Columns:
                    - timestamp_utc: datetime64
                    - document_id: str
                    - sentence_id: str
                    - headline: str
                    - entity_id: str
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
                    - label: str
                    - motivation: str
        Returns:
            Processed DataFrame. Schema:
            - index: int
            - Columns:
                - Time Period
                - Date
                - Company
                - Sector
                - Industry
                - Country
                - Ticker
                - Document ID
                - Headline
                - Quote
                - Motivation
                - Theme
        """
        # Filter unlabeled sentences
        df = df.loc[df["label"] != 'unclear'].copy()
        if df.empty:
            return df

        # Process timestamps
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize(None)

        # Sort and format
        sort_columns = ["entity_name", "timestamp_utc", "label"]
        df = df.sort_values(by=sort_columns).reset_index(drop=True)


        # Replace company placeholders
        df["motivation"] = df.apply(replace_company_placeholders, axis=1)

        # Add formatted columns
        df["Time Period"] = df["timestamp_utc"].dt.strftime("%b %Y")
        df["Date"] = df["timestamp_utc"].dt.strftime("%Y-%m-%d")

        df = df.rename(
            columns={
                "document_id": "Document ID",
                "entity_name": "Entity",
                "headline": "Headline",
                "text": "Quote",
                "motivation": "Motivation",
                "label": "Theme",
            }
        )

        # Select and order columns
        export_columns = [
            "Time Period",
            "Date",
            "Entity",
            "Document ID",
            "Headline",
            "Quote",
            "Motivation",
            "Theme",
        ]

        sort_columns = ["Date", "Time Period", "Entity", "Document ID", "Headline", "Quote"]
        df = df[export_columns].sort_values(sort_columns).reset_index(drop=True)        
        
        return df
