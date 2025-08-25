from pandas import DataFrame
from typing import List, Optional, Union
import pandas as pd
from bigdata_client.models.entities import Company
from bigdata_client.query import Document
from bigdata_client.models.search import DocumentType, SortBy
from bigdata_research_tools.search.query_builder import create_date_ranges, build_batched_query, EntitiesToSearch
from bigdata_client.models.advanced_search_query import ListQueryComponent
from bigdata_research_tools.search.search_utils import filter_search_results
from bigdata_research_tools.search.screener_search import filter_company_entities

from bigdata_research_tools.search import run_search
from tqdm.notebook import tqdm


def search_enhanced(companies:Union[list[str], list[Company]], keywords: list[str], sentences: Optional[list[str]], control_entities:Optional[Union[list[str], list[Company]]], start_date: str, end_date: str, scope = DocumentType.NEWS, fiscal_year: Optional[int] = None, sources: Optional[List[str]] = None, freq:str='M', sort_by: SortBy = SortBy.RELEVANCE, document_limit:int=100, batch_size:int = 1, enhance_search:bool = True, **kwargs):
    """
    Enhanced search function that can retrieve and process document chunks with context.
    
    Args:
        companies: List of company IDs or Company objects to search for
        keywords: List of keywords to search for
        sentences: Optional list of sentences for similarity search
        control_entities: Optional list of control entities to include
        start_date: Start date for the search (YYYY-MM-DD)
        end_date: End date for the search (YYYY-MM-DD)
        scope: Document type to search (NEWS, FILINGS, TRANSCRIPTS)
        fiscal_year: Optional fiscal year for filings
        sources: Optional list of source IDs to filter by
        freq: Frequency for date ranges ('D', 'W', 'M', etc.)
        sort_by: How to sort the results
        document_limit: Maximum number of documents to retrieve
        batch_size: Size of query batches
        enhance_search: Whether to enhance search results with context from full documents
        **kwargs: Additional arguments
        
    Returns:
        DataFrame: DataFrame with search results and optionally enhanced context
    """

    if scope == DocumentType.NEWS:
        if fiscal_year is not None:
            raise ValueError(
                f"`fiscal_year` must be None when `document_scope` is `{scope.value}`"
            )

    date_ranges = create_date_ranges(start_date, end_date, freq=freq)

    entities_config = EntitiesToSearch(companies=companies) if isinstance(companies[0], str) else EntitiesToSearch(companies=[company.id for company in companies])

    control_entities_config = EntitiesToSearch(companies=control_entities) if isinstance(control_entities[0], str) else EntitiesToSearch(companies=[entity.id for entity in control_entities]) if control_entities else None


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

    print(f"About to run {len(batched_query)*len(date_ranges)} queries")
    print("Example Query:", batched_query[0], "over date range:", date_ranges[0])

    # Run concurrent search
    results = run_search(
        batched_query,
        date_ranges=date_ranges,
        limit=document_limit,
        scope=scope,
        sortby=sort_by,
    )

    results, entities = filter_search_results(results)
    # Filter entities to only include COMPANY entities
    entities = filter_company_entities(entities)

    needs_company_filtering = scope not in (
            DocumentType.FILINGS,
            DocumentType.TRANSCRIPTS,
        )

    df_sentences = process_search_results(
            results=results,
            entities=entities,
            companies=companies+control_entities if needs_company_filtering else None,
            document_type=scope,
            enhance_search=enhance_search,
        )

    return df_sentences

def process_search_results(
    results: List[Document],
    entities: List[ListQueryComponent],
    companies: Optional[Union[list[Company], list[str]]] = None,
    document_type: DocumentType = DocumentType.NEWS,
    enhance_search: bool = True,
) -> DataFrame:
    
    entity_key_map = {entity.id: entity for entity in entities}
    companies_ids = [company.id for company in companies] if isinstance(companies[0], Company) else companies
    
    # Store document chunks by document_id for context enhancement if needed
    document_chunks_cache = {}
    
    rows = []
    for result in tqdm(results, desc=f"Processing {document_type} results..."):
        # Only download annotated dict if we need context enhancement
        if enhance_search:
            try:
                annotated_dict = result.download_annotated_dict()
                if annotated_dict:
                    document_chunks_cache[result.id] = extract_chunks_from_annotated_dict(annotated_dict)
                else:
                    print(f"Warning: annotated_dict is empty for document {result.id}")
            except Exception as e:
                print(f"Warning: Could not download annotated dict for document {result.id}: {e}")
        
        # Process document chunks
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

                    # Prepare row data
                    row_data = {
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
                    
                    # If enhance_search is enabled, add context from previous/next chunks
                    if enhance_search and result.id in document_chunks_cache:
                        chunks_df = document_chunks_cache[result.id]
                        prev_text, next_text = get_chunk_context(chunks_df, chunk.chunk)
                        
                        # Add contextualized text combining previous, current, and next chunks
                        row_data["contextualized_chunk_text"] = " ".join([t for t in [prev_text, chunk.text, next_text] if t])
                    else:
                        row_data["contextualized_chunk_text"] = chunk.text
                    
                    # Collect information in standard format
                    rows.append(row_data)
            else:
                # Process standard entities
                for chunk_entity in chunk_entities:
                    entity_key = entity_key_map.get(chunk_entity["key"])

                    if not entity_key:
                        continue  # Skip if entity is not found
                    
                    # # if entity isn't in our original watchlist, skip
                    if companies and chunk_entity["key"] not in companies_ids:
                        continue

                    # Exclude the entity from other entities
                    other_entities = [
                        e for e in chunk_entities if e["name"] != chunk_entity["name"]
                    ]

                    # Prepare row data
                    row_data = {
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
                        'source_name': result.source.name,
                        'source_rank': result.source.rank,
                        'url': result.url}
                    
                    # If enhance_search is enabled, add context from previous/next chunks
                    if enhance_search and result.id in document_chunks_cache:
                        chunks_df = document_chunks_cache[result.id]
                        prev_text, next_text = get_chunk_context(chunks_df, chunk.chunk)
                        
                        # Add contextualized text combining previous, current, and next chunks
                        row_data["contextualized_chunk_text"] = " ".join([t for t in [prev_text, chunk.text, next_text] if t])
                    else:
                        row_data["contextualized_chunk_text"] = chunk.text
                    
                    # Collect information in standard format
                    rows.append(row_data)

    ## what if I am enhancing both chunk 1 and chunk 2 from the same document?

    if not rows:
        raise ValueError("No rows to process")

    df = DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)

    # Deduplicate by quote text as well
    df = df.drop_duplicates(
        subset=["timestamp_utc", "document_id", "text", "entity_id"]
    )

    df['date'] = pd.to_datetime(df['timestamp_utc']).dt.date
    
    return df.reset_index(drop=True)

def extract_chunks_from_annotated_dict(annotated_dict):
    """
    Given a document dict from download_annotated_dict(), returns a DataFrame
    with columns: 'chunk_number', 'text'
    Assumes 'content' > 'body' is a list of chunks in order.
    """
    chunks = annotated_dict.get('content', {}).get('body', [])
    if not chunks:
        print(f"Warning: No chunks found in annotated_dict {annotated_dict}")
        return pd.DataFrame(columns=['chunk_number', 'text'])
        
    # Process chunks into DataFrame rows
    chunk_rows = [
        {'chunk_number': i, 'text': chunk.get('text', '')} 
        for i, chunk in enumerate(chunks)
    ]
    
    # Create DataFrame with proper types
    df = pd.DataFrame(chunk_rows)
    if not df.empty:
        df['chunk_number'] = df['chunk_number'].astype(int)
    
    return df

def get_chunk_context(chunks_df, chunk_number):
    """
    Helper function to safely retrieve previous and next chunk text
    for a given chunk number from a DataFrame of chunks.
    
    Args:
        chunks_df: DataFrame with columns 'chunk_number' and 'text'
        chunk_number: The number of the current chunk
        
    Returns:
        tuple: (prev_text, next_text) - The text of previous and next chunks or None
    """
    chunk_number = int(chunk_number)
    
    # Query for previous and next chunks
    prev_chunk = chunks_df[chunks_df['chunk_number'] == chunk_number - 1]
    next_chunk = chunks_df[chunks_df['chunk_number'] == chunk_number + 1]
    
    # Get text if chunks exist
    prev_text = prev_chunk.iloc[0]['text'] if not prev_chunk.empty else None
    next_text = next_chunk.iloc[0]['text'] if not next_chunk.empty else None
    
    return prev_text, next_text