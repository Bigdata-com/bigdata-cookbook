"""
Copyright (C) 2025 RavenPack | Bigdata.com. All rights reserved.
Author: Alessandro Bouchs (abouchs@ravenpack.com), Jelena Starovic (jstarovic@ravenpack.com)

Modified for Trump Reelections analysis - removed fiscal year validation restraint
"""

from dataclasses import dataclass
from itertools import chain,zip_longest
from typing import List, Optional, Tuple, Type, Dict  
import pandas as pd
from bigdata_client.daterange import AbsoluteDateRange
from bigdata_client.models.advanced_search_query import QueryComponent
from bigdata_client.models.search import DocumentType
from bigdata_client.models.entities import (
    Person, 
    Place,
    Organization, 
    Product, 
    Concept
)

from bigdata_client.query import (
    Any,
    Entity,
    FiscalYear,
    Keyword,
    ReportingEntity,
    Similarity,
    Source,
    Topic
)

from bigdata_research_tools.client import bigdata_connection

@dataclass
class EntitiesToSearch:
    people: Optional[List[str]] = None
    product: Optional[List[str]] = None
    org: Optional[List[str]] = None
    place: Optional[List[str]] = None
    topic: Optional[List[str]] = None
    concepts: Optional[List[str]] = None
    companies: Optional[List[str]] = None

    @staticmethod
    def get_entity_type_map() -> Dict[str, Type]:
        return {
            'people': Person,
            'product': Product,
            'org': Organization,
            'place': Place,
            'topic': Topic,
            'concepts': Concept, 
            'companies': Entity
        }


def build_similarity_queries(sentences: List[str]) -> List[Similarity]:
    """
    Processes a list of sentences to create a list of Similarity query objects, ensuring no duplicates.

    Args:
        sentences (List[str] or str):
            A list of sentences or a single sentence string. If a single string is provided,
            it is converted into a list containing that string.

    Returns:
        List[Similarity]:
            A list of Similarity query objects, one for each unique sentence in the input.

    Operation:
        1. Converts a single string input to a list.
        2. Deduplicates the list of sentences.
        3. Creates a Similarity query object for each unique sentence.
    """
    if isinstance(sentences, str):
        sentences = [sentences]

    sentences = list(set(sentences))  # Deduplicate
    queries = [Similarity(sentence) for sentence in sentences]
    return queries


def build_batched_query(
    sentences: List[str], 
    keywords: Optional[List[str]],
    entities: Optional[EntitiesToSearch],
    control_entities: Optional[EntitiesToSearch],
    sources: Optional[List[str]],
    batch_size: int,
    fiscal_year: Optional[int],
    scope: DocumentType,
    custom_batches: Optional[List[EntitiesToSearch]],
) -> List[QueryComponent]:
    """
    Builds a list of batched query objects based on the provided parameters.

    Args:
        sentences (Optional[List[str]]):
            Sentences for creating similarity queries.
        keywords (Optional[List[str]]):
            Keywords for constructing keyword queries.
        control_entities (EntityConfig):
            Config of control entities of different types (people, companies, organisations..)
        sources (Optional[List[str]]):
            List of sources for constructing source queries.
        entities (EntityConfig):
            Config of entities of different types (people, companies, organisations..)
        batch_size (int, optional):
            Number of entities per batch when auto-batching. Defaults to 10.
        fiscal_year (int, optional):
            Fiscal year to filter queries.
        scope (DocumentType, optional):
            Document type scope (e.g., ALL, TRANSCRIPTS). Defaults to ALL.
        custom_batches (Optional[List[EntitiesToSearch]]):
            Config of custom entity batches of different types (people, companies, organisations..)

    Returns:
        List[QueryComponent]: List of expanded query components.    
    """

    # Early validation: ensure only one of entities or custom_batches is used
    if entities and custom_batches:
        raise ValueError("Only one of `entities` or `custom_batches` should be provided, not both.")

    # REMOVED: _validate_parameters(document_scope=scope, fiscal_year=fiscal_year)
    # This allows more flexible usage without fiscal year constraints

    # Step 1: Build base queries (similarity, keyword, source)
    base_queries, keyword_query, source_query = _build_base_queries(sentences, keywords, sources)
    
    # Step 2: Build control entity query
    control_query = _build_control_entity_query(control_entities, scope=scope) if control_entities else None
    
    # Step 3: Build entity batch queries
    entity_batch_queries = _build_entity_batch_queries(entities, custom_batches, batch_size, scope)
    
    # Step 4: Combine everything into expanded queries
    queries_expanded = _expand_queries(
        (base_queries, keyword_query, source_query), 
        entity_batch_queries, 
        control_query,
        source_query,
        fiscal_year
    )
    
    return queries_expanded

def _validate_parameters(
    document_scope: DocumentType = None, fiscal_year: int = None
) -> None:
    """
    Validates parameters based on predefined rules.
    Will raise a ValueError if any of the rules are violated.
    Will return None otherwise.
    
    NOTE: This function is preserved but not used in build_batched_query
    to allow more flexible usage patterns.
    """
    # Skip validation if document_scope is not provided
    if document_scope is None:
        return

    if document_scope in [DocumentType.FILINGS, DocumentType.TRANSCRIPTS]:
        if fiscal_year is None:
            raise ValueError(
                f"`fiscal_year` is required when `document_scope` is `{document_scope.value}`"
            )

    if document_scope == DocumentType.NEWS:
        if fiscal_year is not None:
            raise ValueError(
                f"`fiscal_year` must be None when `document_scope` is `{document_scope.value}`"
            )
        
def _build_base_queries(
    sentences: Optional[List[str]], 
    keywords: Optional[List[str]],
    sources: Optional[List[str]]
) -> Tuple[List[QueryComponent], Optional[QueryComponent], Optional[QueryComponent]]:
    """Build the base queries from sentences, keywords, and sources."""
    # Create similarity queries from sentences
    queries = build_similarity_queries(sentences) if sentences else []
    
    # Create keyword query
    keyword_query = Any([Keyword(word) for word in keywords]) if keywords else None
    
    # Create source query
    source_query = Any([Source(source) for source in sources]) if sources else None
    
    return queries, keyword_query, source_query

def _get_entity_ids(
        entity_names: List[str],
        entity_type: Type,  
) -> list[Type]:
    bigdata = bigdata_connection()
    entity_ids = []

    lookup_map = {
        Place: bigdata.knowledge_graph.find_places,
        Product: bigdata.knowledge_graph.find_products,
        Person: bigdata.knowledge_graph.find_people,
        Organization: bigdata.knowledge_graph.find_organizations,
        Topic: bigdata.knowledge_graph.find_topics,
        Concept: bigdata.knowledge_graph.find_concepts, 
        Entity: bigdata.knowledge_graph.find_companies,  
        ReportingEntity: bigdata.knowledge_graph.find_companies, 
    }

    lookup_func = lookup_map.get(entity_type)
    if not lookup_func:
        return []

    for name in entity_names:
        entity = next(iter(lookup_func(name)), None)
        if entity is not None:
            if entity_type in (Entity, ReportingEntity):
                entity = entity_type(entity.id)

            entity_ids.append(entity)

    return entity_ids

def _build_control_entity_query(
    control_entities: EntitiesToSearch,
    scope: DocumentType = DocumentType.ALL,
) -> QueryComponent:
    """Build a query for control entities."""
    
    entity_ids = []
    comp_ids = []
    if control_entities.people:
        people_ids = _get_entity_ids(control_entities.people, Person)
        if people_ids: 
            entity_ids.extend(people_ids)
        
    if control_entities.product: 
        prod_ids = _get_entity_ids(control_entities.product, Product)
        if prod_ids: 
            entity_ids.extend(prod_ids)

    if control_entities.companies:
        entity_type = _get_entity_type(scope)
        comp_ids = _get_entity_ids(control_entities.companies,entity_type)
        if comp_ids: 
            entity_ids.extend(comp_ids)

    if control_entities.place:
        place_ids = _get_entity_ids(control_entities.place, Place)
        if place_ids: 
            entity_ids.extend(place_ids)

    if control_entities.org:
        orga_ids = _get_entity_ids(control_entities.org, Organization)
        if orga_ids: 
            entity_ids.extend(orga_ids)

    if control_entities.topic:
        topic_ids = _get_entity_ids(control_entities.topic, Topic)
        if topic_ids: 
            entity_ids.extend(topic_ids)
    
    if control_entities.concepts:
        concept_ids = _get_entity_ids(control_entities.concepts, Concept)
        if concept_ids: 
            entity_ids.extend(concept_ids)

    control_query = Any(entity_ids)
    return control_query

def _build_entity_batch_queries(
    entities: EntitiesToSearch, 
    custom_batches: List[EntitiesToSearch],
    batch_size: int,
    scope: DocumentType,
) -> List[Optional[QueryComponent]]:
    """Build entity batch queries from either custom batches or auto-batched entities."""

    # If no entities specified, return a single None to ensure at least one iteration
    if not entities and not custom_batches:
        return [None]
    
    # If using custom batches, process them
    if custom_batches:
        return _build_custom_batch_queries(custom_batches, scope)
    
    # Otherwise, auto-batch the entities
    return _auto_batch_entities(entities, batch_size, scope)

def _get_entity_type(scope: DocumentType) -> type:
    """Determine the entity type based on document scope."""
    return (
        ReportingEntity
        if scope in (DocumentType.TRANSCRIPTS, DocumentType.FILINGS)
        else Entity
    )

def _build_custom_batch_queries(
    custom_batches: List[EntitiesToSearch],
    scope: DocumentType
) -> List[QueryComponent]:
    """Build entity queries from a list of EntitiesToSearch objects."""
    entity_type_map = EntitiesToSearch.get_entity_type_map()
    
    def get_entity_ids_for_attr(entity_config: EntitiesToSearch, attr_name: str, entity_class) -> List[int]:
        """Get entity IDs for a specific attribute."""
        entity_names = getattr(entity_config, attr_name, None)
        if not entity_names:
            return []
        
        entity_type = _get_entity_type(scope) if entity_class == Entity else entity_class
        return _get_entity_ids(entity_names, entity_type)
    
    batch_queries = []
    for entity_config in custom_batches:
        # Use chain to flatten all entity IDs from all attributes
        all_entities = list(chain.from_iterable(
            get_entity_ids_for_attr(entity_config, attr_name, entity_class)
            for attr_name, entity_class in entity_type_map.items()
        ))
        
        if all_entities:
            batch_queries.append(Any(all_entities))
    
    return batch_queries if batch_queries else [None]

def _auto_batch_entities(
    entities: EntitiesToSearch,
    batch_size: int,
    scope: DocumentType = DocumentType.ALL,
) -> List[QueryComponent]:
    """Auto-batch entities by type using the specified batch size."""
    
    # Create batches for each entity type
    all_entity_batches = []
    
    for attr_name, entity_class in EntitiesToSearch.get_entity_type_map().items():
        entity_names = getattr(entities, attr_name, None)
        if not entity_names:
            continue
            
        # Get valid entity IDs
        entity_type = _get_entity_type(scope) if entity_class == Entity else entity_class
        entity_ids = _get_entity_ids(entity_names, entity_type)
        
        # Split into batches and add to collection
        if entity_ids:
            batches = [entity_ids[i:i + batch_size] for i in range(0, len(entity_ids), batch_size)]
            all_entity_batches.append(batches)
    
    if not all_entity_batches:
        return []
    
    # Combine batches across entity types using zip_longest
    return [
        Any([entity for batch in batch_group for entity in batch])
        for batch_group in zip_longest(*all_entity_batches, fillvalue=[])
        if any(batch for batch in batch_group)  # Skip empty batch groups
    ]

def _expand_queries(
    base_queries_tuple: Tuple[List[QueryComponent], Optional[QueryComponent], Optional[QueryComponent]],
    entity_batch_queries: Optional[List[Optional[QueryComponent]]] = None,  
    control_query: Optional[QueryComponent] = None,
    source_query: Optional[QueryComponent] = None,
    fiscal_year: Optional[int] = None
) -> List[QueryComponent]:
    """Expand all query components into the final list of queries."""
    base_queries, keyword_query, source_query = base_queries_tuple
    queries_expanded = []

    for entity_batch_query in entity_batch_queries or [None]:
        for base_query in base_queries or [None]:
            expanded_query = base_query or None
            # Add entity batch
            if entity_batch_query:
                expanded_query = (
                    expanded_query & entity_batch_query
                    if expanded_query
                    else entity_batch_query
                )
                # Add keyword and control queries
            if keyword_query:
                expanded_query = (
                    expanded_query & keyword_query if expanded_query else keyword_query
                )
            if control_query:
                expanded_query = (
                    expanded_query & control_query if expanded_query else control_query
                )

            if source_query:
                expanded_query = (
                    expanded_query & source_query if expanded_query else source_query
                )

            # Add fiscal year filter if provided
            if fiscal_year:
                expanded_query = (
                    expanded_query & FiscalYear(fiscal_year) if expanded_query else None
                )

            # Append the expanded query to the final list
            queries_expanded.append(expanded_query)

    return queries_expanded
    
def create_date_intervals(
    start_date: str, end_date: str, freq: str
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Generates date intervals based on a specified frequency within a given start and end date range.

    Args:
        start_date (str):
            The start date in 'YYYY-MM-DD' format.
        end_date (str):
            The end date in 'YYYY-MM-DD' format.
        freq (str):
            The frequency for intervals. Supported values:
                - 'Y': Yearly intervals.
                - 'M': Monthly intervals.
                - 'W': Weekly intervals.
                - 'D': Daily intervals.

    Returns:
        List[Tuple[pd.Timestamp, pd.Timestamp]]:
            A list of tuples, where each tuple contains the start and end timestamp
            of an interval. The intervals are inclusive of the start and exclusive of the next start.

    Raises:
        ValueError: If the provided frequency is invalid.

    Operation:
        1. Converts the `start_date` and `end_date` strings to `pd.Timestamp` objects.
        2. Adjusts the frequency for yearly ('Y') and monthly ('M') intervals to align with period starts:
           - 'Y' → 'AS' (Year Start).
           - 'M' → 'MS' (Month Start).
        3. Uses `pd.date_range` to generate a range of dates based on the frequency.
        4. Creates tuples representing start and end times for each interval:
           - The start time is set to midnight (00:00:00).
           - The end time is set to the last second of the interval (23:59:59).
        5. Ensures the final interval includes the specified `end_date`.

    Notes:
        - The intervals are inclusive of the start and exclusive of the next start time.
        - For invalid frequencies, a `ValueError` is raised to indicate the issue.
    """
    # Convert start and end dates to pandas Timestamps
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Adjust frequency for yearly and monthly to use appropriate start markers
    # 'AS' for year start, 'MS' for month start
    adjusted_freq = freq.replace("Y", "AS").replace("M", "MS")

    # Generate date range based on the adjusted frequency
    try:
        date_range = pd.date_range(start=start_date, end=end_date, freq=adjusted_freq)
    except ValueError:
        raise ValueError("Invalid frequency. Use 'Y', 'M', 'W', or 'D'.")

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


def create_date_ranges(
    start_date: str, end_date: str, freq: str
) -> List[AbsoluteDateRange]:
    """
    Generates a list of `AbsoluteDateRange` objects based on the specified frequency.

    Args:
        start_date (str):
            The start date in 'YYYY-MM-DD' format.
        end_date (str):
            The end date in 'YYYY-MM-DD' format.
        freq (str):
            The frequency for dividing the date range. Supported values:
                - 'Y': Yearly.
                - 'M': Monthly.
                - 'W': Weekly.
                - 'D': Daily.

    Returns:
        List[AbsoluteDateRange]:
            A list of `AbsoluteDateRange` objects, where each object represents
            a time range between two dates as determined by the specified frequency.

    Operation:
        1. Calls `create_date_intervals` to generate a list of date intervals.
        2. Converts each interval (start and end tuple) into an `AbsoluteDateRange` object.
        3. Returns a list of these `AbsoluteDateRange` objects.
    """
    intervals = create_date_intervals(start_date, end_date, freq=freq)
    return [AbsoluteDateRange(start, end) for start, end in intervals] 