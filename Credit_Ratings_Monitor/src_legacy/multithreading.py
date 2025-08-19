import datetime
import functools
import itertools
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union, Tuple
from tqdm.notebook import tqdm
import pandas as pd
import pydantic
from bigdata_client import Bigdata
from bigdata_client.connection import RequestMaxLimitExceeds
from bigdata_client.daterange import AbsoluteDateRange, RollingDateRange
from bigdata_client.document import Document
from bigdata_client.models.advanced_search_query import QueryComponent, \
    ListQueryComponent
from bigdata_client.models.search import DocumentType, SortBy
from bigdata_client.query import Similarity, FiscalYear, ReportingEntity, Entity
from bigdata_client.query_type import QueryType
import requests
import math

MAX_KG_BATCH_SIZE = 100
PYDANTIC_NON_ENTITY_KEY_PATTERN = r'\b[A-Z0-9]{6}(?=\.COMP\.entityType)'
TARGET_ENTITY_MASK = 'Target Company'
OTHER_ENTITY_MASK = 'Other Company'


def build_similarity_queries(sentences: List[str]) -> List[Similarity]:
    if isinstance(sentences, str):
        sentences = [sentences]

    sentences = list(set(sentences))  # Deduplicate
    queries = [Similarity(sentence) for sentence in sentences]
    return queries


def build_batched_similarity_query(
        sentences: List[str],
        entity_keys: List[str],
        batch_size: int = 10,
        fiscal_year: int = None,
        scope: DocumentType = DocumentType.ALL,
) -> List[QueryComponent]:
    queries = build_similarity_queries(sentences)

    entity_keys_batched = [entity_keys[i:i + batch_size]
                           for i in range(0, len(entity_keys), batch_size)]
    queries_expanded = []
    for batch in entity_keys_batched:
        for query in queries:
            if scope == DocumentType.TRANSCRIPTS:
                query &= functools.reduce(lambda x, y: x | y,
                                      [ReportingEntity(entity_key) for
                                       entity_key in batch])
            elif scope == DocumentType.NEWS:
                query &= functools.reduce(lambda x, y: x | y,
                                      [Entity(entity_key) for
                                       entity_key in batch])
            if fiscal_year:
                query &= FiscalYear(fiscal_year)
            queries_expanded.append(query)
    return queries_expanded


def run_search(
        bigdata: Bigdata,
        query: QueryComponent,
        date_range: Optional[Union[AbsoluteDateRange,
        RollingDateRange]] = None,
        sortby: SortBy = SortBy.RELEVANCE,
        scope: DocumentType = DocumentType.ALL,
        limit: int = 10,
) -> List[Document]:
    results = bigdata.search.new(query=query,
                                 date_range=date_range,
                                 sortby=sortby,
                                 scope=scope).run(limit=limit)
    return results


def run_search_concurrent(
    bigdata: Bigdata,
    queries: List[QueryComponent],
    date_range: Optional[Union[AbsoluteDateRange, List[AbsoluteDateRange], RollingDateRange]] = None,
    sortby: SortBy = SortBy.RELEVANCE,
    scope: DocumentType = DocumentType.ALL,
    limit: int = 10,
) -> List[List[Document]]:
    """
    Run searches concurrently for multiple queries and date ranges.

    Parameters:
    - bigdata: Bigdata instance.
    - queries: List of queries to execute.
    - date_range: Single date range or list of date ranges.
    - sortby: Sorting criteria.
    - scope: Scope of the search.
    - limit: Maximum number of results per query.

    Returns:
    - List of lists of documents, one per query.
    """
    # If date_range is a single range, wrap it in a list for uniform handling
    if isinstance(date_range, (AbsoluteDateRange, RollingDateRange)):
        date_range = [date_range]

    # IO-bound operation, so use ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(run_search,
                            bigdata=bigdata,
                            query=query,
                            date_range=dr,
                            sortby=sortby,
                            scope=scope,
                            limit=limit)
            for query in queries
            for dr in date_range  # Iterate over all date ranges
        ]
        logging.info('Futures submitted!')
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                logging.error(f"Error processing a query: {e}")
    logging.info(f"Search complete for {len(queries)} queries and {len(date_range)} date ranges.")
    return results

def run_all_query_date_combinations(
    bigdata, 
    queries: List, 
    sortby, 
    scope, 
    limit: int, 
    start_date: str, 
    end_date: str, 
    freq: str = 'D'
):
    """
    Run searches for all combinations of queries and date ranges.

    Parameters:
    - bigdata: The data source for the search.
    - queries: List of query objects.
    - sortby: Sorting criteria.
    - scope: Scope of the search.
    - limit: Limit for each query result set.
    - start_date: Start date in 'YYYY-MM-DD' format.
    - end_date: End date in 'YYYY-MM-DD' format.
    - freq: Frequency for date intervals ('Y', 'M', 'W', 'D').

    Returns:
    - results_list: Combined results of all searches.
    """
    # Generate date intervals
    date_ranges = create_date_ranges(start_date, end_date, freq)

    # Pass all queries and date ranges to run_search_concurrent
    results = run_search_concurrent(
        bigdata=bigdata,
        queries=queries,
        date_range=date_ranges,  # Pass the list of date ranges
        sortby=sortby,
        scope=scope,
        limit=limit
    )

    return results

def create_monthly_periods(start_date: str,
                           end_date: str) -> List[Tuple[pd._libs.tslibs.timestamps.Timestamp, pd._libs.tslibs.timestamps.Timestamp]]:
    month_starts = pd.date_range(start=start_date, end=end_date, freq='MS')

    periods = []
    for start in month_starts:
        # End of the month is the start of the next month minus one day
        end = start + pd.offsets.MonthEnd(1)
        periods.append((start.date(), end.date()))
    return periods


def create_monthly_date_ranges(start_date: str,
                               end_date: str) -> List[AbsoluteDateRange]:
    data_ranges = create_monthly_periods(start_date, end_date)
    absolute_data_ranges = [AbsoluteDateRange(start,end)
                            for start, end in data_ranges]
    return absolute_data_ranges

def create_date_intervals(
    start_date: str, 
    end_date: str, 
    freq: str
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Create date intervals for a given frequency.

    Parameters:
    - start_date (str): The start date in 'YYYY-MM-DD' format.
    - end_date (str): The end date in 'YYYY-MM-DD' format.
    - freq (str): Frequency string ('Y' for yearly, 'M' for monthly, 'W' for weekly, 'D' for daily).

    Returns:
    - List[Tuple[pd.Timestamp, pd.Timestamp]]: List of start and end date tuples.
    """
    # Convert start and end dates to pandas Timestamps
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    # Adjust frequency for yearly and monthly to use appropriate start markers
    adjusted_freq = {'Y': 'AS', 'M': 'MS'}.get(freq, freq)  # 'AS' for year start, 'MS' for month start
    
    # Generate date range based on the adjusted frequency
    try:
        date_range = pd.date_range(start=start_date, end=end_date, freq=adjusted_freq)
    except ValueError:
        raise ValueError("Invalid frequency. Use 'Y', 'M', 'W', or 'D'.")

    # Create intervals
    intervals = []
    for i in range(len(date_range) - 1):
        intervals.append(
            (date_range[i].replace(hour=0, minute=0, second=0), 
             (date_range[i + 1] - pd.Timedelta(seconds=1)).replace(hour=23, minute=59, second=59))
        )

    # Handle the last range to include the full end_date
    intervals.append((
        date_range[-1].replace(hour=0, minute=0, second=0), 
        end_date.replace(hour=23, minute=59, second=59)
    ))
    
    return intervals

def create_date_ranges(
    start_date: str, 
    end_date: str, 
    freq: str
) -> List[AbsoluteDateRange]:
    """
    Create a list of AbsoluteDateRange objects for the given frequency.

    Parameters:
    - start_date (str): The start date in 'YYYY-MM-DD' format.
    - end_date (str): The end date in 'YYYY-MM-DD' format.
    - freq (str): Frequency string ('Y' for yearly, 'M' for monthly, 'W' for weekly, 'D' for daily).

    Returns:
    - List[AbsoluteDateRange]: List of AbsoluteDateRange objects.
    """
    intervals = create_date_intervals(start_date, end_date, freq=freq)
    return [AbsoluteDateRange(start, end) for start, end in intervals]


def postprocess_search_results(
        results: List[List[Document]]
) -> Tuple[List[Document], List[ListQueryComponent]]:
    # Flatten the list of result lists
    results = list(itertools.chain.from_iterable(results))
    # Collect all entities in the chunks
    entity_keys = collect_entity_keys(results)
    # Look up the entities using Knowledge Graph
    entities = look_up_entities_binary_search(entity_keys)

    # Filter only COMPANY Entities
    entities, topics = filter_company_entities(entities)
    return results, entities, topics


def collect_entity_keys(results: List[Document]) -> List[str]:
    entity_keys = set(entity.key
                      for result in results
                      for chunk in result.chunks
                      for entity in chunk.entities
                      if entity.query_type == QueryType.ENTITY)
    entity_keys = list(entity_keys)
    return entity_keys


def look_up_entities_binary_search(
        entity_keys: List[str]
) -> List[ListQueryComponent]:
    """
    Look up entities using the Knowledge Graph API in a binary search manner.

    :param entity_keys: A list of entity keys to look up
    :return: the list of entities and a list of non-entities
    """
    bigdata = Bigdata()

    entities = []
    non_entities = []

    def dfs(batch: List[str]) -> None:
        """
        Recursively lookup entities in a depth-first search manner.

        :param batch: A batch of entity keys to lookup
        :return:
        """
        try:
            batch_lookup = bigdata.knowledge_graph.get_entities(batch)
            entities.extend(batch_lookup)
        except pydantic.ValidationError as e:
            non_entities_found = re.findall(PYDANTIC_NON_ENTITY_KEY_PATTERN,
                                            str(e))
            non_entities.extend(non_entities_found)
            batch_refined = [key
                             for key in batch
                             if key not in non_entities]
            dfs(batch_refined)
        except (json.JSONDecodeError, RequestMaxLimitExceeds):
            time.sleep(5)
            if len(batch) == 1:
                non_entities.extend(batch)
            else:
                mid = len(batch) // 2
                dfs(batch[:mid])  # First half
                dfs(batch[mid:])  # Second half

    for batch_ in range(0, len(entity_keys), MAX_KG_BATCH_SIZE):
        dfs(entity_keys[batch_:batch_ + MAX_KG_BATCH_SIZE])

    # Deduplicate
    entities = list(
        {entity.id: entity
         for entity in entities
         if hasattr(entity, 'id')}.values()
    )

    return entities


def filter_company_entities(
        entities: List[ListQueryComponent]
) -> List[ListQueryComponent]:
    return [entity for entity in entities
            if hasattr(entity, 'entity_type') and
            entity.entity_type == 'COMP'], [entity for entity in entities
            if hasattr(entity, 'entity_type') and
            entity.entity_type != 'COMP']


def mask_entity_coordinates(input_df, column_masked_text, mask_target,
                            mask_other):
    i = 1
    entity_counter = {}
    input_df['other_entities_map'] = None
    for idx, row in input_df.iterrows():
        text = row['text']
        entities = row['entities']
        entities.sort(key=lambda x: x['start'],
                      reverse=True)
        masked_text = text

        target_start = []
        target_end = []
        other_entity_map = []
        for entity in entities:

            if entity['key'] == row['rp_entity_id']:
                target_start.append(entity['start'])
                target_end.append(entity['end'])

        for entity in entities:
            start, end = entity['start'], entity['end']
            if entity['key'] == row['rp_entity_id']:
                masked_text = masked_text[:start] + mask_target + masked_text[
                                                                  end:]


            elif (entity['key'] != row['rp_entity_id']) & (
                    start not in target_start) & (end not in target_end):
                if entity['key'] not in entity_counter:
                    entity_counter[entity['key']] = i
                    masked_text = masked_text[
                                  :start] + f'{mask_other}_{entity_counter[entity["key"]]}' + masked_text[
                                                                                              end:]
                    other_entity_map.append(
                        (entity_counter[entity["key"]], entity['name']))
                    i += 1
                else:
                    masked_text = masked_text[
                                  :start] + f'{mask_other}_{entity_counter[entity["key"]]}' + masked_text[
                                                                                              end:]
                    other_entity_map.append(
                        (entity_counter[entity["key"]], entity['name']))

        input_df.at[idx, column_masked_text] = masked_text
        input_df.at[
            idx, 'other_entities_map'] = other_entity_map if other_entity_map else None  ##beta!!

    return input_df


def mask_sentences(df: pd.DataFrame,
                   target_entity_mask: str,
                   other_entity_mask: str) -> pd.DataFrame:
    df['text'] = df['text'].str.replace('{', '', regex=False)
    df['text'] = df['text'].str.replace('}', '', regex=False)

    df = mask_entity_coordinates(input_df=df,
                                 column_masked_text='masked_text',
                                 mask_target=target_entity_mask,
                                 mask_other=other_entity_mask)

    df['masked_text'] = df['masked_text'].apply(
        lambda x: x.replace('{', '').replace('}', '')
    )
    df = df[df.masked_text != 'to_remove']
    df['text'] = df['text'].apply(
        lambda x: x.replace('{', '').replace('}', '')
    )
    df = df[df.text != 'to_remove']
    return df


def build_dataframe_reporting_entity(
        results: List[Document],
        entities: List[ListQueryComponent],
    topics:List[ListQueryComponent]
) -> pd.DataFrame:
    entity_key_map = {entity.id: entity for entity in entities}
    topic_key_map = {entity.id: entity for entity in topics}

    rows = []
    for result in results:
        for chunk_index, chunk in enumerate(result.chunks):
            # Build a list of entities present in the chunk
            chunk_entities = [{'key': entity.key,
                               'name': entity_key_map[entity.key].name,
                               'ticker': entity_key_map[entity.key].ticker,
                               'country': entity_key_map[entity.key].country,
                               'start': entity.start,
                               'end': entity.end}
                              for entity in chunk.entities
                              if entity.key in entity_key_map]
            chunk_topics = [{'key': entity.key,
                               'name': topic_key_map[entity.key].name,
                               'entity_type': topic_key_map[entity.key].entity_type,
                               #'country': entity_key_map[entity.key].country,
                               'start': entity.start,
                               'end': entity.end}
                              for entity in chunk.entities
                              if entity.key in topic_key_map]

            if not chunk_entities:
                continue  # Skip if no entities are mapped

            # Process each reporting entity
            for re_key in result.reporting_entities:
                reporting_entity = entity_key_map.get(re_key)
                if not reporting_entity:
                    continue  # Skip if reporting entity is not found

                # Exclude the reporting entity from other entities
                other_entities = [e
                                  for e in chunk_entities
                                  if e['name'] != reporting_entity.name]

                # Collect all necessary information in the row
                rows.append({
                    'timestamp_utc': result.timestamp,
                    'rp_document_id': result.id,
                    'sentence_id': f'{result.id}-{chunk_index}',
                    'headline': result.headline,
                    'rp_entity_id': re_key,
                    'entity_name': reporting_entity.name,
                    'entity_sector': reporting_entity.sector,
                    'entity_industry': reporting_entity.industry,
                    'entity_country': reporting_entity.country,
                    'entity_ticker': reporting_entity.ticker,
                    'text': chunk.text,
                    'other_entities': ', '.join(e['name']
                                                for e in other_entities),
                    'entities': chunk_entities,
                    'topics': [topic['name'] for topic in chunk_topics],
                    'topic_type': [topic['entity_type'] for topic in chunk_topics],
                })
    if not rows:
        raise ValueError('No rows to process')
    df = pd.DataFrame(rows)
    df = df.sort_values('timestamp_utc').reset_index(drop=True)
    df = df.drop_duplicates(subset=['timestamp_utc', 'rp_document_id',
                                    'sentence_id', 'rp_entity_id'])
    df = mask_sentences(df,
                        target_entity_mask=TARGET_ENTITY_MASK,
                        other_entity_mask=OTHER_ENTITY_MASK)
    df = df.reset_index(drop=True)
    return df


def build_dataframe(results: List[Document],
                    entities: List[ListQueryComponent],
                   topics:List[ListQueryComponent]) -> pd.DataFrame:
    entity_key_name_ticker_map = {entity.id: {'name': entity.name,
                                              'ticker': entity.ticker,
                                             'sector': entity.sector,'industry': entity.industry,'country': entity.country}
                                  for entity in entities}
    
    topic_key_map = {entity.id: entity for entity in topics}

    rows = []
    for result in results:
        chunk_texts_with_entities = []
        for chunk_index, chunk in enumerate(result.chunks):
            chunk_entities = []
            for entity in chunk.entities:
                name_and_ticker = entity_key_name_ticker_map.get(entity.key,
                                                                 'Unknown')
                if name_and_ticker != 'Unknown':
                    chunk_entities.append({
                        'key': entity.key,
                        'name': name_and_ticker['name'],
                        'ticker': name_and_ticker['ticker'],
                        'sector': name_and_ticker['sector'],'industry': name_and_ticker['industry'],'country': name_and_ticker['country'],
                        'start': entity.start,
                        'end': entity.end
                    })
                else:
                    chunk_topics = [{'key': entity.key,
                               'name': topic_key_map[entity.key].name,
                               'entity_type': topic_key_map[entity.key].entity_type,
                               #'country': entity_key_map[entity.key].country,
                               'start': entity.start,
                               'end': entity.end}
                              for entity in chunk.entities
                              if entity.key in topic_key_map]
            if chunk_entities:
                chunk_texts_with_entities.append((chunk.text,
                                                  chunk_entities,
                                                  chunk.chunk,
                                                 chunk_topics))

        for chunk_text, chunk_entities, chunk_index, chunk_topics in chunk_texts_with_entities:
            for entity in chunk_entities:
                other_entities = [e
                                  for e in chunk_entities
                                  if e['name'] != entity['name']]
                rows.append({
                    'timestamp_utc': result.timestamp,
                    'rp_document_id': result.id,
                    'sentence_id': f"{result.id}-{chunk_index}",
                    'headline': result.headline,
                    'rp_entity_id': entity['key'],
                    'entity_name': entity['name'],
                    'entity_sector': entity.get('sector',None),
                    'entity_industry': entity.get('industry',None),
                    'entity_country': entity.get('country',None),
                    'entity_ticker': entity['ticker'],
                    'text': chunk_text,
                    'other_entities': ', '.join([e['name']
                                                 for e in other_entities]),
                    'entities': chunk_entities,
                    'topics': [topic['name'] for topic in chunk_topics],
                    'topic_type': [topic['entity_type'] for topic in chunk_topics],
                    'source_name': result.source.name,
                    'source_rank': result.source.rank,
                    'url': result.url
                                             })
    df = pd.DataFrame(rows)
    df = df.sort_values('timestamp_utc').reset_index(drop=True)
    df = df.drop_duplicates(subset=['timestamp_utc', 'rp_document_id',
                                    'sentence_id', 'rp_entity_id'])
    # df = mask_sentences(df,
    #                     target_entity_mask=TARGET_ENTITY_MASK,
    #                     other_entity_mask=OTHER_ENTITY_MASK)
    
    df['date'] = pd.to_datetime(df.timestamp_utc.dt.date)
    df = df.reset_index(drop=True)
    
    return df

#