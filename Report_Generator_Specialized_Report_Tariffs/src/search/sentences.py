import itertools
import json
import logging
import re
import time
from typing import List, Tuple
import pandas as pd
import pydantic
from bigdata_client import Bigdata, Company
from bigdata_client.connection import RequestMaxLimitExceeds
from bigdata_client.document import Document
from bigdata_client.models.advanced_search_query import ListQueryComponent
from bigdata_client.query_type import QueryType
from bigdata_client.models.search import DocumentType

MAX_KG_BATCH_SIZE = 50
PYDANTIC_NON_ENTITY_KEY_PATTERN = r'\b[A-Z0-9]{6}(?=\.COMP\.entityType)'
TARGET_ENTITY_MASK = 'Target Company'
OTHER_ENTITY_MASK = 'Other Company'
_bigdata = None

def get_big_data_conn():
    global _bigdata
    if _bigdata:
       return _bigdata
    else:
       _bigdata = Bigdata()
       return _bigdata

def collect_entity_keys(results: List[Document]) -> List[str]:
    """
    Collect all entity keys from the search results.

    :param results: A list of search results
    :return: A list of entity keys
    """
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
    bigdata = get_big_data_conn()

    entities = []
    non_entities = []
    # TODO non_entities never gets used. Is that intentional?

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
        except Exception as e:
            logging.error(f'Error in batch {batch}')
            time.sleep(60)  # Wait for a minute
            dfs(batch)

    logging.info(f'Split into batches of {MAX_KG_BATCH_SIZE} entities')
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
    """
    Filter only COMPANY entities from the list of entities.

    :param entities: A list of entities to filter
    :return: A list of COMPANY entities
    """
    return [entity for entity in entities
            if hasattr(entity, 'entity_type') and
            entity.entity_type == 'COMP']


def postprocess_search_results(
        results: List[List[Document]]
) -> Tuple[List[Document], List[ListQueryComponent]]:
    """
    Postprocess the search results to filter only COMPANY entities.

    :param results: A list of search results
    :return: A tuple of the filtered search results and the entities
    """
    # Flatten the list of result lists
    results = list(itertools.chain.from_iterable(results))
    # Collect all entities in the chunks
    entity_keys = collect_entity_keys(results)
    # Look up the entities using Knowledge Graph
    entities = look_up_entities_binary_search(entity_keys)

    # Filter only COMPANY Entities
    entities = filter_company_entities(entities)
    return results, entities


def mask_entity_coordinates(input_df, column_masked_text, mask_target,
                            mask_other, document_type):
    """
    Mask the target entity and other entities in the text.

    :param input_df: The input DataFrame
    :param column_masked_text: The column name for the masked text
    :param mask_target: The mask for the target entity
    :param mask_other: The mask for other entities
    :return: The masked DataFrame
    """
    i = 1
    entity_counter = {}
    input_df[column_masked_text] = None
    input_df['other_entities_map'] = None

    # Ensure columns are compatible with string/object assignments
    input_df[column_masked_text] = input_df[column_masked_text].astype(
        "object")
    input_df['other_entities_map'] = input_df['other_entities_map'].astype(
        "object")
    entity_id = 'rp_entity_id' if document_type in (DocumentType.TRANSCRIPTS, DocumentType.FILINGS) else 'entity_id'
    
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

            if entity['key'] == row[entity_id]:
                target_start.append(entity['start'])
                target_end.append(entity['end'])

        for entity in entities:
            start, end = entity['start'], entity['end']
            if entity['key'] == row[entity_id]:
                masked_text = masked_text[:start] + mask_target + masked_text[
                    end:]

            elif (entity['key'] != row[entity_id]) & (
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
            idx, 'other_entities_map'] = other_entity_map if other_entity_map else None  # beta!!

    return input_df


def mask_sentences(df: pd.DataFrame,
                   target_entity_mask: str,
                   other_entity_mask: str,
                   document_type: str) -> pd.DataFrame:
    """
    Mask the target entity and other entities in the text.

    :param df: The input DataFrame
    :param target_entity_mask: The mask for the target entity
    :param other_entity_mask: The mask for other entities
    :return: The masked DataFrame
    """
    df['text'] = df['text'].str.replace('{', '', regex=False)
    df['text'] = df['text'].str.replace('}', '', regex=False)

    df = mask_entity_coordinates(input_df=df,
                                 column_masked_text='masked_text',
                                 mask_target=target_entity_mask,
                                 mask_other=other_entity_mask, 
                                 document_type=document_type)

    df['masked_text'] = df['masked_text'].apply(
        lambda x: x.replace('{', '').replace('}', '')
    )
    df = df[df.masked_text != 'to_remove']
    df['text'] = df['text'].apply(
        lambda x: x.replace('{', '').replace('}', '')
    )
    df = df[df.text != 'to_remove']
    return df


def build_dataframe_entity(
        results: List[Document],
        entities: List[ListQueryComponent],
        companies: List[Company]
) -> pd.DataFrame:
    """
    Build a DataFrame from the search results and entities.

    :param results: A list of search results
    :param entities: A list of entities
    :return: The DataFrame
    """
    entity_key_map = {entity.id: entity for entity in entities}

    rows = []
    for result in results:
        for chunk_index, chunk in enumerate(result.chunks):
            # Build a list of entities present in the chunk
            chunk_entities = [{'key': entity.key,
                               'name': entity_key_map[entity.key].name,
                               'ticker': entity_key_map[entity.key].ticker,
                               'start': entity.start,
                               'end': entity.end}
                              for entity in chunk.entities
                              if entity.key in entity_key_map]

            if not chunk_entities:
                continue  # Skip if no entities are mapped

            for chunk_entity in chunk_entities:
                entity_key = entity_key_map.get(chunk_entity['key'])

                if not entity_key:
                    continue  # Skip if entity is not found

                # if entity isn't in our original watchlist, skip
                if entity_key not in companies:
                    continue

                # Exclude the entity from other entities
                other_entities = [e
                                  for e in chunk_entities
                                  if e['name'] != chunk_entity['name']]

                # Collect all necessary information in the row
                rows.append({
                    'timestamp_utc': result.timestamp,
                    'document_id': result.id,
                    'rp_document_id': None,
                    'sentence_id': f'{result.id}-{chunk_index}',
                    'headline': result.headline,
                    'rp_entity_id': None,
                    'entity_id': chunk_entity['key'],
                    'entity_name': entity_key.name,
                    'entity_sector': entity_key.sector,
                    'entity_industry': entity_key.industry,
                    'entity_country': entity_key.country,
                    'entity_ticker': entity_key.ticker,
                    'text': chunk.text,
                    'other_entities': ', '.join(e['name']
                                                for e in other_entities),
                    'entities': chunk_entities
                })

    if not rows:
        logging.info('No rows to process')
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values('timestamp_utc').reset_index(drop=True)

    # Deduplicate by quote text as well
    df = df.drop_duplicates(subset=['timestamp_utc', 'document_id',
                                    'text', 'entity_id'])
    df = mask_sentences(df,
                        target_entity_mask=TARGET_ENTITY_MASK,
                        other_entity_mask=OTHER_ENTITY_MASK,
                        document_type=DocumentType.NEWS)
    df = df.reset_index(drop=True)
    return df


def build_dataframe_reporting_entity(
        results: List[Document],
        entities: List[ListQueryComponent]
) -> pd.DataFrame:
    """
    Build a DataFrame from the search results and entities.

    :param results: A list of search results
    :param entities: A list of entities
    :return: The DataFrame
    """
    entity_key_map = {entity.id: entity for entity in entities}

    rows = []
    for result in results:
        for chunk_index, chunk in enumerate(result.chunks):
            # Build a list of entities present in the chunk
            chunk_entities = [{'key': entity.key,
                               'name': entity_key_map[entity.key].name,
                               'ticker': entity_key_map[entity.key].ticker,
                               'start': entity.start,
                               'end': entity.end}
                              for entity in chunk.entities
                              if entity.key in entity_key_map]

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
                    'entities': chunk_entities
                })
    if not rows:
        raise ValueError('No rows to process')
    df = pd.DataFrame(rows)
    df = df.sort_values('timestamp_utc').reset_index(drop=True)

    # Deduplicate by quote text as well
    df = df.drop_duplicates(subset=['timestamp_utc', 'rp_document_id',
                                    'text', 'rp_entity_id'])
    df = mask_sentences(df,
                        target_entity_mask=TARGET_ENTITY_MASK,
                        other_entity_mask=OTHER_ENTITY_MASK,
                        document_type=DocumentType.TRANSCRIPTS)
    df = df.reset_index(drop=True)
    return df
