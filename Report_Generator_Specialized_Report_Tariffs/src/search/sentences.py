"""Postprocess search results from REST API (no SDK dependencies).

MIGRATION NOTE: SDK imports removed. Works with list[dict] REST documents.
Entity lookup via BigdataRestClient.get_entities_by_id if needed.
"""

import logging
import pandas as pd
from typing import List, Dict, Any

TARGET_ENTITY_MASK = 'Target Company'
OTHER_ENTITY_MASK = 'Other Company'


def postprocess_search_results(
    results: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Postprocess REST API search results into a DataFrame.

    :param results: A list of REST document dicts (from run_universe_search or similar)
    :return: DataFrame with document/chunk data
    """
    if not results:
        logging.info('No results to postprocess')
        return pd.DataFrame()
    
    # If results is already a DataFrame, return it
    if isinstance(results, pd.DataFrame):
        return results
    
    # Otherwise convert list of dicts to DataFrame
    df = pd.DataFrame(results)
    return df


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
    entity_id_col = 'entity_id'
    
    for idx, row in input_df.iterrows():
        text = row.get('text', '')
        entities = row.get('entities', [])
        if isinstance(entities, list):
            entities.sort(key=lambda x: x.get('start', 0), reverse=True)
        masked_text = text

        target_start = []
        target_end = []
        other_entity_map = []
        for entity in entities:

            if entity.get('key') == row.get(entity_id_col):
                target_start.append(entity.get('start'))
                target_end.append(entity.get('end'))

        for entity in entities:
            start, end = entity.get('start'), entity.get('end')
            if entity.get('key') == row.get(entity_id_col):
                masked_text = masked_text[:start] + mask_target + masked_text[
                    end:]

            elif (entity.get('key') != row.get(entity_id_col)) & (
                    start not in target_start) & (end not in target_end):
                if entity.get('key') not in entity_counter:
                    entity_counter[entity.get('key')] = i
                    masked_text = masked_text[
                        :start] + f'{mask_other}_{entity_counter[entity.get("key")]}' + masked_text[
                        end:]
                    other_entity_map.append(
                        (entity_counter[entity.get("key")], entity.get('name')))
                    i += 1
                else:
                    masked_text = masked_text[
                        :start] + f'{mask_other}_{entity_counter[entity.get("key")]}' + masked_text[
                        end:]
                    other_entity_map.append(
                        (entity_counter[entity.get("key")], entity.get('name')))

        input_df.at[idx, column_masked_text] = masked_text
        input_df.at[
            idx, 'other_entities_map'] = other_entity_map if other_entity_map else None

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
        lambda x: x.replace('{', '').replace('}', '') if isinstance(x, str) else x
    )
    df = df[df.masked_text != 'to_remove']
    df['text'] = df['text'].apply(
        lambda x: x.replace('{', '').replace('}', '') if isinstance(x, str) else x
    )
    df = df[df.text != 'to_remove']
    return df


def build_dataframe_entity(
    results: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Build a DataFrame from REST API search results.

    :param results: A list of REST document dicts
    :return: The DataFrame
    """
    df = postprocess_search_results(results)
    if df.empty:
        return df
    
    # Apply entity masking if entities column exists
    if 'entities' in df.columns:
        df = mask_sentences(df,
                            target_entity_mask=TARGET_ENTITY_MASK,
                            other_entity_mask=OTHER_ENTITY_MASK,
                            document_type="news")
    
    return df


def build_dataframe_reporting_entity(
    results: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Build a DataFrame from REST API search results for reporting entities (filings/transcripts).

    :param results: A list of REST document dicts
    :return: The DataFrame
    """
    df = postprocess_search_results(results)
    if df.empty:
        return df
    
    # Apply entity masking if entities column exists
    if 'entities' in df.columns:
        df = mask_sentences(df,
                            target_entity_mask=TARGET_ENTITY_MASK,
                            other_entity_mask=OTHER_ENTITY_MASK,
                            document_type="transcripts")
    
    return df
