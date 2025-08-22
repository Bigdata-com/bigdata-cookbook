"""
Copyright (C) 2024, RavenPack | Bigdata.com. All rights reserved.
Author: Alessandro Bouchs (abouchs@ravenpack.com)
"""

from bigdata_client import Bigdata
from bigdata_client.models.search import DocumentType, SortBy
from bigdata_client.query import Keyword, Entity, Any
from bigdata_client.daterange import AbsoluteDateRange

import time
from tqdm.notebook import tqdm
import pandas as pd
import hashlib
import plotly.express as px
from IPython.display import display, HTML
import re
import openai
MODEL_NAME = 'gpt-4o-mini'

def process_dataframe( 
    results):
    """
    Build a dataframe for when no companies are specified.

    Args:
        results (List[Document]): A list of Bigdata search results.

    Returns:
        DataFrame: Screening DataFrame. Schema:
        - Index: int
        - Columns:
            - timestamp_utc: datetime64
            - document_id: str
            - sentence_id: str
            - headline: str
            - text: str
    """
    
    rows = []
    for result in results:
        for chunk in result.chunks:
            # Collect all necessary information in the row
            rows.append({
                    'timestamp_utc': result.timestamp,
                    'rp_document_id': result.id,
                    'sentence_id': f"{result.id}-{chunk.chunk}",
                    'headline': result.headline,
                    'text': chunk.text,
                    'source_name': result.source.name,
                    'source_rank': result.source.rank,
                    'url': result.url})

    if not rows:
        raise ValueError("No rows to process")

    df = pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)

    # Deduplicate by quote text as well
    df = df.drop_duplicates(
        subset=["timestamp_utc", "rp_document_id", "text"]
    )
    df['date'] = pd.to_datetime(pd.to_datetime(df['timestamp_utc']).dt.date)

    df = df.reset_index(drop=True)
    return df


def drop_duplicates(df = pd.DataFrame(), date_column = '', duplicate_set = []):
    df = df.sort_values(by=date_column)
    non_dup_index = []
    dup_index = []
    for i, gp in df.groupby(duplicate_set):
        non_dup_index.append(gp.iloc[0].name)
        dup_index.extend(gp.iloc[1:].index.values)
    deduplicated_df = df.loc[non_dup_index,:].copy().sort_values(by=date_column).reset_index(drop=True)
    duplicates = df.loc[dup_index,:].copy().sort_values(by=date_column).reset_index(drop=True)
    
    return deduplicated_df, duplicates

def reinstate_duplicates(labelled_df= pd.DataFrame(), duplicated_chunks = pd.DataFrame(), date_column = '', duplicate_set = [], fill_cols = ['exposure','motivation']):
    
    final_df = pd.concat([labelled_df, duplicated_chunks], axis=0).sort_values(by=duplicate_set+[date_column]).reset_index(drop=True)
    for col in fill_cols:
        final_df[col] = final_df.groupby(duplicate_set)[col].ffill()
    # final_df['label'] = final_df.groupby(duplicate_set)['label'].ffill()
    # final_df['motivation'] = final_df.groupby(duplicate_set)['motivation'].ffill()
    final_df = final_df.sort_values(by=date_column).reset_index(drop=True)
    
    return final_df

def visualize_tree(tree) -> pd.DataFrame:
    def extract_labels(node, parent_label=''):
        labels.append(node['Label'])
        parents.append(parent_label)
        if 'Children' in node:
            for child in node['Children']:
                extract_labels(child, node['Label'])

    labels = []
    parents = []
    extract_labels(tree)

    df = pd.DataFrame({'labels': labels,
                       'parents': parents})
    fig = px.treemap(df, names='labels', parents='parents')
    fig.show()