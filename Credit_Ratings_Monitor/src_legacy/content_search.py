import pandas as pd
import asyncio
import os
from bigdata_client import Bigdata
from bigdata_client.query import Similarity, All, Entity, Keyword, Document
from bigdata_client.models.search import DocumentType, SortBy
from bigdata_client.daterange import RollingDateRange, AbsoluteDateRange
import collections
from collections import defaultdict, OrderedDict
import requests
from requests.exceptions import HTTPError
import warnings
warnings.filterwarnings('ignore')
import time
import sys
from itertools import islice
import re
from functools import reduce
from tqdm.notebook import tqdm
import json
from pqdm.threads import pqdm
from operator import itemgetter
import configparser
from itertools import product
from datetime import datetime

def create_dates_lists(start_date, end_date, freq='W'):
    from datetime import datetime
    ## Convert start_date and end_date to datetime objects if they are not already
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    # Initialize lists
    start_list = []
    end_list = []
    # Generate date ranges
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    for i in range(len(date_range) - 1):
        start_list.append(date_range[i].strftime('%Y-%m-%d 00:00:00'))
        end_list.append((date_range[i + 1] - pd.Timedelta(days=1)).strftime('%Y-%m-%d 23:59:59'))
    # Handle the last range
    start_list.append(date_range[-1].strftime('%Y-%m-%d 00:00:00'))
    end_list.append(end_date.strftime('%Y-%m-%d 23:59:59'))
    return start_list, end_list
    
def query_enhanced_chunks(rp_document_id, bigdata_credentials):
    """
    Queries the entire document by rp_document_id and retrieves all chunks.
    Returns a DataFrame of all chunks and their text.
    """
    bigdata = bigdata_credentials
    all_chunks = []

    # Search for the document by rp_document_id
    search_doc = bigdata.search.new(Document(rp_document_id))
    results_doc = search_doc.limit_documents(1)

    for doc in results_doc:
        for chunk in doc.chunks:
            all_chunks.append({
                'timestamp_utc': doc.timestamp,
                'rp_documtent_id': doc.id,
                'sentence_id': f"{doc.id}-{chunk.chunk}",
                'chunk_number': chunk.chunk,
                'text': chunk.text
            })

    return pd.DataFrame(all_chunks)

def inject_context_rows(df, bigdata_credentials, batch_size=500, n_jobs=10):
    """
    Injects additional rows for the previous and next chunks into the DataFrame.
    Queries the full document once for each rp_document_id and retrieves all chunks.
    Parallelized using pqdm.
    """
    
    if 'chunk_number' not in df.columns:
        df['chunk_number'] = df['sentence_id'].str.split('-').str[1].astype(int)
    else:
        df['chunk_number'] = df['chunk_number'].astype(int)
    
    new_rows = []
    start_time = time.time()

    # Process rows in batches with a progress bar
    for i in tqdm(range(0, len(df), batch_size), desc="Enanching chunks context"):
        batch = df.iloc[i:i + batch_size]

        # Prepare arguments for parallel processing
        query_args = [(row['rp_document_id'], bigdata_credentials) for _, row in batch.iterrows()]
        
        # Parallelize the queries with pqdm
        results = pqdm(query_args, query_enhanced_chunks, n_jobs=n_jobs, argument_type='args')
        
        # Process each document's chunks
        for row, all_chunks in tqdm(zip(batch.iterrows(), results),total=len(batch), desc='Storing results...'):
            _, row = row
            rp_document_id = row['rp_document_id']
            chunk_number = row['chunk_number']
            # Inject previous chunk as a new row if it exists
            previous_chunk_text = get_chunk_text(all_chunks, chunk_number - 1)
            if previous_chunk_text:
                new_row = row.copy()
                new_row['text'] = previous_chunk_text
                new_row['chunk_number'] = chunk_number - 1
                new_row['sentence_id'] = f"{rp_document_id}-{chunk_number - 1}"
                new_rows.append(new_row)
            
            # Inject next chunk as a new row if it exists
            next_chunk_text = get_chunk_text(all_chunks, chunk_number + 1)
            if next_chunk_text:
                new_row = row.copy()
                new_row['text'] = next_chunk_text
                new_row['chunk_number'] = chunk_number + 1
                new_row['sentence_id'] = f"{rp_document_id}-{chunk_number + 1}"
                new_rows.append(new_row)

        # Ensure no more than batch_size rows are processed in 60 seconds
        if (time.time() - start_time) < 60:
            time.sleep(60 - (time.time() - start_time))
        start_time = time.time()

    # Convert the new rows to a DataFrame
    additional_rows_df = pd.DataFrame(new_rows)

    # Append the additional rows to the original DataFrame
    df = pd.concat([df, additional_rows_df], ignore_index=True)
    df = df.sort_values(['timestamp_utc','rp_document_id','chunk_number']).reset_index(drop=True)
    df = df.drop_duplicates(subset=['timestamp_utc','rp_document_id','chunk_number']).reset_index(drop=True)

    return df

def get_chunk_text(chunks, chunk_number):
    """
    Retrieves the text for a specific chunk_number from the list of chunks.
    """
    for i, row in chunks.iterrows():
        if row['chunk_number'] == chunk_number:
            return row['text']
    return None

def inject_context_text(df, bigdata_credentials, batch_size=500, n_jobs=10):
    """
    Injects the previous and next chunk's text into the 'text' column of each row, 
    concatenating the chunks in the correct order without adding new rows.
    Parallelized using pqdm.
    """
    
    if 'chunk_number' not in df.columns:
        df['chunk_number'] = df['sentence_id'].str.split('-').str[1].astype(int)
    else:
        df['chunk_number'] = df['chunk_number'].astype(int)
    
    
    df['contextualized_chunk_text'] = ""  # Create an empty column to hold the concatenated text
    # This will hold the updated rows with concatenated texts
    updated_rows = []
    start_time = time.time()

    # Process rows in batches with a progress bar
    for i in tqdm(range(0, len(df), batch_size), desc="Enhancing chunks context"):
        batch = df.iloc[i:i + batch_size]

        # Prepare arguments for parallel processing
        query_args = [(row['rp_document_id'], bigdata_credentials) for _, row in batch.iterrows()]
        
        # Parallelize the queries with pqdm
        results = pqdm(query_args, query_enhanced_chunks, n_jobs=n_jobs, argument_type='args')
        
        # Process each document's chunks
        for row, all_chunks in tqdm(zip(batch.iterrows(), results), total=len(batch), desc='Concatenating chunk texts'):
            _, row = row
            rp_document_id = row['rp_document_id']
            chunk_number = row['chunk_number']
            
            # Retrieve the previous and next chunk's text
            previous_chunk_text = get_chunk_text(all_chunks, chunk_number - 1)
            current_chunk_text = get_chunk_text(all_chunks, chunk_number)
            next_chunk_text = get_chunk_text(all_chunks, chunk_number + 1)

            # Concatenate the texts in the correct order (previous + current + next)
            full_text = " ".join([t for t in [previous_chunk_text, current_chunk_text, next_chunk_text] if t])

            # Replace the row's 'text' with the concatenated text
            row['contextualized_chunk_text'] = full_text
            updated_rows.append(row)

        # Ensure no more than batch_size rows are processed in 60 seconds
        if (time.time() - start_time) < 60:
            time.sleep(60 - (time.time() - start_time))
        start_time = time.time()

    # Convert the updated rows to a DataFrame
    updated_df = pd.DataFrame(updated_rows)

    # Ensure the final DataFrame is sorted correctly
    updated_df = updated_df.sort_values(['timestamp_utc', 'rp_document_id', 'chunk_number']).reset_index(drop=True)

    return updated_df