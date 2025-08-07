import argparse
from hashlib import blake2b
import json
import os
from typing import Optional, Union
import pandas as pd
import re
import torch.nn as nn
import numpy as np

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainerState
import torch
from transformers import EvalPrediction, PreTrainedTokenizer, PreTrainedTokenizerFast

import snowflake.connector as sf


try:
    from tools.classification_training import get_logits
except ModuleNotFoundError:
    from classification_training import get_logits


def perform_inference(
        model: nn.Module,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        dataset: Dataset,
        text_column: str,
        device: Optional[str] = None,
        batch_size: int = 8  # Default batch size
        ):
    """
    Performs classification inference over a given dataset. 

    Returns:
    dict: A dictionary containing the prediction results.
    Evaluates the performance of a sequence classification model on a given dataset.

    This function computes the logits, resulting probabilities and corresponding class for each example in the dataset,
    and then computes the performance metrics based on the problem type.

    Args:
        model (AutoModelForSequenceClassification): The pre-trained model for sequence classification.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        dataset (Dataset): The dataset to evaluate the model on.
        text_column (str): The name of the column in the dataset that contains the text.
        device (Optional[str]): The device to run the evaluation on. If None, defaults to 'cuda' if available, else 'cpu'.
        batch_size (int): The batch size to use for evaluation. Defaults to 8.
        
    Returns:
        DatasetDict: The dataset with the logits, corresponding probabilities and class for each class and predicted class

    """     
    
    _device = (device or torch.device('cuda')) if torch.cuda.is_available() else torch.device('cpu')
    model.to(_device)
    model.eval()
    
    dataset = dataset.rename_columns({text_column: '_text_column'}).map(
        get_logits,
        fn_kwargs={'model': model, 'tokenizer': tokenizer},
        batched=True,
        batch_size=batch_size, 
        load_from_cache_file=False,
        cache_file_name=None,
        keep_in_memory=True
    )
    dataset = dataset.filter(lambda x: np.isnan(x['_logits']).sum()==0)

    dataset = dataset.with_format('numpy')
    logits = dataset['_logits']
    pred_probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
    for class_id in range(pred_probs.shape[1]):
        dataset = dataset.add_column(
            f'P(class={class_id})', pred_probs[:, class_id])
    predictions = np.argmax(logits, axis=-1)
    dataset = dataset.add_column('predicted_class', predictions)
    
    return dataset



def clean_text(text: str) -> str:
    """
    Transforms the given sample text by...

    Args:
    - text (str): The sample text to be transformed.

    Returns:
    - str: Transformed text.
    """
    #...  replacing newlines with spaces
    text = text.replace('\n', ' ')

    #... removing ticker symbols ">TICKER" (mostly common at the end of some news flashes)
    text = re.sub(r'>[A-Z]+', '', text)

    #... removing news sources at the end of the sentence "-- SOURCE_NAME"  (mostly common in some news headlines)
    text = re.sub(r'-- \w+\.?\w+\.?\w*$', '', text)

    #... removing numeric trailing patterns like -4-, -5-, etc. (mostly appearing in some news headlines)
    text = re.sub(r' -\d+-$', '', text)

    #... removing some timestamps (more versions could be added)
    text = re.sub(r'\d\d:\d\d EDT\s*', '', text)
    
    #... removing URLs
    text = re.sub(r'https?://\S+|www\.\S+','',text)
    
    
    #... we will also get rid of sentences that have less than 6 words (counting the identified entities)
    text2=re.sub('[^A-Za-z ]+', '', text)
    if len(text2.split())<6:
        text="to_remove"

    return text.strip()



def generate_sql_query(database_annotations, schema_annotations, 
                       from_year_month, to_year_month, entity_list, entity_relevance):
    '''
    Generates a SQL query to retrieve all the sentences from RP Annotations in Snowflake where entities in a given
    list are detected, for documents where they have an entity Relevance score higher than a given threshold

    Parameters:
    - database_annotations (str): Snowflake database where the RP Annotations tables are located
    - schema_annotations (str): Snowflake schema where the RP Annotations tables are located
    - from_year_month (str): Start year-month for filtering data in the SQL query (format: 'YYYY-MM').
    - to_year_month (str): End year-month for filtering data in the SQL query (format: 'YYYY-MM').
    - entity_list (list of str): Entity IDs (RP_ENTITY_IDs) of the entities to be detected in the documents.
    - entity_relevance (int): Entity Relevance threshold 

    Returns:
    - str: SQL query for retrieving the data from Snowflake.

    '''
    
    query = f"""
        WITH documents AS(
            -- 1. Retrieve potential document and entity pairs or direct detections with enough relevance
            SELECT
                timestamp_utc,
                rp_document_id,
                rp_entity_id,
                entity_name
            FROM
                {database_annotations}.{schema_annotations}.entity_data
            WHERE
                yearmonth BETWEEN '{from_year_month}' and '{to_year_month}'
                AND rp_entity_id in {entity_list}
                AND entity_relevance >= {entity_relevance}
                AND entity_detection_type = 'direct'
        ),

        filt_documents AS(
            -- 2. Restrict pairs to documents 
            SELECT
            a.*
            FROM documents a,
                {database_annotations}.{schema_annotations}.doc_analytics b
            WHERE
            yearmonth BETWEEN '{from_year_month}' and '{to_year_month}'
            AND a.rp_document_id = b.rp_document_id
            AND rp_provider_id in ('WEB')
            AND source_rank <= 4
            AND title_similarity_days >= 1
            AND original_language = 'en'
        ),
        
        sentences AS (
            -- 3. Retrieve IDs of the sentences for the selected document and entity pairs
            SELECT 
                a.timestamp_utc, 
                a.rp_document_id, 
                a.rp_entity_id, 
                a.entity_name,
                b.sentence_id, 
                b.coordinates
            FROM 
                filt_documents a,
                {database_annotations}.{schema_annotations}.sentence_entities b
            WHERE 
                yearmonth BETWEEN '{from_year_month}' and '{to_year_month}'
                AND a.rp_document_id = b.rp_document_id
                AND a.rp_entity_id = b.rp_entity_id
                AND b.coordinates IS NOT NULL
        )
        
        -- 4. Retrieve the text of those sentence IDs
        SELECT
            a.timestamp_utc,
            a.rp_document_id,
            a.sentence_id,
            a.rp_entity_id,
            a.entity_name,
            a.coordinates,
            substring(b.text, a.coordinates[0][0] + 1, a.coordinates[0][1] - a.coordinates[0][0]) AS extracted_entity_name,
            b.text
        FROM
            sentences a,
            {database_annotations}.{schema_annotations}.doc_sentences b
        WHERE
            yearmonth BETWEEN '{from_year_month}' and '{to_year_month}'
            AND a.sentence_id = b.sentence_id
        
    """
    
    return query


def execute_snowflake_query(query, database = None, schema = None, fetchResults = False ):
    '''
    Function that executes a SQL query in Snowflake
    - query (str): string with the query to be executed
    - database (str): name of the database where the query will be executed. It can be an empty 
    - schema (str): name of the schema where the query will be executed
    - fetchResults (Bool): If True, the function will also fetch the results of the query and load them into a dataframe.
    
    Output: If featchResults is True, a dataframe with the results from the query
    '''
    conn = sf.connect(
        user = os.environ['SF_USER'],
        password = os.environ['SF_PASSWORD'],
        account = os.environ['SF_ACCOUNT'],
        role = os.environ['SF_ROLE'],
        warehouse = os.environ['SF_WAREHOUSE'],
        database = database,
        schema = schema
    )

    cursor = conn.cursor()
    cursor.execute(query)
    
    if fetchResults:
        res = cursor.fetchall()
        cols = [desc[0].lower() for desc in cursor.description]
        df = pd.DataFrame(res, columns=cols)
    else:
        df = None  
    
    cursor.close()
    conn.close()
    
    return df    
