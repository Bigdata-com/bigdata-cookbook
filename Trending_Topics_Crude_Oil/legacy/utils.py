from . import keywords_functions
from .label_extractor import extract_label
import pandas as pd
import asyncio
import os
from bigdata import Bigdata
from bigdata.query import Similarity, All, Entity
from bigdata.models.search import DocumentType, SortBy
from bigdata.daterange import RollingDateRange, AbsoluteDateRange
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import collections
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import seaborn as sns
from collections import defaultdict
from .cost_estimation import compute_costs
import configparser
from tqdm.notebook import tqdm
import json
from pqdm.processes import pqdm
from operator import itemgetter
import warnings
warnings.filterwarnings('ignore')
import time

def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def create_similarity_query(sentences, entities = []):
    if not sentences:
        raise ValueError("The list of sentences must not be empty.")
    
    def contains_lists(lst):
        for item in lst:
            if isinstance(item, list):
                return True
        return False
    if contains_lists(sentences):
        final_list = list(set(flatten(sentences)))
    else:
        final_list = sentences

    sentences = [Similarity(sample) for sample in final_list]

    query = All(sentences)
    
    if entities:
        ents = [Entity(ent) for ent in entities]
        query = query & All(ents)
    
    return query

def create_parallel_similarity_query(sentences, entities = []):
    if not sentences:
        raise ValueError("The list of sentences must not be empty.")
    
    def contains_lists(lst):
        for item in lst:
            if isinstance(item, list):
                return True
        return False
    if contains_lists(sentences):
        final_list = list(set(flatten(sentences)))
    else:
        final_list = sentences

    sentences = [Similarity(sample) for sample in final_list]
    
    if entities:
        ents = [Entity(ent) for ent in entities]
        query = [sentence & All(ents) for sentence in sentences]
        
    else:
        query=sentences
    
    return query

## PQDM wrapper for parallel query
def run_similarity_search_parallel_topics_new(query, start_date, end_date, bigdata_credentials, freq='Monthly', document_type='News',  limit_documents = 1000, job_number = 10):   
    
    time_frequencies = {'Yearly':'YS','Monthly':'MS','Weekly':'W', 'Daily':'D'}
    
    documents = {'News':DocumentType.NEWS, 'Transcripts':DocumentType.TRANSCRIPTS, 'Filings':DocumentType.FILINGS}
    doctype = documents[document_type]
    freq_query = time_frequencies[freq]
    
    start_list, end_list = create_dates_lists(start_date, end_date, freq=freq_query)
    
    args = []

    for period in range(1,len(start_list)+1,1): #,desc='Loading and saving datasets for '+'US ML'+' universe'):
        start_date = start_list[period-1]
        end_date = end_list[period-1]
        print(f'{start_date} - {end_date}')
        date_range = AbsoluteDateRange(start_date, end_date)
        for query_sentences in query:
            args.append({'query':query_sentences,'date_range':date_range,'bigdata_credentials':bigdata_credentials, 'document_type':doctype,  'limit_documents':limit_documents})

    sentences_final = pqdm(args, run_query_topics, n_jobs=job_number, argument_type='kwargs')

    sentences = pd.concat(sentences_final).sort_values('timestamp_utc').reset_index(drop=True)
    sentences['date'] = pd.to_datetime(pd.to_datetime(sentences.timestamp_utc).dt.date)
    sentences = sentences.drop_duplicates(subset=['timestamp_utc', 'rp_document_id', 'sentence_id', 'rp_entity_id'])
    # Process the sentences to add masked text
    sentences = process_sentences_masking(sentences, target_entity_mask='Target Company', other_entity_mask='Other Company', masking=True)
    
    return sentences

def run_similarity_search_parallel_parsed(query, start_date, end_date, bigdata_credentials, freq='Monthly', document_type='News',  limit_documents = 1000, job_number = 10):   
    
    time_frequencies = {'Yearly':'YS','Monthly':'MS','Weekly':'W', 'Daily':'D'}
    
    documents = {'News':DocumentType.NEWS, 'Transcripts':DocumentType.TRANSCRIPTS, 'Filings':DocumentType.FILINGS}
    doctype = documents[document_type]
    freq_query = time_frequencies[freq]
    
    start_list, end_list = create_dates_lists(start_date, end_date, freq=freq_query)
    
    interval = 15
    total_intervals = (len(start_list) + interval - 1) // interval
    all_sentences = []

    for i in range(total_intervals):
        # Determine the range for the current interval
        start_idx = i * interval
        end_idx = min((i + 1) * interval, len(start_list))

        # Get the sublist for the current interval
        current_start_list = start_list[start_idx:end_idx]
        current_end_list = end_list[start_idx:end_idx]

        
        for attempt in range(3):
            try:
                args = []
                for period in range(start_idx, end_idx):
                    start_date = start_list[period]
                    end_date = end_list[period]
                    print(f'{start_date} - {end_date}')
                    date_range = AbsoluteDateRange(start_date, end_date)
                    for query_sentences in query:
                        args.append({'query': query_sentences, 'date_range': date_range, 'bigdata_credentials': bigdata_credentials, 'document_type': doctype, 'limit_documents': limit_documents})

                sentences_final = pqdm(args, run_query, n_jobs=job_number, argument_type='kwargs')
                
                for err in sentences_final:
                    if isinstance(err, pd.DataFrame):
                        continue
                    else:
                        print(err)
                
                #print(pd.concat(sentences_final).head(5))
                all_sentences.append(pd.concat(sentences_final))
                #print(all_sentences)
                break
            except Exception as e:
                print(f"Error on attempt {attempt + 1} for batch {start_idx} to {end_idx}: {e}")
                if attempt < 3 - 1:
                    time.sleep(60)  # Wait for 100 seconds before retrying
                else:
                    print(f"Failed after {3} attempts for batch {start_idx} to {end_idx}")

        # Wait for 5 seconds after each pqdm run
        time.sleep(5)

    sentences = pd.concat(all_sentences).sort_values('timestamp_utc').reset_index(drop=True)
    sentences['date'] = pd.to_datetime(pd.to_datetime(sentences.timestamp_utc).dt.date)
    sentences = sentences.drop_duplicates(subset=['timestamp_utc', 'rp_document_id', 'sentence_id', 'rp_entity_id'])
    # Process the sentences to add masked text
    sentences = process_sentences_masking(sentences, target_entity_mask='Target Company', other_entity_mask='Other Company', masking=True)
    
    return sentences

## PQDM wrapper for parallel query
def run_similarity_search_parallel_parsed_short_interval(query, start_date, end_date, bigdata_credentials, freq='Monthly', document_type='News',  limit_documents = 1000, job_number = 10):   
    
    time_frequencies = {'Yearly':'YS','Monthly':'MS','Weekly':'W', 'Daily':'D'}
    
    documents = {'News':DocumentType.NEWS, 'Transcripts':DocumentType.TRANSCRIPTS, 'Filings':DocumentType.FILINGS}
    doctype = documents[document_type]
    freq_query = time_frequencies[freq]
    
    start_list, end_list = create_dates_lists(start_date, end_date, freq=freq_query)
    
    interval = 15
    total_intervals = (len(start_list) + interval - 1) // interval
    all_sentences = []
    
    def query_batch_parallel(start_idx, end_idx):
        args = []
        for period in range(start_idx, end_idx):
            start_date = start_list[period]
            end_date = end_list[period]
            print(f'{start_date} - {end_date}')
            date_range = AbsoluteDateRange(start_date, end_date)
            for query_sentences in query:
                args.append({'query': query_sentences, 'date_range': date_range, 'bigdata_credentials': bigdata_credentials, 'document_type': doctype, 'limit_documents': limit_documents})

        sentences_final = pqdm(args, run_query, n_jobs=job_number, argument_type='kwargs')

        for err in sentences_final:
            if isinstance(err, pd.DataFrame):
                continue
            else:
                print(err)

        #print(pd.concat(sentences_final).head(5))
        all_sentences.append(pd.concat(sentences_final))
        

    for i in range(total_intervals):
        # Determine the range for the current interval
        start_idx = i * interval
        end_idx = min((i + 1) * interval, len(start_list))

        # Get the sublist for the current interval
        current_start_list = start_list[start_idx:end_idx]
        current_end_list = end_list[start_idx:end_idx]

        
        for attempt in range(2):
            try:
                query_batch_parallel(start_idx, end_idx)
                break
            except Exception as e:
                
                print(f"Error on attempt {attempt + 1} for batch {start_idx} to {end_idx}: {e}")
                
                if attempt == 1:
                    time.sleep(60)  # Wait for 30 seconds before retrying
                    mid_idx = (start_idx + end_idx) // 2
                    query_batch_parallel(start_idx, mid_idx)
                    query_batch_parallel(mid_idx, end_idx)
                else:
                    print(f"Failed after {attempt+1} attempts for batch {start_idx} to {end_idx}")

        # Wait for 5 seconds after each pqdm run
        time.sleep(5)

    sentences = pd.concat(all_sentences).sort_values('timestamp_utc').reset_index(drop=True)
    sentences['date'] = pd.to_datetime(pd.to_datetime(sentences.timestamp_utc).dt.date)
    sentences = sentences.drop_duplicates(subset=['timestamp_utc', 'rp_document_id', 'sentence_id', 'rp_entity_id'])
    # Process the sentences to add masked text
    sentences = process_sentences_masking(sentences, target_entity_mask='Target Company', other_entity_mask='Other Company', masking=True)
    
    return sentences

## PQDM wrapper for parallel query
def run_similarity_search_parallel_new(query, start_date, end_date, bigdata_credentials, freq='Monthly', document_type='News',  limit_documents = 1000, job_number = 10):   
    
    time_frequencies = {'Yearly':'YS','Monthly':'MS','Weekly':'W', 'Daily':'D'}
    
    documents = {'News':DocumentType.NEWS, 'Transcripts':DocumentType.TRANSCRIPTS, 'Filings':DocumentType.FILINGS}
    doctype = documents[document_type]
    freq_query = time_frequencies[freq]
    
    start_list, end_list = create_dates_lists(start_date, end_date, freq=freq_query)
    
    args = []

    for period in range(1,len(start_list)+1,1): #,desc='Loading and saving datasets for '+'US ML'+' universe'):
        start_date = start_list[period-1]
        end_date = end_list[period-1]
        print(f'{start_date} - {end_date}')
        date_range = AbsoluteDateRange(start_date, end_date)
        for query_sentences in query:
            args.append({'query':query_sentences,'date_range':date_range,'bigdata_credentials':bigdata_credentials, 'document_type':doctype,  'limit_documents':limit_documents})

    sentences_final = pqdm(args, run_query, n_jobs=job_number, argument_type='kwargs')

    sentences = pd.concat(sentences_final).sort_values('timestamp_utc').reset_index(drop=True)
    sentences['date'] = pd.to_datetime(pd.to_datetime(sentences.timestamp_utc).dt.date)
    sentences = sentences.drop_duplicates(subset=['timestamp_utc', 'rp_document_id', 'sentence_id', 'rp_entity_id'])
    # Process the sentences to add masked text
    sentences = process_sentences_masking(sentences, target_entity_mask='Target Company', other_entity_mask='Other Company', masking=True)
    
    return sentences

def lookup_entities(bigdata_credentials,search_list, condition_attribute, conditions):
    bigdata=bigdata_credentials
    candidate_entities = []
    for obj, ent in bigdata.knowledge_graph.autosuggest(search_list).items():
        for entity in ent:
            if hasattr(entity, condition_attribute):
                if (entity.position == conditions[0]) & (entity.employer in conditions[1]):
                    #print(entity)
                    candidate_entities.append(entity.id)
                    
    return candidate_entities


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


# Define the mask processing function, with numbered entities
def mask_entity_coordinates(input_df, column_masked_text, mask_target, mask_other):
    i = 1
    entity_counter = {}
    input_df['other_entities_map'] = None
    for idx, row in input_df.iterrows():
        # print(idx)
        text = row['text']
        entities = row['entities']
        entities.sort(key=lambda x: x['start'], reverse=True)  # Sort entities by start position in reverse order
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
                masked_text = masked_text[:start] + mask_target + masked_text[end:]
            
            
            elif (entity['key'] != row['rp_entity_id'])&(start not in target_start)&(end not in target_end):
                if entity['key'] not in entity_counter:
                    entity_counter[entity['key']] = i
                    masked_text = masked_text[:start] + f'{mask_other}_{entity_counter[entity["key"]]}' + masked_text[end:]
                    other_entity_map.append((entity_counter[entity["key"]],entity['name']))
                    i+=1
                else:
                    masked_text = masked_text[:start] + f'{mask_other}_{entity_counter[entity["key"]]}' + masked_text[end:]
                    other_entity_map.append((entity_counter[entity["key"]],entity['name']))

        input_df.at[idx, column_masked_text] = masked_text
        input_df.at[idx, 'other_entities_map'] = other_entity_map if other_entity_map else None ##beta!!
    
    return input_df

def clean_text(text):
    text = text.replace('{', '').replace('}', '')
    return text

def clean_text_masked(text):
    text = text.replace('{', '').replace('}', '')
    return text

def process_sentences_masking(sentences: pd.DataFrame, target_entity_mask: str,  other_entity_mask: str, masking: bool = False) -> pd.DataFrame:
    sentences['text'] = sentences['text'].str.replace('{','',regex=False)
    sentences['text'] = sentences['text'].str.replace('}','',regex=False)

    if masking:
        sentencesdf = mask_entity_coordinates(input_df = sentences, 
                                column_masked_text = 'masked_text',
                                mask_target = target_entity_mask,
                                mask_other = other_entity_mask,
                               )

        sentencesdf['masked_text'] = sentencesdf['masked_text'].apply(clean_text_masked)

        sentencesdf = sentencesdf[sentencesdf.masked_text!='to_remove']
    else:
        sentencesdf = sentences

    sentencesdf['text'] = sentencesdf['text'].apply(clean_text)
    sentencesdf = sentencesdf[sentencesdf.text!='to_remove']

    return sentencesdf.reset_index(drop = True)

## Run single query -- in parallel wrapper
def run_query(query,date_range,bigdata_credentials, document_type,  limit_documents):
    bigdata = bigdata_credentials

    search = bigdata.search.new(
    query=query,
    date_range= date_range,
    scope=document_type,
    sortby=SortBy.RELEVANCE)
    # Limit the number of documents to retrieve
    results = search.limit_documents(limit_documents)
    # Collect unique entity keys
    entity_keys = set()
    for result in results:
        for chunk in result.chunks:
            for entity in chunk.entities:
                entity_keys.add(entity.key)

    # Lookup entity details by their keys
    entities_lookup = bigdata.knowledge_graph.get_entities(list(entity_keys))

    # Create a mapping of entity keys to names and types
    entity_key_to_name = {}
    for entity in entities_lookup:
        if hasattr(entity, 'id') and hasattr(entity, 'name') and entity.entity_type == 'COMP':
            entity_key_to_name[entity.id] = entity.name

    # Create lists to store the DataFrame rows
    data_rows = []

    # Filter and store the results with known entity names and coordinates
    for result in results:
        chunk_texts_with_entities = []
        for chunk_index, chunk in enumerate(result.chunks):
            chunk_entities = []
            for entity in chunk.entities:
                entity_name = entity_key_to_name.get(entity.key, 'Unknown')
                if entity_name != 'Unknown':
                    chunk_entities.append({
                        'key': entity.key,
                        'name': entity_name,
                        'start': entity.start,
                        'end': entity.end
                    })
            if chunk_entities:
                chunk_texts_with_entities.append((chunk.text, chunk_entities, chunk.chunk)) ##chunk_index or chunk.chunk

        for chunk_text, chunk_entities, chunk_index in chunk_texts_with_entities:
            for entity in chunk_entities:
                other_entities = [e for e in chunk_entities if e['name'] != entity['name']]
                data_rows.append({
                    'timestamp_utc': result.timestamp,
                    'rp_document_id': result.id,
                    'sentence_id': f"{result.id}-{chunk_index}",
                    'headline': result.headline,
                    'rp_entity_id': entity['key'],
                    'entity_name': entity['name'],
                    'text': chunk_text,
                    'other_entities': ', '.join([e['name'] for e in other_entities]),
                    'entities': chunk_entities
                })

    # Create the DataFrame
    df = pd.DataFrame(data_rows)
    #print(df.head(1))
    # Remove duplicate rows
    #df = df.drop_duplicates(subset=['timestamp_utc', 'rp_document_id', 'sentence_id', 'rp_entity_id'])

    #sentences_final.append(df)

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

def reinstate_duplicates(labelled_df= pd.DataFrame(), duplicated_chunks = pd.DataFrame(), date_column = '', duplicate_set = [], fill_cols = ['label','motivation']):
    
    final_df = pd.concat([labelled_df, duplicated_chunks], axis=0).sort_values(by=duplicate_set+[date_column]).reset_index(drop=True)
    for col in fill_cols:
        final_df[col] = final_df.groupby(duplicate_set)[col].ffill()
    # final_df['label'] = final_df.groupby(duplicate_set)['label'].ffill()
    # final_df['motivation'] = final_df.groupby(duplicate_set)['motivation'].ffill()
    final_df = final_df.sort_values(by=date_column).reset_index(drop=True)
    
    return final_df

async def process_sentences(sentences, sentence_column_2, masked_2, system_prompt, path_training, n_expected_response_tokens, batch_size, model, open_ai_credentials):
    sentence_final = []
    cols_to_keep = list(sentences.columns) + ['label','motivation']

    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            sentences_labels_2 = await extract_label(
                sentences=sentences, #[start_idx:end_idx],
                sentence_column=sentence_column_2,
                masked=masked_2,  # defines whether the name of the entity is hidden (masked) from the LLM.
                system_prompt=system_prompt,
                path_call_hash_location=path_training,
                path_result_hash_location=path_training,
                n_expected_response_tokens=n_expected_response_tokens,
                batch_size=batch_size,
                concurrency=200,  # defines the number of calls made simultaneously
                openai_api_key=open_ai_credentials,
                parameters={
                    'model': model,  # gpt-4-0125-preview
                    'temperature': 0,
                    'response_format': {'type': 'json_object'}
                }
            )
            sentences_labels_2 = sentences_labels_2
            sentence_final.append(sentences_labels_2)
            break  # Break out of retry loop if successful

        except Exception as e:
            print(f"Error on attempt {attempt + 1}") #for batch {start_idx} to {end_idx}: {e}")
            if attempt < retry_attempts - 1:
                await asyncio.sleep(100)  # Wait for 100 seconds before retrying
            else:
                print(f"Failed after {retry_attempts} attempts") # for batch {start_idx} to {end_idx}")

    final = pd.concat(sentence_final)[cols_to_keep]
    final['label'] = final['label'].where(final['motivation'].str.contains('Target Company_', regex=False),  'U')
    final['motivation'] = final.apply(unmask_motivation, axis=1)
    
    return final

# Call the async function using an event loop

def run_prompt(sentences, sentence_column_2, masked_2, system_prompt_ai_classification, path_training, n_expected_response_tokens, batch_size, model, open_ai_credentials):
    df_ai_classif = asyncio.run(
        process_sentences(
            sentences=sentences,
            sentence_column_2=sentence_column_2,
            masked_2=masked_2,
            system_prompt=system_prompt_ai_classification,
            path_training=path_training,
            n_expected_response_tokens=n_expected_response_tokens,
            batch_size=batch_size,
            model = model,
            open_ai_credentials = open_ai_credentials
        )
    )
    df_ai_classif.reset_index(inplace = True, drop = True)
    return df_ai_classif

##plot the distribution of labels
def plot_labels_distribution(df, labels_names_dict):
    
    total_records = len(df)

    # Create the main plot
    plt.figure(figsize=(12, 6))

    # Main plot

    rating_c = df['label'].value_counts(normalize=False)
    rating_c = rating_c.sort_index()

    # Create the side plot for percentages
    plt.subplot(1, 2, 1)
    ax1=sns.barplot(x=rating_c.index, y=rating_c.values, palette='viridis', hue=rating_c.index)
    plt.title('Distribution of Labels (Total Records: {})'.format(total_records))
    plt.xlabel('Labels')
    plt.ylabel('Count')
    ax1.set_xticks(range(len(rating_c.index)))
    ax1.set_xticklabels([labels_names_dict[item.get_text()].title() for item in ax1.get_xticklabels()])


    # Add percentages to the top of the bars
    for index, value in enumerate(rating_c.values):
        plt.text(index, value, f'{value:.0f}', ha='center', va='bottom')


    # Calculate percentages
    rating_percentages = df['label'].value_counts(normalize=True) * 100
    rating_percentages = rating_percentages.sort_index()

    # Create the side plot for percentages
    plt.subplot(1, 2, 2)
    ax2 = sns.barplot(x=rating_percentages.index, y=rating_percentages.values, palette='viridis',hue=rating_percentages.index)
    plt.title('Percentage Distribution of Labels')
    plt.xlabel('Labels')
    plt.ylabel('Percentage')
    ax2.set_xticks(range(len(rating_percentages.index)))
    ax2.set_xticklabels([labels_names_dict[item.get_text()].title() for item in ax2.get_xticklabels()])

    # Add percentages to the top of the bars
    for index, value in enumerate(rating_percentages.values):
        plt.text(index, value, f'{value:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()



# Function to add line breaks
def add_line_breaks(text, max_len=50):
    
    if type(text)==str:
        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 > max_len:
                lines.append(current_line)
                current_line = word
            else:
                if current_line:
                    current_line += " "
                current_line += word
        lines.append(current_line)
        return "<br>".join(lines)
    else:
        return None

def lookup_sector_information(df_filtered, bigdata_credentials):
    bigdata = bigdata_credentials
    
    entity_key_to_sector = {}
    entity_key_to_group = {}
    entity_key_to_sector_consolidated = {}
    entity_key_to_name = {}
    
    # Lookup entity details by their keys
    entities_lookup = bigdata.knowledge_graph.get_entities(df_filtered['rp_entity_id'].unique().tolist())
    
    for entity in entities_lookup:

        if hasattr(entity, 'id') and hasattr(entity, 'name') and entity.entity_type == 'COMP':
            entity_key_to_name[entity.id] = entity.name
            if hasattr(entity, 'sector'):
                entity_key_to_sector[entity.id] = entity.sector
                entity_key_to_sector_consolidated[entity.id] = entity.sector
            else:
                entity_key_to_sector_consolidated[entity.id] = entity.industry_group
            
            if hasattr(entity, 'industry_group'):
                entity_key_to_group[entity.id] = entity.industry_group
        
    # Display the resulting DataFrame
    df_filtered['sector'] = df_filtered.rp_entity_id.map(entity_key_to_sector)
    df_filtered['industry_group'] = df_filtered.rp_entity_id.map(entity_key_to_group)
    df_filtered['sector_consolidated'] = df_filtered.rp_entity_id.map(entity_key_to_sector_consolidated)
    
    if 'entity_name' not in df_filtered.columns:
        df_filtered['entity_name'] = df_filtered.rp_entity_id.map(entity_key_to_name)
    
    return df_filtered

def get_chunk_when_max_coverage(df_filtered):
    
    entity_day_counts = df_filtered.groupby(['rp_entity_id', 'date']).sentence_id.nunique().reset_index(name='counts')

    # Identify the day with the maximum mentions for each entity
    max_entity_day = entity_day_counts.loc[entity_day_counts.groupby('rp_entity_id')['counts'].idxmax()]

    # Merge back with the original dataframe to get the text value
    result = pd.merge(max_entity_day, df_filtered, on=['rp_entity_id', 'date'], how='left').drop(columns='counts')
    text_cols = ['headline','text']
    if 'motivation' in result.columns:
        text_cols.append('motivation')
        #result['motivation'] = result.apply(unmask_motivation, axis=1)
    
    result = result.groupby(['rp_entity_id', 'date'])[text_cols].last().reset_index()
    
    return result

##beta!
def unmask_motivation(row=pd.DataFrame(), motivation_col='motivation'):
    import re
    
    if type(row[motivation_col])==str:
        unmasked_string = re.sub(r'Target Company_\d{1,2}', row['entity_name'], row[motivation_col])
        if 'other_entities_map' in row.index:
            if row['other_entities_map']:
                if isinstance(row['other_entities_map'],str):
                    for key, name in eval(row['other_entities_map']):
                        unmasked_string = unmasked_string.replace(f'Other Company_{key}', str(name))
                elif isinstance(row['other_entities_map'],float):
                    pass
                else:
                    for key, name in row['other_entities_map']:
                        unmasked_string = unmasked_string.replace(f'Other Company_{key}', str(name))
    else:
        unmasked_string = None
    
    return unmasked_string
    
    
def top_companies_sector(df_filtered, bigdata_credentials, sector_column): 
    
    entity_news_count = defaultdict(int)

    df_filtered = lookup_sector_information(df_filtered, bigdata_credentials)
    
    df_filtered = df_filtered.loc[~(df_filtered.industry_group.isin(['Media','Publishing'])|df_filtered.entity_name.eq('Newsfile Corp.')|df_filtered.industry_group.eq('Media Agencies'))]
    
    df_news_count = df_filtered.groupby(['rp_entity_id']).agg({'sentence_id': pd.Series.nunique}).reset_index()

    df_news_count.rename(columns={'sentence_id': 'NewsCount'}, inplace=True)

    chunk_text_df = get_chunk_when_max_coverage(df_filtered)
    df_news_count = df_news_count.merge(df_filtered[['rp_entity_id', 'entity_name', sector_column]], how='left', on='rp_entity_id')
    
    df_news_count = df_news_count.merge(chunk_text_df[['rp_entity_id','headline','text', 'motivation']], on='rp_entity_id', how='left')
    
    # Drop duplicates to ensure each company is counted only once
    df_news_count = df_news_count.drop_duplicates(subset=['rp_entity_id'])

    # Get the top 5 companies by sector
    top_5_by_sector = df_news_count.groupby(sector_column, group_keys=False).apply(lambda x: x.nlargest(5, 'NewsCount')).reset_index(drop=True)

    # Rank sectors by the total number of news mentions
    sector_ranking = (top_5_by_sector.groupby(sector_column, group_keys=False)['NewsCount']
                      .sum()
                      .sort_values(ascending=False)
                      .index)[:10]

    # Determine the number of rows needed for 5 columns
    n_cols = 5
    n_rows = (len(sector_ranking) + n_cols - 1) // n_cols

    # Prepare the layout for subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,  # Adjust the number of columns to accommodate the bar plots
        subplot_titles=[f"{sector} <br> {sector_column.replace('_', ' ').capitalize()}" for sector in sector_ranking],  # Title for each bar plot
        vertical_spacing=0.4,
        horizontal_spacing=0.1,
        specs=[[{"type": "bar"}] * n_cols for _ in range(n_rows)],
    )

    # Flatten the sector ranking for easy iteration
    sector_list = list(sector_ranking)

    for i, sector in enumerate(sector_list):
        row = i // n_cols + 1
        col = (i % n_cols) + 1

        # Plot bar chart for top 5 companies in the sector
        sector_data = top_5_by_sector[top_5_by_sector[sector_column] == sector]
        
        # Create custom hover text with text and motivation, adding dynamic line breaks for better readability
        hover_text = [
            f"<b>Entity:</b> {row.entity_name}<br><br><b>Headline</b>:{add_line_breaks(row.headline)}<br><br><b>Motivation</b>: {add_line_breaks(row.motivation)}<br><br><b>Text</b>: {add_line_breaks(row.text)}"
            for idx, row in sector_data.iterrows()
        ]

        fig.add_trace(
            go.Bar(
                x= sector_data['entity_name'],
                y=sector_data['NewsCount'],
                name=f"{sector} {sector_column.replace('_', ' ').capitalize()}",
                showlegend=False,
                hovertext=hover_text,
                hoverinfo="text",
                marker_color='#3366CC',  # Set all bars to the same color
                hoverlabel=dict(font=dict(size=16))  # Set hover text font size
            ),
            row=row, col=col
        )

        fig.update_xaxes(
            tickfont=dict(size=18),
            gridwidth=0.45,
            #autotickangles=[25, 35, 45, 60],
            tickangle=70, 
            gridcolor='LightGray',
            row=row, col=col
        )

        # Update y-axis for each subplot
        fig.update_yaxes(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='LightGray',
            row=row, col=col
        )

    # Update layout
    fig.update_layout(
        height=550 * n_rows,
        title_text=f"Top Companies by {sector_column.replace('_', ' ').capitalize()}",
    )

    # Show the interactive plot
    fig.show()
    
## Obtain volume exposure to a theme and identify a basket of companies
def identify_basket_of_companies(df, dfs_labels,df_names,bigdata_credentials, sector_column,basket_size, start_date, end_date):
    
    df_filt = df.loc[df.label.isin(dfs_labels)].copy().reset_index(drop=True)
    
    df_filt = df_filt.loc[df_filt.date.ge(start_date)&df_filt.date.le(end_date)]
    
    total_vol = df_filt.groupby('rp_entity_id')['sentence_id'].nunique().rename('total_exposure').reset_index()
    overall_vol = df.groupby('rp_entity_id')['sentence_id'].nunique().rename('total_volume').reset_index()
    
    grouped_df = df_filt.groupby(['rp_entity_id', 'label']).size().unstack(fill_value=0).reset_index()
    grouped_df = lookup_sector_information(grouped_df, bigdata_credentials)
    grouped_df.columns.name = None
    grouped_df = grouped_df.loc[~(grouped_df.industry_group.isin(['Media','Publishing'])|grouped_df.entity_name.eq('Newsfile Corp.')|grouped_df.industry_group.eq('Media Agencies'))]
    
    grouped_df = grouped_df.merge(total_vol, on='rp_entity_id', how='left')
    grouped_df = grouped_df.merge(overall_vol, on='rp_entity_id', how='left')
    
    for label, name in zip(dfs_labels, df_names):
        grouped_df[f'{name}_pct'] = grouped_df[label]*100/grouped_df['total_exposure']
        
        chunk_text_df = get_chunk_when_max_coverage(df_filt.loc[df_filt.label.eq(label)])
        grouped_df = pd.concat([grouped_df.set_index('rp_entity_id'), chunk_text_df.rename(columns={'headline':f'headline_{name}','text':f'text_{name}', 'motivation':f'motivation_{name}'}).set_index(
            'rp_entity_id')[[f'headline_{name}', f'text_{name}',f'motivation_{name}']]],axis=1).reset_index()

    ##select top companies in basket
    top_exposed = grouped_df.nlargest(basket_size, ['total_exposure','total_volume'], keep='first').sort_values(f'total_exposure').reset_index(drop=True)
    top_exposed['start_date'] = start_date
    top_exposed['end_date'] = end_date

    return top_exposed.drop('total_volume',axis=1)

def get_daily_volume_by_label(df, dfs_labels,df_names,bigdata_credentials, sector_column, start_date, end_date):
    
    from functools import reduce

    dfs_by_date = []
    
    for label,name in zip(dfs_labels, df_names):
        
        df = df.sort_values(by=['date'])
        df['motivation'] = df.apply(unmask_motivation, motivation_col=f'motivation', axis=1)
        df.date = pd.to_datetime(df.date)
        df_label = df.loc[df.label.eq(label)].copy().reset_index(drop=True)
        
        vol_by_date = df_label.groupby(['date','rp_entity_id']).agg({'sentence_id':'nunique','headline':'last', 'text':'last', 'motivation':'last'}).reset_index()

        # Get all unique dates and entity names
        all_dates = pd.date_range(start=start_date,end=end_date, freq='D')
        all_entities = df_label['rp_entity_id'].unique()

        # Create a full index of all combinations of dates and entity names
        full_index = pd.MultiIndex.from_product([all_dates, all_entities], names=['date', 'rp_entity_id'])

        # Reindex the original DataFrame to this full index
        vol_by_date_full = vol_by_date.set_index(['date', 'rp_entity_id']).reindex(full_index).reset_index()

        # Fill missing values with zero
        vol_by_date_full['sentence_id'] = vol_by_date_full['sentence_id'].fillna(0)
        vol_by_date_full.date = pd.to_datetime(vol_by_date_full.date)
        # Display the resulting DataFrame
        vol_by_date_full = lookup_sector_information(vol_by_date_full, bigdata_credentials)
        
        vol_by_date_full = vol_by_date_full.loc[~(vol_by_date_full.industry_group.isin(['Media','Publishing'])|vol_by_date_full.entity_name.eq('Newsfile Corp.')|vol_by_date_full.industry_group.eq('Media Agencies'))]
        
        #vol_by_date_full['motivation'] = vol_by_date_full.apply(unmask_motivation, motivation_col=f'motivation', axis=1)
        vol_by_date_full = vol_by_date_full.rename(columns={'sentence_id':name, 'headline':f'headline_{name}','text':f'text_{name}','motivation':f'motivation_{name}'})
        
        dfs_by_date.append(vol_by_date_full[['date', 'entity_name', 'rp_entity_id', sector_column, name,f'headline_{name}', f'text_{name}', f'motivation_{name}']])
        
    if len(dfs_by_date)>1:
        
        final_df = reduce(lambda df1,df2: pd.merge(df1,df2,on=['date', 'entity_name','rp_entity_id', sector_column], how='outer'), dfs_by_date)
        final_df[[f'{name}' for name in df_names]] = final_df[[f'{name}' for name in df_names]].fillna(0)
        final_df['total_exposure'] = final_df[[f'{name}' for name in df_names]].abs().sum(axis=1)
    else:
        final_df = dfs_by_date[0]
        final_df[[f'{name}' for name in df_names]] = final_df[[f'{name}' for name in df_names]].fillna(0)
        final_df['total_exposure'] = final_df[[f'{name}' for name in df_names]].abs().sum(axis=1)
    
    return final_df

def display_basket(df):
    cols = []
    for i in df.columns:
        if len(i) == 1:
            cols.append(i)
    tmp = pd.DataFrame()
    for col in cols:
        tmp = pd.concat([tmp, df.sort_values(by=col, ascending=False).head(3)], axis=0)
    display(tmp.drop_duplicates().reset_index(drop=True))

## Create a stacked bar chart showing relative exposure to the theme, ideally positive/negative exposure
def plot_confidence_bar_chart(basket_df, title_theme, df_names, legend_labels):
    
    df = basket_df.copy() 
    
    # Create the figure
    fig = go.Figure()
    
    # Add positive exposure bars
    # Update layout

    # Create the figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=(f"{title_theme}", "Total Volume"),
        horizontal_spacing=0.15
    )

    fig.update_layout(
        barmode='stack',
        title='Positive and Negative Exposure Percentages by Entity',
        xaxis=dict(range=[0, 100], dtick=10),
        yaxis=dict(title='Entity Name'),
        template='plotly_white',
        showlegend=True,
    )
    
    fig.update_layout(yaxis = dict(
        tickfont = dict(
            size=18-(2*len(basket_df)//10))
    ))
    
    fig.update_yaxes(showticklabels=True, row=1, col=1)
    
    colors = ['green','red','grey','orange']
    
    for i, (name, leg_label, color) in enumerate(zip(df_names, legend_labels, colors[:len(df_names)])): ##automate color generator
        base_val = [0]*len(df['entity_name']) if i == 0 else df[[f'{name}_pct' for name in df_names[:i]]].abs().sum(axis=1)
        text_col = f'text_{name}'
        # Create custom hover text with text and motivation, adding dynamic line breaks for better readability
        hover_text = [
            f"<b>{leg_label}:</b> {round(row[f'{name}_pct'],2)}%<br><br><b>Entity:</b>{row['entity_name']}<br><br><b>Headline</b>: {add_line_breaks(row[f'headline_{name}'])}<br><br><b>Motivation</b>: {add_line_breaks(row[ f'motivation_{name}'])}<br><br><b>Text:</b> {add_line_breaks(row[text_col])}" for index, row in df.iterrows()]
        
        fig.add_trace(go.Bar(
            y=df['entity_name'],
            x=df[f'{name}_pct'],
            name=leg_label,
            orientation='h',
            marker=dict(color=color),
            hovertemplate='%{hovertext}<extra></extra>', #f"<b>{title_theme}:</b> %{x:.2f}%<br><br><b>Entity:</b>%{y}<br><br><b>Text:</b> {add_line_breaks(df[text_col])}<extra></extra>",
            text = df[f'{name}_pct'].apply(lambda x: f'{x:.2f}%'),
            hovertext=hover_text,
            base=base_val,
            textposition='inside',
            insidetextanchor='start',
        ), row=1, col=1)

    # Add vertical line at 50%
    fig.add_shape(
        type='line',
        x0=50,
        x1=50,
        y0=-0.5,
        y1=len(df['entity_name']) - 0.5,
        line=dict(color='black', width=3, dash='dash'), row=1, col=1
    )

    fig.add_trace(go.Bar(
        y=df['entity_name'],
        x=df['total_exposure'],
        name='Total Volume',
        orientation='h',
        marker=dict(color='blue'),
        hovertemplate='%{x}<extra></extra>',
        text=df['total_exposure'].apply(lambda x: f'{x}'),
        textposition='inside',
        textangle=0,
        showlegend=False
    ), row=1, col=2)

    # Update layout
    fig.update_layout(
        height=900,width=1400,
        title_text=f"{title_theme} Watchlist",
    )

    fig.update_yaxes(showticklabels=False, row=1, col=2)

    # Show the plot
    fig.show()
    
## Plot net exposure and chunk text over time
def track_basket_sectors_over_time(df,df_names, companies_list,sector_column, sectors_list = []):
    # Create the figure
    fig = go.Figure()
    import matplotlib.colors as mcolors
    import math
    
    if not sectors_list:
        # Rank sectors by the total number of news mentions
        sectors_list = (df.groupby(sector_column, group_keys=False)['total_exposure']
                      .sum()
                      .sort_values(ascending=False)
                      .index)[:8]

    # Plot lines for each entity within each industry group
    for i, group in enumerate(sectors_list):
        group_data = df[df[sector_column] == group]
        unique_entities = group_data.loc[group_data.rp_entity_id.isin(companies_list)]['entity_name'].unique()

        # Define a colormap with up to 20 distinct colors
        color_palette = plt.get_cmap('tab20', 20).colors
        color_palette = [mcolors.rgb2hex(color) for color in color_palette]
        color_map = dict(zip(unique_entities, color_palette[:len(unique_entities)]))

        fig = go.Figure()

        for entity in unique_entities:
            entity_data = group_data[group_data['entity_name'] == entity].copy()
            entity_data['hovertext'] = entity_data.apply(
                lambda row: f"<b>{row['entity_name']}</b><br><b>Number of {df_names[0].title()} Chunks:</b> {row[f'{df_names[0]}']}<br>"
                            f"<b>Number of {df_names[1].title()} Chunks:</b> {row[f'{df_names[1]}']}<br>"
                            f"<b>Headline:</b> {add_line_breaks(row[f'headline_{df_names[0]}'])}<br>"
                            f"<b>Motivation ({df_names[0].title()}):</b> {add_line_breaks(row[f'motivation_{df_names[0]}'])}<br>"            
                            f"<b>Text ({df_names[0].title()}):</b> {add_line_breaks(row[f'text_{df_names[0]}'])}<br>"
                            f"<b>Headline:</b> {add_line_breaks(row[f'headline_{df_names[1]}'])}<br>"            
                            f"<b>Motivation ({df_names[1].title()}):</b> {add_line_breaks(row[f'motivation_{df_names[1]}'])}<br>"
                            f"<b>Text ({df_names[1].title()}):</b> {add_line_breaks(row[f'text_{df_names[1]}'])}<br>", axis=1)

            # Ensure hovertext is not empty
            entity_data['hovertext'] = entity_data['hovertext'].apply(lambda x: x if x.strip() else "No details available")

            # Positive exposure
            fig.add_trace(go.Scatter(
                x=entity_data['date'],
                y=entity_data['net_exposure'].fillna(0),
                mode='lines',
                name=entity,
                hovertext=entity_data['hovertext'],
                hovertemplate='%{hovertext}<extra></extra>',
                line=dict(color=color_map[entity]),
                legendgroup=entity,
            ))
            
            # Add marker trace for non-zero net_exposure
            non_zero_data = entity_data.loc[entity_data['net_exposure'] != 0]
            if not non_zero_data.empty:
                fig.add_trace(go.Scatter(
                    x=non_zero_data['date'],
                    y=non_zero_data['net_exposure'],
                    mode='markers',
                    name=entity,
                    hovertext=non_zero_data['hovertext'],
                    hovertemplate='%{hovertext}<extra></extra>',
                    marker=dict(color=color_map[entity], size=12),
                    legendgroup=entity,
                    showlegend=False
                ))
            
            # Add marker trace for zero net_exposure with non-null text_positive_exp
            zero_nonnull_data = entity_data.loc[(entity_data['net_exposure'] == 0) & (entity_data[f'text_{df_names[0]}'].notnull())]
            if not zero_nonnull_data.empty:
                fig.add_trace(go.Scatter(
                    x=zero_nonnull_data['date'],
                    y=zero_nonnull_data['net_exposure'],
                    mode='markers',
                    name=entity,
                    hovertext=zero_nonnull_data['hovertext'],
                    hovertemplate='%{hovertext}<extra></extra>',
                    marker=dict(color=color_map[entity], size=12),
                    legendgroup=entity,
                    showlegend=False
                ))

        # Update layout
        fig.update_layout(
            height=500,
            title_text=f"{group}",
            template='plotly_white'
        )

        # Show the plot
        fig.show()

def top_companies_time(df_filtered):
    # Convert timestamp to datetime and extract daily period
    df_filtered['timestamp_utc'] = pd.to_datetime(df_filtered['timestamp_utc'])
    df_filtered['Day'] = df_filtered['timestamp_utc'].dt.to_period('D')

    # Group by day and entity, then count unique documents
    grouped = df_filtered.groupby(['Day', 'entity_name']).agg({'sentence_id': pd.Series.nunique}).reset_index()
    grouped.rename(columns={'sentence_id': 'DocumentCount'}, inplace=True)

    # Ensure all days are represented for each entity
    all_days = pd.date_range(start=grouped['Day'].min().to_timestamp(), end=grouped['Day'].max().to_timestamp(), freq='D').to_period('D')
    all_entities = grouped['entity_name'].unique()
    full_index = pd.MultiIndex.from_product([all_days, all_entities], names=['Day', 'entity_name'])

    # Reindex the DataFrame to include all days and entities, filling missing values with 0
    daily_volume = grouped.set_index(['Day', 'entity_name']).reindex(full_index, fill_value=0).reset_index()

    # Convert Day to datetime for plotting
    daily_volume['Day'] = daily_volume['Day'].dt.to_timestamp()
    daily_volume = daily_volume.sort_values(['Day', 'entity_name'])

    # Identify the top 4 entities by total volume
    top_entities = (daily_volume.groupby('entity_name')['DocumentCount'].sum()
                    .nlargest(4)
                    .index
                    .tolist())

    # Filter the DataFrame to include only the top 4 entities
    top_entities_data = daily_volume[daily_volume['entity_name'].isin(top_entities)]

    # Create a mapping of entity names to texts and motivations
    entity_texts_motivations = df_filtered.groupby(['entity_name', 'Day']).apply(lambda x: x[['headline','text', 'motivation']].to_dict('records')).to_dict()

    # Create subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=top_entities, shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.03, horizontal_spacing=0.1)

    # Plot each entity's daily document volume
    for i, entity_id in enumerate(top_entities):
        entity_data = top_entities_data[top_entities_data['entity_name'] == entity_id]
        row = i // 2 + 1
        col = i % 2 + 1

        # Create hover text for each point
        hover_text = []
        marker_dates = []
        marker_counts = []
        for date, count in zip(entity_data['Day'], entity_data['DocumentCount']):
            if (entity_id, date.to_period('D')) in entity_texts_motivations:
                records = entity_texts_motivations[(entity_id, date.to_period('D'))]
                hover_text.append(
                    "<br><br>".join(
                        [
                            f"<b>Headline</b>: {add_line_breaks(record['headline'])}<br><br><b>Motivation</b>: {add_line_breaks(record['motivation'])}<br><br><b>Text</b>: {add_line_breaks(record['text'])}"
                            for record in records
                        ]
                    )
                )
                marker_dates.append(date)
                marker_counts.append(count)

        # Add line trace
        fig.add_trace(go.Scatter(x=entity_data['Day'], y=entity_data['DocumentCount'], mode='lines', name=entity_id, line=dict(width=3)), row=row, col=col)

        # Add marker trace
        fig.add_trace(go.Scatter(x=marker_dates, y=marker_counts, mode='markers', name=entity_id, hovertext=hover_text, hoverinfo="text", marker=dict(size=10)), row=row, col=col)

    # Update layout
    fig.update_layout(height=2000, width=2000, title_text="Daily Document Volume by Top 4 Entities", showlegend=False)

    # Show the plot
    fig.show()

### NEW FUNCTION
def provider_user_net(df_filtered_ai_cost_reduction,
                      df_filtered_ai_providers,bigdata_cred,
                      dash=False):
    
    # Exclude 'motivation' column during the merge
    cols_to_merge = [
        'timestamp_utc', 'rp_document_id', 'sentence_id', 'rp_entity_id',
        'entity_name', 'text', 'masked_text'
    ]
    merged_df = pd.merge(df_filtered_ai_cost_reduction[cols_to_merge],
                         df_filtered_ai_providers[cols_to_merge],
                         how='outer',
                         indicator=True)

    df_filtered_ai_cost_reduction = lookup_sector_information(df_filtered_ai_cost_reduction, bigdata_cred)
    df_filtered_ai_providers = lookup_sector_information(df_filtered_ai_providers, bigdata_cred)
    
    df_filtered_ai_cost_reduction = df_filtered_ai_cost_reduction.loc[~(df_filtered_ai_cost_reduction.industry_group.isin(['Media','Publishing'])|df_filtered_ai_cost_reduction.entity_name.eq('Newsfile Corp.')|df_filtered_ai_cost_reduction.industry_group.eq('Media Agencies'))]
    
    df_filtered_ai_providers = df_filtered_ai_providers.loc[~(df_filtered_ai_providers.industry_group.isin(['Media','Publishing'])|df_filtered_ai_providers.entity_name.eq('Newsfile Corp.')|df_filtered_ai_providers.industry_group.eq('Media Agencies'))]
    
    # Filter out common rows
    df_filtered_ai_cost_reduction_without_common = df_filtered_ai_cost_reduction.copy(
        deep=True)
    df_filtered_ai_providers_without_common = df_filtered_ai_providers.copy(
        deep=True)

    df_filtered_ai_cost_reduction_without_common[
        'entity_name'] = df_filtered_ai_cost_reduction_without_common[
            'entity_name'] + '_user'
    df_filtered_ai_providers_without_common[
        'entity_name'] = df_filtered_ai_providers_without_common[
            'entity_name'] + '_provider'

    # Concatenate the dataframes
    df_filtered = pd.concat([
        df_filtered_ai_cost_reduction_without_common,
        df_filtered_ai_providers_without_common
    ],
                            ignore_index=True)

    # Create a network graph
    G = nx.Graph()

    # Initialize a dictionary to count co-mentions and store chunk text and motivations
    co_mention_counts = defaultdict(lambda: {
        'count': 0,
        'texts': [],
        'motivations': defaultdict(list),
        'headlines': defaultdict(list)
    })

    # Define a function to truncate text around co-mentions
    def truncate_text(text, entity1, entity2, window=10000):
        text_lower = text.lower()
        idx1 = text_lower.find(entity1.lower())
        idx2 = text_lower.find(entity2.lower())

        if idx1 == -1 or idx2 == -1:
            return text[:window] + "..." if len(text) > window else text

        start = max(0, min(idx1, idx2) - window // 2)
        end = min(len(text), max(idx1, idx2) + len(entity2) + window // 2)
        return text[start:end] + ("..." if end < len(text) else "")

    # Process the DataFrame to identify co-mentions
    for sentence_id, group in df_filtered.groupby('sentence_id'):
        entities = group['entity_name'].unique()
        entity_id_map = dict(zip(group['entity_name'], group['entity_name']))
        if len(entities) > 1:
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    # Ensure only provider-user pairs are considered
                    if ('_provider' in entities[i] and '_user'
                            in entities[j]) or ('_user' in entities[i] and
                                                '_provider' in entities[j]):
                        pair = tuple(sorted([entities[i], entities[j]]))
                        co_mention_counts[pair]['count'] += 1
                        truncated_text = truncate_text(group.iloc[0]['text'],
                                                       entities[i],
                                                       entities[j])
                        co_mention_counts[pair]['texts'].append(truncated_text)
                        for entity in entities:
                            entity_motivations = group[
                                group['entity_name'] ==
                                entity]['motivation'].tolist()
                            entity_headlines = group[
                                group['entity_name'] ==
                                entity]['headline'].tolist()
                            for idx, (motivation, headline) in enumerate(zip(
                                    entity_motivations, entity_headlines)):
                                entity_motivations[idx] = motivation.replace(
                                    f'Target Company_', entity_id_map[entity])
                                # Remove the '_user' and '_provider' tags from the motivation
                                entity_motivations[idx] = entity_motivations[
                                    idx].replace('_user',
                                                 '').replace('_provider', '')
                            co_mention_counts[pair]['motivations'][entity].extend(entity_motivations)
                            co_mention_counts[pair]['headlines'][entity].extend(entity_headlines)

    # Add edges to the graph based on co-mentions
    for pair, data in co_mention_counts.items():
        if data['count'] >= 1:  # Adjust threshold here if needed
            G.add_edge(pair[0],
                       pair[1],
                       weight=data['count'],
                       texts=data['texts'],
                       motivations=data['motivations'],
                       headlines = data['headlines'])

    # Check if the graph is empty
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        print("The graph is empty. Please check the data and conditions.")
    else:
        # Get the positions of the nodes using a spring layout
        pos = nx.spring_layout(G, k=0.1, seed=42, dim=3)

        # Create edge traces
        edge_x = []
        edge_y = []
        edge_z = []

        for edge in G.edges(data=True):
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]

        edge_trace = go.Scatter3d(x=edge_x,
                                  y=edge_y,
                                  z=edge_z,
                                  line=dict(width=2, color='lightgrey'),
                                  hoverinfo='none',
                                  mode='lines')

        # Create node traces
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_annotations = []
        node_color = []

        for node in G.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            clean_node_text = node.replace('_user',
                                           '').replace('_provider', '')
            node_text.append(clean_node_text)
            node_color.append(
                'royalblue' if node.endswith('_user') else 'tomato')
            # Remove duplicate chunks and truncate text
            unique_texts = [
                chunk for edge in G.edges(node, data=True)
                for chunk in edge[2]['texts']
            ]
            unique_texts = list(dict.fromkeys(
                unique_texts))  # Remove duplicates while preserving order
            formatted_texts = [
                f"<b>Text {i+1}:</b><br>" +
                '<br>'.join([text[j:j + 80] for j in range(0, len(text), 80)])
                for i, text in enumerate(unique_texts)
            ]

            # Format motivations
            formatted_motivations = []
            motivations_added = set()
            for i, text in enumerate(unique_texts):
                for edge in G.edges(node, data=True):
                    motivations = edge[2]['motivations'][node]
                    for motivation in motivations:
                        if motivation not in motivations_added:
                            formatted_motivations.append(
                                f"<b>Motivation {len(formatted_motivations)+1}:</b><br>"
                                + '<br>'.join([
                                    motivation[j:j + 80]
                                    for j in range(0, len(motivation), 80)
                                ]))
                            motivations_added.add(motivation)
            
            formatted_headlines = []
            headlines_added = set()
            for i, text in enumerate(unique_texts):
                for edge in G.edges(node, data=True):
                    headlines = edge[2]['headlines'][node]
                    for headline in headlines:
                        if headline not in headlines_added:
                            formatted_headlines.append(
                                f"<b>Headline {len(formatted_headlines)+1}:</b><br>"
                                + '<br>'.join([
                                    headline[j:j + 80]
                                    for j in range(0, len(headline), 80)
                                ]))
                            headlines_added.add(headline)

            node_annotations.append('<br><br>'.join([formatted_headline+'<br><br>'+formatted_motivation+'<br><br>'+formatted_text for formatted_headline, formatted_text, formatted_motivation in zip(formatted_headlines, formatted_texts, formatted_motivations)]))

        node_trace = go.Scatter3d(x=node_x,
                                  y=node_y,
                                  z=node_z,
                                  mode='markers+text',
                                  text=node_text,
                                  hovertext=node_annotations,
                                  hoverinfo='text',
                                  marker=dict(color=node_color,
                                              size=8,
                                              opacity=0.8,
                                              line_width=1))

        # Create the figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=20, r=20,
                            t=40),  # Reduced left and right margins
                annotations=[
                    dict(showarrow=False,
                         xref="paper",
                         yref="paper",
                         x=0.005,
                         y=-0.002)
                ],
                scene=dict(
                    xaxis=dict(showgrid=True,
                               zeroline=True,
                               backgroundcolor='#D3D3D3',
                               gridcolor='rgba(255, 255, 255, 0.1)'),
                    yaxis=dict(showgrid=True,
                               zeroline=True,
                               backgroundcolor='#D3D3D3',
                               gridcolor='rgba(255, 255, 255, 0.1)'),
                    zaxis=dict(showgrid=True,
                               zeroline=True,
                               backgroundcolor='#D3D3D3',
                               gridcolor='rgba(255, 255, 255, 0.1)'),
                ),height=800,  # Increase height
                width=1000   # Increase width
            ))

        if dash:
            return fig
        else:
            fig.show()

### display topic network        
def obtain_company_topic_links(df,labels, topic_blacklist, bigdata):
    if isinstance(df['topics'].iloc[0], str):
        df['topics'] = df['topics'].apply(eval)
        exploded_topics = df.explode('topics')
    
    else:
        exploded_topics = df.explode('topics')
    
    df_filt = exploded_topics.loc[~(exploded_topics.topics.isin(topic_blacklist))].copy()
    
    df_count = df_filt.loc[df_filt.label.isin(['Y','N'])].groupby(['rp_entity_id', 'topics']).agg({
    'sentence_id': pd.Series.nunique,
    'label': lambda x: x.value_counts().idxmax(),
    'other_entities_map':'first'
}).reset_index().rename(columns={'sentence_id': 'count', 'label': 'major_label'})
    
    df_count = lookup_sector_information(df_count, bigdata)
    
    df_count = df_count.merge(get_chunk_when_max_coverage(df.loc[df.label.isin(labels)]).drop('date',axis=1), on='rp_entity_id',how='left')

    #df_count['motivation_unmasked'] = df_count.apply(unmask_motivation, axis=1)
    
    tmp = []
    for label in labels:
        
    # Prepare the node size and color information
        entity_sizes = df.loc[df.label.eq(label)].groupby('rp_entity_id').agg({'sentence_id': pd.Series.nunique,}).reset_index()
        entity_sizes=lookup_sector_information(entity_sizes, bigdata)
        entity_sizes = entity_sizes.merge(get_chunk_when_max_coverage(df.loc[df.label.eq(label)]).drop('date',axis=1), on='rp_entity_id',how='left')
        #entity_sizes['motivation_unmasked'] = entity_sizes.apply(unmask_motivation, axis=1)
        
        tmp.append(entity_sizes)
    
    entity_sizes = pd.concat(tmp, axis=0).reset_index(drop=True)

    topic_sizes = df_count.groupby(['topics']).agg({'entity_name':'nunique','headline':'first','text':'first'}).reset_index()
    
    # Determine color based on majority label
    def determine_color(label):
        return 'green' if label == labels[0] else 'red'

    entity_colors = df_count.groupby('entity_name')['major_label'].apply(lambda x: determine_color(x.mode()[0])).reset_index()
    topic_colors = df_count.groupby('topics')['major_label'].apply(lambda x: determine_color(x.mode()[0])).reset_index()
    
    return df_count, entity_sizes, topic_sizes, entity_colors, topic_colors


def generate_networkx_network(df_count, entity_sizes, topic_sizes, entity_colors, topic_colors):
    # Create NetworkX graph
    G = nx.Graph()

    # Add nodes for rp_entity_ids
    for idx, row in entity_sizes.iterrows():
        motivation_with_breaks = add_line_breaks(row['motivation']).replace('<br>', '\n')
        text_with_breaks = add_line_breaks(row['text']).replace('<br>', '\n')
        headline_with_breaks = add_line_breaks(row['headline']).replace('<br>', '\n') 

        G.add_node(row['entity_name'], label=row['entity_name'].replace(" ", "\n"), value=row['sentence_id'], size=row['sentence_id']*10,
                   headline=row['headline'],
                   text=row['text'],motivation=row['motivation'],
                   title=f"Entity: {row['entity_name']}\n\nNumber of chunks: {row['sentence_id']}\n\nHeadline: {headline_with_breaks}\n\nMotivation: {motivation_with_breaks}\n\nText: {text_with_breaks}",
                   shape='circle', color=entity_colors.loc[entity_colors['entity_name'] == row['entity_name'], 'major_label'].values[0], physics=False)

    # Add nodes for topics
    for idx, row in topic_sizes.loc[topic_sizes.entity_name.gt(2)].iterrows():

        text_with_breaks = add_line_breaks(row['text']).replace('<br>', '\n')

        G.add_node(row['topics'], size=row['entity_name']*4,value=row['entity_name'],
                   headline=row['headline'],
                   text = f"<b>Headline:</b> {row['headline']}<br><br><b>Text:</b> {row['text']}<br><br><b>Number of chunks:</b> {row['entity_name']}",motivation=None,
                   title=f"Topic: {row['topics']}\n\nNumber of chunks: {row['entity_name']}\n\nHeadline: {row['headline']}\n\nText: {text_with_breaks}",
                   shape = 'diamond',color='blue', physics=False) #determine_color(topic_colors.loc[topic_colors['topics'] == row['topics'], 'major_label'].values[0]))

    # Add edges
    for idx, row in df_count.loc[df_count.topics.isin(topic_sizes.loc[topic_sizes.entity_name.gt(2)].topics.unique())].iterrows():
        G.add_edge(row['entity_name'], row['topics']) ##not sure I need text info here

    # Get 3D positions for the nodes
    pos = nx.spring_layout(G, k=0.1, iterations=10, seed=1)
    # Remove nodes without edges
    nodes_to_remove = [node for node, degree in G.degree() if degree == 0]
    G.remove_nodes_from(nodes_to_remove)
    
    return G

## Run single query -- in parallel wrapper
def run_query_topics(query,date_range,bigdata_credentials, document_type,  limit_documents):
    bigdata = bigdata_credentials
    search = bigdata.search.new(
        query=query,
        date_range= date_range,
        scope=document_type,
        sortby=SortBy.RELEVANCE,
    )

    # Limit the number of documents to retrieve
    results = search.limit_documents(limit_documents)

    # Collect unique entity keys
    entity_keys = set()
    for result in results:
        for chunk in result.chunks:
            for entity in chunk.entities:
                entity_keys.add(entity.key)

    # Lookup entity details by their keys
    entities_lookup = bigdata.knowledge_graph.get_entities(list(entity_keys))

    # Create a mapping of entity keys to names and types
    entity_key_to_name = {}
    for entity in entities_lookup:
        if hasattr(entity, 'id') and hasattr(entity, 'name') and entity.entity_type == 'COMP':
            entity_key_to_name[entity.id] = entity.name

    # Create a mapping of additional topics and concepts, matching entity keys to names and types
    topic_key_to_name = {}
    for entity in entities_lookup:
        if hasattr(entity, 'id') and entity.entity_type != 'COMP': #in ['TOPC', 'ECON','FINC','PLT','TECH','SOCI','LAWS','CURR','PEOP','PLCE','ORGA','SUST','CMDT', 'BUSI']:
            topic_key_to_name[entity.id] = entity.name
    # Create lists to store the DataFrame rows
    data_rows = []

    # Filter and store the results with known entity names and coordinates
    for result in results:
        chunk_texts_with_entities = []
        for chunk_index, chunk in enumerate(result.chunks):
            chunk_entities = []
            topic_entities = []
            for entity in chunk.entities:
                entity_name = entity_key_to_name.get(entity.key, 'Unknown')
                if entity_name != 'Unknown':
                    chunk_entities.append({
                        'key': entity.key,
                        'name': entity_name,
                        'start': entity.start,
                        'end': entity.end
                    })
                topic_name = topic_key_to_name.get(entity.key, 'Unknown')
                if topic_name != 'Unknown':
                    topic_entities.append({
                        'key': entity.key,
                        'name': topic_name,
                        'start': entity.start,
                        'end': entity.end
                    })
                else:
                    topic_entities.append(None)
            if chunk_entities:
                chunk_texts_with_entities.append((chunk.text, chunk_entities, chunk.chunk, topic_entities, chunk.section_metadata, chunk.speaker))

        for chunk_text, chunk_entities, chunk_index, chunk_topics, chunk_section, chunk_speaker in chunk_texts_with_entities:
            for entity in chunk_entities:
                other_entities = [e for e in chunk_entities if e['name'] != entity['name']]
                data_rows.append({
                    'timestamp_utc': result.timestamp,
                    'rp_document_id': result.id,
                    'sentence_id': f"{result.id}-{chunk_index}",
                    'headline': result.headline,
                    'rp_entity_id': entity['key'],
                    'entity_name': entity['name'],
                    'text': chunk_text,
                    'other_entities': ', '.join([e['name'] for e in other_entities]),
                    'entities': sorted(chunk_entities, key=itemgetter('start'), reverse=True),
                    'reporting_entity': result.reporting_entities,
                    'section': chunk_section,
                    'speaker': chunk_speaker,
                    'topics': list(set([e['name'] for e in chunk_topics if e])),
                })

    # Create the DataFrame
    df = pd.DataFrame(data_rows)

    # Remove duplicate rows
    df = df.drop_duplicates(subset=['timestamp_utc', 'rp_document_id', 'sentence_id', 'rp_entity_id'])

    #sentences_final.append(df)

    return df

## PQDM wrapper for parallel query
def run_similarity_search_parallel_topics(query, start_date, end_date, bigdata_credentials, freq='Monthly', document_type='News',  limit_documents = 1000, job_number = 10):   
    
    time_frequencies = {'Yearly':'YS','Monthly':'MS','Weekly':'W', 'Daily':'D'}
    
    documents = {'News':DocumentType.NEWS, 'Transcripts':DocumentType.TRANSCRIPTS, 'Filings':DocumentType.FILINGS}
    doctype = documents[document_type]
    freq_query = time_frequencies[freq]
    
    start_list, end_list = create_dates_lists(start_date, end_date, freq=freq_query)
    
    args = []

    for period in range(1,len(start_list)+1,1): #,desc='Loading and saving datasets for '+'US ML'+' universe'):
        start_date = start_list[period-1]
        end_date = end_list[period-1]
        print(f'{start_date} - {end_date}')
        date_range = AbsoluteDateRange(start_date, end_date)
        args.append({'query':query,'date_range':date_range,'bigdata_credentials':bigdata_credentials, 'document_type':doctype,  'limit_documents':limit_documents})

    sentences_final = pqdm(args, run_query_topics, n_jobs=job_number, argument_type='kwargs')
    
    sentences = pd.concat(sentences_final).sort_values('timestamp_utc').reset_index(drop=True)
    sentences['date'] = pd.to_datetime(pd.to_datetime(sentences.timestamp_utc).dt.date)
    
    # Process the sentences to add masked text
    sentences = process_sentences_masking(sentences, target_entity_mask='Target Company', other_entity_mask='Other Company', masking=True)
    
    return sentences

def generate_plotly_network_chart(G, legend = []):

    # Get 3D positions for the nodes
    pos = nx.spring_layout(G, dim=2, k=0.1, iterations=10, seed=1)

    # Create Plotly traces for nodes and edges
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1])
        edge_y.extend([y0, y1])


    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color='rgb(211, 211, 211)'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_z = []
    node_text = []
    node_color = []
    node_size = []
    node_annotations = []
    node_shape = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_color.append(G.nodes[node]['color'])
        node_size.append(G.nodes[node]['size'])  # Adjust size for visibility,
        node_shape.append(G.nodes[node]['shape'])
        # Remove duplicate chunks and truncate text
        # for edge in G.edges(node, data=True):
        #     print(edge)
        #     print(edge[2]['text'])
    #     unique_texts = [f"<b>Text:</b> {add_line_breaks(node['text'])} <br>"]
    #     unique_texts = list(dict.fromkeys(unique_texts))  # Remove duplicates while preserving order

    #     # Format motivations
    #     formatted_motivations = []
    #     motivations_added = set()
    #     for i, text in enumerate(unique_texts):
    #         for edge in G.edges(node, data=True):
    #             motivations = edge[2]['motivation']
    #             formatted_motivations.append(f"<b>Motivation:</b> {add_line_breaks(motivations)}<br>")
    #             motivations_added.add(motivations)

        if G.nodes[node]['motivation']:
            node_annotations.append(f"<b>Entity:</b> {node} <br><br><b>Headline:</b> {G.nodes[node]['headline']} <br><br><b>Motivation:</b> {add_line_breaks(G.nodes[node]['motivation'])}" + '<br><br>' + f"<b>Text:</b> {add_line_breaks(G.nodes[node]['text'])}")
        else:
            node_annotations.append(f"<b>Topic:</b> {node} <br><br>{add_line_breaks(G.nodes[node]['text'])}" + '<br><br>')

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        text=[f'<b>{x.replace(" ", "<br>")}</b>' for x in node_text],
        mode='markers+text',
        hovertext=node_annotations,
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line=dict(width=0, color='gray'),
            symbol=node_shape,
        )
    )
    color_scatters = []
    for color, name in zip(['green','red', 'blue'],legend):
        tmp = go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            size=100,
            color=color,
            symbol='circle'
        ),
        name=name, showlegend=True)

        color_scatters.append(tmp)

    legend_annotations = [] #dict(showarrow=False, x=0.1, y=1, text='<b>Legend</b>', xanchor='left', xref='paper', yref='paper')
    for i, (color, symbol, name) in enumerate(zip(['green','red', 'blue'],['&#9679;', '&#9679;','&#9670;'], legend)):
        tmp = dict(showarrow=False, x=0.1, y=1-(i+1)*0.02, text=f'<span style="font-size:30px; color:{color};">{symbol}</span>&nbsp;&nbsp;&nbsp;<span style="font-size:18px;">{name}</span>', xanchor='left', xref='paper', yref='paper')
        legend_annotations.append(tmp)

    # Create the Plotly figure
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(template='simple_white',
                    titlefont_size=16,
                    showlegend=False,
                    legend=dict(
                            itemsizing='constant'
                        ),
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=legend_annotations,
                    scene=dict(
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False),
                        zaxis=dict(showgrid=False, zeroline=False),
                    ),
                    height=1400,  # Increase height
                    width=1600   # Increase width
                )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.show()

def generate_pyvis_chart(G, legend):
    # Get 3D positions for the nodes
    pos = nx.spring_layout(G, k=0.1, iterations=10, seed=2)
    # Remove nodes without edges
    nodes_to_remove = [node for node, degree in G.degree() if degree == 0]
    G.remove_nodes_from(nodes_to_remove)

    from pyvis.network import Network
    net = Network('1000px', '1600px',notebook=True, cdn_resources='in_line', neighborhood_highlight=True)
    net.from_nx(G)
    # # Customize hover colors
    # for node in net.nodes:
    #     node['color'] = {
    #         'background': node['color'],
    #         'border': node['color'],
    #         'highlight': {
    #             'background': node['color'],  # Color when node is hovered
    #             'border': node['color']          # Border color when node is hovered
    #         }
    #     }

    # Add Legend Nodes
    step = 130
    x = -600
    y = -600
    num_legend_nodes = 3

    # Function to convert color name to RGBA with transparency
    def to_rgba(color, alpha=0.5):
        colors = {
            'red': 'rgba(255, 0, 0, {})'.format(alpha),
            'green': 'rgba(0, 100, 0, {})'.format(alpha),
            'blue': 'rgba(0, 0, 255, {})'.format(alpha)
        }
        return colors.get(color, color)  # Default to given color if not in dict


    legend_nodes = []
    for legend_node, topic, color, shape, size in zip(range(num_legend_nodes), ['Company Positively Affected', 'Company Negatively Affected', 'Topic'],['green','red', 'blue',], ['circle','circle','diamond'], [5,5,45]):
        legend_nodes.append(
            (
                net.num_nodes() + legend_node, 
                {
                    'id': topic, 
                    'label': str(topic),
                    'size': size, 
                    'fixed': True, # So that we can move the legend nodes around to arrange them better
                    'physics': False, 
                    'x': x, 
                    'y': f'{y + legend_node*step}px',
                    'shape': shape, 
                    'widthConstraint': 90, 
                    'font': {'size': 18},
                    'color': to_rgba(color, 0.6)}) #{'background': color,'border': color,'highlight': {'background': color,'border': color}}})
        )

    for node in legend_nodes:
        net.add_node(node[0], **node[1])
    net.barnes_hut(overlap=1, damping=0,spring_strength=0.6, gravity=-2000,
                  spring_length=250)
    net.toggle_physics(False)
    net.set_options('{"layout":{"randomSeed":1}}')

    return net.show('interactive_network.html')

def infer_sentiment_from_trained_model(df, model_name, task = 'sentiment-analysis', text_col = 'augmented_text'):
    
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    tokenizer = AutoTokenizer.from_pretrained(str(model_name))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_name))
    
        # Create a sentiment analysis pipeline
    model_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, top_k=3, max_length=512, truncation=True)

    df_filt, df_duplicates = drop_duplicates(df, 'timestamp_utc',['rp_entity_id',text_col])
    
    df_sent = df_filt[text_col].apply(extract_sentiment_from_pipeline, args=(model_pipeline,)).apply(pd.Series)
    df_filt = df_filt.join(df_sent)
    
    df_sentiment = reinstate_duplicates(df_filt, df_duplicates, 'timestamp_utc',['rp_entity_id',text_col], [df_sent.columns])
    df_sentiment['label_sent'] = df_sentiment[df_sent.columns].idxmax(axis=1)
    return df_sentiment

def extract_sentiment_from_pipeline(text, model_pipeline):
    result = model_pipeline(text)
    sentiment_dict = {element['label'].lower():element['score'] for element in result[0]}
    sentiment_dict['sentiment'] = sentiment_dict['positive']-sentiment_dict['negative']
    #sentiment_dict['predicted_label'] = result[0][0]['label']
    return sentiment_dict

PATTERNS_REPLACEMENTS = [
    (r'\n', ' '),
    (r'>[A-Z]+', ''),
    (r' -\d+-$', ''),
    (r'\d\d:\d\d EDT\s*', ''),
    (r'-- \w+\.?\w+\.?\w*$', ''),
    ]

def replace(text: pd.Series, pattern: str, replacement: str):
    return text.str.replace(pattern, replacement, regex=True)


def identify_patterns(text: pd.Series, patterns: list[str]) -> pd.DataFrame:
    pattern_freq = []
    for pattern in patterns:
        pattern_freq.append(
            100*text.str.contains(pattern, regex=True).value_counts(normalize=True).rename(pattern))

    pattern_freq.append(
        100*text.str.contains('|'.join(patterns), regex=True).value_counts(normalize=True).rename('any'))
    pattern_freq = pd.concat(pattern_freq, axis=1)
    pattern_freq.index.name = 'Detected'
    pattern_freq.columns.name = 'Pattern'
    return pattern_freq

def get_chunk_when_max_and_min_sentiment(df_filtered):
    # Group by entity and calculate the idxmax and idxmin for sentiment
    max_idx = df_filtered.groupby('rp_entity_id')['sentiment'].idxmax()
    min_idx = df_filtered.groupby('rp_entity_id')['sentiment'].idxmin()

    # Extract the rows corresponding to max and min sentiment
    max_sentiment_rows = df_filtered.loc[max_idx]
    min_sentiment_rows = df_filtered.loc[min_idx]

    # Rename columns to distinguish between max and min sentiment rows
    max_sentiment_rows = max_sentiment_rows.rename(columns={
        'text': 'max_sentiment_text', 
        'sentiment': 'max_sentiment', 
        'motivation': 'max_sentiment_motivation',
        'headline': 'max_sentiment_headline'
    })
    
    min_sentiment_rows = min_sentiment_rows.rename(columns={
        'text': 'min_sentiment_text', 
        'sentiment': 'min_sentiment', 
        'motivation': 'min_sentiment_motivation',
        'headline': 'min_sentiment_headline'
    })

    # Merge the max and min sentiment rows back together
    result = pd.merge(max_sentiment_rows, min_sentiment_rows, on='rp_entity_id', suffixes=('_max', '_min'))

    # Select relevant columns and reset index
    relevant_columns = [
        'rp_entity_id', 
        'max_sentiment_headline', 
        'max_sentiment_text', 
        'max_sentiment', 
        'max_sentiment_motivation', 
        'min_sentiment_headline', 
        'min_sentiment_text', 
        'min_sentiment', 
        'min_sentiment_motivation'
    ]
    result = result[relevant_columns].reset_index(drop=True)

    return result

def display_sentiment_df(df, top_bottom):
    return pd.concat([df.sort_values(by='average_sentiment', ascending=False).head(top_bottom), df.sort_values(by='average_sentiment', ascending=False).tail(top_bottom)], axis=0).reset_index(drop=True)

def daily_sentiment_dataframe(df,bigdata_credentials, start_date, end_date):
    
    df = df.sort_values(by=['timestamp_utc'])
    df.date = pd.to_datetime(df.date)
    # Calculate daily average sentiment
    daily_sent_mean = df.groupby(['date', 'rp_entity_id']).agg({
        'sentiment': 'mean',
        'headline': lambda x:list(x),
        'text': lambda x: list(x),
        'motivation': lambda x:list(x)
    }).reset_index()

    # Collect individual sentiments as a separate column
    individual_sentiments = df.groupby(['date', 'rp_entity_id']).agg({
        'sentiment': lambda x: list(round(x,2))
    }).reset_index()

    # Merge the mean sentiment and individual sentiments
    daily_sent = pd.merge(daily_sent_mean, individual_sentiments, on=['date', 'rp_entity_id'], suffixes=('', '_all'))

    # Get all unique dates and entity names
    all_dates = pd.date_range(start=start_date,end=end_date, freq='D')
    all_entities = df['rp_entity_id'].unique()

    # Create a full index of all combinations of dates and entity names
    full_index = pd.MultiIndex.from_product([all_dates, all_entities], names=['date', 'rp_entity_id'])

    # Reindex the original DataFrame to this full index
    sent_by_date_full = daily_sent.set_index(['date', 'rp_entity_id']).reindex(full_index).reset_index()

    # Fill missing values with zero
    sent_by_date_full['sentiment'] = sent_by_date_full['sentiment'].fillna(0)
    sent_by_date_full.date = pd.to_datetime(sent_by_date_full.date)
    # Display the resulting DataFrame
    sent_by_date_full = lookup_sector_information(sent_by_date_full, bigdata_credentials)
    
    sent_by_date_full = sent_by_date_full.loc[~(sent_by_date_full.industry_group.isin(['Media','Publishing'])|sent_by_date_full.entity_name.eq('Newsfile Corp.')|sent_by_date_full.entity_name.eq('Data Centre World Ltd.')|sent_by_date_full.industry_group.eq('Media Agencies'))]
    
    return sent_by_date_full
    
def plot_sentiment_by_sector(df, companies_list,sector_column, sectors_list = []):
    # Create the figure
    fig = go.Figure()
    import matplotlib.colors as mcolors
    import math
    
    # if not sectors_list:
    #     # Rank sectors by the total number of news mentions
    #     sectors_list = (df.groupby(sector_column, group_keys=False)['total_exposure']
    #                   .sum()
    #                   .sort_values(ascending=False)
    #                   .index)[:8]

    # Plot lines for each entity within each industry group
    for i, group in enumerate(sectors_list):
        group_data = df[df[sector_column] == group]
        unique_entities = group_data.loc[group_data.rp_entity_id.isin(companies_list)]['entity_name'].unique()

        # Define a colormap with up to 20 distinct colors
        color_palette = plt.get_cmap('tab20', 20).colors
        color_palette = [mcolors.rgb2hex(color) for color in color_palette]
        color_map = dict(zip(unique_entities, color_palette[:len(unique_entities)]))

        fig = go.Figure()

        for entity in unique_entities:
            entity_data = group_data[group_data['entity_name'] == entity].copy()
            
            # Function to create hovertext
            def create_hovertext(row):
                details = []
                if isinstance(row['headline'], list):
                    for i in range(len(row['headline'])):
                        details.append(
                f"<b>Headline ({i+1}):</b> {add_line_breaks(row['headline'][i])}<br>"
                f"<b>Motivation ({i+1}):</b> {add_line_breaks(row['motivation'][i])}<br>"
                f"<b>Sentiment ({i+1}):</b> {row['sentiment_all'][i]}<br>"
                f"<b>Text ({i+1}):</b> {add_line_breaks(row['text'][i])}<br><br>"
            )
                    details_str = ''.join(details)
    
                    return (f"<b>Date:</b> {row['date'].date()}<br><b>Entity:</b> {row['entity_name']}<br><b>Average Sentiment:</b> {round(row['sentiment'], 2)}<br><br>"
                            f"{details_str}")
                else:
                    return "No details available"


            entity_data['hovertext'] = entity_data.apply(create_hovertext, axis=1)
            # Ensure hovertext is not empty
            entity_data['hovertext'] = entity_data['hovertext'].apply(lambda x: x if x.strip() else "No details available")

            # Positive exposure
            fig.add_trace(go.Scatter(
                x=entity_data['date'],
                y=entity_data['sentiment'].fillna(0),
                mode='lines',
                name=entity,
                hovertext=entity_data['hovertext'],
                hovertemplate='%{hovertext}<extra></extra>',
                line=dict(color=color_map[entity]),
                legendgroup=entity,
            ))
            
            # Add marker trace for non-zero net_exposure
            non_zero_data = entity_data.loc[entity_data['sentiment'] != 0]
            if not non_zero_data.empty:
                fig.add_trace(go.Scatter(
                    x=non_zero_data['date'],
                    y=non_zero_data['sentiment'],
                    mode='markers',
                    name=entity,
                    hovertext=non_zero_data['hovertext'],
                    hovertemplate='%{hovertext}<extra></extra>',
                    marker=dict(color=color_map[entity], size=12),
                    legendgroup=entity,
                    showlegend=False
                ))
            
            # Add marker trace for zero net_exposure with non-null text_positive_exp
            zero_nonnull_data = entity_data.loc[(entity_data['sentiment'] == 0) & (entity_data[f'text'].notnull())]
            if not zero_nonnull_data.empty:
                fig.add_trace(go.Scatter(
                    x=zero_nonnull_data['date'],
                    y=zero_nonnull_data['sentiment'],
                    mode='markers',
                    name=entity,
                    hovertext=zero_nonnull_data['hovertext'],
                    hovertemplate='%{hovertext}<extra></extra>',
                    marker=dict(color=color_map[entity], size=12),
                    legendgroup=entity,
                    showlegend=False
                ))

        # Update layout
        fig.update_layout(
            height=500,
            title_text=f"{group}",
            template='plotly_white'
        )

        # Show the plot
        fig.show()
        
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_sentiment_bar_chart(df, bigdata_credentials, start_date, end_date, title, top_bottom = 5):
    # Calculate average sentiment for each entity
    
    avg_sent = average_sentiment_df(df, bigdata_credentials, start_date, end_date)
    
    all_scores = df.groupby('rp_entity_id').agg({
        'sentiment': lambda x: list(round(x,2))
    }).reset_index().rename(columns={'sentiment':'all_scores'})
    
    avg_sent = avg_sent.merge(all_scores, on='rp_entity_id', how='left')
    
    if top_bottom:
        avg_sent = pd.concat([avg_sent.sort_values(by='average_sentiment', ascending=False).head(top_bottom), avg_sent.sort_values(by='average_sentiment', ascending=False).tail(top_bottom)], axis=0).reset_index(drop=True)
    else:
        avg_sent = avg_sent.copy()
        
    fig = go.Figure()
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        horizontal_spacing=0.15,
        subplot_titles=("Average Sentiment", "Sentiment Distribution")
    )

    # Define colors and orientations
    avg_sent['color'] = avg_sent['average_sentiment'].apply(lambda x: 'green' if x > 0 else 'red')
    avg_sent['direction'] = avg_sent['average_sentiment'].apply(lambda x: x if x > 0 else -x)
    avg_sent = avg_sent.sort_values(by=['average_sentiment'], ascending=[True])
    
    # Define colors and orientation based on sentiment direction
    avg_sent['positive_sentiment'] = avg_sent['average_sentiment'].apply(lambda x: x if x >= 0 else 0)
    avg_sent['negative_sentiment'] = avg_sent['average_sentiment'].apply(lambda x: x if x < 0 else 0)
    
    # Calculate the distribution of positive and negative sentiments
    avg_sent['positive_count'] = avg_sent['all_scores'].apply(lambda x: sum([1 for score in x if score > 0]))
    avg_sent['negative_count'] = avg_sent['all_scores'].apply(lambda x: sum([1 for score in x if score < 0]))
    avg_sent['neutral_count'] = avg_sent['all_scores'].apply(lambda x: sum([1 for score in x if score == 0]))
    avg_sent['total_count'] = avg_sent['positive_count'] + avg_sent['negative_count']+ avg_sent['neutral_count']

    avg_sent['positive_ratio'] = avg_sent['positive_count'] / avg_sent['total_count']
    avg_sent['negative_ratio'] = avg_sent['negative_count'] / avg_sent['total_count']
    avg_sent['neutral_ratio'] = avg_sent['neutral_count'] / avg_sent['total_count']

# Prepare hover text
    avg_sent['hover_text'] = (
        "<b>Entity Name:</b> " + avg_sent['entity_name'] + "<br>" +
        "<b>Average Sentiment:</b> " + avg_sent['average_sentiment'].round(2).astype(str) + "<br>" +
        "<b>Max Sentiment:</b> " + avg_sent['max_sentiment'].round(2).astype(str) + "<br>" +
        "<b>Max Sentiment Headline:</b> " + avg_sent['max_sentiment_headline'].apply(lambda x: add_line_breaks(x)) + "<br>" +
        "<b>Max Sentiment Text:</b> " +avg_sent['max_sentiment_text'].apply(lambda x: add_line_breaks(x)) + "<br><br>" +
        "<b>Min Sentiment:</b> " + avg_sent['min_sentiment'].round(2).astype(str) + "<br>" +
        "<b>Min Sentiment Headline:</b> " + avg_sent['min_sentiment_headline'].apply(lambda x: add_line_breaks(x)) + "<br>" +
        "<b>Min Sentiment Text:</b> " + avg_sent['min_sentiment_text'].apply(lambda x: add_line_breaks(x))
    )
    
    if (avg_sent['positive_sentiment']>0).sum()>0:
        # Add positive sentiment bars
        fig.add_trace(go.Bar(
            y=avg_sent['entity_name'],
            x=avg_sent['positive_sentiment'],
            name='Positive Sentiment',
            orientation='h',
            marker=dict(color='green'),
            hovertemplate=avg_sent['hover_text'] + "<extra></extra>",
            text=avg_sent['positive_sentiment'].round(2).astype(str),
            textposition='inside',
            legendgroup='Positive Sentiment',  # Separate legend entry for positive sentiment
            showlegend=True
        ), row=1, col=1)

    # Add negative sentiment bars
    if (avg_sent['negative_sentiment']<0).sum()>0:
        fig.add_trace(go.Bar(
            y=avg_sent['entity_name'],
            x=avg_sent['negative_sentiment'],
            name='Negative Sentiment',
            orientation='h',
            marker=dict(color='red'),
            hovertemplate=avg_sent['hover_text'] + "<extra></extra>",
            text=avg_sent['negative_sentiment'].round(2).astype(str),
            textposition='inside',
            legendgroup='Negative Sentiment',  # Separate legend entry for negative sentiment
            showlegend=True
        ), row=1, col=1)
        
    # Plot the relative distribution of positive and negative sentiment in the second subplot
    for i, row in avg_sent.iterrows():
        fig.add_trace(go.Bar(
            y=[row['entity_name']],
            x=[row['positive_ratio']],
            name='Positive Ratio',
            orientation='h',
            marker=dict(color='green'),
            showlegend=False,
            base=0,
            hoverinfo = 'text',
            hovertext = f"<b>Entity Name:</b> {row['entity_name']}<br>"
                    f"<b>Positive Ratio:</b> {row['positive_ratio']:.2%}<br>",
            text = f"{row['positive_ratio']:.2%}",
            legendgroup='Positive Sentiment'
        ), row=1, col=2)
        
        fig.add_trace(go.Bar(
            y=[row['entity_name']],
            x=[row['neutral_ratio']],
            name='Neutral Ratio',
            orientation='h',
            marker=dict(color='grey'),
            base=row['positive_ratio'],
            showlegend=False,
            hoverinfo = 'text',
            hovertext = f"<b>Entity Name:</b> {row['entity_name']}<br>"
                    f"<b>Neutral Ratio:</b> {row['neutral_ratio']:.2%}<br>",
            text = f"{row['neutral_ratio']:.2%}",
            legendgroup='Neutral Sentiment'
        ), row=1, col=2)

        fig.add_trace(go.Bar(
            y=[row['entity_name']],
            x=[row['negative_ratio']],
            name='Negative Ratio',
            orientation='h',
            marker=dict(color='red'),
            base = row['positive_ratio']+row['neutral_ratio'],
            showlegend=False,
            hoverinfo = 'text',
            hovertext = f"<b>Entity Name:</b> {row['entity_name']}<br>"
                    f"<b>Negative Ratio:</b> {row['negative_ratio']:.2%}<br>",
            text = f"{row['negative_ratio']:.2%}",
            legendgroup='Negative Sentiment'
        ), row=1, col=2)
        
    # Adjust layout for the second subplot
    fig.update_xaxes(title_text="Distribution", row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)

    # Update layout to center bars around zero
    fig.update_layout(
        title=title,
        xaxis=dict(
            title='Sentiment',
            range=[-1, 1] if (avg_sent['negative_sentiment']<0).sum()>0 else [0,1],  # Adjust the range as needed based on your data
            zeroline=True,
            zerolinecolor='black'
        ),
        yaxis=dict(title='Entity'),
        barmode='overlay',  # Overlay bars for better visualization
        template='plotly_white'
    )
                               # Update layout
    fig.update_layout(
        height=900,width=1400
    )

    # Show the plot
    fig.show()

    
def average_sentiment_df(df_sentiment, bigdata_credentials, start_date, end_date):
    
    df_sentiment = df_sentiment.loc[df_sentiment.date.ge(start_date)&df_sentiment.date.le(end_date)]
    
    avg_sentiment = df_sentiment.groupby('rp_entity_id').agg(
    average_sentiment=('sentiment', 'mean'),
    # max_sentiment=('sentiment', 'max'),
    # min_sentiment=('sentiment', 'min')
    ).reset_index()

    avg_sentiment = lookup_sector_information(avg_sentiment, bigdata_credentials)
    avg_sentiment = avg_sentiment.loc[~(avg_sentiment.industry_group.isin(['Media','Publishing'])|avg_sentiment.entity_name.eq('Newsfile Corp.')|avg_sentiment.entity_name.eq('Data Centre World Ltd.')|avg_sentiment.industry_group.eq('Media Agencies'))]
    sent_text = get_chunk_when_max_and_min_sentiment(df_sentiment)
    avg_sentiment = avg_sentiment.merge(sent_text, on=['rp_entity_id'], how='left')
    
    avg_sentiment['start_date'] = start_date
    avg_sentiment['end_date'] = end_date
    
    return avg_sentiment.sort_values('average_sentiment', ascending=False)

def plot_top_bottom_sentiment(df, bigdata_credentials, top_bottom, title, sentiment_col = 'sentiment'):
    
    df = lookup_sector_information(df, bigdata_credentials)
    df = df.loc[~(df.industry_group.isin(['Media','Publishing'])|df.entity_name.eq('Newsfile Corp.')|df.entity_name.eq('Data Centre World Ltd.')|df.industry_group.eq('Media Agencies'))]
    
    # Find the combination with the highest score (one per entity)
    highest_scores = df.sort_values(by=sentiment_col, ascending=False).drop_duplicates(subset=['rp_entity_id']).head(top_bottom).copy().reset_index().sort_values(by=sentiment_col, ascending=True)

    # Find the combination with the lowest score (one per entity)
    lowest_scores = df.sort_values(by=sentiment_col, ascending=True).drop_duplicates(subset=['rp_entity_id']).head(top_bottom).copy().reset_index().sort_values(by=sentiment_col, ascending=False)
    
    fig = go.Figure()
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.4, 0.4],
        horizontal_spacing=0.15,
        subplot_titles=("Highest Sentiment", "Lowest Sentiment"))

# Prepare hover text
    for i, (data, color, name) in enumerate(zip([highest_scores, lowest_scores], ['green','red'], ["Highest Sentiment", "Lowest Sentiment"])):
    
        data['hover_text'] = (
            "<b>Entity Name:</b> " + data['entity_name'] + "<br><br>" +
            "<b>Sentiment:</b> " + data[sentiment_col].round(2).astype(str) + "<br><br>" +
            "<b>Headline:</b> " + data['headline'].apply(lambda x: add_line_breaks(x)) + "<br><br>" +
            "<b>Text:</b> " +data['text'].apply(lambda x: add_line_breaks(x)) + "<br><br>")
    
        fig.add_trace(go.Bar(
                y=data['entity_name'],
                x=data[sentiment_col],
                name=name,
                orientation='h',
                marker=dict(color=color),
                hovertemplate=data['hover_text'] + "<extra></extra>",
                text=data[sentiment_col].round(2).astype(str),
                textposition='inside',
                legendgroup=name,  # Separate legend entry for positive sentiment
                showlegend=True
            ), row=1, col=i+1)


    # Update layout to center bars around zero
    fig.update_layout(
        title=title,
        xaxis=dict(zeroline=False,
            zerolinecolor='black'
        ),
        barmode='overlay',  # Overlay bars for better visualization
        template='plotly_white'
    )
                               # Update layout
    fig.update_layout(
        height=900,width=1400
    )

    # Show the plot
    fig.show()