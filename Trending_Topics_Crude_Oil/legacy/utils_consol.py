from . import keywords_functions
from .label_extractor import extract_label
import pandas as pd
import asyncio
import os
from bigdata import Bigdata
from bigdata.query import Similarity, All
from bigdata.models.search import DocumentType, SortBy
from bigdata.daterange import RollingDateRange, AbsoluteDateRange
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
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
import logging
import sys

def create_similarity_query(sentences):
    if not sentences:
        raise ValueError("The list of sentences must not be empty.")

    query = Similarity(sentences[0])

    for sentence in sentences[1:]:
        query &= Similarity(sentence)
    
    return query

def create_similarity_query_new(sentences):
    if not sentences:
        raise ValueError("The list of sentences must not be empty.")
        
    def flatten(xss):
        return [x for xs in xss for x in xs]
    
    def contains_lists(lst):
        for item in lst:
            if isinstance(item, list):
                return True
        return False
    if contains_lists(sentences):
        final_list = list(set(list(sum(flatten(sentences), ()))))
    else:
        final_list = sentences

    sentences = [Similarity(sample) for sample in final_list]

    query = All(sentences)
    
    return query


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

# # Define the mask processing function
# def mask_entity_coordinates(input_df, column_masked_text, mask_target, mask_other, entity_key_to_name):
#     for idx, row in input_df.iterrows():
#         text = row['text']
#         entities = row['entities']
#         entities.sort(key=lambda x: x['start'], reverse=True)  # Sort entities by start position in reverse order
#         masked_text = text

#         for entity in entities:
#             entity_name = entity_key_to_name.get(entity['key'])
#             start, end = entity['start'], entity['end']
#             if entity['key'] == row['rp_entity_id']:
#                 masked_text = masked_text[:start] + mask_target + masked_text[end:]
#             else:
#                 masked_text = masked_text[:start] + mask_other + masked_text[end:]

#         input_df.at[idx, column_masked_text] = masked_text
#     return input_df

# Define the mask processing function, with numbered entities
def mask_entity_coordinates(input_df, column_masked_text, mask_target, mask_other, entity_key_to_name):
    i = 1
    entity_counter = {}
    for idx, row in input_df.iterrows():
        # print(idx)
        text = row['text']
        entities = row['entities']
        entities.sort(key=lambda x: x['start'], reverse=True)  # Sort entities by start position in reverse order
        masked_text = text

        target_start = []
        target_end = []
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
                    i+=1
                else:
                    masked_text = masked_text[:start] + f'{mask_other}_{entity_counter[entity["key"]]}' + masked_text[end:]

        input_df.at[idx, column_masked_text] = masked_text
    return input_df

def clean_text(text):
    text = text.replace('{', '').replace('}', '')
    return text

def clean_text_masked(text):
    text = text.replace('{', '').replace('}', '')
    return text

def process_sentences_masking(sentences: pd.DataFrame, entity_key_to_name, target_entity_mask: str,  other_entity_mask: str, masking: bool = False) -> pd.DataFrame:
    sentences['text'] = sentences['text'].str.replace('{','',regex=False)
    sentences['text'] = sentences['text'].str.replace('}','',regex=False)

    if masking:
        sentencesdf = mask_entity_coordinates(input_df = sentences, 
                                column_masked_text = 'masked_text',
                                mask_target = target_entity_mask,
                                mask_other = other_entity_mask,
                                entity_key_to_name = entity_key_to_name
                               )

        sentencesdf['masked_text'] = sentencesdf['masked_text'].apply(clean_text_masked)

        sentencesdf = sentencesdf[sentencesdf.masked_text!='to_remove']
    else:
        sentencesdf = sentences

    sentencesdf['text'] = sentencesdf['text'].apply(clean_text)
    sentencesdf = sentencesdf[sentencesdf.text!='to_remove']

    return sentencesdf.reset_index(drop = True)

# def run_similarity_search(query, start_date, end_date, bigdata_credentials, freq = 'MS', limit_documents = 1000):
#     sentences_final = []
#     bigdata = bigdata_credentials
#     start_list, end_list = create_dates_lists(start_date, end_date, freq=freq)

#     for period in tqdm(range(1,len(start_list)+1,1)): #,desc='Loading and saving datasets for '+'US ML'+' universe'):
#         start_date = start_list[period-1]
#         end_date = end_list[period-1]
#         date_range = AbsoluteDateRange(start_date, end_date)
#         search = bigdata.content_search.new_from_query(
#             query=query,
#             date_range= date_range,
#             scope=DocumentType.NEWS,
#             sortby=SortBy.RELEVANCE
#         )

#         # Limit the number of documents to retrieve
#         results = search.limit_documents(limit_documents)

#         # Collect unique entity keys
#         entity_keys = set()
#         for result in results:
#             for chunk in result.chunks:
#                 for entity in chunk.entities:
#                     entity_keys.add(entity.key)

#         # Lookup entity details by their keys
#         entities_lookup = bigdata.knowledge_graph.get_entities(list(entity_keys))

#         # Create a mapping of entity keys to names and types
#         entity_key_to_name = {}
#         for entity in entities_lookup:
#             if hasattr(entity, 'id') and hasattr(entity, 'name') and entity.entity_type == 'COMP':
#                 entity_key_to_name[entity.id] = entity.name

#         # Create lists to store the DataFrame rows
#         data_rows = []

#         # Filter and store the results with known entity names and coordinates
#         for result in results:
#             chunk_texts_with_entities = []
#             for chunk_index, chunk in enumerate(result.chunks):
#                 chunk_entities = []
#                 for entity in chunk.entities:
#                     entity_name = entity_key_to_name.get(entity.key, 'Unknown')
#                     if entity_name != 'Unknown':
#                         chunk_entities.append({
#                             'key': entity.key,
#                             'name': entity_name,
#                             'start': entity.start,
#                             'end': entity.end
#                         })
#                 if chunk_entities:
#                     chunk_texts_with_entities.append((chunk.text, chunk_entities, chunk_index))

#             for chunk_text, chunk_entities, chunk_index in chunk_texts_with_entities:
#                 for entity in chunk_entities:
#                     other_entities = [e for e in chunk_entities if e['name'] != entity['name']]
#                     data_rows.append({
#                         'timestamp_utc': result.timestamp,
#                         'rp_document_id': result.id,
#                         'sentence_id': f"{result.id}-{chunk_index}",
#                         'rp_entity_id': entity['key'],
#                         'entity_name': entity['name'],
#                         'text': chunk_text,
#                         'other_entities': ', '.join([e['name'] for e in other_entities]),
#                         'entities': chunk_entities
#                     })

#         # Create the DataFrame
#         df = pd.DataFrame(data_rows)

#         # Remove duplicate rows
#         df = df.drop_duplicates(subset=['timestamp_utc', 'rp_document_id', 'sentence_id', 'rp_entity_id'])

#         # Process the sentences to add masked text
#         sentences = process_sentences_masking(df, entity_key_to_name, target_entity_mask='Target Company', other_entity_mask='Other Company', masking=True, )

#         sentences_final.append(sentences)

#     sentences = pd.concat(sentences_final)
    
#     return sentences

def run_similarity_search(query, start_date, end_date, bigdata_credentials, freq = 'Monthly', document_type='News',  limit_documents = 1000):
    sentences_final = []
    time_frequencies = {'Yearly':'YS','Monthly':'MS','Weekly':'W', 'Daily':'D'}
    documents = {'News':DocumentType.NEWS, 'Transcripts':DocumentType.TRANSCRIPTS, 'Filings':DocumentType.FILINGS}
    doctype = documents[document_type]
    freq_query = time_frequencies[freq]
    bigdata = bigdata_credentials
    start_list, end_list = create_dates_lists(start_date, end_date, freq=freq_query)

    for period in tqdm(range(1,len(start_list)+1,1)): #,desc='Loading and saving datasets for '+'US ML'+' universe'):
        start_date = start_list[period-1]
        end_date = end_list[period-1]
        print(f'{start_date} - {end_date}')
        date_range = AbsoluteDateRange(start_date, end_date)
        search = bigdata.search.new(
            query=query,
            date_range= date_range,
            scope=doctype,
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
                        'rp_entity_id': entity['key'],
                        'entity_name': entity['name'],
                        'text': chunk_text,
                        'other_entities': ', '.join([e['name'] for e in other_entities]),
                        'entities': chunk_entities
                    })

        # Create the DataFrame
        df = pd.DataFrame(data_rows)

        # Remove duplicate rows
        df = df.drop_duplicates(subset=['timestamp_utc', 'rp_document_id', 'sentence_id', 'rp_entity_id'])

        sentences_final.append(df)

    sentences = pd.concat(sentences_final).sort_values('timestamp_utc').reset_index(drop=True)
    sentences['date'] = pd.to_datetime(pd.to_datetime(sentences.timestamp_utc).dt.date)
    
    # Process the sentences to add masked text
    sentences = process_sentences_masking(sentences, entity_key_to_name, target_entity_mask='Target Company', other_entity_mask='Other Company', masking=True, )
    
    #sentences = drop_duplicates(sentences, 'timestamp_utc', ['rp_entity_id','text'])
    
    return sentences

# def drop_duplicates(df = pd.DataFrame(), date_column = '', duplicate_set = []):
#     df = df.sort_values(by=date_column)
    
#     df_filt = df.groupby(duplicate_set).first().reset_index().sort_values(by=date_column).reset_index(drop=True)
    
#     return df_filt

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

def reinstate_duplicates(labelled_df= pd.DataFrame(), duplicated_chunks = pd.DataFrame(), date_column = '', duplicate_set = []):
    
    final_df = pd.concat([labelled_df, duplicated_chunks], axis=0).sort_values(by=duplicate_set+[date_column]).reset_index(drop=True)
    final_df['label'] = final_df.groupby(duplicate_set)['label'].ffill()
    final_df = final_df.sort_values(by=date_column).reset_index(drop=True)
    
    return final_df

async def process_sentences(sentences, sentence_column_2, masked_2, system_prompt, path_training, n_expected_response_tokens, batch_size, model, open_ai_credentials):
    sentence_final = []

    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            sentences_labels_2 = await extract_label(
                sentences=sentences,
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
            logger.error(f"Error on attempt {attempt + 1} : {e}")
            if attempt < retry_attempts - 1:
                await asyncio.sleep(100)  # Wait for 100 seconds before retrying
            else:
                logger.error(f"Failed after {retry_attempts} attempts")



    return pd.concat(sentence_final)

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
    ax1.set_xticklabels([labels_names_dict[item.get_text()] for item in ax1.get_xticklabels()])


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
    ax2.set_xticklabels([labels_names_dict[item.get_text()] for item in ax2.get_xticklabels()])

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
    result = result.groupby(['rp_entity_id', 'date'])[['text','motivation']].last().reset_index()
    
    return result

def unmask_motivation(df=pd.DataFrame(), motivation_col='motivation'):
    import re
    
    if type(df[motivation_col])==str:
        unmasked_string = re.sub(r'Target Company_\d{1,2}', df['entity_name'], df[motivation_col])
        
    else:
        unmasked_string = None
    
    return unmasked_string
    
    
def top_companies_sector(df_filtered, bigdata_credentials, sector_column): 
    
    entity_news_count = defaultdict(int)

    df_filtered = lookup_sector_information(df_filtered, bigdata_credentials)
    
    df_news_count = df_filtered.groupby(['rp_entity_id']).agg({'sentence_id': pd.Series.nunique}).reset_index()

    df_news_count.rename(columns={'sentence_id': 'NewsCount'}, inplace=True)

    
    #df_news_count = df_news_count.loc[~df_news_count['sector'].isin(['Technology'])]

    chunk_text_df = get_chunk_when_max_coverage(df_filtered)
    df_news_count = df_news_count.merge(df_filtered[['rp_entity_id', 'entity_name', sector_column]], how='left', on='rp_entity_id')
    
    df_news_count = df_news_count.merge(chunk_text_df[['rp_entity_id','text', 'motivation']], on='rp_entity_id', how='left')
    
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
        vertical_spacing=0.2,
        horizontal_spacing=0.1,
        specs=[[{"type": "bar"}] * n_cols for _ in range(n_rows)]
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
            f"<b>Entity:</b> {row.entity_name}<br><br><b>Text</b>: {add_line_breaks(row.text)}<br><br><b>Motivation</b>: {add_line_breaks(unmask_motivation(row))}"
            for idx, row in sector_data.iterrows()
        ]

        fig.add_trace(
            go.Bar(
                x=sector_data['entity_name'],
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
        height=900 * n_rows,
        title_text=f"Top Companies by {sector_column.replace('_', ' ').capitalize()}",
    )

    # Show the interactive plot
    fig.show()
    
## Obtain volume exposure to a theme and identify a basket of companies
def identify_basket_of_companies(df, dfs_labels,df_names,bigdata_credentials, sector_column,basket_size, start_date, end_date):
    
    df_filt = df.loc[df.label.isin(dfs_labels)].copy().reset_index(drop=True)
    
    total_vol = df_filt.groupby('rp_entity_id')['sentence_id'].nunique().rename('total_exposure').reset_index()
    
    grouped_df = df_filt.groupby(['rp_entity_id', 'label']).size().unstack(fill_value=0).reset_index()
    grouped_df = lookup_sector_information(grouped_df, bigdata_credentials)
    grouped_df.columns.name = None
    
    grouped_df = grouped_df.merge(total_vol, on='rp_entity_id', how='left')
    
    for label, name in zip(dfs_labels, df_names):
        grouped_df[f'{name}_pct'] = grouped_df[label]*100/grouped_df['total_exposure']
        
        chunk_text_df = get_chunk_when_max_coverage(df_filt.loc[df_filt.label.eq(label)])
        
        grouped_df = pd.concat([grouped_df.set_index('rp_entity_id'), chunk_text_df.rename(columns={'text':f'text_{name}', 'motivation':f'motivation_{name}'}).set_index(
            'rp_entity_id')[[f'text_{name}',f'motivation_{name}']]],axis=1).reset_index()
        
        grouped_df[f'motivation_{name}'] = grouped_df.apply(unmask_motivation, motivation_col=f'motivation_{name}', axis=1)
        
#     #select those companies that have at least 1 chunk per month on average, remove those companies that have 50% positive and 50% negative exposure
#     total_months = round(((pd.to_datetime(end_date).year-pd.to_datetime(start_date).year)*365 +
#                           (pd.to_datetime(end_date).month - pd.to_datetime(start_date).month)*30 +
#                           (pd.to_datetime(end_date).day - pd.to_datetime(start_date).day))/30,0)
    
#     top_exposed = grouped_df.loc[grouped_df.total_exposure.ge(total_months)].reset_index(drop=True).copy()
#     top_exposed = top_exposed.sort_values('entity_name').reset_index(drop=True)

    ##select top companies in basket
    top_exposed = grouped_df.nlargest(basket_size, 'total_exposure', keep='all').sort_values('entity_name').reset_index(drop=True)

    return top_exposed

def get_daily_volume_by_label(df, dfs_labels,df_names,bigdata_credentials, sector_column, start_date, end_date):
    
    from functools import reduce

    dfs_by_date = []
    
    for label,name in zip(dfs_labels, df_names):
        
        df = df.sort_values(by=['date'])
        df.date = pd.to_datetime(df.date)
        df_label = df.loc[df.label.eq(label)].copy().reset_index(drop=True)
        
        vol_by_date = df_label.groupby(['date','rp_entity_id']).agg({'sentence_id':'nunique', 'text':'last', 'motivation':'last'}).reset_index()

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
        vol_by_date_full['motivation'] = vol_by_date_full.apply(unmask_motivation, motivation_col=f'motivation', axis=1)
        vol_by_date_full = vol_by_date_full.rename(columns={'sentence_id':name, 'text':f'text_{name}','motivation':f'motivation_{name}'})
        
        dfs_by_date.append(vol_by_date_full[['date', 'entity_name', 'rp_entity_id', 'industry_group', name, f'text_{name}', f'motivation_{name}']])
        
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
    display(df.sort_values(by='total_exposure', ascending=False).head(5))

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
        xaxis=dict(title='Pricing Power', range=[0, 100], dtick=10),
        yaxis=dict(title='Entity Name'),
        template='plotly_white',
        showlegend=True
    )
    for i, (name, leg_label, color) in enumerate(zip(df_names, legend_labels, ['green','red'])):
        base_val = 0 if i==0 else df[f'{df_names[i-1]}_pct']
        text_col = f'text_{name}'
        # Create custom hover text with text and motivation, adding dynamic line breaks for better readability
        hover_text = [
            f"<b>{leg_label}:</b> {round(row[f'{name}_pct'],2)}%<br><br><b>Entity:</b>{row['entity_name']}<br><br><b>Text:</b> {add_line_breaks(row[text_col])}<br><br><b>Motivation</b>: {add_line_breaks(unmask_motivation(row, f'motivation_{name}'))}" for index, row in df.iterrows()]
        
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
def track_basket_sectors_over_time(df, companies_list, sector_column, sectors_list = []):
    # Create the figure
    fig = go.Figure()
    import matplotlib.colors as mcolors
    import math
    
    if not sectors_list:
        # Rank sectors by the total number of news mentions
        sectors_list = (df.groupby(sector_column, group_keys=False)['total_exposure']
                      .sum()
                      .sort_values(ascending=False)
                      .index)[:10]

    # Plot lines for each entity within each industry group
    for i, group in enumerate(sectors_list):
        group_data = df[df['industry_group'] == group]
        unique_entities = group_data.loc[group_data.rp_entity_id.isin(companies_list)]['entity_name'].unique()

        # Define a colormap with up to 20 distinct colors
        color_palette = plt.get_cmap('tab20', 20).colors
        color_palette = [mcolors.rgb2hex(color) for color in color_palette]
        color_map = dict(zip(unique_entities, color_palette[:len(unique_entities)]))

        fig = go.Figure()

        for entity in unique_entities:
            entity_data = group_data[group_data['entity_name'] == entity].copy()
            entity_data['hovertext'] = entity_data.apply(
                lambda row: f"<b>{row['entity_name']}</b><br><b>Number of Positive Chunks:</b> {row['positive_exp']}<br>"
                            f"<b>Number of Negative Chunks:</b> {row['negative_exp']}<br>"
                            f"<b>Text (Positive):</b> {add_line_breaks(row['text_positive_exp'])}<br>"
                            f"<b>Motivation (Positive):</b> {add_line_breaks(unmask_motivation(row,'motivation_positive_exp'))}<br>"
                            f"<b>Text (Negative):</b> {add_line_breaks(row['text_negative_exp'])}<br>"
                            f"<b>Motivation (Negative):</b> {add_line_breaks(unmask_motivation(row,'motivation_negative_exp'))}<br>", axis=1)

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
            zero_nonnull_data = entity_data.loc[(entity_data['net_exposure'] == 0) & (entity_data['text_positive_exp'].notnull())]
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
            height=450,
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
    entity_texts_motivations = df_filtered.groupby(['entity_name', 'Day']).apply(lambda x: x[['text', 'motivation', 'entity_name']].to_dict('records')).to_dict()


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
                            f"<b>Text </b>: {add_line_breaks(record['text'])}<br><br><b>Motivation</b>: {add_line_breaks(unmask_motivation(record))}"
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
    
def provider_user_net(df_filtered_ai_cost_reduction, df_filtered_ai_providers):
    
    # Exclude 'motivation' column during the merge
    cols_to_merge = ['timestamp_utc', 'rp_document_id', 'sentence_id', 'rp_entity_id', 'entity_name', 'text', 'masked_text']
    merged_df = pd.merge(df_filtered_ai_cost_reduction[cols_to_merge], df_filtered_ai_providers[cols_to_merge], how='outer', indicator=True)

    # Filter out common rows
    df_filtered_ai_cost_reduction_without_common = df_filtered_ai_cost_reduction.copy(deep = True)
    df_filtered_ai_providers_without_common = df_filtered_ai_providers.copy(deep = True)

    df_filtered_ai_cost_reduction_without_common['entity_name'] = df_filtered_ai_cost_reduction_without_common['entity_name'] + '_user'
    df_filtered_ai_providers_without_common['entity_name'] = df_filtered_ai_providers_without_common['entity_name'] + '_provider'

    # Concatenate the dataframes
    df_filtered = pd.concat([df_filtered_ai_cost_reduction_without_common, df_filtered_ai_providers_without_common], ignore_index=True)
    
    print(df_filtered)

    # Create a network graph
    G = nx.Graph()

    # Initialize a dictionary to count co-mentions and store chunk text and motivations
    co_mention_counts = defaultdict(lambda: {'count': 0, 'texts': [], 'motivations': defaultdict(list)})

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
                    if ('_provider' in entities[i] and '_user' in entities[j]) or ('_user' in entities[i] and '_provider' in entities[j]):
                        pair = tuple(sorted([entities[i], entities[j]]))
                        co_mention_counts[pair]['count'] += 1
                        truncated_text = truncate_text(group.iloc[0]['text'], entities[i], entities[j])
                        co_mention_counts[pair]['texts'].append(truncated_text)
                        for entity in entities:
                            entity_motivations = group[group['entity_name'] == entity]['motivation'].tolist()
                            for idx, motivation in enumerate(entity_motivations):
                                entity_motivations[idx] = motivation.replace(f'Target Company_', entity_id_map[entity])
                            co_mention_counts[pair]['motivations'][entity].extend(entity_motivations)

    # Add edges to the graph based on co-mentions
    for pair, data in co_mention_counts.items():
        if data['count'] >= 1:  # Adjust threshold here if needed
            G.add_edge(pair[0], pair[1], weight=data['count'], texts=data['texts'], motivations=data['motivations'])

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

        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            line=dict(width=4, color='gray'),
            hoverinfo='none',
            mode='lines'
        )

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
            node_text.append(node)
            node_color.append('blue' if node.endswith('_user') else 'red')
            # Remove duplicate chunks and truncate text
            unique_texts = [chunk for edge in G.edges(node, data=True) for chunk in edge[2]['texts']]
            unique_texts = list(dict.fromkeys(unique_texts))  # Remove duplicates while preserving order
            formatted_texts = [f"<b>Text {i+1}:</b><br>" + '<br>'.join([text[j:j+80] for j in range(0, len(text), 80)]) for i, text in enumerate(unique_texts)]

            # Format motivations
            formatted_motivations = []
            motivations_added = set()
            for i, text in enumerate(unique_texts):
                for edge in G.edges(node, data=True):
                    motivations = edge[2]['motivations'][node]
                    for motivation in motivations:
                        if motivation not in motivations_added:
                            formatted_motivations.append(f"<b>Motivation {len(formatted_motivations)+1}:</b><br>" + '<br>'.join([motivation[j:j+80] for j in range(0, len(motivation), 80)]))
                            motivations_added.add(motivation)

            node_annotations.append('<br><br>'.join(formatted_texts) + '<br><br>' + '<br><br>'.join(formatted_motivations))

        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode='markers+text',
            text=node_text,
            hovertext=node_annotations,
            hoverinfo='text',
            marker=dict(
                color=node_color,
                size=15,
                line_width=2
            )
        )

        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Network of Companies Mentioned in AI Cost-Cutting Context',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=20, r=20, t=40),  # Reduced left and right margins
                            annotations=[dict(
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002 )],
                            scene=dict(
                                xaxis=dict(showgrid=False, zeroline=False),
                                yaxis=dict(showgrid=False, zeroline=False),
                                zaxis=dict(showgrid=False, zeroline=False),
                            ),
                            height=1600,  # Increase height
                            width=2000   # Increase width
                        )
        )

        fig.show()