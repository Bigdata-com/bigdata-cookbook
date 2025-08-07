from openai import OpenAI
from datetime import datetime
import calendar
from bigdata.query import Keyword, Source, Any, Document
import re
import pandas as pd
import asyncio
import os
from bigdata import Bigdata
from bigdata.query import Similarity, All, Entity, Source, Document, Keyword, Any
from bigdata.models.search import DocumentType, SortBy
from bigdata.daterange import RollingDateRange, AbsoluteDateRange
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from collections import defaultdict
import configparser
from tqdm.notebook import tqdm
import json
from pqdm.processes import pqdm
from operator import itemgetter
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime
import json
import openai
import os
import jinja2
from IPython.core.display import display, HTML
import pandas as pd
import unicodedata
import re
from typing import Optional
from . import *
from . import keywords_functions
import numpy as np
import aiohttp
from tqdm.notebook import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio
from aiolimiter import AsyncLimiter
from concurrent.futures import ThreadPoolExecutor
from openai import AsyncOpenAI


def run_search_keyword_parallel(keywords, start_query, end_query, bigdata_credentials, limit=100, job_number=10):
    start_list, end_list = create_dates_lists(pd.to_datetime(start_query), pd.to_datetime(end_query), freq='D')
    args = []

    # List to store results with keywords
    chunks_with_keywords = []

    for period in range(1, len(start_list) + 1):
        start_date = start_list[period - 1]
        end_date = end_list[period - 1]
        print(f'{start_date} - {end_date}')
        date_range = AbsoluteDateRange(start_date, end_date)

        for keyword in keywords:
            # Modify the args to include the keyword information
            args.append({'query': Keyword(keyword), 'date_range': date_range, 'bigdata_credentials': bigdata_credentials, 'limit': limit, 'keyword': keyword})

    # Run the search in parallel
    chunks_final = pqdm(args, run_single_search, n_jobs=job_number, argument_type='kwargs')

    # Concatenate all chunks
    chunks = pd.concat(chunks_final).sort_values('timestamp').reset_index(drop=True)
    chunks = chunks.drop_duplicates(subset=['timestamp', 'headline', 'text'])

    # Convert 'timestamp' to datetime and extract the date
    chunks['date'] = pd.to_datetime(chunks['timestamp']).dt.date

    # Count chunks by date and keyword
    daily_keyword_counts = chunks.groupby(['date', 'keyword']).size().reset_index(name='number_of_chunks')

    return chunks, daily_keyword_counts

def run_single_search(query, date_range,bigdata_credentials, limit, keyword = ''):
    chunk_results = []
    bigdata = bigdata_credentials
    search = bigdata.search.new(
                    query=query,
                    date_range= date_range,
                    scope=DocumentType.NEWS,
                    sortby=SortBy.RELEVANCE,
                )
    # Limit the number of documents to retrieve
    results = search.limit_documents(limit)
    for result in results:
        for chunk in result.chunks:
            chunk_results.append({'timestamp':result.timestamp, 'rp_document_id':result.id,'headline':result.headline,'chunk_number':chunk.chunk, 'sentence_id':f'{result.id}-{chunk.chunk}', 'source_id':result.source.key,'source_name':result.source.name, 'text':chunk.text})
        if result.cluster:
            for cluster_doc in result.cluster:
                for chunk in cluster_doc.chunks:
                    chunk_results.append({'timestamp':cluster_doc.timestamp, 'rp_document_id':cluster_doc.id,'headline':cluster_doc.headline, 'chunk_number':chunk.chunk,'sentence_id':f'{cluster_doc.id}-{chunk.chunk}', 'source_id':cluster_doc.source.key,'source_name':cluster_doc.source.name, 'text':chunk.text})

    all_chunks = pd.DataFrame(chunk_results)
    all_chunks['keyword'] = keyword
    return all_chunks
## Deprecated
def run_search(query, start_query, end_query, limit=100):
    start_list, end_list = create_dates_lists(pd.to_datetime(start_query), pd.to_datetime(end_query), freq='D')
    similarity_search_sources=[]
    monthly_report_ids_sim = []
    for period in range(1,len(start_list)+1,1): #,desc='Loading and saving datasets for '+'US ML'+' universe'):
        start_date = start_list[period-1]
        end_date = end_list[period-1]
        print(f'{start_date} - {end_date}')
        date_range = AbsoluteDateRange(start_date, end_date)

        search = bigdata_cred.search.new(
                query=query,
                date_range= date_range,
                scope=DocumentType.NEWS,
                sortby=SortBy.RELEVANCE,
            )

        # Limit the number of documents to retrieve
        results = search.limit_documents(limit)
        for result in results:
            #if (any(x in result.headline for x in accepted_keywords_in_headlines)):
            #print(result.headline)
            monthly_report_ids_sim.append(result.id)
            for chunk in result.chunks:
                similarity_search_sources.append({'timestamp':result.timestamp, 'rp_document_id':result.id,'headline':result.headline,'chunk_number':chunk.chunk, 'sentence_id':f'{result.id}-{chunk.chunk}', 'source_id':result.source.key,'source_name':result.source.name, 'text':chunk.text})
            if result.cluster:
                for cluster_doc in result.cluster:
                    #if (any(x in cluster_doc.headline for x in accepted_keywords_in_headlines)):
                    #print(cluster_doc.headline)
                    monthly_report_ids_sim.append(cluster_doc.id)
                    for chunk in cluster_doc.chunks:
                        similarity_search_sources.append({'timestamp':cluster_doc.timestamp, 'rp_document_id':cluster_doc.id,'headline':cluster_doc.headline,'chunk_number':chunk.chunk, 'sentence_id':f'{cluster_doc.id}-{chunk.chunk}', 'source_id':cluster_doc.source.key,'source_name':cluster_doc.source.name, 'text':chunk.text})
    
    similarity_sources = pd.DataFrame(similarity_search_sources)
    return similarity_sources

## Deprecated
def run_search_by_keyword(keywords, start_query, end_query, limit=100):
    start_list, end_list = create_dates_lists(pd.to_datetime(start_query), pd.to_datetime(end_query), freq='D')
    chunk_results=[]
    for keyword in keywords:
        query = Keyword(keyword)
        for period in range(1,len(start_list)+1,1): #,desc='Loading and saving datasets for '+'US ML'+' universe'):
            start_date = start_list[period-1]
            end_date = end_list[period-1]
            print(f'{start_date} - {end_date}')
            date_range = AbsoluteDateRange(start_date, end_date)

            search = bigdata_cred.search.new(
                    query=query,
                    date_range= date_range,
                    scope=DocumentType.NEWS,
                    sortby=SortBy.RELEVANCE,
                )

            # Limit the number of documents to retrieve
            results = search.limit_documents(limit)
            for result in results:
                for chunk in result.chunks:
                    chunk_results.append({'timestamp':result.timestamp, 'rp_document_id':result.id,'headline':result.headline,'chunk_number':chunk.chunk, 'sentence_id':f'{result.id}-{chunk.chunk}', 'source_id':result.source.key,'source_name':result.source.name, 'text':chunk.text})
                if result.cluster:
                    for cluster_doc in result.cluster:
                        for chunk in cluster_doc.chunks:
                            chunk_results.append({'timestamp':cluster_doc.timestamp, 'rp_document_id':cluster_doc.id,'headline':cluster_doc.headline, 'chunk_number':chunk.chunk,'sentence_id':f'{cluster_doc.id}-{chunk.chunk}', 'source_id':cluster_doc.source.key,'source_name':cluster_doc.source.name, 'text':chunk.text})

        all_chunks = pd.DataFrame(chunk_results)
    all_chunks = all_chunks.drop_duplicates(subset=['timestamp','headline','text'])
    return all_chunks

def filter_headline_keywords(headlines, keywords):
    import re
    results = []
    for x in headlines:
        for w in keywords:
            if re.search("\\b{}\\b".format(w), x.lower()):
                #print(x.lower())
                results.append(x)
                break
    return results

def query_entire_document(documents,bigdata_credentials, limit=100):

    query_doc = Any([Document(id_doc) for id_doc in documents])
    bigdata = bigdata_credentials

    all_report_text=[]

    search_doc = bigdata.search.new(query_doc)

    results_doc = search_doc.limit_documents(limit)
    for doc in results_doc:
        #print(doc)
        for chunk in doc.chunks:
            all_report_text.append({'timestamp':doc.timestamp, 'rp_document_id':doc.id,'headline':doc.headline,'chunk_number':chunk.chunk, 'sentence_id':f'{doc.id}-{chunk.chunk}', 'text':chunk.text})
            if doc.cluster:
                for cluster_doc in doc.cluster:
                    #print(cluster_doc)
                    for chunk in cluster_doc.chunks:
                            all_report_text.append({'timestamp':cluster_doc.timestamp, 'rp_document_id':cluster_doc.id,'headline':cluster_doc.headline,'chunk_number':chunk.chunk, 'sentence_id':f'{cluster_doc.id}-{chunk.chunk}','text':chunk.text})

    reports_text = pd.DataFrame(all_report_text)
    reports_text = reports_text.drop_duplicates(subset=['headline', 'text'])
    
    full_reports = reports_text.groupby(['timestamp','rp_document_id','headline']).apply(lambda x: '\n'.join(x.sort_values('chunk_number', ascending=True)['text'])).rename('text').reset_index()
    
    unique_reports = full_reports.groupby("headline").apply(lambda x: x.loc[x['text'].str.len().idxmax()]).reset_index(drop=True).sort_values('timestamp').reset_index(drop=True)
    
    return unique_reports

def extract_summary(text_to_summarize, system_prompt, model, yearmonth, number_of_reports, api_key):
    # Initialize OpenAI client
    client = OpenAI(api_key= api_key)
    # Because of the way we construct the prompt, the keywords may be split into multiple concepts
    # For example "fraud in healthcare" may be split into "fraud" and "healthcare", each with 
    # different keywords suggestions.
    system_prompt_1 = system_prompt.replace('[DATE]', yearmonth)
    system_prompt_1 = system_prompt_1.replace('[N]', str(number_of_reports))
    #print(system_prompt_1)
    
    response = client.chat.completions.create(
        model = model, #"gpt-4o-2024-05-13", #"gpt-4-1106-preview", #"gpt-3.5-turbo-1106", #"gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125", "gpt-4-1106-preview", "gpt-4-0125-preview"
        messages = [
                {
                  "role": "system",
                  "content": system_prompt_1
                },
            {"role":"user",
             "content":text_to_summarize}
        ],
        temperature = 0,
        response_format={ "type":"json_object"}
            )
    ##hash the results!!
    summary=response.model_dump()
    final_summary = eval(summary['choices'][0]['message']['content'])[f'{yearmonth}']
    table = eval(summary['choices'][0]['message']['content'])['table']
    df = pd.DataFrame(index = table[0], data=table[1], columns=['Performance'])

    return final_summary, df

def search_keyword(report_headline_keywords, start_query, end_query, report_headlines_list,bigdata_credentials, job_number = 10, limit=750):
    
    chunk_text, daily_keyword_counts = run_search_keyword_parallel(keywords = report_headline_keywords, start_query = start_query, end_query= end_query, bigdata_credentials=bigdata_credentials, limit=limit, job_number=job_number)
    
    #post_processed_text = chunk_text.loc[chunk_text.headline.isin(report_headlines_list)]
    
    monthly_report_ids = chunk_text.rp_document_id.unique()
    
    return chunk_text, daily_keyword_counts 

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

def create_parallel_query_objects(sentences, query_type = 'Similarity', entities = []):
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

    if query_type == 'Similarity':
        sentences = [Similarity(sample) for sample in final_list]
    elif query_type == 'Keyword':
        sentences = [Keyword(sample) for sample in final_list]
    else:
        raise ValueError("The query type is not specified.")
    
    if entities:
        ents = [Entity(ent) for ent in entities]
        query = [sentence & Any(ents) for sentence in sentences]
        
    else:
        query=sentences
    
    return query



def retrieve_news_details(row_numbers, original_df):
    details = []
    for row_number in row_numbers:
        matching_row = original_df.loc[original_df['row_number'] == row_number]
        if not matching_row.empty:
            details.append({
                'source': matching_row['source_name'].values[0],
                'headline': matching_row['headline'].values[0],
                'text': matching_row['text'].values[0]
            })
    return details

def flatten_trending_topics(df, original_df):
    flattened_data = []
    
    for index, row in df.iterrows():
        date = row['Date']
        topics = row['trending_topics']
        
        for topic in topics:
            day_in_review = "\n".join(topic.get('day_in_review', []))  # Get the 'Day in Review' and join as a single string
            
            # Calculate Volume_Score for the topic on this day
            volume_score = len(topic['cited_news'])
            
            for news in topic['cited_news']:
                row_number = news['row_number']  # Extract the row number
                
                # Find the original entry in the original_df based on the row number
                matching_row = original_df.loc[original_df['row_number'] == row_number]

                # If a match is found, extract the relevant details; otherwise, assign NaN
                if not matching_row.empty:
                    source = matching_row['source_name'].values[0]
                    headline = matching_row['headline'].values[0]
                    text = matching_row['text'].values[0]
                else:
                    source = np.nan  # Assign NaN to source if no match is found
                    headline = np.nan
                    text = np.nan
                
                flattened_row = {
                    'Date': date,
                    'Day_in_Review': day_in_review,  # Add 'Day in Review'
                    'Topic': topic['topic'],
                    'Summary': topic['summary'],  # Use the detailed summary from the topic
                    'Source': source,
                    'Headline': headline,
                    'Text': text,
                    'Volume_Score': volume_score  # Add the Volume Score
                }
                flattened_data.append(flattened_row)
    
    # Convert the flattened data into a DataFrame
    flattened_df = pd.DataFrame(flattened_data)
    
    return flattened_df


def generate_text_summary(text, topic, api_key, model='gpt-4o-mini-2024-07-18'):
    client = openai.Client(api_key=api_key)
    
    # Create the prompt incorporating the topic
    prompt = (
        f"You are an expert summarizer. Given the following text, please provide a concise, one-row summary relevant to the topic '{topic}'. "
        f"The summary should be part of a larger analysis report that explains the significance of this text within the context of market movements.\n\n"
        f"Text: {text}\n\n"
        f"Return the result as a JSON object with the following structure:\n"
        f'{{ "summary": "Your summary here"}}'
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    summary = response.model_dump()
    
    try :
        summary_dict = eval(summary['choices'][0]['message']['content'])['summary']
    except :
        summary_dict = ''
    
    return summary_dict

def add_text_summaries_to_df(df, api_key):
    df['Text_Summary'] = df.apply(lambda row: generate_text_summary(row['Text'], row['Topic'], api_key), axis=1)
    return df

def generate_advanced_novelty_score(topic, previous_topics_dict, api_key, model='gpt-4o-mini-2024-07-18'):
    client = openai.Client(api_key=api_key)
    
    # Format the previous topics dictionary into a string for the prompt
    previous_topics_formatted = "\n".join([f"{date}: {', '.join(topics)}" for date, topics in previous_topics_dict.items()])
    
    # Create the prompt that compares the current topic with previous topics
    prompt = (
        f"You are an expert analyst tasked with determining the novelty of topics in the crude oil market. "
        f"The current topic is: '{topic}'. Below are the topics that have been mentioned on previous days:\n\n"
        f"{previous_topics_formatted if previous_topics_formatted else 'No previous topics provided.'}\n\n"
        f"**Important Considerations:**\n"
        f"1. **Temporal Distance**: Consider how many days have passed since the last mention of similar topics. If a topic hasn't been mentioned for several days and reappears with new information or updates, it may be considered 'New' again.\n"
        f"2. **Topic Specificity**: Distinguish between general topics (e.g., 'Oil Production Uncertainty') and more specific elements (e.g., 'Venezuelan Election'). If a topic combines familiar elements with new, specific details, assess the novelty based on the new elements.\n"
        f"3. **Combining and Differentiating Topics**: If a topic includes elements that have been discussed previously but introduces new aspects or a different context (e.g., new geopolitical developments, elections), weigh the novelty based on the new components.\n\n"
        f"**Decision Criteria:**\n"
        f"- If there are no previous topics provided, or if the current topic introduces new elements after a significant time gap, consider it 'New'.\n"
        f"- If the topic reappears but with additional new information or developments, consider it 'Moderate'.\n"
        f"- If the topic has been mentioned recently and repeatedly without significant new information, consider it 'Old'.\n\n"
        f"Return the response as an object with the following structure:\n"
        f'{{ "novelty_score": "Old" | "Moderate" | "New" }}'
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    novelty_score = response.model_dump()
    try:
        novelty_score_dict = json.loads(novelty_score['choices'][0]['message']['content'])['novelty_score']
    except:
        novelty_score_dict = "Error"  # Default to "Old" if there's an error
    
    return novelty_score_dict

def add_advanced_novelty_scores_to_df(df, api_key):
    df = df.sort_values(by='Date')
    novelty_scores = []
    
    # Dictionary to store previous topics by date
    previous_topics_dict = {}
    
    # Iterate through each unique date in the DataFrame
    for date in df['Date'].unique():
        # Update previous topics dictionary (topics from all previous days)
        previous_topics = df[df['Date'] < date].groupby('Date')['Topic'].unique().to_dict()
        
        # Filter the DataFrame for the current date
        daily_df = df[df['Date'] == date]
        
        # Process each topic on the current date
        for _, row in daily_df.iterrows():
            topic = row['Topic']
            
            # Generate the novelty score using the OpenAI API, comparing with previous topics
            novelty_score = generate_advanced_novelty_score(topic, previous_topics, api_key)
            novelty_scores.append(novelty_score)
        
        # Add current day's topics to the previous_topics_dict for the next day's comparison
        previous_topics_dict[date] = daily_df['Topic'].unique().tolist()
    
    # Add the novelty scores to the DataFrame
    df['Novelty_Score'] = novelty_scores
    
    return df

def generate_market_impact_and_magnitude(text, topic, api_key, model='gpt-4o-mini-2024-07-18'):
    client = openai.Client(api_key=api_key)
    
    # Create the prompt for determining market impact and magnitude with a focus on oil prices
    prompt = (
        f"You are an expert market analyst with deep knowledge of the oil market, tasked with evaluating the impact of the topic '{topic}' from the perspective of a trader holding oil. "
        f"Consider both the topic and the provided text to assess how this information might influence global oil supply, demand, and consequently, oil prices. "
        f"Keep in mind the following:\n"
        f"- An increase in supply generally leads to a decrease in oil prices (Negative impact).\n"
        f"- A decrease in supply generally leads to an increase in oil prices (Positive impact).\n"
        f"- An increase in demand generally leads to an increase in oil prices (Positive impact).\n"
        f"- A decrease in demand generally leads to a decrease in oil prices (Negative impact).\n\n"
        f"Based on these principles, determine whether the impact on oil prices is Positive (price increase), Negative (price decrease). "
        f"Additionally, evaluate the magnitude of this impact as follows:\n"
        f"- High: The topic is expected to have a significant impact on the overall global oil market.\n"
        f"- Medium: The topic is likely to have a moderate impact on the market, affecting certain regions or segments.\n"
        f"- Low: The topic is expected to have a limited impact, possibly affecting only a specific segment of the market.\n"
        f"- Neutral: No significant impact is anticipated.\n\n"
        f"Consider both direct and indirect effects, including geopolitical risks, market sentiment, and potential shifts in supply and demand dynamics.\n\n"
        f"Text: {text}\n\n"
        f"Make your decision by carefully analyzing both the topic and the text. "
        f"Provide your analysis and return **ONLY** the result as an object with the following structure:\n"
        f'{{ "impact": "Positive" | "Negative", "magnitude": "High" | "Medium" | "Low" }}'
    )


    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert market analyst specializing in the oil market."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    impact_and_magnitude = response.model_dump()
    
    try:
        impact_magnitude_dict = json.loads(impact_and_magnitude['choices'][0]['message']['content'])
    except:
        impact_magnitude_dict = {"impact": "Error", "magnitude": "Error"}  # Default to Neutral if there's an error
    
    return impact_magnitude_dict

def add_market_impact_to_df(df, api_key):
    df = df.sort_values(by='Date')
    impact_magnitude_dict = {}
    
    # Group by Topic and Date to generate a single impact and magnitude score for each topic
    grouped = df.groupby(['Date', 'Topic'])
    
    for (date, topic), group in grouped:
        # Concatenate the 'Summary' text for the topic on the given date
        combined_text = group['Summary']
        
        # Generate the impact and magnitude scores using the OpenAI API
        impact_magnitude = generate_market_impact_and_magnitude(combined_text, topic, api_key)
        
        # Store the scores in the dictionary
        impact_magnitude_dict[(date, topic)] = impact_magnitude
    
    # Map the impact and magnitude back to the DataFrame
    df['Impact_Score'] = df.apply(lambda row: impact_magnitude_dict[(row['Date'], row['Topic'])]['impact'], axis=1)
    df['Magnitude_Score'] = df.apply(lambda row: impact_magnitude_dict[(row['Date'], row['Topic'])]['magnitude'], axis=1)
    
    # Reorder the last columns
    # Specify the exact order you want for the last columns
    final_column_order = ['Text_Summary', 'Volume_Score', 'Novelty_Score', 'Impact_Score', 'Magnitude_Score']
    
    # Get the current column order
    current_columns = df.columns.tolist()
    
    # Reorder the columns by placing the 'final_column_order' at the end
    reordered_columns = current_columns[:-5] + final_column_order
    
    # Apply the reordering to the DataFrame
    df = df[reordered_columns]
    
    return df

def clean_text(text):
    # Normalize the text to remove any weird encodings
    text = unicodedata.normalize('NFKD', text)
    
    # Check if the dollar sign has already been replaced, and only replace if it hasn't
    if not r'\$' in text:
        text = text.replace('$', r'\$')
    
    # Remove any unintended italic or mathematical symbols by replacing them
    text = re.sub(r'[\u2061-\u2064\u0338-\u0339\u2212-\u2213\u200E-\u200F]', '', text)
    
    # Ensure spacing is correct by replacing multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text


# Function to generate a report for a single date
def generate_html_report(date, day_in_review, topics, template_path="./report_template.html"):
    # Load the Jinja2 template
    template_loader = jinja2.FileSystemLoader(searchpath="./")
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template(template_path)
    
    # Render the template with data
    html_output = template.render(
        date=date,
        day_in_review=day_in_review,
        topics=topics
    )
    
    return html_output

# Helper functions for sorting
def novelty_score_value(novelty_score):
    score_map = {'New': 3, 'Moderate': 2, 'Old': 1}
    return score_map.get(novelty_score, 0)  # Default to 0 if novelty_score is not recognized

def magnitude_value(magnitude):
    score_map = {'High': 3, 'Medium': 2, 'Low': 1, 'Neutral': 0}
    return score_map.get(magnitude, 0)  # Default to 0 if magnitude is not recognized

def number_of_news_value(number_of_news):
    return number_of_news  # Use directly for sorting

def prepare_data_for_report(df, ranking_criteria, report_date: Optional[str] = None, impact_filter: Optional[str] = None):
    # Define the mapping for novelty scores
    novelty_mapping = {
        'New': 'Novel',
        'Moderate': 'Moderate',
        'Old': 'Repeat'
    }
    
    # Remove rows where 'Source' or 'Text_Summary' is NaN or empty
    df = df.dropna(subset=['Source', 'Text_Summary'])
    df = df[(df['Source'].str.strip() != '') & (df['Text_Summary'].str.strip() != '')]

    # Apply impact filter if provided
    if impact_filter is not None:
        if impact_filter == 'positive_impact':
            df = df[df['Impact_Score'].str.lower() == 'positive']
        elif impact_filter == 'negative_impact':
            df = df[df['Impact_Score'].str.lower() == 'negative']

    reports = []
    criteria_map = {
        'novelty': lambda x: novelty_score_value(novelty_mapping.get(x['novelty_score'], x['novelty_score'])),
        'magnitude': lambda x: magnitude_value(x['magnitude']),
        'volume': lambda x: number_of_news_value(x['number_of_news'])
    }

    # Convert report_date to datetime.date if provided
    if report_date is not None:
        try:
            report_date = datetime.strptime(report_date, "%Y-%m-%d").date()
            # Filter DataFrame by date
            df = df[df['Date'] == report_date]
        except ValueError:
            print(f"Invalid date format: {report_date}. Expected format is YYYY-MM-DD.")
            return []

    grouped = df.groupby('Date')

    for date, group in grouped:
        topics = []
        top_topics = group.groupby('Topic').agg({
            'Summary': 'first',
            'Day_in_Review': 'first',
            'Novelty_Score': 'first',
            'Magnitude_Score': 'first',
            'Impact_Score': 'first'
        }).reset_index()
        
        # Extract "Day in Review" for the report from the first topic (assuming it is consistent across topics)
        day_in_review = top_topics['Day_in_Review'].iloc[0].split('\n') if 'Day_in_Review' in top_topics.columns else []
        
        for _, topic_row in top_topics.iterrows():
            topic = topic_row['Topic']
            summary = topic_row['Summary']
            # Apply the novelty mapping
            novelty_score = novelty_mapping.get(topic_row['Novelty_Score'], topic_row['Novelty_Score'])
            magnitude = topic_row['Magnitude_Score']
            impact = topic_row['Impact_Score']
            
            # Filter out rows with empty 'Source' or 'Text_Summary'
            associated_news = group[group['Topic'] == topic].dropna(subset=['Source', 'Text_Summary'])
            associated_news = associated_news[(associated_news['Source'].str.strip() != '') & (associated_news['Text_Summary'].str.strip() != '')]
            
            sources_and_summaries = associated_news[['Source', 'Text_Summary']].values.tolist()
            
            # Calculate the number of news items after cleaning
            number_of_news = len(associated_news)
            
            topics.append({
                'topic': topic,
                'summary': summary,
                'novelty_score': novelty_score,
                'magnitude': magnitude,
                'impact': impact,
                'sources_and_summaries': sources_and_summaries,
                'number_of_news': number_of_news
            })
        
        # Sort topics based on the provided ranking criteria
        topics = sorted(topics, key=lambda x: tuple(criteria_map[crit](x) for crit in ranking_criteria), reverse=True)
        
        reports.append({
            'date': date.strftime('%Y-%m-%d'),
            'day_in_review': day_in_review,
            'topics': topics
        })

    return reports

def label_topics_and_cite_news(text_to_analyze, model, date, number_of_reports, api_key):
    topic_labeling_prompt = (
    f"""You are an expert analyst tasked with identifying the underlying causes of market movements in the crude oil market based on a set of daily news reports.

    Your objective is to analyze the provided text and group related news articles into broad but coherent topics that represent specific causes or events influencing the market on the date [DATE]. The goal is to create a manageable number of topics by clustering news items that share common themes, causes, or impacts, while avoiding unnecessary granularity.

Text Description:
- The input text includes [N] market reports, each explicitly structured with its Timestamp, Source, Headline, Text, and Row Number. Each report is clearly separated by a '--- Report End ---' marker. The association between the Source, Headline, and Text must be preserved and not altered.

Instructions:
1. **Identify and Group Related News**: Analyze the news reports and group them into broader topics based on similar causes, events, or market impacts. Avoid creating too many small or overly specific topics. Instead, focus on clustering news items that share significant similarities into comprehensive topics.

2. **Create Descriptive and Coherent Topic Labels**: For each topic, generate a label that captures the overall theme or cause of the market movement. The label should be broad enough to encompass all related news items within the cluster, yet specific enough to accurately reflect the key driver behind the market activity.

3. **Avoid Over-Specification**: While ensuring coherence within each topic, avoid creating overly specific labels that could fragment related news into multiple smaller topics. Aim to consolidate similar events or developments under a single, broader topic.

4. **Emphasize Market Impact**: Ensure that each topic label communicates the market impact of the events or developments. For example, if multiple reports discuss geopolitical tensions in different regions, they could be grouped under a single topic like 'Geopolitical Tensions Affecting Global Oil Supply'.

Examples of Broader Topics:
- 'Geopolitical Tensions Affecting Global Oil Supply and Prices'
- 'U.S. Economic Indicators and Their Impact on Oil Demand'
- 'OPEC+ Production Decisions and Global Market Reactions'
- 'Changes in U.S. Crude Inventories and Market Responses'
- 'Market Volatility Driven by Algorithmic Trading and Technical Factors'

Output:
Return the response as a JSON object with the following structure:
{{
    "topics": [
        {{
            "topic": "Clustered Cause or Event (e.g., Geopolitical Tensions Affecting Global Oil Supply and Prices)",
            "cited_news": [
                {{"row_number": Y}},
                {{"row_number": Z}},
                // All related row numbers...
            ]
        }},
        {{
            "topic": "Clustered Cause or Event (e.g., Changes in U.S. Crude Inventories and Market Responses)",
            "cited_news": [
                {{"row_number": A}},
                {{"row_number": B}},
                // All related row numbers...
            ]
        }},
        // More clustered topics...
    ]
}}
"""
)
    # Prepare the prompt
    prompt_prepared = topic_labeling_prompt.replace('[DATE]', date).replace('[N]', str(number_of_reports))
    
    # Send request to OpenAI API
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_prepared},
            {"role": "user", "content": text_to_analyze}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    # Extract and process the response
    topics = eval(response.model_dump()['choices'][0]['message']['content'])['topics']
    
    return topics


def generate_topic_summaries_by_group(df, model, api_key):
    summaries = []

    # Group by 'Date' and 'Topic' to generate summaries topic by topic
    grouped = df.groupby(['Date', 'topic'])

    for (date, topic), group in grouped:
        topic_summary_prompt = (
            f"""You are an expert analyst tasked with summarizing the specific and distinct causes of market movements in the crude oil market for the topic: '{topic}' on {date}.

            Instructions:
            - Analyze all the provided news articles related to this topic.
            - Provide a detailed, in-depth analytical summary that explains why this specific cause or event influenced the crude oil market on {date}.
            - Synthesize information from all related news reports.

            Output:
            Return the summary as a string.
            """
        )
        
        # Prepare the text by concatenating all related news for this topic
        related_news = "\n\n".join(group['text_with_reference'].tolist())
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Send request to OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": topic_summary_prompt},
                {"role": "user", "content": related_news}
            ],
            response_format={"type": "text"},
            temperature=0
        )
        
        # Extract and process the response
        summary = response.model_dump()['choices'][0]['message']['content'].strip()
        
        summaries.append({
            "date": date,
            "topic": topic,
            "summary": summary,
            "number_of_news": len(group),
            "cited_news": group['row_number'].tolist()
        })
    
    return summaries



def generate_day_in_review_by_date(summaries, model, api_key):
    day_in_reviews = []
    
    # Group by 'Date' to generate a day in review for each day
    grouped = pd.DataFrame(summaries).groupby('date')

    for date, group in grouped:
        day_in_review_prompt = (
            f"""You are an expert analyst tasked with summarizing the key events of the day for the crude oil market on {date}.

            Instructions:
            - Analyze the provided topic summaries.
            - Focus on the most important market-moving events and data points.
            - Create a list of bullet points summarizing these key events.

            Output:
            Return the summary as a list of bullet points.
            """
        )
        
        group['summary'] = group['summary'].fillna('').astype(str)
        
        # Combine all topic summaries for the day
        all_summaries = "\n\n".join(group['summary'].tolist())
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Send request to OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": day_in_review_prompt},
                {"role": "user", "content": all_summaries}
            ],
            response_format={"type": "text"},
            temperature=0
        )
        
        # Extract and process the response
        day_in_review = response.model_dump()['choices'][0]['message']['content'].strip().split('\n')
        
        day_in_reviews.append({
            "date": date,
            "day_in_review": day_in_review
        })
    
    return day_in_reviews

def process_trending_topics(unique_reports, start_query, end_query, model='gpt-4o-mini-2024-07-18', api_key=None):
    # Initialize your date range
    start_list, end_list = create_dates_lists(pd.to_datetime(start_query), pd.to_datetime(end_query), freq='D')

    # DataFrames to store results
    trending_topics_df = pd.DataFrame(columns=['Date', 'trending_topics'])
    
    final_merged_df = pd.DataFrame()

    # Preprocess the unique_reports DataFrame
    unique_reports['Date'] = unique_reports['timestamp'].apply(lambda x: x.date())
    unique_reports['row_number'] = unique_reports.index  # Add row number as an additional column

    unique_reports['text_with_reference'] = (
        '--- Report Start ---\n'
        'Row Number: ' + unique_reports['row_number'].astype(str) +  # Include row number
        '\nTimestamp: ' + unique_reports.timestamp.astype(str) + 
        '\nSource: ' + unique_reports.source_name + 
        '\nHeadline: ' + unique_reports.headline +
        '\nText: ' + unique_reports.text.astype(str) +
        '\n--- Report End ---'
    )

    # Convert dates to datetime objects
    dates = [date_str for date_str in unique_reports['Date'].unique()]

    # Iterate over each date in the range
    for period in tqdm(dates):
        reports = unique_reports.loc[unique_reports['Date'] == period]
        if reports.empty:
            continue
        
        # Extract labeled topics and cited news for each topic
        labeled_topics = label_topics_and_cite_news(
            text_to_analyze="\n\n".join(reports['text_with_reference']),
            model=model,
            date=period.strftime('%Y-%m-%d'),
            number_of_reports=len(reports),
            api_key=api_key
        )

        # Convert labeled topics to a DataFrame
        topics_df = pd.DataFrame(labeled_topics)
        
        # Explode cited_news to get each row_number in a separate row
        topics_df = topics_df.explode('cited_news')
        # Extract row_number from the cited_news dictionary
        topics_df['row_number'] = topics_df['cited_news'].apply(lambda x: x['row_number'])

        # Merge back with the original DataFrame to retain all information
        merged_df = pd.merge(reports, topics_df, on='row_number', how='left')
        
        # print(merged_df)
        # Generate summaries for each topic grouped by date and topic
        summaries = generate_topic_summaries_by_group(merged_df, model, api_key)
        summaries = pd.DataFrame(summaries)

        # Merge summaries back with the original DataFrame
        merged_summaries_df = pd.merge(merged_df, summaries, on=['date', 'topic'], how='left')

        # Generate the day in review for each date
        day_in_reviews = generate_day_in_review_by_date(merged_summaries_df, model, api_key)
        day_in_reviews_df = pd.DataFrame(day_in_reviews)
    
        
        date_final_df = pd.merge(merged_summaries_df, day_in_reviews_df, on='date', how='left')
        final_merged_df = pd.concat([final_merged_df, date_final_df], ignore_index=True)

    return final_merged_df

async def fetch_relevance(client, text_to_analyze, current_prompt, model, limiter):
    async with limiter:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": current_prompt},
                    {"role": "user", "content": text_to_analyze}
                ],
                temperature=0
            )

            relevance = eval(response.model_dump()['choices'][0]['message']['content'].strip())
            return relevance['relevance']
        except Exception as e:
            print(response.model_dump()['choices'][0]['message']['content'])
            print(f"Failed to fetch relevance for text: {text_to_analyze}\nError: {str(e)}")
            return None

async def filter_irrelevant_news(df, model, api_key):
    df = df.reset_index(drop=True)
    df['row_number'] = df.index

    filter_prompt = (
        f"You are an expert analyst specializing in the crude oil market. "
        f"Your task is to filter out news articles that are not related to the crude oil market. "
        f"The following text contains a news article including its Timestamp, Source, Headline, Text, and Row Number. "
        f"Your job is to determine if the content of the article is relevant to the crude oil market. "
        f"Consider the article relevant if it specifically discusses or impacts the crude oil market, including aspects like production, supply, demand, prices, geopolitical events affecting oil, and major industry trends.\n\n"
        f"Instructions:\n"
        f"1. Analyze the news article provided in the text.\n"
        f"2. If the article is relevant to the crude oil market, return a JSON object with 'relevance': 'Relevant'.\n"
        f"3. If the article is not relevant, return a JSON object with 'relevance': 'Irrelevant'.\n\n"
        f"Output:\n"
        f"Return a JSON object with a single key 'relevance' and its value should be either 'Relevant' or 'Irrelevant'.\n\n"
        f"Example Output:\n"
        f"{{'relevance': 'Relevant'}}\n"
        f"{{'relevance': 'Irrelevant'}}\n"
    )

    relevant_rows = []

    semaphore = asyncio.Semaphore(100)  # Limit to 10 concurrent tasks
    limiter = AsyncLimiter(max_rate=100, time_period=1)  # Limit to 10 requests per second

    client = AsyncOpenAI(api_key=api_key)  # Initialize AsyncOpenAI client

    async with aiohttp.ClientSession() as session:
        tasks = []
        for _, row in df.iterrows():
            text_to_analyze = row['text']
            current_prompt = filter_prompt.replace('[N]', '1')
            task = fetch_relevance(client, text_to_analyze, current_prompt, model, limiter)
            tasks.append(task)

        # Use tqdm to show a progress bar
        results = []
        for task in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Filtering News"):
            result = await task
            results.append(result)

        # Collect relevant row numbers based on the results
        for i, relevance in enumerate(results):
            if relevance == 'Relevant':
                relevant_rows.append(df.loc[i, 'row_number'])

    # Filter the DataFrame based on relevant row numbers
    filtered_df = df[df['row_number'].isin(relevant_rows)]

    # Remove the 'row_number' column before returning the DataFrame
    filtered_df = filtered_df.drop(columns=['row_number'])

    return filtered_df

