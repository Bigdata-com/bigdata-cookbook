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
import numpy as np

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

def search_keyword_and_filter_headline(report_headline_keywords, start_query, end_query, report_headlines_list,bigdata_credentials, job_number = 10, limit=750):
    
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

def extract_trending_topics(text_to_analyze, model, yearmonth, number_of_reports, api_key, main_theme):
    
    top_topics_prompt = (
    f""""You are an expert analyst tasked with identifying the specific and distinct underlying causes of market movements in the {main_theme} based on a set of daily news reports.

    Your goal is to analyze the provided text and determine the key reasons behind the market's movements for the date [DATE]. Focus on identifying distinct causes or events that drove the market. Avoid combining different causes into broad categories; instead, create separate topics for each specific cause or event, even if they are related to the same news.

Text Description:
- The input text includes [N] market reports, each explicitly structured with its Timestamp, Source, Headline, Text, and Row Number. Each report is clearly separated by a '--- Report End ---' marker. The association between the Source, Headline, and Text must be preserved and not altered.

Instructions:
1. **The Day in Review:** Summarize the key events of the day in bullet points. Focus on the most important market-moving events and data points, particularly those that explain why the market moved.

2. **Identify and Analyze Specific Causes of Market Movements:** Extract the most frequently mentioned specific events, data points, or geopolitical developments across all reports that directly influenced the {main_theme}. For each identified topic, provide a **detailed, in-depth analytical summary** that explains why this topic is significant for the {main_theme} on [DATE]. The summary should synthesize information from all related news reports, offering a comprehensive explanation of the specific causes and implications for market movements.

3. **Cite All Related News:** For each identified cause or driver, list **all** news items that are relevant to the topic by their Row Number. Do not include the full text, headline, or source—just the Row Number.

Output:
1. **Key Drivers of Market Movements:** Return a list of the main causes or drivers of market movements, each accompanied by a **detailed, in-depth analytical summary** based on all related news reports.
2. **Cited News:** For each driver, return a list of **all** news items relevant to that topic, identified only by their Row Numbers. Also, include the **total number of news items** related to each topic.
3. **Day in Review:** Include a list of bullet points summarizing the key events of the day as part of the output.

Format:
Return the response as a JSON object with the following structure:
{{
    "date": "[DATE]",
    "trending_topics": [
        {{
            "day_in_review": [
                "Unique Bullet point 1",
                "Unique Bullet point 2",
                "..."
            ],
            "topic": "Specific Cause or Event",
            "summary": "A single detailed, in-depth analytical summary explaining why this specific cause or event influenced the {main_theme} on [DATE].",
            "number_of_news": X,  # Total number of news items related to this topic
            "cited_news": [
                {{"row_number": Y}},
                {{"row_number": Z}},
                // All related row numbers...
            ]
        }},
        // More specific causes...
    ]
}}
"""
)

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Prepare the prompt
    top_topics_prompt_1 = top_topics_prompt.replace('[DATE]', yearmonth)
    top_topics_prompt_1 = top_topics_prompt_1.replace('[N]', str(number_of_reports))
    
    # Send request to OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": top_topics_prompt_1
            },
            {
                "role": "user",
                "content": text_to_analyze
            }
        ],
        response_format={ "type":"json_object"},
        temperature=0
    )
    
    # Extract and process the response
    summary = response.model_dump()

    # Parse the content assuming it is a JSON-like structure
    trending_topics = eval(summary['choices'][0]['message']['content'])['trending_topics']

    return trending_topics

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
                    'Text': text
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
    df['Text_summary'] = df.apply(lambda row: generate_text_summary(row['Text'], row['Topic'], api_key), axis=1)
    return df

def generate_advanced_novelty_score(topic, previous_topics_dict, main_theme, api_key, model='gpt-4o-mini-2024-07-18'):
    client = openai.Client(api_key=api_key)
    
    # Format the previous topics dictionary into a string for the prompt
    previous_topics_formatted = "\n".join([f"{date}: {', '.join(topics)}" for date, topics in previous_topics_dict.items()])
    
    # Create the prompt that compares the current topic with previous topics
    prompt = (
    f"You are an expert analyst tasked with determining the novelty of topics in the {main_theme}. "
    f"The current topic is: '{topic}'. Below are the topics that have been mentioned on previous days:\n\n"
    f"{previous_topics_formatted if previous_topics_formatted else 'No previous topics provided.'}\n\n"
    f"**Important Considerations:**\n"
    f"1. **Temporal Distance**: Consider how many days have passed since the last mention of similar topics. If a topic hasn't been mentioned for several days and reappears with new information or updates, it may be considered 'New' again.\n"
    f"2. **Topic Specificity**: Distinguish between general topics (e.g., 'Market Uncertainty') and more specific elements (e.g., 'Venezuelan Election'). If a topic combines familiar elements with new, specific details, assess the novelty based on the new elements.\n"
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
            {"role": "system", "content": "You are an expert market analyst specializing in the analysis of {main_theme}."},
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

def add_advanced_novelty_scores_to_df(df, main_theme, api_key):
    df = df.sort_values(by='Date')
    novelty_scores = []
    
    # Dictionary to store previous topics by date
    previous_topics_dict = {}
    
    # Iterate through each unique date in the DataFrame
    for date in tqdm(df['Date'].unique()):
        # Update previous topics dictionary (topics from all previous days)
        previous_topics = df[df['Date'] < date].groupby('Date')['Topic'].unique().to_dict()
        
        # Filter the DataFrame for the current date
        daily_df = df[df['Date'] == date]
        
        # Process each topic on the current date
        for _, row in daily_df.iterrows():
            topic = row['Topic']
            
            # Generate the novelty score using the OpenAI API, comparing with previous topics
            novelty_score = generate_advanced_novelty_score(topic, previous_topics, main_theme, api_key)
            novelty_scores.append(novelty_score)
        
        # Add current day's topics to the previous_topics_dict for the next day's comparison
        previous_topics_dict[date] = daily_df['Topic'].unique().tolist()
    
    # Add the novelty scores to the DataFrame
    df['Novelty_Score'] = novelty_scores
    
    return df

def generate_market_impact_and_magnitude(text, topic, main_theme, api_key, model='gpt-4o-mini-2024-07-18'):
    client = openai.Client(api_key=api_key)
    
    # Create the prompt for determining market impact and magnitude with a focus on oil prices
    prompt = (
    f"You are an expert market analyst with deep knowledge of the {main_theme}, tasked with evaluating the impact of the topic '{topic}' from the perspective of a stakeholder in this market. "
    f"Consider both the topic and the provided text to assess how this information might influence market dynamics, including any potential effects on key factors such as pricing, investor sentiment, and economic indicators within the {main_theme}. "
    f"Keep in mind the following considerations:\n"
    f"- Positive Impact: The topic is expected to drive favorable market conditions, such as an increase in prices, improvement in market sentiment, or strengthening of key economic indicators.\n"
    f"- Negative Impact: The topic is likely to create adverse market conditions, such as a decrease in prices, deterioration in market sentiment, or weakening of key economic indicators.\n"
    f"- Neutral Impact: The topic is not expected to cause significant changes in the market conditions.\n\n"
    f"Based on these considerations, determine whether the impact on the market is Positive, Negative, or Neutral. "
    f"Additionally, evaluate the magnitude of this impact as follows:\n"
    f"- High: The topic is expected to have a significant impact on the overall market.\n"
    f"- Medium: The topic is likely to have a moderate impact, affecting certain sectors or segments of the market.\n"
    f"- Low: The topic is expected to have a limited impact, possibly affecting only specific elements of the market.\n"
    f"- Neutral: No significant impact is anticipated.\n\n"
    f"Consider both direct and indirect effects, including potential shifts in investor behavior, regulatory responses, and broader economic implications.\n\n"
    f"Text: {text}\n\n"
    f"Make your decision by carefully analyzing both the topic and the text. "
    f"Provide your analysis and return **ONLY** the result as an object with the following structure:\n"
    f'{{ "impact": "Positive" | "Negative" | "Neutral", "magnitude": "High" | "Medium" | "Low" | "Neutral" }}'
)




    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert market analyst specializing in the analysis of {main_theme}."},
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

def add_market_impact_to_df(df, main_theme, api_key):
    df = df.sort_values(by='Date')
    impact_magnitude_dict = {}
    
    # Group by Topic and Date to generate a single impact and magnitude score for each topic
    grouped = df.groupby(['Date', 'Topic'])
    
    for (date, topic), group in tqdm(grouped):
        # Concatenate all text for the topic on the given date
        # text_list = group['Text'].fillna("").astype(str).tolist()
        
        # Concatenate all text for the topic on the given date
        # combined_text = " ".join(text_list)
        
        combined_text = group['Summary']
        
        # Generate the impact and magnitude scores using the OpenAI API
        impact_magnitude = generate_market_impact_and_magnitude(combined_text, topic, main_theme, api_key)
        
        # Store the scores in the dictionary
        impact_magnitude_dict[(date, topic)] = impact_magnitude
    
    # Map the impact and magnitude back to the DataFrame
    df['Impact'] = df.apply(lambda row: impact_magnitude_dict[(row['Date'], row['Topic'])]['impact'], axis=1)
    df['Magnitude'] = df.apply(lambda row: impact_magnitude_dict[(row['Date'], row['Topic'])]['magnitude'], axis=1)
    
    return df

# Function to clean and normalize text
def clean_text(text):
    # Normalize the text to remove any weird encodings
    text = unicodedata.normalize('NFKD', text)
    # Replace dollar signs with the word 'dollars'
    text = text.replace('$', '\$')
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

# Function to prepare the data for the report with customizable ranking and optional date filter
def prepare_data_for_report(df, ranking_criteria, report_date: Optional[str] = None):
    reports = []
    criteria_map = {
        'novelty': lambda x: novelty_score_value(x['novelty_score']),
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
            'Magnitude': 'first',
            'Impact': 'first'
        }).reset_index()
        
        # Extract "Day in Review" for the report from the first topic (assuming it is consistent across topics)
        day_in_review = top_topics['Day_in_Review'].iloc[0].split('\n') if 'Day_in_Review' in top_topics.columns else []
        
        for _, topic_row in top_topics.iterrows():
            topic = topic_row['Topic']
            summary = topic_row['Summary']
            novelty_score = topic_row['Novelty_Score']
            magnitude = topic_row['Magnitude']
            impact = topic_row['Impact']
            associated_news = group[group['Topic'] == topic].sort_values(by='Source', ascending=True)
            
            sources_and_summaries = associated_news[['Source', 'Text_summary']].values.tolist()
            
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

def process_trending_topics(unique_reports, start_query, end_query, main_theme, model='gpt-4o-mini-2024-07-18', api_key=None):
    # Initialize your date range
    start_list, end_list = create_dates_lists(pd.to_datetime(start_query), pd.to_datetime(end_query), freq='D')

    # DataFrames to store results
    trending_topics_df = pd.DataFrame(columns=['Date', 'trending_topics'])

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
    dates = [datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').date() for date_str in start_list]

    # Iterate over each date in the range
    for period in tqdm(dates):
        reports = unique_reports.loc[unique_reports['Date'] == period]
        trending_topics_dict = {}
        text_to_analyze = "\n\n".join(reports['text_with_reference'])
        yearmonth = period.strftime('%Y-%m-%d')
        number_of_reports = len(reports)

        if number_of_reports == 0:
            continue

        # Extract trending topics instead of a summary
        trending_topics = extract_trending_topics(text_to_analyze, model, yearmonth, number_of_reports, api_key, main_theme)

        trending_topics_dict['Date'] = period
        trending_topics_dict['trending_topics'] = trending_topics

        trending_topics_df = pd.concat([trending_topics_df, pd.DataFrame([trending_topics_dict])], ignore_index=True)

    return trending_topics_df