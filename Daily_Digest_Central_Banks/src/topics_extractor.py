from datetime import datetime
import re
import pandas as pd
import asyncio
import os
from tqdm.notebook import tqdm
import json
import openai
from openai import OpenAIError
import os
import jinja2
from IPython.core.display import HTML
import pandas as pd
import unicodedata
import re
from typing import Optional
import numpy as np
from tqdm.notebook import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio
import ast


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
                # matching_stats['total_citations'] += 1
                # matching_stats['by_date'][date_str]['total_citations'] += 1
                
                # Find the original entry in the original_df based on the row number
                matching_row = original_df.loc[original_df['row_number'] == row_number]

                # If a match is found, extract the relevant details; otherwise, assign NaN
                if not matching_row.empty:
                    source = matching_row['source_name'].values[0]
                    headline = matching_row['headline'].values[0]
                    text = matching_row['text'].values[0]
                    # matching_stats['successful_matches'] += 1
                    # matching_stats['by_date'][date_str]['successful_matches'] += 1
                else:
                    source = np.nan  # Assign NaN to source if no match is found
                    headline = np.nan
                    text = np.nan
                    # matching_stats['failed_matches'] += 1
                    # matching_stats['by_date'][date_str]['failed_matches'] += 1
                    # if len(matching_stats['failed_row_numbers']) < 20:  # Limit to first 20
                    #     matching_stats['failed_row_numbers'].append(row_number)
                
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


async def generate_text_summary(text, topic, api_key, model='gpt-4o-mini-2024-07-18'):
    client = openai.AsyncOpenAI(api_key=api_key, timeout=25)

    # Create the prompt incorporating the topic
    prompt = (
        f"You are an expert summarizer. Given the following text, please provide a concise, one-row summary relevant to the topic '{topic}'. "
        f"The summary should be part of a larger analysis report that explains the significance of this text within the context of {topic}.\n\n"
        f"Text: {text}\n\n"
        f"Return the result as a JSON object with the following structure:\n"
        f'{{ "summary": "Your summary here"}}'
    )

    # Asynchronous request to OpenAI API
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": prompt}
        ],
        response_format={ "type":"json_object"},
        temperature=0
    )

    summary = response.model_dump()

    try:
        summary_dict = ast.literal_eval(summary['choices'][0]['message']['content'])['summary']
    except:
        summary_dict = ''
    
    return summary_dict

# Asynchronous function to process DataFrame rows
async def add_text_summaries_to_df(df, api_key, model='gpt-4o-mini-2024-07-18'):
    # Create a dictionary to hold the mapping of Text to summaries
    text_to_summary = {}

    # Define an async task for generating summaries
    async def summarize_text(text, topic):
        summary = await generate_text_summary(text, topic, api_key, model)
        return text, summary

    # Step 1: Create a list of tasks for each row where text has not been summarized
    tasks = []
    for idx, row in df.iterrows():
        text = row['Text']
        topic = row['Topic']  # Get the topic for the current row
        
        if text not in text_to_summary:
            tasks.append(summarize_text(text, topic))

    # Step 2: Execute all tasks concurrently with a progress bar
    results = await tqdm_asyncio.gather(*tasks, desc="Generating text summaries", total=len(tasks))

    # Step 3: Store the summaries in the dictionary
    text_to_summary = {text: summary for text, summary in results}

    # Step 4: Map the generated summaries back to the DataFrame
    df['Text_Summary'] = df['Text'].map(text_to_summary)

    return df


async def generate_advanced_novelty_score(client, topic, previous_topics_dict, main_theme, model='gpt-4o-mini-2024-07-18'):
    # Format the previous topics dictionary into a string for the prompt
    previous_topics_formatted = "\n".join([f"{date}: {', '.join(topics)}" for date, topics in previous_topics_dict.items()])

    # Create the prompt that compares the current topic with previous topics
    prompt = (
        f"You are an expert analyst tasked with determining the novelty of topics in the context of '{main_theme}'. "
        f"The current topic is: '{topic}'. Below are the topics that have been mentioned on previous days:\n\n"
        f"{previous_topics_formatted if previous_topics_formatted else 'No previous topics provided.'}\n\n"
        f"**Important Considerations:**\n"
        f"1. **Temporal Distance**: Consider how many days have passed since the last mention of similar topics. If a topic hasn't been mentioned for several days and reappears with new information or updates, it may be considered 'New' again.\n"
        f"2. **Topic Specificity**: Distinguish between general topics and more specific elements. If a topic combines familiar elements with new, specific details, assess the novelty based on the new elements.\n"
        f"3. **Combining and Differentiating Topics**: If a topic includes elements that have been discussed previously but introduces new aspects or a different context (e.g., new developments, discoveries), weigh the novelty based on the new components.\n\n"
        f"**Decision Criteria:**\n"
        f"- If there are no previous topics provided, or if the current topic introduces new elements after a significant time gap, consider it 'New'.\n"
        f"- If the topic reappears but with additional new information or developments, consider it 'Moderate'.\n"
        f"- If the topic has been mentioned recently and repeatedly without significant new information, consider it 'Old'.\n\n"
        f"Return the response as an object with the following structure:\n"
        f'{{ "novelty_score": "Old" | "Moderate" | "New" }}'
    )

    # Asynchronous request to OpenAI API
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    try:
        novelty_score_dict = json.loads(response.model_dump()['choices'][0]['message']['content'])['novelty_score']
    except:
        novelty_score_dict = "Error"  # Default to "Error" if there's an issue
    
    return novelty_score_dict

async def add_advanced_novelty_scores_to_df(df, api_key, main_theme):
    df = df.sort_values(by='Date')
    
    client = openai.AsyncOpenAI(api_key=api_key, timeout=25)

    # Dictionary to store previous topics by date
    previous_topics_dict = {}
    
    topic_date_pairs = []
    tasks = []

    # Iterate through each unique date in the DataFrame
    for date in df['Date'].unique():
        
        # Get previous topics (all topics from previous days)
        previous_topics = df[df['Date'] < date].groupby('Date')['Topic'].unique().to_dict()
        # Get unique topics for the current date
        daily_df = df[df['Date'] == date]
        unique_topics = daily_df['Topic'].unique()

        for topic in unique_topics:
            task = generate_advanced_novelty_score(client, topic, previous_topics, main_theme)
            tasks.append(task)
            topic_date_pairs.append((date, topic))

        # Add current day's topics to the previous_topics_dict for the next day's comparison
        previous_topics_dict[date] = daily_df['Topic'].unique().tolist()

    # Execute all tasks asynchronously with progress bar
    novelty_scores = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Calculating Novelty Scores"):
        novelty_scores.append(await task)

    # Build a mapping from (date, topic) to novelty_score
    novelty_score_map = {
        (date, topic): score for (date, topic), score in zip(topic_date_pairs, novelty_scores)
    }

    # Map novelty scores back to the DataFrame
    df['Novelty_Score'] = df.apply(lambda row: novelty_score_map.get((row['Date'], row['Topic']), "Error"), axis=1)

    return df

# Function to run the asynchronous process
def run_add_advanced_novelty_scores(df, api_key, main_theme):
    return asyncio.run(add_advanced_novelty_scores_to_df(df, api_key, main_theme))

def generate_market_impact_and_magnitude(client, text, topic, main_theme, point_of_view, model='gpt-4o-mini-2024-07-18'):
    # Create the prompt for determining market impact and magnitude
    prompt = (
        f"You are an expert market analyst with deep knowledge of the {point_of_view} market, tasked with evaluating the impact of the topic '{topic}' within the context of '{main_theme}'. "
        f"Consider both the topic and the provided text to assess how this information might influence key factors relevant to the {main_theme} and the {point_of_view} market. "
        
        f"Evaluate the **impact** as follows:\n"
        f"- **Positive Impact**: If the information is likely to lead to a price increase or growth in value within the {point_of_view} market.\n"
        f"- **Negative Impact**: If the information is expected to lead to a price decline or reduction in value within the {point_of_view} market.\n\n"
        
        f"Next, evaluate the **magnitude** of this impact:\n"
        f"- **High**: The topic is expected to have a significant, market-wide impact, potentially influencing major segments of the {point_of_view} market within the context of {main_theme}.\n"
        f"- **Medium**: The topic is likely to have a moderate impact, affecting certain regions, sectors, or segments within the market.\n"
        f"- **Low**: The topic is expected to have a limited impact, affecting only a small portion or segment of the {point_of_view} market.\n"
        f"- **Neutral**: No significant impact on the {point_of_view} market is anticipated.\n\n"

        f"Consider both direct and indirect effects, including economic indicators, market sentiment, geopolitical risks, and potential shifts in supply and demand dynamics related to {main_theme}.\n\n"
        
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

def add_market_impact_to_df(df, api_key, main_theme, point_of_view):
    df = df.sort_values(by='Date')
    impact_magnitude_dict = {}
    
    # Group by Topic and Date to generate a single impact and magnitude score for each topic
    grouped = df.groupby(['Date', 'Topic'])
    
    client = openai.Client(api_key=api_key)
    
    for (date, topic), group in grouped:
        # Concatenate the 'Summary' text for the topic on the given date
        combined_text = topic
        
        # Generate the impact and magnitude scores using the OpenAI API
        impact_magnitude = generate_market_impact_and_magnitude(client, combined_text, topic, main_theme, point_of_view)
        
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
    
    # Check if the columns are already in the correct order
    if current_columns[-5:] != final_column_order:
        # Reorder the columns by placing the 'final_column_order' at the end
        reordered_columns = current_columns[:-5] + final_column_order
        df = df[reordered_columns]
    
    return df


# Function to run the asynchronous process
def run_add_market_impact(df, api_key, main_theme, point_of_view):
    return asyncio.run(add_market_impact_to_df(df, api_key, main_theme, point_of_view))

async def extract_trending_topics(text_to_analyze, model, yearmonth, number_of_reports, api_key, main_theme):
    top_topics_prompt = (
    f""""You are an expert analyst tasked with identifying the key topics and specific drivers that influence recent news reports related to the {main_theme}.

        Your goal is to analyze the provided text and determine the most important and distinct **drivers and topics** that are influencing developments around the {main_theme}, with a specific focus on identifying **precise entities and causes** (such as organizations, people, countries, companies, or financial institutions) that are central to these drivers. Only include news items that are directly relevant to the {main_theme}, ensuring the focus is on specific drivers rather than broad categories.

    Text Description:
    - The input text includes [N] news reports, each explicitly structured with its Timestamp, Source, Text, and Row Number. Each report is clearly separated by a '--- Report End ---' marker. The association between the Source and Text must be preserved and not altered.

    Instructions:
    1. **Identify and Analyze Key Drivers and Topics:** Extract the most frequently mentioned **specific entities and driving factors** (organizations, individuals, countries, economic indicators, market forces, geopolitical events, etc.) from the reports. Use these entities and drivers to form distinct topics that explain **why** they are significant for the {main_theme}, ensuring that each topic is based on specific **causes** and **drivers**. 
       - Avoid combining different topics into broad categories like "market trends" or "price fluctuations." Instead, focus on **precise drivers** (e.g., "US interest rate hikes" or "China’s weak demand") and create separate topics for each **specific cause**.
    2. **Driver-Centric Topic Summaries:** For each identified entity or driver, provide a **detailed, in-depth analytical summary** explaining why this entity or factor is driving the development related to the {main_theme}. The summary should synthesize information from all related news reports, offering a comprehensive explanation of the specific **causes**, entities, or developments and their implications.
    3. **Filter Out Unrelated News:** Only include news items that are directly related to the {main_theme} and focus on specific drivers. Do not include reports or summaries that are only loosely connected or lack a clear driver.
    4. **Cite All Related News:** For each identified entity, driver, or topic, list **all** news items that are relevant to that topic by their Row Number. Do not include the full text or source—just the Row Number.

    Output:
    1. **Key Drivers and Entities Related to the {main_theme}:** Return a list of the main driving factors or developments related to the {main_theme}, each accompanied by a **detailed, in-depth analytical summary** based on all related news reports. Ensure that each topic is strongly associated with key **specific entities and driving factors** identified from the reports. Exclude unrelated news.
    2. **Cited News:** For each topic, return a list of **all** news items relevant to that topic, identified only by their Row Numbers. Also, include the **total number of news items** related to each topic.

    Format:
    Return the response as a JSON object with the following structure:
    {{
        "date": "[DATE]",
        "trending_topics": [
            {{
                "topic": "Key Topic or Driver",
                "summary": "A single detailed, in-depth analytical summary explaining the importance of this specific entity or driver related to the {main_theme}.",
                "number_of_news": X,  # Total number of news items related to this topic
                "cited_news": [
                    {{"row_number": Y}},
                    {{"row_number": Z}},
                    // All related row numbers...
                ]
            }},
            // More key topics or drivers...
        ]
    }}
    """
)
    client = openai.AsyncOpenAI(api_key=api_key, timeout=120)

    try:
        # Prepare the prompt
        top_topics_prompt_1 = top_topics_prompt.replace('[DATE]', yearmonth)
        top_topics_prompt_1 = top_topics_prompt_1.replace('[N]', str(number_of_reports))
    
        # Send asynchronous request to OpenAI API
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": top_topics_prompt_1},
                {"role": "user", "content": text_to_analyze}
            ],
            response_format={ "type":"json_object"},
            temperature=0
        )

        # Extract and process the response
        summary = response.model_dump()

        # Parse the content assuming it is a JSON-like structure
        trending_topics = ast.literal_eval(summary['choices'][0]['message']['content'])['trending_topics']

        return trending_topics

    except OpenAIError as e:
        print(f"Timeout error: {e}. Retrying...")
        return await extract_trending_topics(text_to_analyze, model, yearmonth, number_of_reports, api_key, main_theme)

# Helper function to consolidate with automatic retry and batch splitting
async def consolidate_trending_topics_with_retry(batch_str, model, api_key, main_theme, max_retries=3, min_batch_size=10):
    """
    Wrapper for consolidate_trending_topics with automatic retry and batch size reduction.
    
    In case of JSONDecodeError, split the batch in half and retry recursively.
    """
    # Parse batch_str into list of (topic_id, topic_name)
    items = []
    for item in batch_str.split(', '):
        if ': "' in item:
            topic_id = item.split(': "')[0].strip()
            topic_name = item.split(': "')[1].rstrip('"')
            items.append((topic_id, topic_name))
    
    current_batch_size = len(items)
    
    # If the batch is too small, we cannot split it further
    if current_batch_size <= min_batch_size:
        return await consolidate_trending_topics(batch_str, model, api_key, main_theme)
    
    # Try consolidation
    try:
        return await consolidate_trending_topics(batch_str, model, api_key, main_theme)
    
    except json.JSONDecodeError as e:
        print(f"⚠️  JSONDecodeError with batch size {current_batch_size}. Splitting batch in half...")
        
        # Split the batch in half
        mid = len(items) // 2
        batch1_items = items[:mid]
        batch2_items = items[mid:]
        
        # Rebuild batch strings
        batch1_str = ', '.join([f'{tid}: "{tname}"' for tid, tname in batch1_items])
        batch2_str = ', '.join([f'{tid}: "{tname}"' for tid, tname in batch2_items])
        
        print(f"   Batch 1: {len(batch1_items)} items")
        print(f"   Batch 2: {len(batch2_items)} items")
        
        # Process the two sub-batches recursively
        result1 = await consolidate_trending_topics_with_retry(
            batch1_str, model, api_key, main_theme, max_retries, min_batch_size
        )
        result2 = await consolidate_trending_topics_with_retry(
            batch2_str, model, api_key, main_theme, max_retries, min_batch_size
        )
        
        # Merge results
        merged_result = {**result1, **result2}
        return merged_result


# Asynchronous function to consolidate trending topics
async def consolidate_trending_topics(batch_results, model, api_key, main_theme):
    consolidation_prompt = (
    f"""You are an expert analyst tasked with consolidating unique topics related to {main_theme}.
    
    Your goal is to unify topics based on their **underlying causes or drivers** that represent the same or highly similar reasons for developments within the theme {main_theme}. Only merge topics that refer to the **exact same underlying cause or driver**, ensuring you account for specific contexts (e.g., different regions or conflicts) even when the broader cause (like "war" or "economic slowdown") may seem similar.

    Guidelines:

    1. **Identify Similar Topics Based on Specific Causes and Drivers:** Review the topic labels and identify those that are **strictly related** based on shared **specific underlying causes or drivers** (e.g., US oil inventories drop, Russia-Ukraine War, Chinese weak demand). 
       - **Only consolidate** topics that refer to the **same driver or cause in the same context** (e.g., "US oil inventories drop" or "China’s weak demand"). Avoid consolidating based on broad categories like "war" or "economic slowdown"—instead, focus on the **specific context** such as region, country, or event.
       - Focus on shared **precise causes or drivers** (e.g., macroeconomic factors, geopolitical events, company actions, specific conflicts) in the topic labels to determine whether consolidation is appropriate.
    2. **Handle Duplicate Topics with Different Labels:** If two or more topics refer to the same cause but are labeled differently, merge them into a **single standardized label** that reflects the shared driver or cause, **only if the context is the same**.
       - Example: "China’s Slow Growth Hurts Oil Prices" and "Oil Prices Drop as China’s Demand Weakens" could be merged into "China’s Weak Demand Causes Oil Price Drop." However, "War in Ukraine Affects Gas Prices" should not be merged with "Conflict in the Middle East Affects Oil Supply" because they involve different contexts.
    3. **Avoid Overgeneralizing Based on Broad Outcomes or Drivers:** Do not consolidate topics based solely on general drivers like "war" or "economic slowdown." Each **specific cause** must remain distinct (e.g., separate conflicts in different regions or specific economic factors in different countries) unless they are part of the **same underlying context**.
    4. **Consolidate Topics Based on Precise Drivers:** Merge topics only when they clearly share the **exact same cause or driver in the same context** (e.g., region, conflict, economic factor), creating a single, coherent label that emphasizes the **why** behind the development.
    5. **Provide a Mapping for Original Topics:** For each original topic label, map all the original topics linked to the same new, consolidated topic.

    Output:
    Return the response as a JSON object with the following structure:
    {{
        "consolidated_topics": {{
            "consolidated_topic_1": [
                "original_topic_1",
                "original_topic_3"
            ],
            "consolidated_topic_2": [
                "original_topic_2",
                "original_topic_4"
            ],
            // More consolidated topics...
        }}
    }}
    """
    )

    # Initialize OpenAI client
    client = openai.AsyncOpenAI(api_key=api_key)
    
    # Send asynchronous request to OpenAI API
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": consolidation_prompt
            },
            {
                "role": "user",
                "content": json.dumps(batch_results)
            }
        ],
        response_format={ "type":"json_object"},
        #max_tokens=4000,
        temperature=0
    )
    
    # Extract and process the response
    summary = response.model_dump()

    # # Parse the content assuming it is a JSON-like structure
    # consolidated_topics = ast.literal_eval(summary['choices'][0]['message']['content'])['consolidated_topics']
    
    try:
        content = json.loads(summary['choices'][0]['message']['content'])
        consolidated_topics = content.get('consolidated_topics', {})
        
    except json.JSONDecodeError as e:
        # Re-raise the JSON error to handle it at a higher level with retry
        raise
    except Exception as e:
        print(f"⚠️  Unexpected error in consolidate_trending_topics: {e}")
        raise

    return consolidated_topics


async def summarize_grouped_summaries(df, model, api_key, main_theme, yearmonth):
    # Step 1: Group by 'Topic' and gather unique summaries for each topic
    grouped_summaries = df.groupby('Topic')['Summary'].unique().to_dict()

    summarized_topics = {}

    # Define a task function for summarization
    async def summarize_topic(topic, summaries):
        # Combine all summaries into one string
        summaries_text = ' '.join(summaries)

        # Define the prompt to ask the LLM to summarize the summaries centered around the topic
        summarization_prompt = (
            f"""You are an expert analyst tasked with writing a concise summary centered around the topic "{topic}".
            
            Here are the summaries related to this topic:
            {summaries_text}

            Your task:
            1. Write a single, coherent summary that captures the key points from all the provided summaries.
            2. Ensure the summary is centered on the topic "{topic}" and highlights important entities, events, or trends related to it.
            3. The final summary should be clear, concise, and informative, without deviating from the topic.

            Output:
            Return the response as a JSON object with the structure: {{ "summary": "<your final summary here>" }}.
            """
        )

        # Initialize OpenAI client (same as before)
        client = openai.AsyncOpenAI(api_key=api_key)

        # Send the prompt to the LLM
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": summarization_prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        # Extract the LLM's response
        summary_response = response.model_dump()

        # Parse the JSON-like structure to extract the summary
        summary_content = json.loads(summary_response['choices'][0]['message']['content'])['summary']

        # Return the summarized content for the topic
        return topic, summary_content

    # Step 2: Create a list of tasks for each topic
    tasks = [summarize_topic(topic, summaries) for topic, summaries in grouped_summaries.items()]

    # Step 3: Execute all tasks concurrently with tqdm for progress tracking
    results = await tqdm_asyncio.gather(*tasks, desc="Summarizing topics", total=len(tasks))

    # Step 4: Store the summarized content for each topic
    summarized_topics = {topic: summary for topic, summary in results}

    # Step 5: Replace the summaries in the original dataframe
    df['Summary'] = df['Topic'].map(summarized_topics)

    return df

async def generate_title_for_summary(summary, model, api_key):
    title_prompt = (
    f"""You are an expert tasked with generating a journalistic, engaging, and driver-focused title for this report summary.

    Your goal is to create a concise and compelling title that captures the **core drivers** and **key points** discussed in the summary, focusing on the **primary cause or driver** behind the event. Avoid including secondary or unrelated details that are not central to the main topic.

    Guidelines:
    1. **Focus on the Core Driver and Main Event:** Generate a title that captures the most important driver or cause behind the main event in the summary. Ensure the title remains tightly aligned with the main topic and avoids extraneous details (e.g., "geopolitical tensions" if they are not a primary driver).
    2. **Include Important Entities and Key Values Only When Relevant:** Mention the key organizations, companies, individuals, or financial assets involved, but only if they are central to the event. If the summary includes numerical data (e.g., interest rates, percentages, stock prices), include these values to provide more context, but do not overpopulate the title with secondary information.
    3. **Avoid Irrelevant or Secondary Information:** Ensure the title is focused and does not include information that is unrelated or not significant to the main theme of the summary. The title should be clean and concise.
    4. **Engage the Reader:** The title should sound natural and engaging, like a headline from a leading news outlet, and should spark interest without becoming too technical or overloaded with unnecessary details.

    Here is the summary to generate a title for:
    {summary}

    Output:
    Return the response as a JSON object with the following structure:
    {{
        "summary": "{summary}",
        "title": "Generated journalistic and driver-focused title based on the summary."
    }}
    """
)


    # Initialize OpenAI client
    client = openai.AsyncOpenAI(api_key=api_key)

    # Send the prompt to the LLM to generate the title for a single summary
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": title_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    # Process the response
    result = response.model_dump()

    # Parse the JSON response to get the generated title
    title_data = json.loads(result['choices'][0]['message']['content'])

    return title_data['title']

# Function to create titles from unique summaries and map them back to the original DataFrame
async def create_titles_from_summaries(df, model='gpt-4o-mini', api_key=None):
    # Step 1: Extract unique summaries
    unique_summaries = df['Summary'].unique()

    # Step 2: Create a dictionary to store titles for each unique summary
    summary_to_title = {}

    # Define an async task for generating titles
    async def generate_title(summary):
        title = await generate_title_for_summary(summary, model, api_key)
        return summary, title

    # Step 3: Create a list of tasks for each unique summary
    tasks = [generate_title(summary) for summary in unique_summaries]

    # Step 4: Execute all tasks concurrently with a progress bar
    results = await tqdm_asyncio.gather(*tasks, desc="Generating titles", total=len(tasks))

    # Step 5: Store the generated titles in the dictionary
    summary_to_title = {summary: title for summary, title in results}

    # Step 6: Map the generated titles back to the original DataFrame
    # Replace the 'Topic' column with the titles based on the 'Summary' column
    df['Topic_labels'] = df['Topic']
    df['Topic'] = df['Summary'].map(summary_to_title)

    return df

def detect_entities_in_texts(texts, api_key, model):
    client = openai.OpenAI(api_key=api_key, timeout=15)

    entity_detection_prompt = (
        "You are an expert in detecting entities from text. "
        "Your task is to extract all entities such as countries, central banks, organizations, individuals, and financial assets (e.g., currencies, stocks, bonds, commodities). "
        "Return the entities in a JSON object with the structure: {'entities': ['...']}.\n\n"
    )

    entities_results = []
    for text in tqdm(texts, desc="Detecting Entities"):
        prompt = entity_detection_prompt + f"Text: {text}"
        result = fetch_entities(client, prompt, model)
        entities_results.append(result)

    return entities_results

# Fetch entities from LLM synchronously
def fetch_entities(client, prompt, model):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        entities_result = ast.literal_eval(response.model_dump()['choices'][0]['message']['content'])
        return entities_result['entities']
    except Exception as e:
        print(f"Error detecting entities: {e}")
        return []

# Check subject relevance (row by row)
def check_subject_relevance(df, matched_topics, model, api_key):
    client = openai.OpenAI(api_key=api_key, timeout=15)

    match_prompt_template = (
        "Forget about all the previous prompts."
        "You are an expert in financial analysis. "
        "You have been provided a topic and a text along with entities detected in both. "
        "Determine whether the subject of the topic and the text are closely related. "
        "Base your decision on the entities and the context of both the topic and the text. "
        "Return the output as a JSON object with the structure: {'related': 'Yes' or 'No', 'motivation': '...'}.\n\n"
    )

    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking Subject Relevance"):
        text_to_match = row['Text']

        # Get matched topic for this text
        if idx < len(matched_topics):
            matched_topic = matched_topics[idx]
        else:
            matched_topic = None

        # Only check relevance for matched topics
        if matched_topic:
            prompt = match_prompt_template + f"\n\nText: {text_to_match}\nTopic: {matched_topic}\n"
            result = fetch_subject_relevance(client, prompt, model)
            results.append(result)
        else:
            results.append({'related': 'No', 'motivation': 'No topic matched or entities aligned.'})

    return results

# Fetch subject relevance synchronously
def fetch_subject_relevance(client, prompt, model):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        relevance_result = ast.literal_eval(response.model_dump()['choices'][0]['message']['content'])
        return relevance_result
    except Exception as e:
        print(f"Error checking subject relevance: {e}")
        return {'related': 'No', 'motivation': 'Error during relevance check'}

# Main function to process the reports
def process_reports(df, model, api_key, main_theme):
    # Extract the list of topics and text summaries
    topics = df['Topic'].unique().tolist()
    texts = df['Text'].tolist()

    # Step 1: Detect entities from the topics (row by row)
    topic_entities = detect_entities_in_texts(topics, api_key, model)

    # Step 2: Detect entities from the text summaries (row by row)
    text_entities = detect_entities_in_texts(texts, api_key, model)

    # Step 3: Match text to topic based on shared entities
    matched_topics = []
    matched_entities = []
    for text_entity_list in text_entities:
        matched_topic = None
        matched_entity_set = set()
        for idx, topic_entity_list in enumerate(topic_entities):
            # Match if text entities overlap with topic entities
            entity_overlap = set(text_entity_list) & set(topic_entity_list)
            if entity_overlap:
                matched_topic = topics[idx]
                matched_entity_set = entity_overlap
                break
        matched_topics.append(matched_topic)
        matched_entities.append(matched_entity_set)

    # Ensure length of matched_topics matches df
    if len(matched_topics) < len(df):
        matched_topics.extend([None] * (len(df) - len(matched_topics)))
        matched_entities.extend([set()] * (len(df) - len(matched_entities)))

    # Step 4: Check subject relevance between matched topics and text (row by row)
    subject_relevance_results = check_subject_relevance(df, matched_topics, model, api_key)

    # Ensure motivations and relevance lists have the correct length
    motivations = [result['motivation'] for result in subject_relevance_results]
    relevance = [result['related'] == 'Yes' for result in subject_relevance_results]

    # Ensure the lengths match
    if len(motivations) < len(df):
        motivations.extend(["No motivation available"] * (len(df) - len(motivations)))
    if len(relevance) < len(df):
        relevance.extend([False] * (len(df) - len(relevance)))

    df['Matched_Topic'] = matched_topics
    df['Matched_Entities'] = matched_entities
    df['Motivation'] = motivations
    df['Relevance'] = relevance

    return df

# Entry point for processing the reports
def process_matching(reports, model, api_key, main_theme):
    matched_reports = process_reports(reports, model, api_key, main_theme)
    return matched_reports

async def phase1_extract_topics(unique_reports, start_query, end_query, main_theme, model='gpt-4o-mini', api_key=None, batches=5, freq='D'):
    """
    PHASE 1: Topic Extraction
    Extracts topics from reports using the LLM for each date.
    
    Returns:
        tuple: (all_results, trending_topics_df, unique_reports_with_date)
    """
    start_list = pd.date_range(start=start_query, end=end_query, freq=freq).strftime('%Y-%m-%d %H:%M:%S').tolist()
    
    # DataFrame to store the trending topics for each date
    trending_topics_df = pd.DataFrame(columns=['Date', 'trending_topics'])
    
    # Extract dates and row numbers
    unique_reports = unique_reports.copy()
    unique_reports['Date'] = unique_reports['timestamp'].apply(lambda x: x.date())
    unique_reports['row_number'] = unique_reports.index
    
    # Add a text_with_reference field for each report
    unique_reports['text_with_reference'] = (
        '--- Report Start ---\n'
        'Row Number: ' + unique_reports['row_number'].astype(str) +
        '\nText: ' + unique_reports['text'].astype(str) +
        '\n--- Report End ---'
    )
    
    # Convert start_list to a list of datetime objects
    dates = [datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').date() for date_str in start_list]
    
    all_tasks = []
    
    # Loop through each date and collect reports
    for period in dates:
        reports = unique_reports.loc[unique_reports['Date'] == period]
        yearmonth = period.strftime('%Y-%m-%d')
        number_of_reports = len(reports)
        
        if number_of_reports == 0:
            continue
        
        batch_size = len(reports) // batches + (1 if len(reports) % batches != 0 else 0)
        
        # Create async tasks for all batches for this date
        tasks = []
        for i in range(0, len(reports), batch_size):
            batch_reports = reports.iloc[i:i + batch_size]
            text_to_analyze = "\n\n".join(batch_reports['text_with_reference'])
            
            # Create async tasks for extracting trending topics and summaries
            task = extract_trending_topics(text_to_analyze, model, yearmonth, len(batch_reports), api_key, main_theme)
            tasks.append(task)
        
        all_tasks.append((period, tasks))
    
    # Run all extract tasks concurrently for all dates and batches with tqdm
    all_results = {}
    for period, tasks in all_tasks:
        results = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Extracting Topics for {period}"):
            result = await task
            results.append(result)
        all_results[period] = [item for sublist in results for item in sublist]
    

    
    # Collect all topics and summaries from all batches across all dates
    for period, batch_results in all_results.items():
        # Add the extracted topics and summaries to trending_topics_df
        trending_topics_dict = {
            'Date': period,
            'trending_topics': batch_results
        }
        trending_topics_df = pd.concat([trending_topics_df, pd.DataFrame([trending_topics_dict])], ignore_index=True)
    
    return all_results, trending_topics_df, unique_reports


def phase2_flatten_topics(trending_topics_df, unique_reports):
    """
    PHASE 2: Flattening
    Converts hierarchical structure into flat DataFrame.
    
    Returns:
        tuple: (flattened_raw_topics_df, total_cited_for_summary)
    """
    # Calculate total cited for summary
    all_cited_row_numbers = set()
    for _, row in trending_topics_df.iterrows():
        topics = row['trending_topics']
        for topic in topics:
            for news in topic.get('cited_news', []):
                row_num = news.get('row_number')
                if row_num is not None:
                    all_cited_row_numbers.add(row_num)
    
    total_cited_for_summary = len(all_cited_row_numbers)
    
    # Flatten the DataFrame
    flattened_raw_topics_df = flatten_trending_topics(trending_topics_df, unique_reports)
    
    return flattened_raw_topics_df, total_cited_for_summary


async def phase3_consolidate_topics(flattened_raw_topics_df, model='gpt-4o-mini', api_key=None, main_theme='energy markets', 
                                    consolidation_batch_size=50, consolidation_round_batch_size=80, max_consolidation_rounds=8):
    """
    PHASE 3: Consolidation
    Consolidates topics using the LLM in batches and iterative rounds.
    
    Args:
        consolidation_batch_size: Initial batch size for consolidation (default: 50)
        consolidation_round_batch_size: Batch size for iterative rounds (default: 80)
        max_consolidation_rounds: Maximum number of consolidation rounds (default: 8)
    
    Returns:
        tuple: (flattened_consolidated_df, final_mapping, consolidation_stats)
    """
    print("Consolidating topics...")
    CONSOLIDATION_BATCH_SIZE = consolidation_batch_size
    MAX_CONSOLIDATION_ROUNDS = max_consolidation_rounds
    
    # Get unique topics
    unique_topics_and_summaries = flattened_raw_topics_df.Topic.unique()
    

    
    # Create mapping from topic_X to topic name
    topic_num_to_string_mapping = {f'topic_{i + 1}': topic for i, topic in enumerate(unique_topics_and_summaries)}
    
  
    
    # Prepare batches for consolidation
    topic_id_label_pairs = list(zip(topic_num_to_string_mapping.keys(), topic_num_to_string_mapping.values()))
    
    # Sort alphabetically by topic name (second element of tuple)
    topic_id_label_pairs = sorted(topic_id_label_pairs, key=lambda x: x[1])
    

    
    # Add a progress bar for batch consolidation
    batch_tasks = []
    batch_ranges = []
    for batch_start in range(0, len(topic_id_label_pairs), CONSOLIDATION_BATCH_SIZE):
        batch = topic_id_label_pairs[batch_start:batch_start + CONSOLIDATION_BATCH_SIZE]
        batch_str = ', '.join([f'{topic_id}: "{topic}"' for topic_id, topic in batch])
        batch_tasks.append(consolidate_trending_topics_with_retry(batch_str, model, api_key, main_theme))
        batch_ranges.append(batch)
    
    # Run all batch consolidations with a progress bar
    batch_results = await tqdm_asyncio.gather(*batch_tasks, desc="Consolidating topic batches", total=len(batch_tasks))
    
    # Helper: resolve topic ids/names to original topic ids
    def resolve_to_ids(items, mapping, track_unresolved=False):
        import re
        reverse_map = {v: k for k, v in mapping.items()}
        resolved = []
        unresolved = []
        for item in items:
            # Try direct match first
            if item in mapping:
                resolved.append(item)
                continue
            
            # Try reverse map (topic name -> topic ID)
            if item in reverse_map:
                resolved.append(reverse_map[item])
                continue
            
            # Try to extract topic ID from mixed format: 'topic_X: "Topic Name"'
            mixed_pattern = r'^topic_(\d+)(?:\s*:\s*["\'].*?["\'])?$'
            match = re.match(mixed_pattern, str(item).strip())
            if match:
                topic_id = f'topic_{match.group(1)}'
                if topic_id in mapping:
                    resolved.append(topic_id)
                    continue
            
            # Try to extract topic name from mixed format: 'topic_X: "Topic Name"'
            name_pattern = r'^topic_\d+\s*:\s*["\'](.+?)["\']$'
            match = re.match(name_pattern, str(item).strip())
            if match:
                topic_name = match.group(1)
                if topic_name in reverse_map:
                    resolved.append(reverse_map[topic_name])
                    continue
            
            # If nothing matches, it's unresolved
            if track_unresolved:
                unresolved.append(item)
            continue
        
        if track_unresolved:
            return resolved, unresolved
        return resolved
    
    
    all_resolved_ids = set()
    all_unresolved_items = []
    
    consolidated_batches = {}
    for batch_idx, batch_result in enumerate(batch_results, 1):
        if not batch_result or len(batch_result) == 0:
            print(f"  ⚠️  Batch {batch_idx}: batch_result is EMPTY or None!")
            print(f"    Type: {type(batch_result)}")
            print(f"    Value: {batch_result}")
            continue
        
        for k, v in batch_result.items():
            resolved_ids, unresolved = resolve_to_ids(v, topic_num_to_string_mapping, track_unresolved=True)
            all_resolved_ids.update(resolved_ids)
            all_unresolved_items.extend(unresolved)
            
            if k in consolidated_batches:
                consolidated_batches[k].extend(resolved_ids)
            else:
                consolidated_batches[k] = resolved_ids
    
    missing_after_resolve = set(topic_num_to_string_mapping.keys()) - all_resolved_ids
    
    if missing_after_resolve:
     
        topics_reinserted = 0
        for topic_id in missing_after_resolve:
            topic_name = topic_num_to_string_mapping[topic_id]
            # Add as standalone topic in consolidated_batches
            if topic_name in consolidated_batches:
                # If it already exists, add the ID to the list
                consolidated_batches[topic_name].append(topic_id)
            else:
                # Create new entry
                consolidated_batches[topic_name] = [topic_id]
            topics_reinserted += 1
        
 
    
    # Consolidate again across batches
    current_dict = consolidated_batches
    current_mapping = topic_num_to_string_mapping.copy()
    
    rounds = 0
    round_stats = []
    
    while True:
        rounds += 1
        consolidated_topic_num_to_string_mapping = {f'topic_{i + 1}': topic for i, topic in enumerate(current_dict.keys())}
        new_id_label_pairs = list(consolidated_topic_num_to_string_mapping.items())
        
        topics_before_round = len(current_dict)
        all_topic_ids_in_round = set()
        for topic_list in current_dict.values():
            all_topic_ids_in_round.update(topic_list)
        
        # Exit conditions: active only from round 4
        if rounds >= 4:
            if len(new_id_label_pairs) <= CONSOLIDATION_BATCH_SIZE or rounds >= MAX_CONSOLIDATION_ROUNDS:
                # print(f"\n  Round {rounds}: Exiting loop (topics={topics_before_round}, batch_size={CONSOLIDATION_BATCH_SIZE}, max_rounds={MAX_CONSOLIDATION_ROUNDS})")
                break
        
        consolidated_topic_id_label_pairs = list(zip(consolidated_topic_num_to_string_mapping.keys(), consolidated_topic_num_to_string_mapping.values()))
        
        # Sort alphabetically by topic name (second element of tuple)
        consolidated_topic_id_label_pairs = sorted(consolidated_topic_id_label_pairs, key=lambda x: x[1])
        
        # ===== SIMPLE BATCHING =====
        # Configurable batch size for rounds
        ROUND_BATCH_SIZE = consolidation_round_batch_size
        round_batch_tasks = []
        round_batch_ranges = []
        
        
        # Divide into batches of ROUND_BATCH_SIZE
        num_round_batches = (len(consolidated_topic_id_label_pairs) + ROUND_BATCH_SIZE - 1) // ROUND_BATCH_SIZE
        
        for batch_start in range(0, len(consolidated_topic_id_label_pairs), ROUND_BATCH_SIZE):
            batch = consolidated_topic_id_label_pairs[batch_start:batch_start + ROUND_BATCH_SIZE]
            batch_str = ', '.join([f'{topic_num}: "{topic_label}"' for topic_num, topic_label in batch])
            round_batch_tasks.append(consolidate_trending_topics_with_retry(batch_str, model, api_key, main_theme))
            round_batch_ranges.append(batch)
        
        # Execute all batches of the round in parallel
        round_batch_results = await tqdm_asyncio.gather(*round_batch_tasks, desc=f"Round {rounds} batches", total=len(round_batch_tasks))
        
        # Merge all batch results
        next_consolidation_result = {}
        for batch_result in round_batch_results:
            for k, v in batch_result.items():
                if k in next_consolidation_result:
                    # If the key already exists, extend the list
                    next_consolidation_result[k].extend(v)
                else:
                    next_consolidation_result[k] = v

        merged = {}
        all_orig_ids_in_merged = set()
        unresolved_in_round = []
        
        for k, v in next_consolidation_result.items():
            ids = []
            for item in v:
                orig_ids = []
                found = False
                
                if item in consolidated_topic_num_to_string_mapping:
                    topic_name = consolidated_topic_num_to_string_mapping[item]
                    if topic_name in current_dict:
                        orig_ids = current_dict[topic_name]
                        found = True
                
                elif item in current_dict:
                    orig_ids = current_dict[item]
                    found = True
                
                else:
                    import re
                    mixed_pattern = r'^topic_(\d+)(?:\s*:\s*["\'].*?["\'])?$'
                    match = re.match(mixed_pattern, str(item).strip())
                    if match:
                        topic_id = f'topic_{match.group(1)}'
                        if topic_id in consolidated_topic_num_to_string_mapping:
                            topic_name = consolidated_topic_num_to_string_mapping[topic_id]
                            if topic_name in current_dict:
                                orig_ids = current_dict[topic_name]
                                found = True
                    
                    if not found:
                        name_pattern = r'^topic_\d+\s*:\s*["\'](.+?)["\']$'
                        match = re.match(name_pattern, str(item).strip())
                        if match:
                            topic_name = match.group(1)
                            if topic_name in current_dict:
                                orig_ids = current_dict[topic_name]
                                found = True
                
                if not found:
                    unresolved_in_round.append((item, str(item)))
                
                ids.extend(orig_ids)
            seen = set()
            ids_dedup = [x for x in ids if not (x in seen or seen.add(x))]
            all_orig_ids_in_merged.update(ids_dedup)
            merged[k] = ids_dedup
        
        topics_after_round = len(merged)
        missing_in_round = all_topic_ids_in_round - all_orig_ids_in_merged

        round_stats.append({
            'round': rounds,
            'topics_before': topics_before_round,
            'topics_after': topics_after_round,
            'orig_ids_before': len(all_topic_ids_in_round),
            'orig_ids_after': len(all_orig_ids_in_merged),
            'missing': len(missing_in_round)
        })
        
        # Reinsert lost topics into merged, so they will be consolidated in the next round
        if missing_in_round:
            topics_reinserted = 0
            for topic_id in missing_in_round:
                topic_name = topic_num_to_string_mapping[topic_id]
                # Add as standalone topic in merged
                if topic_name in merged:
                    # If it already exists, add the ID to the list
                    merged[topic_name].append(topic_id)
                else:
                    # Create new entry
                    merged[topic_name] = [topic_id]
                topics_reinserted += 1
        
        # Check if reduction is less than 10%, then stop (only from round 4)
        if rounds >= 4 and topics_before_round > 0:
            reduction_pct = ((topics_before_round - topics_after_round) / topics_before_round) * 100
            if reduction_pct < 10.0:
                # print(f"  Round {rounds}: ⚠️  Reduction below 10% ({reduction_pct:.1f}%), early exit from loop")
                current_dict = merged
                break
        
        current_dict = merged
        current_mapping = {f'topic_{i + 1}': topic for i, topic in enumerate(merged.keys())}
    

    
    # Find all original topics that were lost during consolidation
    all_topic_nums_in_current = set()
    for topic_list in current_dict.values():
        all_topic_nums_in_current.update(topic_list)
    
    missing_topic_ids = set(topic_num_to_string_mapping.keys()) - all_topic_nums_in_current
    
    if missing_topic_ids:

        # Track duplicates
        duplicates_found = 0
        
        for topic_id in missing_topic_ids:
            topic_name = topic_num_to_string_mapping[topic_id]
            
            # Check if this topic name already exists in current_dict
            if topic_name in current_dict:
                # Append to existing list instead of overwriting
                current_dict[topic_name].append(topic_id)
                duplicates_found += 1
            else:
                # Create new entry
                current_dict[topic_name] = [topic_id]

    
    # Build the final mapping from original topic label to final consolidated label
    final_mapping = {}
    all_topic_nums_in_final = set()
    for consolidated_topic, topic_nums in current_dict.items():
        all_topic_nums_in_final.update(topic_nums)
        for topic_num in topic_nums:
            actual_topic = topic_num_to_string_mapping.get(topic_num)
            if actual_topic:
                final_mapping[actual_topic] = consolidated_topic
    
   
    
    # Apply the final mapping to the original dataframe
    topics_before_consolidation = flattened_raw_topics_df['Topic'].nunique()
    rows_before_consolidation = len(flattened_raw_topics_df)
    
    
    flattened_raw_topics_df['Topic'] = flattened_raw_topics_df['Topic'].map(final_mapping)
    
    
    topics_nan = flattened_raw_topics_df['Topic'].isna().sum()
   
    
    topics_after_consolidation = flattened_raw_topics_df['Topic'].nunique()
    rows_after_consolidation = len(flattened_raw_topics_df)
    
    consolidation_stats = {
        'topics_before': topics_before_consolidation,
        'topics_after': topics_after_consolidation,
        'rows_before': rows_before_consolidation,
        'rows_after': rows_after_consolidation,
        'topics_with_nan': topics_nan,
        'round_stats': round_stats
    }
    
    return flattened_raw_topics_df, final_mapping, consolidation_stats


async def phase4_summarize_topics(flattened_df, model='gpt-4o-mini', api_key=None, main_theme='energy markets', yearmonth='2024-01'):
    """
    PHASE 4: Summarization
    Generates consolidated summaries for each topic.
    
    Returns:
        pd.DataFrame: DataFrame with updated Summaries
    """
    print("Summarizing text for each topic...")
    flattened_df = await summarize_grouped_summaries(flattened_df, model, api_key, main_theme, yearmonth)
    return flattened_df


async def phase5_generate_titles(flattened_df, model='gpt-4o-mini', api_key=None):
    """
    PHASE 5: Title Generation
    Generates journalistic titles from summaries.
    
    Returns:
        pd.DataFrame: DataFrame with Topic titles
    """
    flattened_df = await create_titles_from_summaries(flattened_df, model=model, api_key=api_key)
    return flattened_df


async def phase6_postprocess(flattened_df, api_key=None, main_theme='energy markets', total_cited_for_summary=0, unique_reports=None):
    """
    PHASE 6: Post-processing
    Adds Day in Review, Text Summaries and removes NaN.
    
    Returns:
        pd.DataFrame: Final cleaned DataFrame
    """
    print("Generating Day in Review summaries...")
    flattened_df = add_day_in_review_to_df(flattened_df, api_key=api_key, main_theme=main_theme)
    
    print("Adding one-line summaries to DataFrame...")
    flattened_df = await add_text_summaries_to_df(flattened_df, api_key=api_key)
    
    
    flattened_df = flattened_df.dropna().reset_index(drop=True)
    
    
    return flattened_df


async def process_all_trending_topics_modular(unique_reports, start_query, end_query, main_theme, model='gpt-4o-mini', api_key=None, batches=5, freq='D', phases='all',
                                             consolidation_batch_size=50, consolidation_round_batch_size=80, max_consolidation_rounds=8):
    """
    Modular workflow to process trending topics.
    
    Args:
        batches: Number of batches per date in extraction phase (default: 5)
        consolidation_batch_size: Initial batch size for consolidation (default: 50)
        consolidation_round_batch_size: Batch size for iterative rounds (default: 80)
        max_consolidation_rounds: Maximum number of consolidation rounds (default: 8)
        phases: 'all' to execute all phases, or list of phases to execute
                e.g. ['phase1', 'phase2', 'phase3'] or 'phase1,phase2,phase3'
    
    Returns:
        dict: Dictionary with all intermediate results of executed phases
    """
    results = {}
    
    # Parse phases parameter
    if phases == 'all':
        phases_to_run = ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6']
    elif isinstance(phases, str):
        phases_to_run = [p.strip() for p in phases.split(',')]
    else:
        phases_to_run = phases
    

    
    # Phase 1: Topic Extraction
    if 'phase1' in phases_to_run:

        all_results, trending_topics_df, unique_reports_with_date = await phase1_extract_topics(
            unique_reports, start_query, end_query, main_theme, model, api_key, batches, freq
        )
        results['phase1'] = {
            'all_results': all_results,
            'trending_topics_df': trending_topics_df,
            'unique_reports': unique_reports_with_date
        }
    
    # Phase 2: Flattening
    if 'phase2' in phases_to_run:

        if 'phase1' not in results:
            raise ValueError("Phase 2 requires results from Phase 1. Execute Phase 1 first or pass data manually.")
        
        flattened_raw_topics_df, total_cited = phase2_flatten_topics(
            results['phase1']['trending_topics_df'],
            results['phase1']['unique_reports']
        )
        results['phase2'] = {
            'flattened_df': flattened_raw_topics_df,
            'total_cited': total_cited
        }
    
    # Phase 3: Consolidation
    if 'phase3' in phases_to_run:

        if 'phase2' not in results:
            raise ValueError("Phase 3 requires results from Phase 2. Execute Phase 1 and 2 first or pass data manually.")
        
        flattened_consolidated_df, final_mapping, consolidation_stats = await phase3_consolidate_topics(
            results['phase2']['flattened_df'],
            model, api_key, main_theme,
            consolidation_batch_size, consolidation_round_batch_size, max_consolidation_rounds
        )
        results['phase3'] = {
            'flattened_df': flattened_consolidated_df,
            'final_mapping': final_mapping,
            'consolidation_stats': consolidation_stats
        }
    
    # Phase 4: Summarization
    if 'phase4' in phases_to_run:

        if 'phase3' not in results:
            raise ValueError("Phase 4 richiede i risultati di Phase 3.")
        
        # Determina yearmonth dalla prima data nel DataFrame
        first_date = results['phase3']['flattened_df']['Date'].min()
        yearmonth = first_date.strftime('%Y-%m') if hasattr(first_date, 'strftime') else '2024-01'
        
        flattened_summarized_df = await phase4_summarize_topics(
            results['phase3']['flattened_df'],
            model, api_key, main_theme, yearmonth
        )
        results['phase4'] = {
            'flattened_df': flattened_summarized_df
        }
    
    # Phase 5: Title Generation
    if 'phase5' in phases_to_run:

        if 'phase4' not in results:
            raise ValueError("Phase 5 richiede i risultati di Phase 4.")
        
        flattened_titled_df = await phase5_generate_titles(
            results['phase4']['flattened_df'],
            model, api_key
        )
        results['phase5'] = {
            'flattened_df': flattened_titled_df
        }
    
    # Phase 6: Post-processing
    if 'phase6' in phases_to_run:

        if 'phase5' not in results:
            raise ValueError("Phase 6 richiede i risultati di Phase 5.")
        
        total_cited = results.get('phase2', {}).get('total_cited', 0)
        unique_reports_data = results.get('phase1', {}).get('unique_reports')
        
        flattened_final_df = await phase6_postprocess(
            results['phase5']['flattened_df'],
            api_key, main_theme, total_cited, unique_reports_data
        )
        results['phase6'] = {
            'flattened_df': flattened_final_df
        }
    

    
    # Return final dataframe if all phases completed
    if 'phase6' in results:
        results['final_df'] = results['phase6']['flattened_df']
    
    return results


def run_process_all_trending_topics_modular(unique_reports, start_query, end_query, main_theme, model='gpt-4o-mini', api_key=None, batches=5, freq='D', phases='all',
                                           consolidation_batch_size=50, consolidation_round_batch_size=80, max_consolidation_rounds=8):
    """
    Synchronous wrapper for process_all_trending_topics_modular.
    """
    return asyncio.run(process_all_trending_topics_modular(unique_reports, start_query, end_query, main_theme, model, api_key, batches, freq, phases,
                                                           consolidation_batch_size, consolidation_round_batch_size, max_consolidation_rounds))



async def run_full_trending_topics_pipeline(
    unique_reports, 
    start_query, 
    end_query, 
    main_theme, 
    model='gpt-4o-mini', 
    api_key=None, 
    batches=5, 
    freq='D',
    consolidation_batch_size=50, 
    consolidation_round_batch_size=80, 
    max_consolidation_rounds=8
):
    """
    Automatically executes all 6 phases of the trending topics pipeline and returns the final DataFrame.
    
    This function simplifies workflow usage by automatically executing:
    - Phase 1: Topic Extraction
    - Phase 2: Flattening
    - Phase 3: Consolidation
    - Phase 4: Summarization
    - Phase 5: Title Generation
    - Phase 6: Post-processing (Day in Review, Text Summaries, cleanup)
    
    Args:
        unique_reports: DataFrame with filtered news
        start_query: Start date (format 'YYYY-MM-DD')
        end_query: End date (format 'YYYY-MM-DD')
        main_theme: Main theme (e.g. 'Energy - Gas and Power')
        model: LLM model to use (default: 'gpt-4o-mini')
        api_key: OpenAI API key
        batches: Number of batches per date in extraction phase (default: 5)
        freq: Time frequency ('D' for daily, 'W' for weekly, etc.)
        consolidation_batch_size: Initial batch size for consolidation (default: 50)
        consolidation_round_batch_size: Batch size for iterative rounds (default: 80)
        max_consolidation_rounds: Maximum number of consolidation rounds (default: 8)
    
    Returns:
        pd.DataFrame: Final processed DataFrame with all topics, summaries, titles and scores
    
    Example:
        ```python
        final_df = await run_full_trending_topics_pipeline(
            unique_reports=filtered_reports,
            start_query='2025-11-05',
            end_query='2025-11-12',
            main_theme='Energy - Gas and Power',
            model='gpt-4o-mini',
            api_key=OPENAI_API_KEY,
            batches=20,
            consolidation_batch_size=50
        )
        ```
    """
    
    # PHASE 1: Topic Extraction
    all_results, trending_topics_df, unique_reports_with_date = await phase1_extract_topics(
        unique_reports=unique_reports,
        start_query=start_query,
        end_query=end_query,
        main_theme=main_theme,
        model=model,
        api_key=api_key,
        batches=batches,
        freq=freq
    )
    
    # PHASE 2: Flattening
    flattened_raw_topics_df, total_cited = phase2_flatten_topics(
        trending_topics_df=trending_topics_df,
        unique_reports=unique_reports_with_date
    )
    
    # PHASE 3: Consolidation
    flattened_consolidated_df, final_mapping, consolidation_stats = await phase3_consolidate_topics(
        flattened_raw_topics_df=flattened_raw_topics_df,
        model=model,
        api_key=api_key,
        main_theme=main_theme,
        consolidation_batch_size=consolidation_batch_size,
        consolidation_round_batch_size=consolidation_round_batch_size,
        max_consolidation_rounds=max_consolidation_rounds
    )
    
    # PHASE 4: Summarization
    flattened_with_summaries = await phase4_summarize_topics(
        flattened_df=flattened_consolidated_df,
        model=model,
        api_key=api_key,
        main_theme=main_theme
    )
    
    # PHASE 5: Title Generation
    flattened_with_titles = await phase5_generate_titles(
        flattened_df=flattened_with_summaries,
        model=model,
        api_key=api_key
    )
    
    # PHASE 6: Post-processing
    final_df = await phase6_postprocess(
        flattened_df=flattened_with_titles,
        api_key=api_key,
        main_theme=main_theme,
        total_cited_for_summary=total_cited,
        unique_reports=unique_reports_with_date
    )
    
    return final_df


def process_full_trending_topics_pipeline(
    unique_reports, 
    start_query, 
    end_query, 
    main_theme, 
    model='gpt-4o-mini', 
    api_key=None, 
    batches=5, 
    freq='D',
    consolidation_batch_size=50, 
    consolidation_round_batch_size=80, 
    max_consolidation_rounds=8
):
    """
    Synchronous wrapper for run_full_trending_topics_pipeline.
    
    Executes all 6 phases of the pipeline automatically and returns the final DataFrame.
    This is the simplest version to use for processing trending topics end-to-end.
    
    Args:
        unique_reports: DataFrame with filtered news
        start_query: Start date (format 'YYYY-MM-DD')
        end_query: End date (format 'YYYY-MM-DD')
        main_theme: Main theme (e.g. 'Energy - Gas and Power')
        model: LLM model to use (default: 'gpt-4o-mini')
        api_key: OpenAI API key
        batches: Number of batches per date in extraction phase (default: 5)
        freq: Time frequency ('D' for daily, 'W' for weekly, etc.)
        consolidation_batch_size: Initial batch size for consolidation (default: 50)
        consolidation_round_batch_size: Batch size for iterative rounds (default: 80)
        max_consolidation_rounds: Maximum number of consolidation rounds (default: 8)
    
    Returns:
        pd.DataFrame: Final processed DataFrame
    
    Example:
        ```python
        final_df = process_full_trending_topics_pipeline(
            unique_reports=filtered_reports,
            start_query='2025-11-05',
            end_query='2025-11-12',
            main_theme='Energy - Gas and Power',
            model='gpt-4o-mini',
            api_key=OPENAI_API_KEY
        )
        ```
    """
    return asyncio.run(run_full_trending_topics_pipeline(
        unique_reports=unique_reports,
        start_query=start_query,
        end_query=end_query,
        main_theme=main_theme,
        model=model,
        api_key=api_key,
        batches=batches,
        freq=freq,
        consolidation_batch_size=consolidation_batch_size,
        consolidation_round_batch_size=consolidation_round_batch_size,
        max_consolidation_rounds=max_consolidation_rounds
    ))


async def fetch_relevance(client, text_to_analyze, current_prompt, model, semaphore):
    async with semaphore:  # Limit to 1000 concurrent tasks
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": current_prompt},
                    {"role": "user", "content": text_to_analyze}
                ],
                temperature=0
            )

            relevance = ast.literal_eval(response.model_dump()['choices'][0]['message']['content'].strip())
            return relevance['relevance']
        except Exception as e:
            # Log the error for debugging purposes
            import warnings
            warnings.warn(f"Error in fetch_relevance for model {model}: {str(e)}. Returning None (will be filtered out).")
            return None


def generate_day_in_review(text, api_key, main_theme, model='gpt-4o-mini-2024-07-18'):
    client = openai.Client(api_key=api_key)
    
    # Create the prompt to summarize the key events of the day based on unique topics and summaries
    prompt = (
    f"You are an expert market analyst specializing in the {main_theme}. "
    f"Your task is to summarize the **most significant** market-moving events for the day based on the provided topics and summaries. "
    f"Rather than rewriting, focus on **condensing** the information and highlighting the most impactful aspects, such as price movements, supply disruptions, demand fluctuations, geopolitical tensions, or unexpected data releases that are relevant to the {main_theme}. "
    f"Only include key points that would be most relevant to a trader or analyst trying to understand the core reasons behind market shifts in the {main_theme}.\n\n"
    f"Text: {text}\n\n"
    f"Provide the final output as a **concise list** of bullet points, avoiding repetition or unnecessary details."
)

    # Use the OpenAI API to generate the Day in Review section
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"You are an expert market analyst specializing in {main_theme}."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    try:
        # Extract the bullet points from the response
        day_in_review = response.model_dump()['choices'][0]['message']['content'].strip()
    except:
        day_in_review = "Error generating Day in Review"  # Return an error message if generation fails
    
    return day_in_review

def add_day_in_review_to_df(df, api_key, main_theme):
    
    day_in_review_dict = {}
    
    # Group by Date to generate a single Day in Review for each unique date
    grouped = df.groupby('Date')
    
    for date, group in grouped:
        # Concatenate unique 'Summary' and 'Topic' text for the given date
        
        group['Topic'] = group['Topic'].fillna('')
        group['Summary'] = group['Summary'].fillna('')
        
        unique_summaries = [str(summary) for summary in group['Summary'].unique().tolist()]
        unique_topics = [str(topic) for topic in group['Topic'].unique().tolist()]


        # Combine unique topics and summaries
        combined_text = "\n\n".join(unique_topics) + "\n\n" + "\n\n".join(unique_summaries)
        
        # Generate the Day in Review using the OpenAI API
        day_in_review = generate_day_in_review(combined_text, api_key, main_theme)
        
        # Store the Day in Review in the dictionary
        day_in_review_dict[date] = day_in_review
    
    # Map the Day in Review back to the DataFrame
    df['Day_in_Review'] = df['Date'].map(day_in_review_dict)
    
    return df

# Asynchronous function to filter irrelevant news articles with a concurrency limit
async def filter_irrelevant_news(df, model, api_key, main_theme, semaphore_size=1000):
    df = df.reset_index(drop=True)
    df['row_number'] = df.index

    filter_prompt = (
        f"You are an expert analyst specializing in the {main_theme}. "
        f"Your task is to filter out news articles that are not related to {main_theme}. "
        f"Determine if the article discusses or impacts {main_theme}, considering aspects like production, supply, demand, prices, geopolitical events, and major industry trends.\n\n"
        f"Return a JSON object with 'relevance': 'Relevant' or 'Irrelevant'."
    )

    client = openai.AsyncOpenAI(api_key=api_key, timeout=15)

    # Limit to `semaphore_size` concurrent API requests (500, 1000, etc.)
    semaphore = asyncio.Semaphore(semaphore_size)

    tasks = []
    for _, row in df.iterrows():
        text_to_analyze = row['text']
        task = fetch_relevance(client, text_to_analyze, filter_prompt, model, semaphore)
        tasks.append(task)

    # Execute tasks asynchronously with a progress bar
    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Filtering News"):
        result = await task
        results.append(result)

    # Collect relevant row numbers based on the results
    relevant_rows = [df.loc[i, 'row_number'] for i, relevance in enumerate(results) if relevance == 'Relevant']
    
    # Count statistics
    total_count = len(results)
    relevant_count = len(relevant_rows)
    irrelevant_count = sum(1 for r in results if r == 'Irrelevant')
    error_count = sum(1 for r in results if r is None)
    
    # Filter the DataFrame based on relevant row numbers
    filtered_df = df[df['row_number'].isin(relevant_rows)].drop(columns=['row_number']).reset_index(drop=True)
    
    # Print summary
    print(f"\n📊 Filtering Summary:")
    print(f"   Total records: {total_count}")
    print(f"   Relevant: {relevant_count}")
    print(f"   Irrelevant: {irrelevant_count}")
    if error_count > 0:
        print(f"   ⚠️  Errors (filtered out): {error_count}")
        print(f"   💡 Tip: Check model name and API key if error count is high")

    return filtered_df

# Function to process all reports at once using the semaphore to control concurrency
def process_all_reports(reports, model, api_key, main_theme, semaphore_size=1000):
    # Process the reports without batch limits, controlling concurrency with the semaphore
    filtered_reports = asyncio.run(filter_irrelevant_news(reports, model, api_key, main_theme, semaphore_size))

    return filtered_reports

_intialization_sent = False 
def notebook_initialized():
    from importlib.metadata import version
    from bigdata_client import Bigdata
    from bigdata_client import tracking_services

    try:
        bigdata = Bigdata()
        global _intialization_sent
        if not _intialization_sent:
            trace = tracking_services.TraceEvent(event_name = "BigdataCookbookExecution", 
                       properties={"bigdataResearchToolsVersion": version("bigdata_research_tools"),
                                    "bigdataClientVersion": version("bigdata-client"),
                                   "cookbook_name": "DailyDigestCrudeOil"
                                  })
            
            tracking_services.send_trace(bigdata_client = bigdata, trace = trace)
            _intialization_sent = True
    except Exception as e:
        pass
        
notebook_initialized()