import asyncio
import json
import logging
import os
import re
import time

import pandas as pd
from openai import AsyncOpenAI
from src_legacy.common.const import compose_labeling_system_prompt
from src_legacy.multithreading import TARGET_ENTITY_MASK

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_NAME = 'gpt-4o-mini'
SEMAPHORE_COUNT = 1000

UNKNOWN_LABEL = 'unclear'

openai_client = AsyncOpenAI()


def stringify_label_summaries(label_summaries):
    return [f'{label}: {summary}'
            for label, summary in label_summaries.items()]


async def make_request(system_prompt, prompt, semaphore):
    async with semaphore:
        try:
            response = await openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=MODEL_NAME,
                response_format={"type": "json_object"}
            )
            # Print or process the response
            r_str = response.model_dump()['choices'][0]['message']['content']
            r = re.sub('```', '', r_str)
            r = re.sub('json', '', r_str)
            return r
        except Exception as e:
            return None


async def make_request_with_backoff(system_prompt, prompt, semaphore,
                                    max_retries=3):
    retries = 0
    while retries < max_retries:
        result = await make_request(system_prompt, prompt, semaphore)
        if result is not None:
            return result  # Success, return the result

        retries += 1
        logging.info(f"Retrying... (Attempt {retries}/{max_retries})")
        # Exponential backoff before retrying
        wait_time = 2 ** retries
        await asyncio.sleep(wait_time)
    logging.error("Max retries reached. Request failed.")
    return None


def post_process_data_frame(df):
    # Filter out unlabelled sentences
    df = df.loc[df['label'] != UNKNOWN_LABEL].copy()
    # Remove timezone information - Excel compatibility
    df['timestamp_utc'] = df['timestamp_utc'].dt.tz_localize(None)
    # Sort by entity name and label
    sort_columns = ['entity_name', 'timestamp_utc', 'label']
    df = df.sort_values(by=sort_columns).reset_index(drop=True)

    # Replace company name
    df['motivation'] = df.apply(
        lambda row: row['motivation'].replace(TARGET_ENTITY_MASK,
                                              row['entity_name']),
        axis=1
    )

    # Rename and reorder columns
    df['Time Period'] = df['timestamp_utc'].dt.strftime('%b %Y')
    df['Date'] = df['timestamp_utc'].dt.strftime('%Y-%m-%d')
    df['Company'] = df['entity_name']
    df['Sector'] = df['entity_sector']
    df['Industry'] = df['entity_industry']
    df['Country'] = df['entity_country']
    df['Ticker'] = df['entity_ticker']
    df['Document ID'] = df['rp_document_id']
    df['Headline'] = df['headline']
    df['Quote'] = df['text']
    df['Motivation'] = df['motivation']
    df['Theme'] = df['label']
    # df['Revenue Generation'] = df['revenue_generation']
    # df['Cost Efficiency'] = df['cost_efficiency']

    export_columns = [
        'Time Period',
        'Date',
        'Company',
        'Sector',
        'Industry',
        'Country',
        'Ticker',
        'Document ID',
        'Headline',
        'Quote',
        'Motivation',
        'Theme',
        # 'Revenue Generation',
        # 'Cost Efficiency'
    ]
    df = df[export_columns].copy()
    return df


async def run_requests(prompts, system_prompt):
    # Control concurrency (adjust the semaphore value according to API limits)
    semaphore = asyncio.Semaphore(
        SEMAPHORE_COUNT)  # Number of concurrent requests
    tasks = []
    for prompt in prompts:
        tasks.append(make_request_with_backoff(system_prompt,
                                               prompt,
                                               semaphore))

    # Gather and run the requests concurrently
    results = await asyncio.gather(*tasks)
    return results


def process_request(prompts, system_prompt):
    tic = time.perf_counter()
    responses = asyncio.run(run_requests(prompts, system_prompt))
    toc = time.perf_counter() - tic
    print(f"Completed {len(prompts)} requests in {toc:.2f} seconds.")
    return responses


def deserialize_responses(responses):
    
    response_mapping = {}
    for response in responses:
        try:
            deserialized_response = json.loads(response)
        except json.JSONDecodeError:
            continue
        for k, v in deserialized_response.items():
            try:
                response_mapping[k] = {key: v[key] for key in v.keys()}
            except KeyError:
                response_mapping[k] ={key: '' for key in v.keys()}

    df_labels = pd.DataFrame.from_dict(response_mapping, orient='index')
    df_labels.index = df_labels.index.astype(int)

    return df_labels


def get_system_prompt(theme_tree, main_theme):
    label_summaries = extract_label_summaries(theme_tree)
    label_summaries = stringify_label_summaries(label_summaries)
    label_summaries = label_summaries[1:]  # Remove the root - AI Solutions
    return compose_labeling_system_prompt(main_theme, label_summaries)


def get_prompts(df, columns=None,):
    if columns is None:
        columns = ['id', 'text']  # Default to id and text, if no columns are provided
    
    return [
        json.dumps({col: row[col] for col in columns})
        for _, row in df.iterrows()]
