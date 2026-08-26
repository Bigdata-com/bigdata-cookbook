import asyncio
import json
import logging
import os
import re
import time
from typing import List

import pandas as pd
from openai import AsyncOpenAI

from src.label.label_prompts import compose_labeling_system_prompt
from src.mindmap.themes import extract_terminal_summaries, stringify_label_summaries
from src.openai_compat import DEFAULT_LLM_MODEL

MODEL_NAME = DEFAULT_LLM_MODEL
SEMAPHORE_COUNT = 1000

UNKNOWN_LABEL = 'unclear'

# OpenAI client will be initialized when needed with provided API key
openai_client = None


async def make_request(system_prompt, prompt, semaphore, api_key=None):
    """
    Make a request to the LLM.

    :param system_prompt: The system prompt
    :param prompt: The prompt
    :param semaphore: The async semaphore
    :param api_key: The OpenAI API key
    :return: The result
    """
    global openai_client
    
    # Initialize client if not already done or if new API key provided
    if openai_client is None or api_key:
        if not api_key:
            # Try to get from environment variable as fallback
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("API key must be provided either as parameter or OPENAI_API_KEY environment variable")
        openai_client = AsyncOpenAI(api_key=api_key)
    
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
                                    max_retries=3, api_key=None):
    """
    Make a request with exponential backoff.

    :param system_prompt: The system prompt
    :param prompt: The prompt
    :param semaphore: The semaphore
    :param max_retries: The maximum number of retries
    :param api_key: The OpenAI API key
    :return: The result
    """
    retries = 0
    while retries < max_retries:
        result = await make_request(system_prompt, prompt, semaphore, api_key)
        if result is not None:
            return result  # Success, return the result

        retries += 1
        logging.info(f"Retrying... (Attempt {retries}/{max_retries})")
        # Exponential backoff before retrying
        wait_time = 2 ** retries
        await asyncio.sleep(wait_time)
    logging.error("Max retries reached. Request failed.")
    return None


async def run_requests(prompts, system_prompt, api_key=None):
    """
    Run the requests concurrently.

    :param prompts: The prompts
    :param system_prompt: The system prompt
    :param api_key: The OpenAI API key
    :return: The results
    """
    # Control concurrency (adjust the semaphore value according to API limits)
    semaphore = asyncio.Semaphore(
        SEMAPHORE_COUNT)  # Number of concurrent requests
    tasks = []
    for prompt in prompts:
        tasks.append(make_request_with_backoff(system_prompt,
                                               prompt,
                                               semaphore,
                                               api_key=api_key))

    # Gather and run the requests concurrently
    results = await asyncio.gather(*tasks)
    return results


def process_request(prompts, system_prompt, api_key=None):
    """
    Process the requests using LLM.

    :param prompts: The prompts
    :param system_prompt: The system prompt
    :param api_key: The OpenAI API key
    :return: The responses
    """
    tic = time.perf_counter()
    responses = asyncio.run(run_requests(prompts, system_prompt, api_key))
    toc = time.perf_counter() - tic
    print(f"Completed {len(prompts)} requests in {toc:.2f} seconds.")
    return responses


def deserialize_patent_labels(responses) -> pd.DataFrame:
    """
    Deserialize patent labelling into a data frame.

    :param responses: The responses
    :return: The DataFrame
    """
    response_mapping = {}
    i = 0
    for response in responses:
        try:
            deserialized_response = json.loads(response)
        except json.JSONDecodeError:
            i += 1
            continue

        response_mapping[i] = {"Relevant": deserialized_response["RELEVANT"],
                               "Explanation": deserialized_response["explanation"]}
        i += 1
       
    df_labels = pd.DataFrame.from_dict(response_mapping, orient='index')
    df_labels.index = df_labels.index.astype(int)
    
    return df_labels

def new_deserialize_patent_labels(responses):
    """
    Deserialize JSON responses into a DataFrame.
    """
    response_mapping = {}
    
    for i, response in enumerate(responses):
        try:
            data = json.loads(response)
            response_mapping[i] = {k: v for k, v in data.items()}
        except json.JSONDecodeError:
            continue
            
    return pd.DataFrame.from_dict(response_mapping, orient='index')

def deserialize_responses(responses):
    """
    Deserialize the responses into a data frame.

    :param responses: The responses
    :return: The DataFrame
    """
    response_mapping = {}
    for response in responses:
        try:
            deserialized_response = json.loads(response)
        except json.JSONDecodeError:
            continue
            
        for k, v in deserialized_response.items():                
            try:
                response_mapping[k] = {
                    'motivation': v['motivation'],
                    'label': v['label'],
                }
            except KeyError:
                response_mapping[k] = {
                    'motivation': '',
                    'label': UNKNOWN_LABEL,
                }

    df_labels = pd.DataFrame.from_dict(response_mapping, orient='index')
    df_labels.index = df_labels.index.astype(int)

    return df_labels


def get_system_prompt(theme_tree, main_theme):
    """
    Generate the system prompt for the labeling system.

    :param theme_tree: The theme tree.
    :param main_theme: The main theme.
    :return: The system prompt.
    """
    terminal_summaries = extract_terminal_summaries(theme_tree)
    terminal_summaries = stringify_label_summaries(terminal_summaries)

    return compose_labeling_system_prompt(main_theme, terminal_summaries)

def get_company_prompts(df) -> List[str]:
    """
    Generate the prompts for the labeling system.

    :param df: The data frame.
    :return: A list of prompts.
    """
    return [json.dumps({'sentence_id': i, 'text': text, 'Target Company': company})
        for i, (text, company) in enumerate(zip(df['text'], df['entity_name']))]

def get_prompts(df) -> List[str]:
    """
    Generate the prompts for the labeling system.

    :param df: The data frame.
    :return: A list of prompts.
    """
    return [json.dumps({'sentence_id': i, 'text': text})
            for i, text in enumerate(df['masked_text'])]