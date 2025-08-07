import asyncio
import os
from typing import Any, Coroutine, Optional, Union

import pandas as pd

from tools.asyncio_openai_joint import run_prompts

from tools.completion_processor import (
    completion_to_dataframe 
    )
from tools.prompt_generator import (
     create_prompt_generator,
    )



async def async_layer(
        all_prompt_messages,
        parameters,
        concurrency,
        api_key,
        path_call_hash,
        path_result_hash,
        timeout,
        max_retries):
    '''
    This function is an asynchronous layer that runs prompts using OpenAI API.
    '''
    completions = await run_prompts(
        all_prompt_messages,
        parameters,
        concurrency=concurrency,
        api_key=api_key,
        path_call_hash=path_call_hash,
        path_result_hash=path_result_hash,
        timeout=timeout,
        max_retries=max_retries,
        )
    return completions

async def async_extract_label(
        sentences: pd.DataFrame,
        system_prompt: list[dict[str, str]],
        n_expected_response_tokens: int,
        openai_api_key:str,
        path_call_hash_location: str,
        path_result_hash_location: str,
        parameters: dict = {
            'model': "gpt-3.5-turbo-1106",
            'temperature': 0,
            'response_format': {'type': 'json_object'}},
        masked: bool = False,
        sentence_column: str = 'text',
        context_window_size: int = 16385,
        n_completion_tokens: int = 4096,
        batch_size: int = 50,
        shuffle_chunk: bool = False,
        shuffle_all_data: bool = False,
        chunk_seed: Optional[int] = None,
        data_seed: Optional[int] = None,
        group_key: Optional[str] = None,
        concurrency: int = 1,
        timeout: int = 60,
        max_retries: int = 5
        ) -> pd.DataFrame:
    '''
    This function is used to extract labels from sentences asynchronously using OpenAI API.

    Parameters:
    sentences (pd.DataFrame): DataFrame containing the sentences to be processed.
    system_prompt (str): system prompts to be used for the OpenAI API.
    n_expected_response_tokens (int): Number of expected response tokens.
    openai_api_key (str): OpenAI API key.
    path_call_hash_location (str): Path to the location of the call hash.
    path_result_hash_location (str): Path to the location of the result hash.
    parameters (dict): Parameters to be used for the OpenAI API. Can contain any of the parameters used for the OpenAI API.
    masked (bool): If True, the entity names in the sentences are masked. Default is False.
    sentence_column (str): Column name in the DataFrame that contains the sentences. Default is 'text'.
    context_window_size (int): Size of the context window. Default is 16385.
    n_completion_tokens (int): Number of completion tokens. Default is 4096.
    batch_size (int): Size of the batch to be processed at once. Default is 50.
    shuffle_chunk (bool): If True, the chunks of data are shuffled. Default is False.
    shuffle_all_data (bool): If True, all data is shuffled. Default is False.
    chunk_seed (Optional[int]): Seed for chunk shuffling. Default is None.
    data_seed (Optional[int]): Seed for data shuffling. Default is None.
    group_key (Optional[str]): Key to group the data. Default is None.
    concurrency (int): Number of concurrent requests to be made to the OpenAI API. Default is 1.
    timeout (int): Timeout for the OpenAI API requests in seconds. Default is 60.
    max_retries (int): Maximum number of retries for the OpenAI API requests. Default is 5.

    Returns:
    pd.DataFrame: DataFrame containing the sentences with the extracted labels.
    '''

    
    prompt_preamble = [
    {
        "role": "system",
        "content": system_prompt
    }
    ]

    model = parameters['model']
    sentences_unique = sentences.copy().drop_duplicates(
        subset=['rp_entity_id', sentence_column])
    print('Total sentences:', sentences.shape[0])
    print('Unique sentences:', sentences_unique.shape[0])

    if masked:
        sentence_prompt_template = '{id};{masked_entity_name};"{filled_'+sentence_column+'}";'
    else:
        sentence_prompt_template = '{id};{entity_name};"{'+sentence_column+'}";'

    sentence_prompt_generator = create_prompt_generator(
        sentences=sentences_unique,
        model=model,
        prompt=prompt_preamble,
        sentence_prompt_template=sentence_prompt_template,
        masked=masked,
        context_window_size=context_window_size,
        n_completion_tokens=n_completion_tokens,
        n_expected_tokens_per_response=n_expected_response_tokens,
        chunk_size=batch_size,
        shuffle_chunk=shuffle_chunk,
        seed_chunk=chunk_seed,
        seed_data=data_seed,
        shuffle_data=shuffle_all_data,
        group_key=group_key)


    all_prompt_messages, all_sentence_batches = list(zip(*list(sentence_prompt_generator)))
    print('Number of generated prompts', len(all_prompt_messages))

    close_loop_at_the_end = False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        close_loop_at_the_end = True

    task = async_layer(
        all_prompt_messages,
        parameters,
        concurrency=concurrency,
        api_key=openai_api_key,
        path_call_hash=path_call_hash_location,
        path_result_hash=path_result_hash_location,
        timeout=timeout,
        max_retries=max_retries,
        )
    completions = await task

    if close_loop_at_the_end:
        loop.close()

    completion_df_list = []
    for idx, completion in enumerate(completions):
        completion_df = completion_to_dataframe(completion)
        completion_df_list.append(completion_df)

    combined_df = pd.concat(
        [sentence.merge(completion, how='left', left_on=['id'], right_index=True).assign(batch_idx=idx)
        for idx, (sentence, completion) in enumerate(zip(all_sentence_batches, completion_df_list))
        ], axis=0).reset_index(drop=True)

    sentences_w_label = sentences.merge(
        combined_df[['rp_entity_id', sentence_column]
                    + combined_df.columns[sentences.shape[1]:].tolist()],
        how='left',
        on=['rp_entity_id', sentence_column]
        )
    return sentences_w_label


def extract_label(**kwargs) -> Union[Coroutine, pd.DataFrame]:
    '''
    This function is a wrapper for the async_extract_label function. It runs the coroutine if an event loop is already running, 
    otherwise it creates a new event loop and runs the coroutine.

    Parameters:
    **kwargs: Keyword arguments to be passed to the async_extract_label function.

    Returns:
    Coroutine if an event loop is already running, else returns the result of the async_extract_label function as a DataFrame.
    '''
    coroutine = async_extract_label(**kwargs)
    try:
        _ = asyncio.get_running_loop()
        return coroutine
    except RuntimeError:
        result = asyncio.run(coroutine)
        return result