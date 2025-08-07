import json
import os
import re
from datetime import datetime
from hashlib import blake2b
from itertools import chain, product
from typing import Callable, Generator, Literal, Optional

import pandas as pd

from tools import llmutils
from tools.experiment_tracker import experiment_tracker
from tools.prompt_generator import *
from tools.label_extractor import *

import tiktoken




def prompt_generator_cost(
        sentences: pd.DataFrame,
        prompt: list[dict[str, str]],
        masked: bool,
        role: str,
        n_input_tokens: int,
        n_preamble_tokens: int,
        n_completion_tokens: int,
        n_expected_response_tokens: int,
        sentence_prompt_template: str,
        context_window_size: int,
        model: str,
        chunk_size: Optional[int] = None,
        shuffle_chunk: bool = False,
        seed_chunk: Optional[int] = None,
        shuffle_data: bool = False,
        seed_data: Optional[int] = None,
        ) -> Generator[list[dict[str, str]], None, None]:
    """
    Function to generate chunks of sentences that fit within the remaining token count limit.
    
    Parameters
    ----------
    sentences : pd.DataFrame
        DataFrame of sentences to chunk.
    prompt : list[dict[str, str]]
        Initial condition setting up the conversation with the model.
    masked : bool
        Indicates if the prompt sentences are masked.
    role : str
        The role setting who the model is in the conversation.
    n_input_tokens : int
        The maximum number of preamble tokens that can be input to the model.
    n_preamble_tokens : int
        The number of preamble tokens in the conversation.
    n_completion_tokens : int
        The maximum number of tokens allowed in the response.
    n_expected_response_tokens : int
        The expected number of tokens in the response.
    sentence_prompt_template : str
        The template used to write the prompts.
    context_window_size : int
        The size of the context window used to process the conversation.
    model : str
        The openAI model used.
    chunk_size : Optional[int]
        The size of the chunks into which the data is divided. (default is None)
    shuffle_chunk : bool
        Whether to shuffle the data within a chunk. (default is False)
    seed_chunk : Optional[int]
        The random seed for chunk shuffling. (default is None)
    shuffle_data : bool
        Whether to shuffle the data before chunking. (default is False)
    seed_data : Optional[int]
        The random seed for data shuffling. (default is None)
            
    Returns
    -------
    Generator[list[dict[str, str]], None, None]
        Yields (prompt + last_message, data_chunk)
    """ 

    sentence_processor = format_with_fstring(
        fstring=sentence_prompt_template, masked=masked)

    num_rows = len(sentences)
    for reserved_column_name in ['_PROMPT', '_INDEX']:
        if reserved_column_name in sentences.columns:
            raise KeyError(
                f'Reserved column name "{reserved_column_name}" in dataframe')

    sentences['_INDEX'] = sentences.index.values
    if shuffle_data:
        data = sentences.sample(
            frac=1, random_state=seed_data)
    else:
        data = sentences
    if chunk_size is None:
        chunk_size = num_rows

    i = 0
    while i < num_rows:
        data_chunk = data.iloc[i:i+chunk_size].reset_index(drop=True)
        data_chunk['_PROMPT'] = sentence_processor(data_chunk)  # This can be expensive when chunk_size is None
        n_tokens_per_row = data_chunk['_PROMPT'].apply(
            llmutils.count_user_message_tokens, model=model).to_frame('input_tokens')
        n_tokens_per_row['response_tokens'] = n_expected_response_tokens
        n_tokens = n_tokens_per_row.cumsum()

        chunk_size_adjusted = min(
            sum(n_tokens['response_tokens'] < n_completion_tokens),
            sum((n_preamble_tokens + n_tokens['input_tokens']) < n_input_tokens),
            sum((n_preamble_tokens + n_tokens.sum(axis=1)) < context_window_size),
            )

        data_chunk = data_chunk.iloc[:chunk_size_adjusted].reset_index(drop=True)
        # print('Getting', i, i+ chunk_size_adjusted)
        i = i + chunk_size_adjusted
        if shuffle_chunk:
            data_chunk = data_chunk.sample(
                frac=1, random_state=seed_chunk).reset_index(drop=True)
            # Process again to have the order of ids correct.
            data_chunk['_PROMPT'] = sentence_processor(data_chunk)

        last_messsage = [
            {
                "role": role,
                "content": "\n".join(data_chunk['_PROMPT'])
            }
        ]
        yield prompt + last_messsage, data_chunk


# @track_experiment
def create_prompt_generator_costs(
        sentences: pd.DataFrame,
        model: str,
        prompt: list[dict[str, str]],
        sentence_prompt_template: str,
        n_expected_tokens_per_response: int,
        n_completion_tokens: int =4096,
        context_window_size: int = 4096,
        role: Literal['user', 'system', 'assistant'] = 'user',
        chunk_size: Optional[int] = None,
        shuffle_chunk: bool = False,
        seed_chunk: Optional[int] = None,
        seed_data: Optional[int] = None,
        shuffle_data: bool = False,
        group_key: Optional[str] = None,
        masked: bool = False,
        ):
    """
    Function to create a generator the yields chunks of sentences that fit within the remaining
    token count limit.
    """
    seed_chunk = seed_chunk if shuffle_chunk else None
    seed_data = seed_data if shuffle_data else None

    n_preamble_tokens = llmutils.num_tokens_from_messages(prompt, model=model)

    n_input_tokens = context_window_size
    if group_key is None:
        sentence_prompt_generator = prompt_generator(
            sentences,
            model=model,
            prompt=prompt,
            sentence_prompt_template=sentence_prompt_template,
            masked=masked,
            role=role,
            n_preamble_tokens=n_preamble_tokens,
            context_window_size=context_window_size,
            n_input_tokens=n_input_tokens,
            n_completion_tokens=n_completion_tokens,
            n_expected_response_tokens=n_expected_tokens_per_response,
            chunk_size=chunk_size,
            shuffle_chunk=shuffle_chunk,
            seed_chunk=seed_chunk,
            seed_data=seed_data,
            shuffle_data=shuffle_data
            )
    else:

        sentence_prompt_generator = []
        for _, sentence_group in sentences.groupby(group_key, sort=False):
            sentence_prompt_generator.append(
                prompt_generator(
                sentence_group,
                model=model,
                prompt=prompt,
                sentence_prompt_template=sentence_prompt_template,
                masked=masked,
                role=role,
                n_preamble_tokens=n_preamble_tokens,
                context_window_size=context_window_size,
                n_input_tokens=n_input_tokens,
                n_completion_tokens=n_completion_tokens,
                n_expected_response_tokens=n_expected_tokens_per_response,
                chunk_size=chunk_size,
                shuffle_chunk=shuffle_chunk,
                seed_chunk=seed_chunk,
                seed_data=seed_data,
                shuffle_data=shuffle_data
                )
            )
        sentence_prompt_generator = chain(*sentence_prompt_generator)
    return sentence_prompt_generator

async def async_extract_costs(
        sentences: pd.DataFrame,
        system_prompt: list[dict[str, str]],
        n_expected_response_tokens: int,
        parameters: dict = {
            'model':'gpt-4o',
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

    sentence_prompt_generator = create_prompt_generator_costs(
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


    all_prompt_messages, all_sentence_batches  = list(zip(*list(sentence_prompt_generator)))


    return all_prompt_messages, all_sentence_batches 


def extract_costs(**kwargs) -> Union[Coroutine, pd.DataFrame]:
    '''
    This function is a wrapper for the async_extract_label function. It runs the coroutine if an event loop is already running, 
    otherwise it creates a new event loop and runs the coroutine.

    Parameters:
    **kwargs: Keyword arguments to be passed to the async_extract_label function.

    Returns:
    Coroutine if an event loop is already running, else returns the result of the async_extract_label function as a DataFrame.
    '''
    coroutine = async_extract_costs(**kwargs)
    try:
        _ = asyncio.get_running_loop()
        return coroutine
    except RuntimeError:
        result = asyncio.run(coroutine)
        return result
    
    
async def process_sentences(sentences, sentence_column_2, masked_2, system_prompt, n_expected_response_tokens, batch_size, model):
    all_prompt_messages = []
    all_sentence_batches = []
    sentences_labels_2 = await extract_costs(
        sentences=sentences,
        sentence_column=sentence_column_2,
        masked=masked_2,  # defines whether the name of the entity is hidden (masked) from the LLM.
        system_prompt=system_prompt,
        n_expected_response_tokens=n_expected_response_tokens,
        batch_size=batch_size,
        concurrency=200,  # defines the number of calls made simultaneously
        parameters={
            'model': model,  # gpt-4-0125-preview
            'temperature': 0,
            'response_format': {'type': 'json_object'}
        }
    )
    all_prompt_messages.append(sentences_labels_2[0])
    all_sentence_batches.append(pd.concat(sentences_labels_2[1]))
    #sentence_final.append(sentences_labels_2[1][0])


        # await asyncio.sleep(60)  # Wait for 60 second between batches to avoid hitting rate limits

    return all_prompt_messages, len(pd.concat(all_sentence_batches))


def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))


def compute_costs(sentences, sentence_column_2, system_prompt, model, masked_2 = True, n_expected_response_tokens = 100, batch_size = 10,
                   cost_prompt_tokens = 0.01 / 1000, cost_completion_tokens = 0.03 / 1000, completion_token_limit_placeholder = None, output = False):
    
    all_prompt_messages, all_sentence_batches = asyncio.run(
    process_sentences(
        sentences=sentences,
        sentence_column_2=sentence_column_2,
        masked_2=masked_2,
        system_prompt=system_prompt,
        n_expected_response_tokens=n_expected_response_tokens,
        batch_size=batch_size,
        model = model
        )
    )
    
    costs_models_dict = {'gpt-4o' : [0.005/1000, 0.015/1000],
                        'gpt-4o-2024-05-13' : [0.005/1000, 0.015/1000],
                        'gpt-3.5-turbo-0125' : [0.0005/1000, 0.0015/1000],
                        'gpt-3.5-turbo-instruct' : [0.0015/1000, 0.0020/1000],
                        'gpt-4-turbo' : [0.01/1000, 0.03/1000],
                        'gpt-4-0125-preview' : [0.01/1000, 0.03/1000],
                        'gpt-4-1106-preview' : [0.01/1000, 0.03/1000]}
    
    cost_prompt_tokens, cost_completion_tokens = costs_models_dict[model]
    tokenizer = tiktoken.encoding_for_model(model)

    prompts_text = [message['content'] for prompt in all_prompt_messages for conversation in prompt for message in conversation]

    # Calculate number of tokens for prompts
    prompt_tokens = sum(count_tokens(prompt, tokenizer) for prompt in prompts_text)
    
    completion_tokens =  24.8014 * all_sentence_batches # Here it's the average token length computed from previous iteration
    
    if completion_token_limit_placeholder is not None :
        completion_tokens =  completion_token_limit_placeholder * all_sentence_batches


    # Calculate costs
    total_cost_prompt_tokens = prompt_tokens * cost_prompt_tokens
    total_cost_completion_tokens = completion_tokens * cost_completion_tokens

    # Total cost
    total_cost = total_cost_prompt_tokens + total_cost_completion_tokens

    print(f'Total prompt tokens: {prompt_tokens}')
    print(f'Total completion tokens: {completion_tokens}')
    print(f'Total cost: ${total_cost:.4f}')
    
    if output:
        return round(total_cost,4)