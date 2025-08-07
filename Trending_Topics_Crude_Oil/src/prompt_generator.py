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


def generate_indices():
    """Iterates through combinations of the alphabet letters progressively increasing the length of combinations."""
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    index_length = 1

    while True:
        # Generate all possible combinations of letters with the current length
        combinations = product(alphabet, repeat=index_length)

        # Convert the combinations to strings
        index_strings = [''.join(combination) for combination in combinations]

        # Yield each index string
        for index_string in index_strings:
            yield index_string

        # Increase the index length for the next iteration
        index_length += 1



def calculate_prompt_hash(prompt: list[dict[str, str]]) -> str:
    """Calculate a hash for the given prompt using BLAKE2b algorithm.

    Args:
        prompt: A list of dictionaries containing prompt data.

    Returns:
        A hexadecimal string representing the hash of the prompt.
    """
    hasher = blake2b(digest_size=10)
    hasher.update(str(prompt).encode('utf-8'))
    return hasher.hexdigest()


def create_folder_if_not_exists(path: str):
    """Create folder for the given path if it does not already exist.

    Args:
        path: A string representing folder path.
    """
    path_folder = os.path.dirname(path)
    os.makedirs(path_folder, exist_ok=True)


def save_hashed_prompt(
        prompt: list[dict[str, str]],
        path_hashed_prompts: str,
        prompt_hash: Optional[str] = None,
        **kwargs
        ):
    """Saves a JSON file containing the prompt hash and additional metadata.

    Args:
        prompt: A list of dictionaries containing prompt data.
        path_hashed_prompts: Folder path where the JSON file will be saved.
        prompt_hash: An optional hash value for the prompt. If not provided, will be calculated.
        **kwargs: Additional data to be saved along with the prompt.
    """

    create_folder_if_not_exists(path_hashed_prompts)

    if prompt_hash is None:
        prompt_hash = calculate_prompt_hash(prompt)
    _path_prompt = path_hashed_prompts + prompt_hash + '.json'

    if not os.path.isfile(_path_prompt):
        with open(_path_prompt, 'w', encoding='utf-8') as file:

            time_now = datetime.now()
            hashed_prompt = {
                'hash': prompt_hash,
                'prompt': prompt,
                'first_hashed_utc': time_now.strftime("%Y-%m-%d %H:%M"),
                }
            json.dump(hashed_prompt | kwargs, file, indent=4)


def count_tokens_in_sentence(
        sentence_promts: pd.Series,
        model: str,
        ) -> pd.Series:
    """Counts tokens in the provided sentences using a specified model.

    Args:
        sentence_promts: Pandas Series containing sentences.
        model: String specifying the model to be used for token counting.

    Returns:
        A Pandas Series containing the number of tokens for each sentence.
    """
    n_tokens = sentence_promts.apply(llmutils.count_user_message_tokens, model=model)
    return n_tokens


def format_with_fstring(fstring: str, masked: bool) -> Callable[[pd.DataFrame], pd.Series]:
    """Generates a function that formats sentences in a DataFrame using the provided f-string.

    Args:
        fstring: The format string to apply.
        masked: If True, apply masking to entity names in sentences.

    Returns:
        A function that takes a DataFrame and returns a Series of formatted sentences.
    """
    def process_sentences(sentences: pd.DataFrame) -> pd.Series:
        sentences['id'] = range(1, len(sentences) + 1)
        missing_columns = unique_variables - (set(sentences.columns) | set(['filled_masked_text', 'masked_entity_name']))
        if missing_columns:
            raise KeyError(
                f'Provided dataframe does not have columns {missing_columns}')

        if not masked:
            return sentences.apply(lambda row: fstring.format(**row), axis=1)
        else:
            sentences['masked_text']=sentences['masked_text'].str.replace('Target Company','{target}',regex=False)
            sentences['masked_entity_name'] = sentences['id'].map(lambda x: 'Target Company_' + str(x))
            # substitute {company_mask} by the masked entity name
            sentences['filled_masked_text'] = sentences.apply(lambda row: row['masked_text'].format(target=row['masked_entity_name']), axis=1)
            sentences['masked_text']=sentences['masked_text'].str.replace('{target}','Target Company',regex=False)

            return sentences.apply(lambda row: fstring.format(**row), axis=1)
    variable_pattern = r'{(.*?)}'
    variables = re.findall(variable_pattern, fstring)
    unique_variables = set(variables)


    return process_sentences


def load_prompt(
        path: str
        ) -> list[dict[str, str]]:
    """
    This function is used to load a JSON file of prompts from a specified path

    :param path: string that represents the path to the JSON file 

    :return: list of dictionaries, where each dictionary contains a single prompt. 
            Each dictionary's key is the prompt id and the value is the prompt text.
    """   
    with open(path, 'r', encoding='utf-8') as file:
        _prompt = json.load(file)
   
    return _prompt


def prompt_generator(
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
def create_prompt_generator(
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

