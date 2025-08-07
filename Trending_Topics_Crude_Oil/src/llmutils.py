import json
import os
import re
from typing import Literal
import openai
import pandas as pd
import tiktoken
from tqdm import tqdm
from interval import interval

import datasets
from datasets import Dataset

def process_sentences(sentences: pd.DataFrame, target_entity_mask: str, other_entity_mask: str, masking: bool = False) -> pd.DataFrame:
    """
    Processes a DataFrame of sentences by:
        - creating the masked version of sentences (in 'masked_text') using the coordinates
                of the entities
        - cleaning the text in each sentence.


    Parameters:
    - sentences (pd.DataFrame): A pandas DataFrame with at least a 'text' column. If masking= True, it also requires
                                at least a column 'coordinates'.
    - target_entity_mask (str): String that will be used in the place of the target entity to be masked (entity in the 'coordinates' column)
    - other_entity_mask (str): String that will be used in place of the other entities
    - masking (bool): if True, sentences with masked entities are created 
            the entity in 'coordinates' is masked with the target_entity_mask string, while entities in 'other_coordinates', if present,
            are masked with the other_entity_mask string.

    Returns:
    - pd.DataFrame: The modified DataFrame with cleaned text in the 'text' column and masked text in 'masked_text' column (if requested)
    """

    sentences['text'] = sentences['text'].str.replace('{','',regex=False)
    sentences['text'] = sentences['text'].str.replace('}','',regex=False)

    if masking:
        sentencesdf = mask_entity_coordinates(input_df = sentences, 
                                column_masked_text = 'masked_text',
                                mask_target = target_entity_mask,
                                mask_other= other_entity_mask
                               )
    
        # note that doing the cleaning can invalidate the coordinates of the entity detections, 
        # which need to be corrected if some text was cleaned before them 
        # (we are ignoring this here because we already prepared the masked text and won't use coordinates further)
        sentencesdf['masked_text'] = sentencesdf['masked_text'].apply(clean_text_masked)
        
        sentencesdf = sentencesdf[sentencesdf.masked_text!='to_remove']
    else:
        sentencesdf=sentences
    
    sentencesdf['text'] = sentencesdf['text'].apply(clean_text)
    sentencesdf = sentencesdf[sentencesdf.text!='to_remove']
    
    return sentencesdf.reset_index(drop = True)

def mask_entity_coordinates(
        input_df: pd.DataFrame,
        column_masked_text: str,
        mask_target: str,
        mask_other: str
        ) -> pd.DataFrame:
    """
    This function masks the entities in the provided coordinates from a given column of text

    Parameters:
        input_df (pd.DataFrame): The input dataframe.
        column_masked_text (str): The column name for the masked text.
        mask_target (str): The target entity mask.
        mask_other (str): The other entities mask.

    Returns:
        pd.DataFrame: The dataframe with masked entities text in column_masked_text .
    """

    df = input_df.copy()
    
    MANDATORY_COLUMNS = [
        COLUMN_SENTENCE_ID:= 'sentence_id',
        COLUMN_RP_ENTITY_ID:= 'rp_entity_id',
        COLUMN_COORDINATES:= 'coordinates',
        COLUMN_TEXT:= 'text']

    OPTIONAL_COLUMNS = [
        COLUMN_COORDINATES_OTHER:= 'other_coordinates'
        ]
    
    for mandatory_column in MANDATORY_COLUMNS:
        assert mandatory_column in df.columns, f"Column {mandatory_column} is missing in the dataframe."
    
    assert column_masked_text not in df.columns, f"Column {column_masked_text} is already in already in the input dataframe, please use other name for the masked text."

    assert df.groupby([COLUMN_SENTENCE_ID, COLUMN_RP_ENTITY_ID, COLUMN_COORDINATES]).transform('size').max() == 1, \
        "There are duplicate entities for the same coordinates in a sentence. Please remove them before proceeding."
    
    # transform strings to list
    if df[COLUMN_COORDINATES].apply(lambda x: isinstance(x, str)).all():
        df['temp_coordinates'] = df[COLUMN_COORDINATES].apply(eval)
   
    
    tqdm.pandas(desc='Getting all sentence coordinates 1-2')

    
    df['temp_coordinates'] = df['temp_coordinates'].progress_apply(
        group_and_select_shortest_overlapping_coordinates)

    tqdm.pandas(desc='Getting all sentence coordinates 2-2')

    if(COLUMN_COORDINATES_OTHER in df.columns):
        # explode the other coordinates if more than one entity is present
        odf = df[~df[COLUMN_COORDINATES_OTHER].isna()].reset_index(drop=True)
        odf[COLUMN_COORDINATES_OTHER] = odf[COLUMN_COORDINATES_OTHER].apply(eval)
        odf = odf[[COLUMN_SENTENCE_ID,COLUMN_RP_ENTITY_ID,COLUMN_COORDINATES_OTHER]].explode(COLUMN_COORDINATES_OTHER)

        odf[COLUMN_COORDINATES_OTHER] = odf[COLUMN_COORDINATES_OTHER].progress_apply(
            group_and_select_shortest_overlapping_coordinates)
        
        odf = odf.groupby([COLUMN_SENTENCE_ID,COLUMN_RP_ENTITY_ID])[COLUMN_COORDINATES_OTHER].sum().reset_index()
        odf= odf.rename(columns={COLUMN_COORDINATES_OTHER:'temp_coordinates_other'})
        df = df.merge(odf,on=[COLUMN_SENTENCE_ID,COLUMN_RP_ENTITY_ID],how='left')

    
    MASK_TARGET = {
            'coordinate_column_name': 'temp_coordinates',
            'mask': mask_target
            }
    MASK_OTHER = {
            'coordinate_column_name': 'temp_coordinates_other',
            'mask': mask_other
            }

    df[column_masked_text] = df.apply(mask_entities, args=(MASK_TARGET, MASK_OTHER), axis=1)
    df = df.drop(columns=['temp_coordinates','temp_coordinates_other'])
    return df


def mask_entities(
        x: list,
        mask_target: dict[str, str],
        mask_other: dict[str, str],
        ):
    """
    Function that does the actual masking.

    Parameters:
        x (list): list with the text and coordinates to be masked 
        mask_target (dict[str, str]): The target entity mask column information.
        mask_other (dict[str, str]): The other entity mask column information.

    Returns:
        str: The text with masked entities.
    """

    text = x['text']

    coordinates_target: list[list[int]] = [i[:] for i in x[mask_target['coordinate_column_name']]]
    coordinates_other = x[mask_other['coordinate_column_name']]
    if isinstance(coordinates_other, list):
        try:
            coordinates_other: list[list[int]] = [i[:] for i in x[mask_other['coordinate_column_name']]]
        except TypeError as e:
            print('We are here!')
            raise e
        #remove coordinates from coordinates_other if they are in coordinates_target
        for coords_target in coordinates_target:
            for coords_other in coordinates_other:
                if coords_target == coords_other:
                    coordinates_other.remove(coords_other)
    else:
        coordinates_other: list[list[int]] = []

    all_coordinates = [(coords, mask_target['mask']) for coords in coordinates_target]
    if coordinates_other:
        all_coordinates += [(coords, mask_other['mask']) for coords in coordinates_other]



    all_coordinates = sorted(all_coordinates, key=lambda x: x[0])

    updated_coordinates = []
    current_index = 0
    for idx, (coords, mask) in enumerate(all_coordinates):
        len_diff = len(mask) - (coords[1] - coords[0])
        if current_index > coords[0]:
            # print('Coordinates overlap')
            len_diff = len(mask) - (coords[1] - current_index - 1)
            text = text[:current_index] + ' ' + mask + text[coords[1]: ]
            coords[0] = current_index + 1
            coords[1] = coords[0] + len(mask)
        # if len_diff < 0:

            # raise Exception('Coordinates overlap')
        else:
            text = text[:coords[0]] + mask + text[coords[1]: ]
            coords[1] = coords[0] + len(mask)
        for remaining_coords in all_coordinates[idx+1: ]:
            remaining_coords[0][0] += len_diff
            remaining_coords[0][1] += len_diff
        current_index = coords[1]
        updated_coordinates.append(coords)

    return text


def group_overlapping_coordinates(coordinates_list: list[list[int]]) -> list[list[int]]:
    coordinate_groups = {interval(a): [] for a in interval.union([interval(x) for x in coordinates_list])}

    for coordinates in coordinates_list:
        for coord_group in coordinate_groups:
            if len(interval(coordinates) & coord_group) > 0:
                coordinate_groups[coord_group].append(coordinates)

    return list(coordinate_groups.values())

def select_shortest_overlapping_coordinates(coordinates_list):
    selected_coordinates = []
    for coordinates in coordinates_list:
        min_distance = None
        selected_coordinate = None
        for start, end in coordinates:
            distance = end - start
            if (min_distance is None) or (distance < min_distance):
                min_distance = distance
                selected_coordinate = [start, end]
        selected_coordinates.append(selected_coordinate)
    return selected_coordinates

def group_and_select_shortest_overlapping_coordinates(
        coordinates_list: list[list[int]]
        ) -> list[list[int]]:
    grouped_coordinates = group_overlapping_coordinates(coordinates_list)
    selected_coordinates = select_shortest_overlapping_coordinates(grouped_coordinates)
    return selected_coordinates


def calculate_n_tokens(prompt: str, model: str = "gpt-3.5-turbo-0301") -> int:
    """
    Calculates the number of tokens for a given prompt string according to a specified language model's tokenizer.

    Parameters:
    - prompt (str): The text for which to calculate the token count.
    - model (str, optional): The model identifier for which the token count needs to be calculated.
      Defaults to "gpt-3.5-turbo-0301".

    Returns:
    - int: The calculated token count.

    Note: The actual token count may vary depending on the tokenizer's version and specifics.
    """
    # Define the function to calculate the number of tokens ->
    # https://platform.openai.com/tokenizer
    # gpt-3.5-turbo = 485 #gpt-3.5-turbo-0301 = 486
    # vs https://platform.openai.com/tokenizer = 522
    encoding = tiktoken.encoding_for_model(model)
    token_count = len(encoding.encode(prompt))
    return token_count


def num_tokens_from_messages(messages: list[dict[str, str]], model: str) -> int:
    """
    Returns the number of tokens used by a list of messages.

    Args:
    - messages (List[Dict[str, str]]): List of messages.
    - model (str): Model name for token counting.

    Returns:
    - int: Total number of tokens.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        #print('base')
        encoding = tiktoken.get_encoding("cl100k_base")
    if ('gpt-3.5' in model) or ('gpt-4' in model):  # model == "gpt-3.5-turbo-0301":
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():

                num_tokens += len(encoding.encode(value))
                #print(key, encoding.encode(value), len(encoding.encode(value)))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant

        # num_tokens += 1 #For some reason we need to add this to match the
        # responses, TODO look up why.
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md
  for information on how messages are converted to tokens.""")

def num_tokens_from_messages_alt(
        messages: list[dict[str, str]],
        model: str  #  = "gpt-3.5-turbo-0301"
        ) -> int:
    """
    Returns the number of tokens used by a list of messages.

    Args:
    - messages (List[Dict[str, str]]): List of messages.
    - model (str): Model name for token counting.

    Returns:
    - int: Total number of tokens.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        #print('base')
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        for key, value in message.items():
            num_tokens += len(encoding.encode(key))
            num_tokens += len(encoding.encode(value))
    num_tokens += 2  # every reply is primed with <im_start>assistant

    return num_tokens


def count_user_message_tokens(string: str, model: str) -> int:
    """
    Count tokens for a single sentence.

    Args:
    - sentence (str): The sentence to count tokens for.

    Returns:
    - int: Number of tokens in the sentence.
    """
    tokens_count = num_tokens_from_messages([{"role": "user", "content": string}], model=model)
    return tokens_count



def clean_text_masked(text: str) -> str:
    """
    Transforms the given sample text by...

    Args:
    - text (str): The sample text to be transformed.

    Returns:
    - str: Transformed text.
    """
    #...  replacing newlines with spaces
    text = text.replace('\n', ' ')

    #... removing ticker symbols ">TICKER" (mostly common at the end of some news flashes)
    text = re.sub(r'>[A-Z]+', '', text)

    #... removing news sources at the end of the sentence "-- SOURCE_NAME"  (mostly common in some news headlines)
    text = re.sub(r'-- \w+\.?\w+\.?\w*$', '', text)

    #... removing numeric trailing patterns like -4-, -5-, etc. (mostly appearing in some news headlines)
    text = re.sub(r' -\d+-$', '', text)

    #... removing some timestamps (more versions could be added)
    text = re.sub(r'\d\d:\d\d EDT\s*', '', text)
    
    #... removing URLs
    text = re.sub(r'https?://\S+|www\.\S+','',text)
    
    
    #... we will also get rid of sentences that have less than 5 words (not counting the identified entities)
    text2 = text.replace("Target Company","")
    text2 = text2.replace("Other Company","")
    text2=re.sub('[^A-Za-z ]+', '', text2)
    if len(text2.split())<5:
        text="to_remove"

    return text.strip()

def clean_text(text: str) -> str:
    """
    Transforms the given sample text by...

    Args:
    - text (str): The sample text to be transformed.

    Returns:
    - str: Transformed text.
    """
    #...  replacing newlines with spaces
    text = text.replace('\n', ' ')

    #... removing ticker symbols ">TICKER" (mostly common at the end of some news flashes)
    text = re.sub(r'>[A-Z]+', '', text)

    #... removing news sources at the end of the sentence "-- SOURCE_NAME"  (mostly common in some news headlines)
    text = re.sub(r'-- \w+\.?\w+\.?\w*$', '', text)

    #... removing numeric trailing patterns like -4-, -5-, etc. (mostly appearing in some news headlines)
    text = re.sub(r' -\d+-$', '', text)

    #... removing some timestamps (more versions could be added)
    text = re.sub(r'\d\d:\d\d EDT\s*', '', text)
    
    #... removing URLs
    text = re.sub(r'https?://\S+|www\.\S+','',text)
    
    
    #... we will also get rid of sentences that have less than 6 words (counting the identified entities)
    text2=re.sub('[^A-Za-z ]+', '', text)
    if len(text2.split())<6:
        text="to_remove"

    return text.strip()


def split_to_batches(
        dataframe: pd.DataFrame,
        max_tokens_per_batch: int
        ) -> list[pd.DataFrame]:
    # ~50 headlies per execution once included the fixe part of the prompt
    # append_batches_to_list -> split_to_batches
    """
    Create batches of rows from a DataFrame based on a maximum token limit.
    fixed_prompt = 1062 = 325+552+185 to give context to the assistant,
    as user and examples as assitant (respectively)
    For gpt-3.5-turbo-0301 the max is 4096
    tokens-1062(fixed_prompt)-234(buffer)=**2800**
    as max_tokens_per_batch => variable sentences part

    Args:
    - dataframe (pd.DataFrame): The input DataFrame to be divided into batches.
    - max_tokens (int): The maximum total tokens allowed in each batch.

    Returns:
    - List[pd.DataFrame]: A list of DataFrames representing the batches.

    Example usage:
    - Replace dataframe and "3800" with your actual DataFrame and max_tokens value
    >>[batches = append_batches_to_list(DF_60_SAMPLE, 1100)]

    ToDo:
    If the response in the batch is the last and it has only one the last
    respoonse is not in json format it only display a number - potential issue
    for one respoinse?
    """

    batches = []
    batch_ids = dataframe.N_TOKENS.cumsum()//max_tokens_per_batch
    for _, batch_data in dataframe.groupby(batch_ids, as_index=False):
        batches.append(batch_data.reset_index(drop=True))
    return batches


def call_openai(
        messages: list[dict[str, str]],
        model: str,
        response_format: Literal['text', 'json_object'] = 'text',
        temperature: float = 0,
        top_p: float = 1,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        timeout: int = 600,
        max_retries: int = 2
        ) -> openai.types.Completion:
    """
    Calls the OpenAI API to get completions for a given list of messages using the specified model and parameters.
    
    Args:
        messages (list[dict[str, str]]): Input messages for the AI to respond to.
        model (str): The model of OpenAI to use for generating responses.
        response_format (Literal['text', 'json_object'], optional): Desired format of the response from the API.
        temperature (float, optional): Temperature for response generation.
        top_p (float, optional): Parameter for nucleus sampling
        frequency_penalty (float, optional): Penalty for frequent words to avoid repetition.
        presence_penalty (float, optional): Penalty to encourage new topics.
        timeout (int, optional): Timeout for the API call.
        max_retries (int, optional): Maximum number of retries for the API call.

    Returns:
        openai.types.Completion: The response object from OpenAI API.
    """
    # Initialize the OpenAI client with the provided API key and settings

    client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY']       , timeout=timeout, max_retries=max_retries)
    response = client.chat.completions.create(
        model=model,
        messages=messages, # type: ignore
        temperature=temperature,
        response_format = {"type": response_format},
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty)
    return response 


def map_float_to_category(
        value: float,
        min_val: float = -1,
        max_val: float = 1,
        num_categories: int = 21) -> int:
    """
    Maps a float value to a category within a specified range and division.

    Args:
        value (float): The value to categorize.
        min_val (float, optional): Minimum value of the range. Defaults to -1.
        max_val (float, optional): Maximum value of the range. Defaults to 1.
        num_categories (int, optional): Number of categories. Defaults to 21.

    Returns:
        int: The category number that the value belongs to.
    """
    # Ensure the value is within the specified range
    value = min(max_val, max(min_val, value))

    # Calculate the width of each category
    category_width = (max_val - min_val) / num_categories

    # Map the value to an integer category
    category = int((value - min_val) / category_width)

    return category


def load_prompt(
        path: str
        ) -> list[dict[str, str]]:
    """
    Loads a prompt from a JSON file at the specified path.

    Args:
        path (str): The file path to load the prompt from.

    Returns:
        list[dict[str, str]]: The loaded prompt.
    """
    with open(path, 'r', encoding='utf-8') as file:
        _prompt = json.load(file)
    return _prompt



def estimate_cost(dataframe_with_token_usage: pd.DataFrame):
    '''
    Function to calculate the cost after running a job using OpenAI API
    (the input is the output of running a job, were sentences have been already
    split into batches in order to optimise for a minimum number of tokens).
     note: costs below as of March 2024. OpenAI introduces new models/changes costs over time
     so the data below will need to be updated.
    '''
    COST_INPUT = {
        'gpt-4-1106-preview': 0.01 / 1000,
        'gpt-3.5-turbo-1106': 0.0015 / 1000,
        'gpt-4-0613': 0.03 / 1000
    }
    COST_OUTPUT = {
        'gpt-4-1106-preview': 0.03 / 1000,
        'gpt-3.5-turbo-1106': 0.002 / 1000,
        'gpt-4-0613': 0.06 / 1000
    }

   
    calls = dataframe_with_token_usage.groupby(['completion_id']).last()
    if any(model not in COST_INPUT.keys() for model in calls.model.unique()):
        raise ValueError(f'Missing pricing information for model.')
    cost_input_token = calls.model.map(COST_INPUT)
    cost_output_token = calls.model.map(COST_OUTPUT)
    prompt_cost = (calls.prompt_tokens * cost_input_token).sum()
    output_cost = (calls.completion_tokens * cost_output_token).sum()
    total_cost = prompt_cost + output_cost
    return {
        "prompt_cost": f'${prompt_cost:.4f}',
        "output_cost": f'${output_cost:.4f}',
        "total_cost": f'${total_cost:.4f}',
    }

