import asyncio
import os
from typing import Any, Coroutine, Optional, Union
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries for the complete implementation
from httpx import ReadTimeout
from typing import Any, Generator, Iterable, Optional, Union
import openai
from tqdm.asyncio import tqdm as async_tqdm
import hashlib
import pickle
from pathlib import Path

# ==============================================================================
# ORIGINAL IMPLEMENTATION - EXACTLY AS IN THE WORKFLOW
# ==============================================================================

def unmask_motivation(row, motivation_col='motivation'):
    """
    Function to remove masking from company names in motivations.
    """
    import re
    
    if isinstance(row[motivation_col], str):
        unmasked_string = re.sub(r'Target Company(_\d{1,2})?', row['entity_name'], row[motivation_col])
        if 'other_entities_map' in row.index:
            if row['other_entities_map']:
                if isinstance(row['other_entities_map'], str):
                    for key, name in eval(row['other_entities_map']):
                        unmasked_string = unmasked_string.replace(f'Other Company_{key}', str(name))
                elif isinstance(row['other_entities_map'], float):
                    pass
                else:
                    for key, name in row['other_entities_map']:
                        unmasked_string = unmasked_string.replace(f'Other Company_{key}', str(name))
    else:
        unmasked_string = None
    
    return unmasked_string


def convert_to_dict(obj) -> Union[list[Any], dict[Any, Any], Any]:
    """
    A function converts an object to a serializable dictionary.
    This is primarily used to convert ChatCompletion object to a dictionary.
    """
    if isinstance(obj, list):
        return [convert_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_dict(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        return {key: convert_to_dict(value) for key, value in obj.__dict__.items()}
    else:
        return obj


async def run_prompts(
        prompts: list,
        parameters: dict,
        concurrency: int,
        api_key: str,
        path_call_hash: str,
        path_result_hash: Optional[str] = None,
        max_retries: int = 1,
        timeout: int = 60,
        retry_delay: int = 1,
        exponential_backoff: bool = False
        ):
    """
    Executes a series of prompts concurrently using an external API, with support for retries on failure.
    """
    
    # Hash generation for caching
    def generate_hash(prompt, params):
        content = str(prompt) + str(params)
        return hashlib.md5(content.encode()).hexdigest()
    
    # Load cached results if available
    def load_cached_result(hash_key, path):
        cache_file = Path(path) / f"{hash_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    # Save result to cache
    def save_cached_result(hash_key, result, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        cache_file = Path(path) / f"{hash_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    
    async def process_single_prompt(prompt, semaphore, client):
        async with semaphore:
            hash_key = generate_hash(prompt, parameters)
            
            # Check cache first
            cached_result = load_cached_result(hash_key, path_call_hash)
            if cached_result:
                return cached_result
            
            for attempt in range(max_retries + 1):
                try:
                    response = await client.chat.completions.create(
                        messages=prompt,
                        **parameters,
                        timeout=timeout
                    )
                    
                    result = convert_to_dict(response)
                    
                    # Save to cache
                    if path_result_hash:
                        save_cached_result(hash_key, result, path_result_hash)
                    
                    return result
                    
                except Exception as e:
                    if attempt < max_retries:
                        delay = retry_delay * (2 ** attempt if exponential_backoff else 1)
                        await asyncio.sleep(delay)
                    else:
                        return None
    
    # Setup
    client = openai.AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    
    # Process all prompts concurrently
    tasks = [process_single_prompt(prompt, semaphore, client) for prompt in prompts]
    results = await async_tqdm.gather(*tasks)
    
    return results


def create_prompt_generator(
        sentences: pd.DataFrame,
        model: str,
        prompt: list,
        sentence_prompt_template: str,
        masked: bool = False,
        context_window_size: int = 16385,
        n_completion_tokens: int = 4096,
        n_expected_tokens_per_response: int = 100,
        chunk_size: int = 50,
        shuffle_chunk: bool = False,
        seed_chunk: Optional[int] = None,
        seed_data: Optional[int] = None,
        shuffle_data: bool = False,
        group_key: Optional[str] = None):
    """
    Creates a generator for prompts based on sentences.
    """
    
    # Calculate tokens per sentence (rough estimation)
    def estimate_tokens(text):
        return len(str(text)) // 4
    
    # Shuffle data if requested
    if shuffle_data:
        if seed_data:
            sentences = sentences.sample(frac=1, random_state=seed_data).reset_index(drop=True)
        else:
            sentences = sentences.sample(frac=1).reset_index(drop=True)
    
    # Process sentences in chunks
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences.iloc[i:i+chunk_size].copy()
        
        # Create user content for this chunk
        user_content = []
        for idx, row in chunk.iterrows():
            if masked:
                text = row.get('masked_text', row.get('text', ''))
                entity_name = 'Target Company'
            else:
                text = row.get('text', '')
                entity_name = row.get('entity_name', 'Target Company')
            
            sentence_text = sentence_prompt_template.format(
                id=row.get('sentence_id', f'sent_{idx}'),
                entity_name=entity_name,
                masked_entity_name='Target Company',
                **{f'filled_{col}': row.get(col, '') for col in chunk.columns}
            )
            user_content.append(sentence_text)
        
        # Create the complete prompt
        messages = prompt + [{"role": "user", "content": "\n".join(user_content)}]
        
        yield messages, chunk


def completion_to_dataframe(completion_response):
    """
    Converts completion response to DataFrame.
    """
    
    if not completion_response or not completion_response.get('choices'):
        return pd.DataFrame()
    
    try:
        content = completion_response['choices'][0]['message']['content']
        
        data = json.loads(content)
        
        # Convert to DataFrame format
        results = []
        for key, value in data.items():
            results.append({
                'id': key,
                'label': value.get('label', 'U'),
                'motivation': value.get('motivation', '')
            })
        
        return pd.DataFrame(results).set_index('id')
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Debug: JSON parsing failed: {e}")
        print(f"Debug: Content that failed: {content}")
        return pd.DataFrame()


async def async_extract_label(
        sentences: pd.DataFrame,
        system_prompt: str,
        n_expected_response_tokens: int,
        openai_api_key: str,
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
    """
    This function is used to extract labels from sentences asynchronously using OpenAI API.
    """

    prompt_preamble = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    model = parameters['model']
    
    # Handle both entity_id and rp_entity_id column names for compatibility
    entity_col = 'entity_id' if 'entity_id' in sentences.columns else 'rp_entity_id'
    
    sentences_unique = sentences.copy().drop_duplicates(
        subset=[entity_col, sentence_column])
    
    # Check if required columns exist
    required_cols = [entity_col, sentence_column]
    missing_cols = [col for col in required_cols if col not in sentences.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if masked:
        sentence_prompt_template = '{id};{masked_entity_name};"{filled_'+sentence_column+'}";'
    else:
        sentence_prompt_template = '{id};{entity_name};"{filled_'+sentence_column+'}";'

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


    
    # Use the run_prompts function with full concurrency
    completions = await run_prompts(
        all_prompt_messages,
        parameters,
        concurrency=concurrency,
        api_key=openai_api_key,
        path_call_hash=path_call_hash_location,
        path_result_hash=path_result_hash_location,
        timeout=timeout,
        max_retries=max_retries,
    )

    completion_df_list = []
    
    for idx, completion in enumerate(completions):

        completion_df = completion_to_dataframe(completion)
        completion_df_list.append(completion_df)

    combined_df = pd.concat(
        [sentence.merge(completion, how='left', left_on=['sentence_id'], right_index=True).assign(batch_idx=idx)
        for idx, (sentence, completion) in enumerate(zip(all_sentence_batches, completion_df_list))
        ], axis=0).reset_index(drop=True)

    sentences_w_label = sentences.merge(
        combined_df[[entity_col, sentence_column]
                    + combined_df.columns[sentences.shape[1]:].tolist()],
        how='left',
        on=[entity_col, sentence_column]
        )
    
    return sentences_w_label


def extract_label(**kwargs) -> Union[Coroutine, pd.DataFrame]:
    """
    This function is a wrapper for the async_extract_label function.
    """
    coroutine = async_extract_label(**kwargs)
    try:
        _ = asyncio.get_running_loop()
        return coroutine
    except RuntimeError:
        result = asyncio.run(coroutine)
        return result


async def process_sentences(sentences, sentence_column_2, masked_2, system_prompt, path_training, n_expected_response_tokens, batch_size, model, open_ai_credentials):
    
    sentence_final = []
    
    # Handle both entity_id and rp_entity_id column names for compatibility
    entity_col = 'entity_id' if 'entity_id' in sentences.columns else 'rp_entity_id'
    
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            sentences_labels_2 = await extract_label(
                sentences=sentences,
                sentence_column=sentence_column_2,
                masked=masked_2,
                system_prompt=system_prompt,
                path_call_hash_location=path_training,
                path_result_hash_location=None,  #
                n_expected_response_tokens=n_expected_response_tokens,
                batch_size=batch_size,
                concurrency=200,  # defines the number of calls made simultaneously
                openai_api_key=open_ai_credentials,
                parameters={
                    'model': model,
                    'temperature': 0,
                    'response_format': {'type': 'json_object'}
                }
            )

            
            sentence_final.append(sentences_labels_2)
            break  # Break out of retry loop if successful

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retry_attempts - 1:
                await asyncio.sleep(100)  # Wait for 100 seconds before retrying
            else:
                raise

    final = pd.concat(sentence_final)
    
    # Check if label and motivation columns exist before processing
    if 'label' in final.columns and 'motivation' in final.columns:
        final['label'] = final['label'].where(final['motivation'].str.contains('Target Company', regex=False, na=False), 'U')
        final['motivation'] = final.apply(unmask_motivation, axis=1)
    else:
        print(f"Warning: Expected columns 'label' and 'motivation' not found. Available columns: {final.columns.tolist()}")
        # Add dummy columns if they don't exist
        if 'label' not in final.columns:
            final['label'] = 'U'
        if 'motivation' not in final.columns:
            final['motivation'] = 'No motivation available'
    
    return final


def run_prompt(sentences, sentence_column_2, masked_2, system_prompt_ai_classification, path_training, n_expected_response_tokens, batch_size, model, open_ai_credentials):

    
    try:
        df_ai_classif = asyncio.run(
            process_sentences(
                sentences=sentences,
                sentence_column_2=sentence_column_2,
                masked_2=masked_2,
                system_prompt=system_prompt_ai_classification,
                path_training=path_training,
                n_expected_response_tokens=n_expected_response_tokens,
                batch_size=batch_size,
                model=model,
                open_ai_credentials=open_ai_credentials
            )
        )
        
        df_ai_classif.reset_index(inplace=True, drop=True)
        return df_ai_classif
    
    except Exception as e:
        raise


def compute_costs(df, text_column, system_prompt, model='gpt-4o-mini-2024-07-18'):
    """
    Estimate costs for DataFrame processing.
    """
    total_chars = df[text_column].str.len().sum()
    system_chars = len(system_prompt)
    
    # Approximation: 4 characters = 1 token
    input_tokens = (total_chars + system_chars * len(df)) / 4
    output_tokens = len(df) * 100  # Estimate 100 output tokens per row
    
    # Approximate costs for GPT-4o-mini (update with current prices)
    input_cost = input_tokens * 0.00015 / 1000  # $0.15 per 1K input tokens
    output_cost = output_tokens * 0.0006 / 1000  # $0.60 per 1K output tokens
    
    total_cost = input_cost + output_cost
    
    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_cost': total_cost
    }


# Trump-specific system prompt for business impact classification
DEFAULT_TRUMP_REELECTION_PROMPT_olod = """You are a financial analyst identifying companies expressing a view on the possible election of Donald Trump as President of the United States and how it would impact their business.
Your task is to determine if, based on the provided sentence, the Target Company is expressing a positive or negative view on Trump's election impact on their business operations.

Instructions:
1. Label each sentence for the Target Company: 'P' if the company mentions that Donald Trump's election will positively affect their business, 'N' if the company mentions that Trump's re-election will negatively impact their business, 'U' if unrelated or unclear.
2. Evaluate each sentence individually, focusing on whether the company is mentioning how Trump's policies or presidency would specifically affect THEIR business.
3. Use only the information in the sentence; do not infer from outside knowledge.
4. Ensure the text expresses a clear view on how the elections would impact the company's business operations, not just general political commentary.
5. Mentioning that the elections are upcoming, or asking a question about the elections does not imply expressing a business impact view.
6. A company that mentions how another company's business may be affected by the elections is not directly expressing their own business impact unless their business is also affected.
7. General economic or market commentary without specific reference to the company's business should be labeled 'U'.
8. You will be given a sentence ID, a company name, and the sentence text, for which you must assign the label. Your output should be a JSON object with a very brief motivation for the choice of the label and the label. The motivation must be one short sentence that starts with the company name and should explain why the label is 'P', 'N', or 'U' for that company, without summarizing the text. Format the JSON like: {{"<ID>": {{"motivation": <motivation>, "label": <label>}}, ...}}.

Example sentences and evaluations:

Example 1:
Sentence: "Target Company executives mentioned that a Trump administration's tax policies would significantly boost our profit margins and allow for expanded operations."
Motivation: Target Company expects their business to benefit from Trump administration tax policies.
Label: 'P'

Example 2:
Sentence: "Target Company's CEO warned that Trump's trade policies could disrupt our supply chain and increase costs substantially for our manufacturing operations."
Motivation: Target Company expects their business operations to be negatively impacted by Trump's trade policies.
Label: 'N'

Example 3:
Sentence: "Target Company analysts noted that the upcoming election between Trump and Biden could create market volatility."
Motivation: Target Company is providing general market commentary without mentioning specific business impact.
Label: 'U'

Example 4:
Sentence: "Target Company reported strong quarterly earnings despite concerns about the presidential election outcome."
Motivation: Target Company's earnings report does not express how Trump's election would affect their business.
Label: 'U'

Example 5:
Sentence: "On the Internet, we see news like the if Trump wins, the trade relations will change and the security situation will change. An of course, the relations with China and Taiwan comes into play. So the security is a big factor."
Motivation: Target_Company's discusses how Trump's election entails a security risk.
Label: 'N'

Example 6:
Sentence: "The former President Trump may win the upcoming presidential election. If that happens, what would be the strategy of Target Company or how will Target Company be prepared for the second Trump administration?"
Motivation: The sentence does not provide information about Target Company's view on the elections.
Label: 'U'
"""


DEFAULT_TRUMP_REELECTION_PROMPT = (
    f"""You are a financial analyst identifying companies expressing a view on the possible election of Donald Trump as President of the United States.
Your task is to determine if, based on the provided sentence, the Target Company is expressing a positive or negative view on Trump's election.

Instructions:
1. Label each sentence for the Target Company: 'P' if the company mention that Donald Trump's election will positively affect their business, 'N' if the company will be negatively impacted by Trump's re-election, 'U' if unrelated.
2. Evaluate each sentence individually, focusing on whether the company is mentioning Donald Trump.
3. Use only the information in the sentence; do not infer from outside knowledge.
4. Ensure the text expresses a clear view on the elections and the possible consequences. 
5. Mentioning that the elections are upcoming, or asking a question about the elections does not imply expressing a view.
6. A company that mentions how another company's business may be affected by the elections is not directly related to the elections unless its business is also affected.
7. You will be given a sentence ID, a company name, and the sentence text, for which you must assign the label. Your output should be a JSON object with a very brief motivation for the choice of the label and the label. The motivation must be one short sentence that starts with the company name and should explain why the label is 'Y', 'N', or 'U' for that company, without summarizing the text. Format the JSON like: {{"<ID>": {{"motivation": <motivation>, "label": <label>}}, ...}}.

Example sentences and evaluations:

Example 1:
Sentence: "Target Company executives mentioned that a Trump administration's tax policies would significantly boost our profit margins and allow for expanded operations."
Motivation: Target Company expects their business to benefit from Trump election.
Label: 'P'

Example 2:
Sentence: "Target Company's CEO warned that Trump's trade policies could disrupt our supply chain and increase costs substantially for our manufacturing operations."
Motivation: Target Company expects their business  to be negatively impacted by Trump's election.
Label: 'N'

Example 3:
Sentence: "Target Company analysts noted that the upcoming election between Trump and Biden could create market volatility."
Motivation: Target Company is providing general market commentary without mentioning specific business impact.
Label: 'U'

Example 4:
Sentence: "Target Company reported strong quarterly earnings despite concerns about the presidential election outcome."
Motivation: Target Company's earnings report does not express how Trump's election would affect their business.
Label: 'U'

Example 5:
Sentence: "On the Internet, we see news like the if Trump wins, the trade relations will change and the security situation will change. An of course, the relations with China and Taiwan comes into play. So the security is a big factor."
Motivation: Target_Company's discusses how Trump's election entails a security risk.
Label: 'N'

Example 6:
Sentence: "The former President Trump may win the upcoming presidential election. If that happens, what would be the strategy of Target Company or how will Target Company be prepared for the second Trump administration?"
Motivation: The sentence does not provide information about Target Company's view on the elections.
Label: 'U'
"""
)





def run_trump_reelection_prompt(sentences, sentence_column_2='text', masked_2=False, 
                              path_training='./output/', n_expected_response_tokens=100, 
                              batch_size=10, model='gpt-4o-mini-2024-07-18', open_ai_credentials=None):
    """
    Convenience wrapper function for Trump reelection business impact classification.
    
    This function uses a custom prompt specifically designed for analyzing how companies
    view Trump's reelection impact on their business operations.
    
    Args:
        sentences (pd.DataFrame): DataFrame containing sentences to classify
        sentence_column_2 (str): Name of the column containing the text (default: 'text')
        masked_2 (bool): If True, use masked text if available (default: False)
        path_training (str): Path to save intermediate results (default: './output/')
        n_expected_response_tokens (int): Number of expected tokens in response (default: 100)
        batch_size (int): Batch size for processing (default: 10)
        model (str): OpenAI model name to use (default: 'gpt-4o-mini-2024-07-18')
        open_ai_credentials (str): OpenAI API key
    
    Returns:
        pd.DataFrame: Original DataFrame with 'label' and 'motivation' columns added
        Labels: 'P' (Positive business impact), 'N' (Negative business impact), 'U' (Unrelated/Unclear)
    
    Example:
        >>> df_labeled = run_trump_reelection_prompt(
        ...     sentences=df,
        ...     open_ai_credentials='your-api-key'
        ... )
    """
    
    # Check if required columns exist
    if sentence_column_2 not in sentences.columns:
        return sentences
    
    try:
        result = run_prompt(
            sentences=sentences,
            sentence_column_2=sentence_column_2,
            masked_2=masked_2,
            system_prompt_ai_classification=DEFAULT_TRUMP_REELECTION_PROMPT,
            path_training=path_training,
            n_expected_response_tokens=n_expected_response_tokens,
            batch_size=batch_size,
            model=model,
            open_ai_credentials=open_ai_credentials
        )
        return result
    except Exception as e:
        raise


# Global variable for compatibility
sentence_column = 'text'


if __name__ == "__main__":
    print("Trump Reelection Labeling module loaded successfully!") 