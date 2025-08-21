
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
    sentences_unique = sentences.copy().drop_duplicates(
        subset=['rp_entity_id', sentence_column])
    
    # Check if required columns exist
    required_cols = ['rp_entity_id', sentence_column]
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
        combined_df[['rp_entity_id', sentence_column]
                    + combined_df.columns[sentences.shape[1]:].tolist()],
        how='left',
        on=['rp_entity_id', sentence_column]
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
    cols_to_keep = list(sentences.columns) + ['label','motivation']

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
            
            sentences_labels_2 = sentences_labels_2
            sentence_final.append(sentences_labels_2)
            break  # Break out of retry loop if successful

        except Exception as e:
            
            if attempt < retry_attempts - 1:
                await asyncio.sleep(100)  # Wait for 100 seconds before retrying
            else:
                raise

    final = pd.concat(sentence_final)[cols_to_keep]
    
    final['label'] = final['label'].where(final['motivation'].str.contains('Target Company', regex=False),  'U')
    
    final['motivation'] = final.apply(unmask_motivation, axis=1)
    
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


# Predefined system prompt for liquid cooling classification (from original workflow)
DEFAULT_LIQUID_COOLING_PROMPT = """You are a financial analyst tasked with identifying companies that are either providers of liquid cooling technology in data centers, or end-users of liquid cooling in data centers. Liquid cooling technology also comprises water and underwater cooling, and immersion cooling, and it is applied to improve the energy efficiency of data centers. Providers develop, manufacture, and supply liquid cooling solutions and design products, such as servers units and data centers, equipped with liquid cooling. End-users operate and invest in data centers where liquid cooling solutions are installed and integrated in the infrastructure.
Your task is to determine if, based on the provided text, the Target Company's business activities are directly involved in providing liquid cooling technology to data centers or being end-user of liquid cooling in data centers.

Instructions:
1. Label each sentence for the Target Company: 'P' if, based on the text, the company provides liquid cooling technology to data centers. Assign an 'A' if the company is an end-user of liquid cooling in its data centers, assign 'U' if the text is NOT related to liquid cooling in data centers.
2. Evaluate each sentence individually, focusing on whether the Target Company is mentioned as a provider or an end-user of liquid cooling technology specifically in data centers.
3. Use only the information in the sentence; do not infer from outside knowledge.
4. Ensure that the text clearly mentions the company as a provider or end-user of liquid cooling in data centers. If the context in which Target Company is mentioned is not clear, assign 'U'.
5. Do not split the topic into core concepts. Pieces of text mentioning liquid cooling without mentioning data centers, or efficient cooling without mentioning liquid cooling, are not related to the topic.
6. Reporting on liquid cooling, or providing a market analysis on liquid cooling, does not imply providing or using liquid cooling in data centers.
7. Carefully analyse the sentence to ensure that the company's activities are explicitly linked to liquid cooling in data centers. If the company is mentioning another company, or a study on liquid cooling, it is likely that the company is a financial analyst, a consulting firm or a media publisher and as such it will not be a provider or end-user of liquid cooling solutions in data centers.
8. You will be given a text ID, a company name, and the text, for which you must assign the label. Your output should be a JSON object with a very brief motivation for the choice of the label and the label. The motivation must be one short sentence that starts with the company name and should explain why the label is 'P', 'A', or 'U' for that company, without summarizing the text. Format the JSON like: {{"<ID>": {{"motivation": <motivation>, "label": <label>}}, ...}}.

Example sentences and evaluations:

Example 1:
Text: "Target Company carved out its own niche by producing high-performance, liquid-cooled servers for demanding tasks."
Motivation: Target_Company provides high-performance liquid cooled servers.
Label: P

Example 2:
Text: "Target Company announced that it was adopting liquid cooling systems for AI data crunching."
Motivation: Target Company is an end-user of liquid cooling technology.
Label: A

Example 3:
Text: "As Target_Company wrote in a note last month, after hosting a webinar for clients on the topic of liquid cooling, "the underlying technology is harder to differentiate from vendor to vendor."
Motivation: The sentence does not mention that Target Company is either providing or using liquid cooling in data centers.
Label: U
"""


# Predefined system prompt for AI cost cutting classification
DEFAULT_AI_COST_CUTTING_PROMPT = """You are a financial analyst tasked with identifying organizations that are either providing AI solutions to help other companies reduce their operational costs, or using AI to reduce their own operational costs.
Your objective is to assess whether, based exclusively on the provided sentence, Target Company's business activities are DIRECTLY involved in providing AI solutions for cost reduction or using AI to reduce its own costs.

1. Label Assignment: For each sentence, assign a label to the Target Company:
   - Assign 'P' if Target Company is actively providing AI solutions to other companies specifically aimed at cost reduction.
   - Assign 'A' if Target Company is actively using AI to directly reduce its own operational costs.
   - Assign 'N' if Target Company is not involved in either providing AI solutions for cost reduction or using AI to reduce its own operational costs.

2. Context Evaluation: Evaluate each sentence individually from the perspective of the Target Company. Focus specifically on whether Target Company is directly providing AI solutions that assist other companies in reducing costs (label as 'P') or actively using AI to achieve its own cost reduction (label as 'A').

3. Information Use: When selecting each label, ONLY use the information available in the corresponding sentence; do NOT incorporate any external knowledge or context not present in the text provided.

4. Direct Involvement: Confirm that Target Company is directly involved in either providing AI solutions for cost reduction or using AI for its own cost reduction. Be mindful that mere statements about potential benefits, general research findings, predicted outcomes, or the importance of AI do not constitute a direct involvement in AI solutions or cost-cutting measures.

5. Avoid Misclassification:
   - Any sentence that includes phrases such as "predicts", "could", "underlines" or "highlights the importance of" should be classified as **not directly involved** in AI solutions.
   - Avoid sentences that portray hypothetical scenarios, future predictions, or general research summaries without evidence of specific AI implementations or projects.
   - Ensure that only sentences that clearly demonstrate active engagement in current AI projects or solutions that relate to operational cost reduction are labeled 'P' or 'A'.

6. Organization Type: Explicitly verify if the sentence pertains to a commercial entity. Organizations such as Media Companies, Consulting Firms, Research Centers, and any other entities focused primarily on research or general AI capabilities should be treated separately and not classified under 'P' or 'A' unless they are providing effective solutions to help companies reduce costs.

7. **Output Format**: You will be given a sentence ID, an organization name, and the sentence text. Your output should conform to the following JSON structure:
   {{"<ID>": {{"motivation": <motivation>, "label": <label>}}, ...}}.
   The motivation must be a concise statement starting with the organization name, explaining the label assigned ('P', 'A', or 'N') without summarizing the sentence text.

Example sentences and evaluations:

Example 1 (Provider):
Text: "Target Company's custom AI models, integrated with automation platforms, provide a reusable unified framework for operational cost reduction solutions. Cloud-native framework with extensible examples provide faster deployment. Cost Savings: Operational efficiency gains and reduction in technology needs translate to cost savings for client companies."
Motivation: Target Company provides AI solutions that help other companies reduce costs through operational efficiency gains.
Label: P

Example 2 (User):
Text: "By the end of 2024, Target Company aims to cut business costs by leveraging AI in service operations. We plan to optimize our delivery team, reducing expenses. Through AI chatbots and automated platforms, we'll provide seamless 24/7 customer support, efficient query resolution, increased productivity, and improved resource allocation for complex tasks."
Motivation: Target Company is reducing operational costs by leveraging AI in various service operations.
Label: A

Example 3 (Not involved):
Text: "Target Company predicts that AI tools could save call centers 80 million USD."
Motivation: Target Company is reporting on AI cost savings potential rather than directly providing or using AI solutions.
Label: N

Example 4 (Not involved):
Text: "The resource from Target Company highlights the importance of implementing AI in business operations."
Motivation: Target Company is discussing AI importance rather than directly providing or using AI cost reduction solutions.
Label: N
"""


def run_liquid_cooling_prompt(sentences, sentence_column_2='text', masked_2=False, 
                              path_training='./output/', n_expected_response_tokens=100, 
                              batch_size=10, model='gpt-4o-mini-2024-07-18', open_ai_credentials=None):
    """
    Convenience wrapper function for liquid cooling classification using the original workflow prompt.
    
    This function uses the exact same prompt and parameters as the original Liquid Cooling workflow,
    but provides default values for easier usage.
    
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
        Labels: 'P' (Provider), 'A' (Adopter/User), 'U' (Unrelated)
    
    Example:
        >>> df_labeled = run_liquid_cooling_prompt(
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
            system_prompt_ai_classification=DEFAULT_LIQUID_COOLING_PROMPT,
            path_training=path_training,
            n_expected_response_tokens=n_expected_response_tokens,
            batch_size=batch_size,
            model=model,
            open_ai_credentials=open_ai_credentials
        )
        return result
    except Exception as e:
        raise


def run_ai_cost_cutting_prompt(sentences, sentence_column_2='text', masked_2=False, 
                               path_training='./output/', n_expected_response_tokens=100, 
                               batch_size=10, model='gpt-4o-mini-2024-07-18', open_ai_credentials=None):
    """
    Convenience wrapper function for AI cost cutting classification using specialized prompt.
    
    This function classifies companies as either providers of AI cost cutting solutions (P),
    users of AI for their own cost reduction (A), or not involved (N).
    
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
        Labels: 'P' (Provider), 'A' (User), 'N' (Not involved)
    
    Example:
        >>> df_labeled = run_ai_cost_cutting_prompt(
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
            system_prompt_ai_classification=DEFAULT_AI_COST_CUTTING_PROMPT,
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
    print("Labeling module loaded successfully!") 