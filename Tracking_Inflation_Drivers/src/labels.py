"""
Copyright (C) 2024, RavenPack | Bigdata.com. All rights reserved.
Author: Alessandro Bouchs (abouchs@ravenpack.com)
"""


import asyncio
import json
import logging
import os
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from openai import AsyncOpenAI

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_NAME = 'gpt-4o-mini'
SEMAPHORE_COUNT = 1000

UNKNOWN_LABEL = 'U'
TARGET_ENTITY_MASK = 'Target Company'

##plot the distribution of labels
def plot_labels_distribution(df, label_column):
    
    total_records = len(df)

    # Create the main plot
    plt.figure(figsize=(12, 6))

    # Main plot

    rating_c = df[label_column].value_counts(normalize=False)
    rating_c = rating_c.sort_index()

    # Create the side plot for percentages
    plt.subplot(1, 2, 1)
    ax1=sns.barplot(x=rating_c.index, y=rating_c.values, palette='viridis', hue=rating_c.index)
    plt.title('Distribution of Labels (Total Records: {})'.format(total_records))
    plt.xlabel('Labels')
    plt.ylabel('Count')
    ax1.set_xticks(range(len(rating_c.index)))
    #ax1.set_xticklabels([labels_names_dict[item.get_text()].title() for item in ax1.get_xticklabels()])


    # Add percentages to the top of the bars
    for index, value in enumerate(rating_c.values):
        plt.text(index, value, f'{value:.0f}', ha='center', va='bottom')


    # Calculate percentages
    rating_percentages = df[label_column].value_counts(normalize=True) * 100
    rating_percentages = rating_percentages.sort_index()

    # Create the side plot for percentages
    plt.subplot(1, 2, 2)
    ax2 = sns.barplot(x=rating_percentages.index, y=rating_percentages.values, palette='viridis',hue=rating_percentages.index)
    plt.title('Percentage Distribution of Labels')
    plt.xlabel('Labels')
    plt.ylabel('Percentage')
    ax2.set_xticks(range(len(rating_percentages.index)))
    #ax2.set_xticklabels([labels_names_dict[item.get_text()].title() for item in ax2.get_xticklabels()])

    # Add percentages to the top of the bars
    for index, value in enumerate(rating_percentages.values):
        plt.text(index, value, f'{value:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


async def make_request_with_backoff_with_schema(system_prompt, prompt, schema, semaphore,
                                    max_retries=3, api_key=None):
    retries = 0
    while retries < max_retries:
        result = await make_request_with_schema(system_prompt, prompt, schema, semaphore, api_key)
        if result is not None:
            return result  # Success, return the result

        retries += 1
        logging.info(f"Retrying... (Attempt {retries}/{max_retries})")
        # Exponential backoff before retrying
        wait_time = 2 ** retries
        await asyncio.sleep(wait_time)
    logging.error("Max retries reached. Request failed.")
    return None

async def run_requests_with_schema(prompts, system_prompt, schema, replacements = None, api_key=None):
    
    system_prompt = generate_prompt(system_prompt, replacements)
    
    # Control concurrency (adjust the semaphore value according to API limits)
    semaphore = asyncio.Semaphore(
        SEMAPHORE_COUNT)  # Number of concurrent requests
    tasks = []
    for prompt in prompts:
        tasks.append(make_request_with_backoff_with_schema(system_prompt,
                                                           prompt,
                                                           schema,
                                                           semaphore,
                                                           api_key=api_key))

    # Gather and run the requests concurrently
    results = await asyncio.gather(*tasks)
    return results


def process_request_with_schema(prompts, system_prompt, schema, replacements = None, api_key=None):
    tic = time.perf_counter()
    responses = asyncio.run(run_requests_with_schema(prompts, system_prompt, schema, replacements, api_key))
    toc = time.perf_counter() - tic
    print(f"Completed {len(prompts)} requests in {toc:.2f} seconds.")
    return responses

async def make_request_with_schema(system_prompt, prompt, schema, semaphore, api_key):
    async with semaphore:
        try:
            client = AsyncOpenAI(api_key=api_key)
            response = await client.chat.completions.create(
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
                model="gpt-4o-mini",
                temperature = 0,
                top_p = 1,
                frequency_penalty = 1,
                presence_penalty = 1,
                response_format={
                    "type": "json_schema",
                    "json_schema": schema}
            )
            # Print or process the response
            r_str = response.model_dump()['choices'][0]['message']['content']
            r = re.sub('```', '', r_str)
            r = r.replace("⟩", "").strip()
            r = re.sub('json', '', r)
            
            data = json.loads(r)
            # Initialize the transformed dictionary
            transformed_dict = {}
            # Loop through each sentence in the "sentences" array
            for sentence in data.get("sentences", []):
                sentence_id = sentence.get("sentence_id")
                # Use the rest of the sentence object directly as the value
                if sentence_id is not None:
                    sentence_copy = sentence.copy()
                    del sentence_copy["sentence_id"]  # Remove sentence_id from the sub-object
                transformed_dict[sentence_id] = sentence_copy
            
            json_result = json.dumps(transformed_dict, indent=4)
            return json_result
        except Exception as e:
            print(e)
            print(response)
#             num_splits = [2,4,6,8,10]
#             for num_split in num_splits:
#                 try:

#                     split_texts = split_text_on_nearest_linebreak(prompt, num_splits=num_split)

#                     # Generate completions for all parts
#                     completions = []
#                     for part in split_texts:
#                         completion = get_chat_completion(client=openai_client, model="gpt-4o-mini-2024-07-18", system_prompt=system_prompt, text_string=part)
#                         completions.append(completion)

#                     # Consolidate the completions into one if more than one split
#                     if len(completions) > 1 and system_prompt_consolidation:
#                         final_completion = consolidate_completions(openai_client, system_prompt_consolidation, completions, model)
#                         r_str = final_completion.model_dump()['choices'][0]['message']['content']
#                         r = re.sub('```', '', r_str)
#                         r = r.replace("⟩", "").strip()
#                         r = re.sub('json', '', r)
#                         print(r)
#                         data = json.loads(r)
#                         # Initialize the transformed dictionary
#                         transformed_dict = {}
#                         # Loop through each sentence in the "sentences" array
#                         for sentence in data.get("sentences", []):
#                             sentence_id = sentence.get("sentence_id")
#                             # Use the rest of the sentence object directly as the value
#                             if sentence_id is not None:
#                                 sentence_copy = sentence.copy()
#                                 del sentence_copy["sentence_id"]  # Remove sentence_id from the sub-object
#                             transformed_dict[sentence_id] = sentence_copy

#                         json_result = json.dumps(transformed_dict, indent=4)
#                         return json_result

#                 except Exception as exception_message:
#                     error_message = str(exception_message)
            return None

system_prompt_consolidation = ("""
You are an expert summarizer and assistant. You will be given message contents with a JSON structure. Your task is to merge them into a single coherent completion that follows the same JSON structure, but the consolidating the text.

Please follow these guidelines while merging:
1. **Consistency**: Ensure that the final completion is logically structured and does not contradict itself.
2. **Flow**: Ensure a smooth flow between the parts. If any information is repeated or overlaps, combine it seamlessly. If any information is missing between the parts, bridge the gap.
3. **Preserve Details**: Make sure to include all relevant details from all completions.
4. **Avoid Redundancy**: If the completions contain redundant information, remove the repetition.
5. **Structure**: Ensure that the final completion is well-organized, using appropriate paragraphs or sections to maintain readability.

Please merge these completions into one consolidated text. Your output should be a well-structured, concise, and accurate message content.

Format the final output as a JSON object with the following structure:{
{{  "<Id>": {
    "summary": "<summary>"
  }
}}
""")
def split_text_on_nearest_linebreak(text_string, num_splits):
    """Splits the text string into `num_splits` parts, with each split occurring at the nearest line break.
    Also appends the start and last part of string1 to string2 for context."""

    split_texts = [text_string]
    
    for _ in range(num_splits - 1):  # We split num_splits - 1 times, the last one is automatic
        new_splits = []
        for text in split_texts:
            mid_index = len(text) // 2

            # Find the closest line break before or after the midpoint
            before_split = text.rfind('/', 0, mid_index)
            after_split = text.find('/', mid_index)

            # Choose the closest split point, favoring the one before the midpoint
            if before_split != -1:
                split_index = before_split
            elif after_split != -1:
                split_index = after_split
            else:
                # If no line break is found, split at the midpoint
                split_index = mid_index

            # Split the string into two parts
            string1 = text[:split_index]
            string2 = text[split_index:]

            # Take the start of string1 and append to the start of string2 (for context)
            start_of_string1 = string1.split('/')[:3]  # First 3 lines of string1
            last_part_of_string1 = string1.split('/')[-3:]  # Last 3 lines of string1

            # Append both to string2 for continuity
            string2 = '/'.join(start_of_string1) + '/'.join(last_part_of_string1) + '/' + string2

            new_splits.extend([string1, string2])  # Add both parts to the new_splits list

        split_texts = new_splits  # Update the split_texts with the newly split texts
    
    return split_texts

def consolidate_completions(client, system_prompt_consolidation, completions, model="gpt-4o-mini-2024-07-18"):
    """Send a consolidation request to merge multiple completions."""
    completion_text = '\n\n'.join([f"Completion {i+1}: {comp}" for i, comp in enumerate(completions)])
    
    consolidation_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt_consolidation},
            {"role": "user", "content": f"Please merge and consolidate these completions into a single response:\n\n{completion_text}"}
        ],
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return consolidation_response

def get_chat_completion(client, model, system_prompt, text_string):
    """Handles chat completion requests, splitting text if needed."""
    # Attempt to generate the completion
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_string}
        ],
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
        # Return the response
    return response.model_dump().get('choices', [{}])[0].get('message', {}).get('content', {})

def stringify_label_summaries(label_summaries):
    return [f'{label}: {summary}'
            for label, summary in label_summaries.items()]


async def make_request(system_prompt, prompt, semaphore, api_key):
    async with semaphore:
        try:
            client = AsyncOpenAI(api_key=api_key)
            response = await client.chat.completions.create(
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

def adjust_brace_position(input_string):
    """
    Adjusts the position of the first closing brace ('}') in the string.
    Removes the first '}' and appends it to the end of the string.

    Args:
        input_string (str): The input string to be modified.

    Returns:
        str: The modified string with the first '}' moved to the end.
    """
    # Find the position of the first '}'
    first_closing_brace_index = input_string.find('}')
    if first_closing_brace_index == -1:
        # If no closing brace is found, return the string as-is
        return input_string

    # Remove the first closing brace
    modified_string = (input_string[:first_closing_brace_index] + input_string[first_closing_brace_index + 1:]).strip()

    # Append the removed closing brace to the end
    modified_string += '}'

    return modified_string

def generate_prompt(system_prompt, replacements):
    if replacements:
        for key, value in replacements.items():
            placeholder = f'[{key}]'
            system_prompt = system_prompt.replace(placeholder, value)
        #print(system_prompt)

        return system_prompt
    else:
        return system_prompt

async def run_requests(prompts, system_prompt, replacements = None, api_key=None):
    
    system_prompt = generate_prompt(system_prompt, replacements)
    
    #print(system_prompt)
    
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


def process_request(prompts, system_prompt, replacements = None, api_key=None):
    tic = time.perf_counter()
    responses = asyncio.run(run_requests(prompts, system_prompt, replacements, api_key))
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
            # except AttributeError:
            #     print(deserialized_response)
            #     print(k)
            #     print(v)

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

def create_label_to_parent_mapping(tree):
    """
    Creates a mapping from each leaf node label to its parent node label.
    
    Args:
        tree (dict): The taxonomy tree structure with 'Label' and 'Children'.
    
    Returns:
        dict: A dictionary mapping child labels to their parent labels.
    """
    mapping = {}
    
    def traverse(node, parent_label=None):
        current_label = node.get('Label')
        children = node.get('Children', [])
        
        if parent_label and not children:
            # Leaf node: map its label to the parent label
            mapping[current_label] = parent_label
        
        for child in children:
            traverse(child, current_label)
    
    traverse(tree)
    return mapping

# Function to map risk_factor to risk_category
def map_risk_category(risk_factor, mapping):
    return mapping.get(risk_factor, 'Not Applicable')

## Assign up/down demand-pull/cost-pull labels
def get_topic_drivers_tag_prompt(main_theme, drivers_tags):
    # Create the prompt for determining market impact and magnitude   
    prompt = (
        f"""You are an expert macroeconomic analyst with deep knowledge of {main_theme} and its driving forces. Your task is to analyse a series of reports about the components of {main_theme} asked with evaluating the main driver of the dynamics involving THEME and TOPIC.
        
    Text Description:
    - The input text includes sentence ID identifier, and a TOPIC_SUMMARY report on how TOPIC is evolving and affecting THEME, and how this can influence the evolution of {main_theme} as a result.

    Instructions:
    1. Based on the provided text, evaluate the main driver picking the most appropriate label that better describes the current dynamic of TOPIC in THEME within the context of {main_theme} from the list:
        \n- {drivers_tags}

    2. Provide a one-sentence explanation of the main-driver that better describes the current dynamic of TOPIC in THEME within the context of {main_theme}. 
    
    Output:
    Return the result as a JSON object with the following structure:
    {{"<sentence_id>": {{"motivation": "<motivation>", "label": "<label>"}},...}}
    """
    )
    return prompt

labeling_system_prompt_template: str = """
Forget all previous prompts.
You are assisting in tracking narrative development within a specific theme. 
Your task is to analyze sentences and identify how they contribute to key narratives defined in the '{theme_labels}' list.

Please adhere to the following guidelines:

1. **Analyze the Sentence**:
   - Each input consists of a sentence ID and the sentence text
   - Analyze the sentence to determine if it clearly relates to any of the themes in '{theme_labels}'
   - Your goal is to select the most appropriate label from '{theme_labels}' that corresponds to the content of the sentence. 
   
2. **Label Assignment**:
   - If the sentence doesn't clearly match any theme in '{theme_labels}', assign the label 'unclear'
   - Evaluate each sentence independently, using only the context within that specific sentence
   - Do not make assumptions beyond what is explicitly stated in the sentence
   - You must not create new labels or choose labels not present in '{theme_labels}'
   - The connection to the chosen narrative must be explicit and clear

3. **Response Format**:
   - Output should be structured as a JSON object with:
     1. A brief motivation for your choice
     2. The assigned label
   - Each entry must start with the sentence ID
   - The motivation should explain why the specific theme was selected based on the sentence content
   - The assigned label should be only the string that precedes the colon in '{theme_labels}'
   - Format your JSON as follows:  [
    "sentence ID": "<sentence_id>",
    "motivation": "<motivation>",
    "label": "<label>",
   ]
   - Ensure all strings in the JSON are correctly formatted with proper quotes
"""
label_schema = {
    "name": "label",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sentence_id": {
                            "type": "integer",
                            "description": "The numeric ID corresponding to the sentence being analyzed."
                        },
                        "motivation": {
                            "type": "string",
                            "description": f"A one sentence explanation justifying the exposure label assigned."
                        },
                        "label": {
                            "type": "string",
                            "description": "The assigned label extracted from the allowed themes as defined in the prompt."
                        },
                    },
                    "required": ["sentence_id", "motivation", "label"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["sentences"],
        "additionalProperties": False
    }
}
def get_labeling_system_prompt(theme_labels: list) -> str: 
    """Generate a system prompt for labeling sentences with narrative labels."""
    return labeling_system_prompt_template.format(
        theme_labels=", ".join(theme_labels),
    )
