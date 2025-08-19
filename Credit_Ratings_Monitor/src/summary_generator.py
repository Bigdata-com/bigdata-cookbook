from concurrent.futures import ThreadPoolExecutor
from src_legacy.completion_processor import *
import openai
import asyncio
from tqdm.notebook import tqdm
import re
import json
system_prompt_consolidation = ("""
You are an expert summarizer and assistant. Your task is to merge multiple message contents into a comprehensive timeline report regarding the credit ratings of companies. Do not alter the structure of the inputs and focus on crafting a cohesive output.

Please follow these guidelines while merging:

1. **Consistency**: Ensure the final timeline is coherent, logically structured, and free from contradictions.
2. **Flow**: Maintain a smooth flow between parts. Seamlessly integrate information, especially where overlaps occur.
3. **Preserve Details**: Include all unique and relevant details from all completions, ensuring nothing important is lost.
4. **Avoid Redundancy**: Identify and eliminate redundant or repeated information. Focus on credit rating actions such as upgrades, downgrades, affirmations. If similar events are described in consecutive entries without new actions, merge them as a single entry. If contradictory events are reported in consecutive days, exclude these misreported events from the list.
5. **Structured Events**: Maintain a clear structure, ensuring the timeline reflects all major credit rating actions. Clearly outline each event with the appropriate date and source attribution.
6. **Highlight Novelty**: If information from different sources confirms the same event, consolidate it into one entry, using multiple sources for added credibility only when they offer new insights.

The final report should be a well-organized and accurate consolidation of the various inputs, capturing all significant changes without unnecessary repetition.

""")

system_prompt_summary_daily = ("""**Task: Report Credit Rating Information with Enhanced Source Tracking**

You are tasked with generating a comprehensive timeline report based on input texts regarding the credit ratings of companies, ensuring the inclusion of news source names and URLs where available. Your output should prioritize data from dates identified as having high novelty, indicating significant changes or developments in credit ratings or outlooks.

**Input Structure:**

- `Ratee Entity`: [Company Name]
- `DateX`: [YYYY-MM-DD hh:mm:ss]
- `TextX`: [Content regarding credit ratings, outlooks, or financial strategies]
- `SourcesX`: [Comma-separated list of source names and URLs]

**Instructions:**

1. **Identify Novelty**:
   - Focus on entries where significant changes or updates to credit ratings, outlooks, or financial strategies are reported.
   - Do not repeat similar updates if they span multiple consecutive days; consolidate into the most impactful date.
   - Ensure that the information spanning multiple consecutive days is not contradictory. Credit rating agencies are unlikely to announce back-to-back credit rating changes within a few days.

2. **Data Consolidation**:
   - Only include dates with distinct credit rating or outlook updates. Highlight changes in ratings, outlook revisions, or major financial movements.
   - Focus on the information that is discussed by more than one source, if available.
   - Ensure that details from multiple sources for the same event are consolidated under one date.

3. **Source Inclusion**:
   - For each credit rating update or outlook change, include all related source names and URLs.
   - Use the format: "[Source Name](URL)" in brackets after each summarized date entry.
   - Exclude entries with contradictory information from different sources.
   - DO NOT infer URLs from outside information. DO NOT use placeholder URLs. Leave the URL blank if no URL is available.
   - Only report Source Names that are given in the text. DO NOT infer sources from outside knowledge.

4. **Content Structure**:
   - Each entry in the summary should contain:
     - **Credit Ratings and Raters**: Summarize all involved raters and their assigned ratings.
     - **Credit Outlooks and Actions**: Emphasize any changes in outlooks or affirmations of ratings.
     - **Key Drivers**: Briefly explain the main factors influencing the rating or outlook decision.

5. **Output Format**:
   - Structure the output as a timeline of key credit rating events.
   - Keep each entry concise, no more than two sentences, while maintaining information clarity.
   - Avoid creating new dates or entries without valid credit data.

**Example Output**:

### Credit Report

- **2024-03-26**: Moody's placed Boeing Co.'s Baa2 senior unsecured rating and Prime-2 short-term rating on review for a potential downgrade due to concerns over their ability to manage debt and deliver enough 737 models, highlighting production challenges ([NBC San Diego](https://example.com), [Bloomberg Government](https://example.com)).
- **2024-04-24**: Moody's downgraded Boeing Co.'s credit rating to Baa3 from Baa2, indicating ongoing challenges and potential cash shortfalls against looming debt, marking a negative outlook ([WCVB.com ](https://example.com), [BNN Bloomberg](https://example.com)).
- ...
""")

system_prompt_summary_daily_deprecated = ("""**Task: Report Credit Rating Information with Focus on High Novelty**

You are tasked with generating a comprehensive report based on a sequence of structured input texts regarding the credit ratings of companies derived from news articles. Your output should prioritize data from dates identified as having high novelty, which indicate significant changes or updates.


The input text follows this structure:

- Ratee Entity: [the company who is the subject of the report]
- Date1: [YYYY-MM-DD hh:mm:ss]
- Text1: [Content]
- Sources1: [Comma-separated list of source names and URLs]
- Date2: [YYYY-MM-DD hh:mm:ss]
- Text2: [Content]
...
      
#### Instructions:
- Generate entries only for those dates that have novel information regarding the ratee entity and its credit ratings or credit outlooks. Consolidate days with similar information to avoid repetition.
- Ensure data integrity by not inventing additional dates or entries and by consolidating all data and rater inputs available for each date.
- If a date does not contain relevant information (e.g., no ratings or outlooks available for the ratee entity), exclude it from the report.
- Combine credit ratings and credit outlooks in the same timeline.
- Remember that the text you summarize has to be clearly related to credit ratings or outlooks assigned to debt instruments issued by the ratee entity.
- Ensure that there is no contradictory information: the same credit rating agency will not upgrade and downgrade the same company within the span of a few days from an announced credit rating decision.
- Focus on the information that is discussed by more than one source, if available.
- If one source reports contrasting information, i.e. a different credit rating from the same rater on the same day or compared to the previous day, discard this source and the text.

The output should adhere to the following structure:

### Credit Report
      - For each relevant date in the dataset, generate an entry no longer than two sentences capturing the following information. 
      - Credit rating and raters: <all pairs of credit ratings and raters related to the ratee>.
      - Credit Outlooks, Credit Actions, and Credit watchlist. Emphasize changes or affirmations of credit ratings and credit outlooks, such as downgrades, upgrades, stable outlooks, affirmed ratings, etc.
      - Short Term Ratings (<short term debt instruments and related credit ratings>), Long Term Ratings (<long term debt instruments and related credit ratings>) and debt instruments mentioned.
      - Short description of the key drivers and the forward guidance provided by the rater, supporting the credit rating or outlook change.
          - Key drivers are factors directly motivating the credit rating or credit outlook decision, and influencing the credit quality of the ratee entity. These include, but are not limited to, aspects such as cash flow generation, insider trading, capital structure changes, etc.
      - Ensure that the description is easy to read and understandable; do not simply reorganize the original input. Do not write more than two sentences.
      - Always quote in brackets ALL source names and URLs of every piece of information that you include in the report.. E.g. ([Seeking Alpha](url), [Nasdaq](url))
      - Discard dates and texts that have no sources quoted.
      - DO NOT infer URLs from outside information. DO NOT use placeholder URLs. Leave the URL blank if no URL is available.
      - Only report Source Names that are given in the text. DO NOT infer sources from outside knowledge.

- Your output should accurately reflect all substantive information per date without generating unnecessary or fabricated entries. Only include dates with available, valid credit rating or credit outlook data for the ratee entity.
- - Generate entries only for those dates that have novel information regarding the ratee entity and its credit ratings or credit outlooks. Consolidate days with similar information to avoid repetition. For example, if the same credit rating agency and credit rating are repeated over a few days, focus on the first date in which you saw this new information.
- Ensure that there is no contradictory information: the same credit rating agency will not upgrade and downgrade the same company within the span of a few days from an announced credit rating decision.
- Focus on the information that is discussed by more than one source, if available.
- If one source reports contrasting information, i.e. a different credit rating from the same rater on the same day or compared to the previous day, discard this source and the text.
""")

daily_summarization_prompt = (f"""
    Forget all previous instructions.
    You are tasked with consolidating and summarizing daily information from a sequence of news extracts related to corporate debt obligations and credit ratings.

    Your primary job is to consolidate information for entries that share the same date into a single cohesive string, ensuring you capture all relevant details retaining the original structure.

    Please follow these guidelines precisely:

    1. **Input Structure**:
        - Each input consists of the following structured fields:
            - Date: [the date and time of the text]
            - Ratee Entity: [the entity on which you should focus your summary]
            - Headline: [the headline of the news article]
            - Source Name: [the name of the source of the news article]
            - Url: [the url of the news article]
            - Text: [the text to read carefully]

    2. **Data Consolidation and Summary**:
        - For each Date, consolidate the texts into a summary string that captures all of the following:
            - Credit rating(s) associated with the ratee, paired with each corresponding rater to ensure clarity.
            - Identify and consolidate any changes in the credit rating, emphasizing actions such as credit rating upgrades, downgrades, or affirmations.
            - Capture credit rating status, such as credit rating in review.
            - Identify any credit outlook terms and changes, such as positive, negative, or stable credit outlook.
            - Capture watchlist placements and specific actions:, such as potential upgrades, downgrades, or unchanged
            - Specific debt instruments rated (i.e. long-term or short-term).
            - Current status of the credit outlook per rater.
            - Current credit watchlist status, if any.
            - Key drivers impacting the credit ratings, e.g. factors directly motivating the credit rating or credit outlook decision, and influencing the credit quality of the ratee entity. These include, but are not limited to, aspects such as cash flow generation, insider trading, capital structure changes, etc.
            - Comments on future guidance.
        - Highlight recent changes in the credit rating and credit outlook.
        - Ensure that you provide accurately the credit ratings issued by each rater.
    
    3. **Source Analysis and Attribution**
        - Focus on the information that is discussed by more than one source.
        - If one source reports contrasting information, i.e. a different credit rating from the same rater, discard this source and the text.
        - ALWAYS report the source name and corresponding URL of EACH article used to create the consolidated text. Add "\nSources: ..." at the end of the string.
            - If you have not used any content from a source, DO NOT include the source name.
            - Quote in brackets ALL source names and URLs of every piece of information that you include in the report. E.g. ([Source Name_X](URL_X), [Source Name_Y](URL_Y))
            - DO NOT infer URL from outside information. 
            - DO NOT complete the string without quoting the source names.
            - Leave the URL section blank if no URL is available in the input text.
        - Avoid using text that does not have a source name attached.
        - Only report Source Names that are given in the text. DO NOT infer sources from outside knowledge.

    4. **Output Format**:
        - Your output should be structured as a JSON object:
        {{
            "<id>": {{
                "daily_summary": "<daily_summary_string_generated>"
        }}
    }}
        - The`<daily_summary_string_generated> should be a single string that integrates all summarized and consolidated information as detailed above.
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
            before_split = text.rfind('\n', 0, mid_index)
            after_split = text.find('\n', mid_index)

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
            start_of_string1 = string1.split('\n')[:3]  # First 3 lines of string1
            last_part_of_string1 = string1.split('\n')[-3:]  # Last 3 lines of string1

            # Append both to string2 for continuity
            string2 = '\n'.join(start_of_string1) + '\n'.join(last_part_of_string1) + '\n' + string2

            new_splits.extend([string1, string2])  # Add both parts to the new_splits list

        split_texts = new_splits  # Update the split_texts with the newly split texts
    
    return split_texts

def consolidate_completions(client, system_prompt_consolidation, completions, model):
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
    # Use consolidation_response instead of response
    return consolidation_response.model_dump().get('choices', [{}])[0].get('message', {}).get('content', {})

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

def verify_information(text_string,system_prompt,
						  OPENAI_API_KEY, model = "gpt-4o-mini-2024-07-18"):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    i = 0
    while i < 5:
        try:
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
            completion = completion_to_dataframe(response.model_dump())
        # # Extracting the summary string from the response by parsing the JSON string
            # json_string = response.model_dump()['choices'][0]['message']['content'] 
            # json_string = re.sub('```', '', json_string)
            # json_string = re.sub('json', '', json_string)
            # json_string = json.loads(json_string)
            # print(json_string)
            # #summary_string = json_string["1"]["summary"]
            return completion
        except Exception as exception_message:
            error_message = str(exception_message)
            if 'context_length_exceeded' or 'string_above_max_length' in error_message:
                # If the error is due to exceeding the token limit, split the text
                print("Text is too long, splitting and retrying...")
                num_splits = [2,4,6,8,10]
                for num_split in num_splits:
                    try:

                        split_texts = split_text_on_nearest_linebreak(text_string, num_splits=num_split)

                        # Generate completions for all parts
                        completions = []
                        for part in split_texts:
                            completion = get_chat_completion(client, model, system_prompt, part)
                            completions.append(completion)

                        # Consolidate the completions into one if more than one split
                        if len(completions) > 1 and system_prompt_consolidation:
                            final_completion = consolidate_completions(client, system_prompt_consolidation, completions, model)
                            return final_completion
                    except Exception as exception_message:
                        error_message = str(exception_message)
#                        if 'context_length_exceeded' or 'string_above_max_length' in error_message:
#                            split_texts = split_text_on_nearest_linebreak(text_string, num_splits=4)
#
#                            # Generate completions for all parts
#                            completions = []
#                            for part in split_texts:
#                                completion = get_chat_completion(client, model, system_prompt, part)
#                                completions.append(completion)
#
#                            # Consolidate the completions into one if more than one split
#                            if len(completions) > 1 and system_prompt_consolidation:
#                                final_completion = consolidate_completions(client, system_prompt_consolidation, completions, model)
#                                return final_completion

            else:
                print(f"An error occurred on attempt {i}: {exception_message}")
                i += 1
    print(f"An error occurred after {i} attempts. Unable to retrieve the full summary.")
    return None


# Asynchronous function to gather summaries for the dataFrame
async def verify_information_with_asynchronous_calls_to_openai(industries_labels_df,analyze_col,system_prompt, OPENAI_API_KEY, model):
    
    def sanitize_input(input_string):
        # Remove invalid control characters
        return re.sub(r'[^\x20-\x7E\r\n]', '', input_string)

    industries_labels_df[analyze_col] = industries_labels_df[analyze_col].apply(sanitize_input)
    
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor() as executor:

        tasks = []
        for text_individual_row in tqdm(industries_labels_df[analyze_col], 
                                        desc="Verifying", 
                                        unit="text_individual_row"):
            
            tasks.append(loop.run_in_executor(executor, 
                                              verify_information, 
                                              text_individual_row,
                                              system_prompt,
											  OPENAI_API_KEY, 
                                              model))
        
        # Gather results while monitoring progress
        summaries_appended = await asyncio.gather(*tasks)
    
    return summaries_appended


# Main function to execute the summarization
def async_llm_pipeline(industries_labels_df,system_prompt,analyze_col, added_columns,replacements,
						   OPENAI_API_KEY, model = "gpt-4o-mini-2024-07-18"):
    
    # if 'id' not in industries_labels_df.columns:
    #     industries_labels_df['id'] = range(len(industries_labels_df))
    
    columns_to_keep = list(industries_labels_df.columns) + added_columns
    
    generated_prompt = generate_prompt(system_prompt, **replacements)
    
    # Assuming dataframe is already defined and contains the necessary data
    summarizations = \
        asyncio.run(verify_information_with_asynchronous_calls_to_openai(
											industries_labels_df,analyze_col,generated_prompt,
											OPENAI_API_KEY, model))
    

    responses = pd.concat(summarizations).reset_index().rename(columns={'index':'id'})
    industries_labels_df = industries_labels_df.merge(responses, how='left', on='id')

    return industries_labels_df[columns_to_keep]
    
def summarize_string(text_string, system_prompt, replacements,
                    OPENAI_API_KEY, model="gpt-4o-mini-2024-07-18", max_retries=5, max_split_retries=5):
    
    generated_prompt = generate_prompt(system_prompt, **replacements)
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    num_splits = [2, 4, 6, 8, 10]  # Increasing split sizes
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": generated_prompt},
                    {"role": "user", "content": text_string}
                ],
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            # Extracting the summary string from the response
            json_string = response.model_dump()['choices'][0]['message']['content'] 
            return json_string  # Return summary if successful

        except Exception as exception_message:
            error_message = str(exception_message)
            
            # Check for length-related error
            if 'context_length_exceeded' in error_message or 'string_above_max_length' in error_message:
                print("Text is too long, splitting and retrying...")

                # Attempt to split text and retry up to `max_split_retries`
                for split_attempt, num_split in enumerate(num_splits[:max_split_retries]):
                    try:
                        split_texts = split_text_on_nearest_linebreak(text_string, num_splits=num_split)

                        # Generate completions for all parts
                        completions = []
                        for part in split_texts:
                            completion = get_chat_completion(client, model, system_prompt, part)
                            completions.append(completion)

                        # Consolidate completions if necessary
                        if len(completions) > 1 and system_prompt_consolidation:
                            final_completion = consolidate_completions(client, system_prompt_consolidation, completions, model)
                            return final_completion
                        
                        # Return the completion if successful and no consolidation needed
                        return completions[0] if completions else None
                    
                    except Exception as nested_exception_message:
                        nested_error = str(nested_exception_message)
                        # Break out if the error isn’t length-related
                        if 'context_length_exceeded' not in nested_error and 'string_above_max_length' not in nested_error:
                            print(f"Encountered a persistent error during split attempt {split_attempt + 1}: {nested_exception_message}")
                            break
            else:
                # Handle non-length-related errors and retry if within `max_retries`
                print(f"An error occurred on attempt {attempt + 1}: {exception_message}")

    # Return None after all retries fail
    print(f"All attempts failed after {max_retries} main retries and {max_split_retries} split retries.")
    return None

def generate_prompt(system_prompt, **replacements):
    
    if replacements:
        for key, value in replacements.items():
            placeholder = f'[{key}]'
            system_prompt = system_prompt.replace(placeholder, value)
        return system_prompt
    else:
        return system_prompt

# ### Reviewed CODE
    
# def summarize_string_splits(text_string, system_prompt, replacements, OPENAI_API_KEY, model="gpt-4o-mini-2024-07-18",
#                     max_retries=5, max_split_retries=5, should_split_initially=False, num_splits=2, split_char = '\n'):
    
#     generated_prompt = generate_prompt(system_prompt, **replacements)
#     client = openai.OpenAI(api_key=OPENAI_API_KEY)
#     retry_splits = [2, 4, 6, 8, 10]  # Increasing split sizes

#     # Initial split if requested
#     if should_split_initially:
#         split_texts = split_text_on_character(text_string, num_splits, split_char)
#     else:
#         split_texts = [text_string]

#     for attempt in range(max_retries):
#         completions = []
#         for part in split_texts:
#             try:
#                 json_string = get_chat_completion(client, model, generated_prompt, part)
#                 completions.append(json_string)
#             except Exception as exception_message:
#                 error_message = str(exception_message)
#                 # Check for length-related error and split logic if necessary
#                 if 'context_length_exceeded' in error_message or 'string_above_max_length' in error_message:
#                     print("Text is too long, splitting and retrying...")
#                     # Handle split and retry with reduced text length recursively
#                     split_result = handle_split_and_retry(client, part, system_prompt, model, max_split_retries, retry_splits)
#                     if split_result:
#                         return completions.append(split_result)

#                 else:
#                     # Handle non-length-related errors
#                     print(f"An error occurred on attempt {attempt + 1}: {exception_message}")
        
#         if len(completions)>1:
#             final_summary = consolidate_completions(client, system_prompt_consolidation, completions, model)
#             return final_summary
#         else:
#             return completions[0]

#     print(f"All attempts failed after {max_retries} main retries and {max_split_retries} split retries.")
#     return None

# def handle_split_and_retry(client, text_string, system_prompt, model, max_split_retries, retry_splits):
#     """Handles splitting the text_string and retrying."""
#     for split_attempt, num_split in enumerate(retry_splits[:max_split_retries]):
#         try:
#             split_texts = split_text_on_character(text_string, num_splits=num_split, split_char = '\n')
#             completions = [get_chat_completion(client, model, system_prompt, part) for part in split_texts]
#             if len(completions) > 1:  # Use consolidation logic if needed
#                 return consolidate_completions(client, system_prompt_consolidation, completions, model)
#             return completions[0] if completions else None
#         except Exception as nested_exception_message:
#             nested_error = str(nested_exception_message)
#             if not ('context_length_exceeded' in nested_error or 'string_above_max_length' in nested_error):
#                 print(f"Error during split attempt {split_attempt + 1}: {nested_exception_message}")
#                 break
#     return None

# def generate_prompt(system_prompt, **replacements):
#     """Generates a prompt with placeholders replaced by actual values."""
#     for key, value in replacements.items():
#         placeholder = f'[{key}]'
#         system_prompt = system_prompt.replace(placeholder, value)
#     return system_prompt

# def split_text_on_character(text_string, num_splits, split_char='\n'):
#     """Splits the text string into `num_splits` parts using the specified character, maintaining context."""
    
#     # Split the text using the specified character
#     segments = text_string.split(split_char)

#     avg_length = len(segments) // num_splits
#     split_texts = []
    
#     for i in range(num_splits):
#         start_idx = i * avg_length
#         end_idx = (i + 1) * avg_length if i < num_splits - 1 else len(segments)

#         # Extracts each segment based on calculated indices
#         segment = segments[start_idx:end_idx]

#         # Add context: previous segment and current segment
#         if i > 0:
#             # Add context from the previous segment for continuity
#             previous_context = segments[max(0, start_idx - 1)]
#             segment = [previous_context] + segment
        
#         final_segment = split_char.join(segment).strip()  # Assembles the final segment
#         split_texts.append(final_segment)
    
#     return split_texts

# def consolidate_completions(client, system_prompt_consolidation, completions, model):
#     """Consolidates multiple completions into a single response."""
#     consolidation_input = '\n\n'.join([f"Completion {i+1}: {comp}" for i, comp in enumerate(completions)])
#     response = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": system_prompt_consolidation},
#             {"role": "user", "content": f"Please merge and consolidate these completions into a single response:\n\n{consolidation_input}"}
#         ],
#         temperature=0,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )
#     return response.model_dump().get('choices', [{}])[0].get('message', {}).get('content', {})

# def get_chat_completion(client, model, system_prompt, text_string):
#     """Handles chat completion requests."""
#     response = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": text_string}
#         ],
#         temperature=0,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )
#     return response.model_dump().get('choices', [{}])[0].get('message', {}).get('content', {})