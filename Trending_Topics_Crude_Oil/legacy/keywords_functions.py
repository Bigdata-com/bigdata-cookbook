import json
import numpy as np
import pandas as pd
import itertools
import string
import re
from openai import OpenAI
import openai
import os
import asyncio
from tqdm.asyncio import tqdm as tqdm_asyncio

def chunk_list(input_list, chunk_size):
    '''
    Function to chuck a list into sub-lists
    
    Parameters:
    - input_list (list): list to be chunked
    - chunk_size (integer): number of elements in each chunk
    
    Returns:
    - list of lists: list with the chunked lists
    '''
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

   
async def fetch_openai_response(theme, openAIkey, system_prompt, seed, rr0):
    client = openai.AsyncOpenAI(api_key=openAIkey)

    # Construct the prompt based on the seed
    if rr0 != "":
        print("Using example output for expansion:", rr0)
        system_prompt2 = (system_prompt + f"""
        Below is just an example output, please expand:
        {str(rr0)}
        """)
    else:
        system_prompt2 = system_prompt
    
    # Make an asynchronous call to OpenAI API
    response = await client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": system_prompt2},
            {"role": "user", "content": theme}
        ],
        temperature=0.25,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        seed=seed,
        response_format={"type": "json_object"}
    )

    return response.model_dump()['choices'][0]['message']['content']

# Main function to generate keywords combinations with asynchronous OpenAI calls
async def generate_keywords_combinations(theme, openAIkey, system_prompt, seeds=[123, 123456, 123456789, 456789, 789]):
    client = openai.AsyncOpenAI(api_key=openAIkey)

    # Placeholder for combined results
    keywords = {}
    rr0 = ""

    # Create a list of tasks for each seed
    tasks = []
    for seed in seeds:
        tasks.append(fetch_openai_response(theme, openAIkey, system_prompt, seed, rr0))

    # Use tqdm to track the progress of asynchronous tasks
    # responses = []
    # for task in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Fetching OpenAI Responses"):
    #     responses.append(await task)
    responses = await asyncio.gather(*tasks)

    print("OpenAI responses fetched successfully.")
    # Process each response
    for idx, rr in enumerate(responses):
        print(f"[run_generate_keywords] Raw response for seed {seeds[idx]}:", rr)
        rr = re.sub('```', '', rr)
        rr = re.sub('json', '', rr)
        try:
            concept_dict = json.loads(rr)
            print(f"[run_generate_keywords] Parsed JSON for seed {seeds[idx]}:", concept_dict)
            rr0 = concept_dict.copy()
            for s in iter(concept_dict):
                rr0[s] = [concept_dict[s][0]]

            for uu in concept_dict:
                keywords[uu] = keywords.get(uu, []) + concept_dict.get(uu, [])
        except Exception as e:
            print(f"[run_generate_keywords] JSON error for seed {seeds[idx]}: {e}")
            pass
    print(f"[run_generate_keywords] Combined keywords dict:", keywords)

    # Create all possible combinations of keywords
    kk = []
    for uu in keywords:
        kk.append(list(set(keywords[uu])))
    print(f"[run_generate_keywords] {len(kk)} keyword lists found.")
    print(f"[run_generate_keywords] Keyword lists: {kk}")
    if len(kk) > 1:
        keywords = list(itertools.product(*kk))
    else:
        keywords = kk[0]

    # Define function to split the keyword list into chunks
    def chunk_list(input_list, chunk_size):
        return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

    # Split keywords for query strings
    keywords_query = []
    if type(keywords[0]) == tuple:
        nkey = int(300 / len(keywords[0]))
        keywords = chunk_list(keywords, nkey)
        for kk in keywords:
            keywords_query.append('("' + '") OR ("'.join('" W/P "'.join(sub) for sub in kk) + '")')
    else:
        nkey = 300
        keywords = chunk_list(keywords, nkey)
        for kk in keywords:
            keywords_query.append('"' + '" OR "'.join(sub for sub in kk) + '"')

    # Flatten and clean up the keywords
    flat_keywords = [keyword for sublist in keywords for keyword in sublist]
    seen = set()
    unique_flat_keywords = []
    for keyword in flat_keywords:
        if keyword not in seen:
            unique_flat_keywords.append(keyword)
            seen.add(keyword)

    # Replace spaces with dashes in unique keywords
    keywords_with_dashes = [keyword.replace(' ', '-') for keyword in unique_flat_keywords]
    
    print(unique_flat_keywords)
    
    # Return the cleaned keywords and the query list
    return keywords_with_dashes, keywords_query

# Entry point to run the asynchronous function
def run_generate_keywords(theme, openAIkey, system_prompt, seeds=[123, 123456, 123456789, 456789, 789]):
    return asyncio.run(generate_keywords_combinations(theme, openAIkey, system_prompt, seeds))


# --- LexiconGenerator: streamlined class for prompt-to-keywords with consolidation ---
class LexiconGenerator:
    def __init__(self, openai_key, model="gpt-4o", seeds=None):
        self.openai_key = openai_key
        self.model = model
        if seeds is None:
            self.seeds = [123, 123456, 123456789, 456789, 789]
        else:
            self.seeds = seeds

    async def _fetch_keywords(self, theme, system_prompt, seed, rr0=None):
        client = openai.AsyncOpenAI(api_key=self.openai_key)
        
        if rr0:
            print("Using example output for expansion:", rr0)
            system_prompt2 = f"{system_prompt}\nBelow is just an example output, please expand:\n{str(rr0)}"
        else:
            system_prompt2 = system_prompt
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt2},
                {"role": "user", "content": theme}
            ],
            temperature=0.25,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            seed=seed,
            response_format={"type": "json_object"}
        )
        return response.model_dump()['choices'][0]['message']['content']

    async def _generate(self, theme, system_prompt):
        keywords = {}
        rr0 = None
        print("[LexiconGenerator] Using seeds:", self.seeds)
        tasks = [self._fetch_keywords(theme, system_prompt, seed, rr0) for seed in self.seeds]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for i, rr in enumerate(responses):
            if isinstance(rr, Exception):
                print(f"[LexiconGenerator] Seed {self.seeds[i]} failed with error: {rr}")
                continue
            print(f"[LexiconGenerator] Raw response for seed {self.seeds[i]}:", rr)
            rr = re.sub(r'```', '', rr)
            rr = re.sub(r'json', '', rr)
            try:
                concept_dict = json.loads(rr)
                print(f"[LexiconGenerator] Parsed JSON for seed {self.seeds[i]}:", concept_dict)
                rr0 = concept_dict.copy()
                for s in concept_dict:
                    rr0[s] = [concept_dict[s][0]]
                for k in concept_dict:
                    keywords[k] = keywords.get(k, []) + concept_dict.get(k, [])
            except Exception as e:
                print(f"[LexiconGenerator] Seed {self.seeds[i]} JSON error: {e}")
                continue
        print(f"[LexiconGenerator] Combined keywords dict:", keywords)
        # Consolidate all keywords
        all_keywords = []
        for klist in keywords.values():
            all_keywords.extend(klist)
        print(f"[LexiconGenerator] All keywords before deduplication:", all_keywords)
        # Remove duplicates, preserve order
        seen = set()
        unique_keywords = []
        for kw in all_keywords:
            if kw not in seen:
                unique_keywords.append(kw)
                seen.add(kw)
        print(f"[LexiconGenerator] Unique keywords after deduplication:", unique_keywords)
        #unique_keywords = [keyword.replace(' ', '-') for keyword in unique_keywords]
        return unique_keywords

    def generate(self, theme, system_prompt):
        """
        Synchronously generate a consolidated lexicon for a theme.
        """
        return asyncio.run(self._generate(theme, system_prompt))
    
def normalize_keywords_single(keyword_list):
    # Lowercase and remove dashes, treat dashed words as a single keyword
    return set(kw.replace('-', '').lower() for kw in keyword_list)

def normalize_keywords_single(keyword_list):
    # Lowercase and remove dashes, treat dashed words as a single keyword
    return set(kw.replace('-', ' ').lower() for kw in keyword_list)

# norm_keywords = normalize_keywords_single(keywords)
# norm_keywords_lex = normalize_keywords_single(keywords_lex)

# only_in_keywords = norm_keywords - norm_keywords_lex
# only_in_keywords_lex = norm_keywords_lex - norm_keywords
# overlap = norm_keywords & norm_keywords_lex

# print(f"Unique to keywords ({len(only_in_keywords)}):", sorted(only_in_keywords))
# print(f"Unique to keywords_lex ({len(only_in_keywords_lex)}):", sorted(only_in_keywords_lex))
# print(f"Overlap ({len(overlap)}):", sorted(overlap))