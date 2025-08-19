from bigdata_client import Bigdata
from bigdata_client.models.search import DocumentType, SortBy
from bigdata_client.query import Keyword, Entity, Any
from bigdata_client.daterange import AbsoluteDateRange

from src_legacy.labels import get_prompts, process_request, deserialize_responses
from src_legacy.portfolio import get_entities, extract_entity_keys
from src_legacy.multithreading import run_search_concurrent, postprocess_search_results, build_dataframe
from src_legacy.content_search import inject_context_text, create_dates_lists

import time
from tqdm.notebook import tqdm
import pandas as pd
import hashlib
import plotly.express as px
from IPython.display import display, HTML
import re
import openai
import json

def run_daily_search_concurrent(bigdata,queries,sortby,scope,limit,start_date,end_date):
    start_dates, end_dates = create_dates_lists(start_date, end_date, 'D')
	
	# Prepare the argument list for parallel queries
    date_ranges = []
    for period in range(len(start_dates)):
        start_date = start_dates[period]
        end_date = end_dates[period]
        date_range = AbsoluteDateRange(start_date, end_date)
        date_ranges.append(date_range)
	
    results_list = []
    i = 0
    for dates in tqdm(date_ranges):
        i += 1
        if i == 100:
            print('Pausing the query...')
            time.sleep(60)
            i = 0
        results = run_search_concurrent(bigdata=bigdata,
                                            queries=queries,
                                            sortby=sortby,
                                            date_range=dates,
                                            scope=scope,
                                            limit=limit)
        results_list += [res for res in results]
    
    return results_list

def post_process_dataframe(df, entity_ids = list()):
    df_final = df.loc[df.rp_entity_id.isin(entity_ids)].reset_index(drop=True)
    return df_final

labeling_prompt = (f"""
Forget all previous instructions.

You are assisting a professional analyst in determining the role of a company in news extracts that describe **credit ratings and corporate debt conditions**.

Your primary task is to categorize the role of the company as either a rater, ratee, or unclear based on the provided text.

Please adhere to these guidelines:

1. **Analyze the Sentence**:
    - Each input consists of a sentence ID, a company name ("Entity_Name"), and the sentence text formatted as:
        - ID: [the id of the text]
        - Entity_Name: [the name of the company whose role you need to evaluate]
        - Headline: [the title of the news article]
        - Text: [the sentence requiring analysis]

    - **Identification of Roles**:
        - Determine the role of "Entity_Name" related to credit ratings and corporate debt:
            - **rater**: Entity_Name is actively involved in assigning or evaluating credit ratings for others. This typically includes credit rating agencies like Fitch Ratings, Moody's, and S&P, which only serve as raters.
            - **ratee**: Entity_Name is clearly the subject of a credit rating, i.e., receiving a credit rating or being explicitly assigned a credit rating. For Entity_Name to be a ratee, it has to be clear that the credit rating discussed is referred to Entity_Name.
            - **unclear**: Ambiguities in the text or references to non-credit ratings (e.g., stock ratings, general mentions) where the role of Entity_Name is not explicit.

    - **Role Descriptions and Examples**:
        - **rater**: 
          - Example: "Entity_Name assigned Other Company a credit rating of AAA."
          - Example: "Other Company has been assigned a credit rating of AAA by Entity_Name."
          - Example: "Entity_Name reviewed the BBB- rating of Other Company, maintaining a stable outlook."
        - **ratee**: 
          - Example: "Entity_Name's credit rating was upgraded to A+ by Another Company."
        - **unclear**: 
          - Example: "Entity_Name mentions a stable BBB+ rating is crucial, without assigning it themselves."
          - Example: "Entity_Name provides insights on corporate debt trends without specific rating involvement."

2. **Response Format**:
    - Output the analysis as a properly formatted JSON object:
    {{
        "<id>": {{
            "motivation": "<explanation of why the label was selected>",
            "label": "<assigned role: rater, ratee, or unclear>"
        }}
    }}
    - Start the motivation with Entity_Name, elaborating why the label was chosen based on sentence content.
    - Ensure accurate JSON syntax: use proper quotes and structure.

3. **Evaluation Instructions**:
    - Assess each sentence independently using provided sentence data.
    - Consider variations and abbreviations of company names:
      - "S&P" for "S&P Global".
      - "Moody's" for "Moody's Corp".
      - "Fitch" for "Fitch Ratings Inc."
      - "Verizon" for "Verizon Communications".
    - Example: "S&P affirmed the AA rating of Other Company." Here, "S&P" stands for "S&P Global" and it clearly acts as the **rater**.
    
""")

validation_prompt = (f"""
    Forget all previous instructions.

You are tasked with validating the role assigned to a company in news extracts that describe **credit ratings and corporate debt conditions**.

Your primary task is to assess whether the label assigned to the company (Entity_Name) is correct or if it needs to be changed based on the provided text and motivation.

Please adhere to these guidelines:

1. **Input Structure**:
    - Each input consists of the following:
        - ID: [the id of the text]
        - Entity_Name: [the name of the company whose role you need to evaluate]
        - Headline: [the title of the news article]
        - Text: [the sentence requiring analysis]
        - Motivation: [the explanation provided for the assigned label]
        - Label: [the current label assigned: 'rater', 'ratee', or 'unclear']

2. **Label Descriptions and Examples**:
    - **'rater'**: If Entity_Name is involved in assigning a credit rating to another company or evaluating that company’s credit rating. Credit rating agencies, such as Fitch Ratings, Moody's, and S&P, always act as raters and never as ratees.
        - *Example*: "Entity_Name assigned Other Company a credit rating of AAA."
        - *Example*: "Other Company has been assigned a credit rating of AAA by Entity_Name."
    - **'ratee'**: If a credit rating agency has assigned a credit rating to Entity_Name. When it is clear that Entity_Name is receiving a credit rating, it must be labelled as ratee. For Entity_Name to be considered a ratee, the text has to explicitly mentioned that the credit rating discussed is referred to Entity_Name.
        - *Example*: "Entity_Name's credit rating was upgraded to A+ by Other Company."
    - **'unclear'**: If Entity_Name is mentioned in contexts where it is not explicitly involved in issuing or receiving a credit rating, or when the roles are ambiguous.
        - *Example*: "The investment-grade Other Company credit rating of BBB+ is relevant for Entity_Name, but does not imply that Entity_Name is a rater."

3. **Validation Process**:
    - Carefully read the provided Text and Motivation.
    - Based on the context in the Text, determine if the assigned Label accurately reflects the role of Entity_Name.
    - Maintain the Label if it is correct.
    - If the Label is incorrect, select the appropriate label ('rater', 'ratee', or 'unclear') based on the text's context focusing on credit ratings and corporate debt instruments.
    - If the label is 'unclear', verify if there is clear evidence of Entity_Name's involvement in rating activities. Correct this to 'rater' or 'ratee' if the Text indicates that a credit rating is explicitly assigned by or to Entity_Name.
        - Example: "Verizon has a BBB+/Baa1 rating by S&P and Moody's." In this case, both S&P and Moody's should be labeled as 'rater'. Change from 'unclear' to 'rater' if incorrectly labeled.
    - Consider name variations and abbreviations such as "S&P" for "S&P Global", "Moody's" for "Moody's Corp", "Fitch" for "Fitch Ratings Inc.", "Verizon" for "Verizon Communications", etc.
    - Credit rating agencies, such as Fitch Ratings, Moody's, and S&P, always act as raters and never as ratees.

4. **Response Format**:
    - Your output should be structured as a JSON object as follows:
    {{
        "<id>": {{
            "validated_label": "<the updated or confirmed label based on your validation>",
            "validated_motivation": "<your rationale for the validated label>"
        }}
    }}
""")

extraction_prompt_1 = (f"""
    Forget all previous instructions.

You are tasked with extracting credit ratings and relevant insights for a single ratee entity from the provided text, based on a single rater entity.

1. **Input Structure**:
    - Each input consists of:
        - ID: [the unique ID for the input text]
        - sentence_key: [a unique identifier for the sentence]
        - text: [the text to read carefully]
        - rater_entity: [the company acting as the rater]
        - ratee_entity: [the company being rated]
        - unclear_entities: [a list of companies that could be either raters or ratees]

2. **Extraction Guidelines**:
    - **Context**:
        - Ensure that the context pertains to **corporate debt obligations** and **corporate credit ratings**.

    - **Data Features to Extract**:
        - **Credit Rating**:
            - Extract any specific credit rating assigned by the **rater_entity** for the **ratee_entity**, such as "AAA," "BBB+," "Baa1," etc.
            - Capture ratings marked as **confirmed** or **reaffirmed** if present.

        - **Credit Rating Action**:
            - Identify any changes in the credit rating, emphasizing actions such as:
                - **Upgrade**: An improvement in the rating.
                - **Downgrade**: A decrease in the rating.
                - **Affirmed**: Rating confirmed with no change.
                - **Corrected**: Adjusted due to an error.
                - **Withdrawn**: The rating is removed.
                - **Reinstated**: A withdrawn rating is restored.

        - **Credit Rating Status**:
            - Capture status changes such as:
                - **Provisional Rating**: A preliminary rating.
                - **Matured or Paid in Full**: When the obligation reaches maturity.
                - **No Rating**: Rating declined or unavailable.
                - **Published**: Officially issued or announced.

        - **Credit Outlook**:
            - Identify any outlook terms such as:
                - **Positive**: Suggests potential improvement.
                - **Negative**: Indicates potential downgrade.
                - **Stable**: No expected change.
                - **Developing**: Change possible based on future events.

        - **Credit Watchlist**:
            - Capture watchlist placements and specific actions:
                - **Watch**: The rating is on a watchlist.
                - **Watch Positive**: Potential upgrade.
                - **Watch Negative**: Suggests downgrade.
                - **Watch Removed**: No longer active.
                - **Watch Unchanged**: Status remains without change in expectation.

    - **Extraction Logic**:
        - For each identified feature (rating, action, status, outlook, watchlist), link the **rater_entity** and **ratee_entity**.
        - If the text contains multiple ratings or statuses, extract each as a separate entry.

3. **Output Format**:
    - Your output should be structured as a JSON object:
    {{
        "<id>": {{
            "sentence_key": "<sentence_key as received in the input>",
            "credit_rating": "<credit_rating_extracted or None>",
            "credit_action": "<credit_action_extracted or None>",
            "credit_status": "<credit_status_extracted or None>",
            "credit_outlook": "<credit_outlook_extracted or None>",
            "credit_watchlist": "<credit_watchlist_extracted or None>",
            "validated_rater": "<the rater entity of the rating extracted>",
            "validated_ratee": "<the company receiving the rating>"
        }}
    }}
    - If no valid details are found, set the corresponding fields as empty strings or `None`.

**Examples**:

1. **Credit Rating and Action Example**:
   - Text: "Moody's upgraded Entity_Name's credit rating to A- from BBB+, reflecting the company's improved financial outlook."
   - Extracted Credit Rating: "A-"
   - Credit Action: "Upgrade"
   - Credit Outlook: None
   - Validated Rater: "Moody's"
   - Validated Ratee: "Entity_Name"

2. **Credit Outlook and Watchlist Example**:
   - Text: "S&P placed Entity_Name on Watch Negative due to potential risks but affirmed its BBB rating."
   - Extracted Credit Rating: "BBB"
   - Credit Action: "Affirmed"
   - Credit Outlook: None
   - Credit Watchlist: "Watch Negative"
   - Validated Rater: "S&P"
   - Validated Ratee: "Entity_Name"

3. **Credit Status Example**:
   - Text: "Entity_Name's provisional rating by Fitch was published this morning, indicating a positive future outlook."
   - Extracted Credit Rating: None
   - Credit Action: None
   - Credit Status: "Provisional Rating"
   - Credit Outlook: "Positive"
   - Validated Rater: "Fitch"
   - Validated Ratee: "Entity_Name"

These examples provide illustrations of extracting all necessary information, focusing on the credit ratings, actions, statuses, outlooks, and watchlists while correctly associating the rater and ratee entities.
""")

extraction_prompt_2 = (f"""
    Forget all previous instructions.
    You are assisting a professional analyst in extracting short-term and long-term credit ratings related to a single ratee entity from the provided text.

    Your primary task is to extract relevant information pertaining specifically to the identified rater entity and ratee entity.

    1. **Input Format**:
        - Each input consists of:
            - ID: [the id of the text]
            - sentence_key: [the sentence key]
            - text: [the text to read carefully]
            - rater_entity: [the company acting as the rater]
            - ratee_entity: [the company acting as the ratee]

    2. **Extraction Guidelines**:
        - Ensure that the text discusses **corporate debt obligations** and **corporate credit ratings**.

        - **Extraction Steps**:
            - **short_term_credit_rating**: Extract any short-term credit rating assigned by the rater entity to the ratee entity. Short-term ratings may include instruments like short-term loans, commercial paper, etc.
            
            - **long_term_credit_rating**: Extract any long-term credit rating associated with the ratee entity discussed by the rater entity.

            - **debt_instrument**: Identify the debt instruments mentioned by the rater entity and referenced in relation to the ratee entity.

        - If no explicit information is available for any fields, return an empty string for that field in the JSON response.

    3. **Response Format**:
        - Your output should be structured as follows:
        {{
            "<id>": {{
                "sentence_key": "<sentence_key as received in the prompt>",
                "short_term_credit_rating": "<short_term_credit_rating_extracted or None>",
                "long_term_credit_rating": "<long_term_credit_rating_extracted or None>",
                "debt_instrument": "<associated_debt_instrument or None>"
            }},
            ...
        }}
        - If no information is available for a specific field, ensure that it is represented as an empty string.
""")

extraction_prompt_3 = (f"""
    Forget all previous instructions.
    You are assisting a professional analyst in extracting relevant insights regarding credit ratings and associated contextual elements for a single ratee entity from the provided text.

    Your primary task is to extract detailed information about the rationale, forward guidance, and key drivers impacting the credit rating of the identified ratee.

    1. **Input Format**:
        - Each input consists of the following fields:
            - ID: [the id of the text]
            - sentence_key: [the sentence key]
            - text: [the text to read carefully]
            - rater_entity: [the company acting as the rater]
            - ratee_entity: [the company acting as the ratee]

    2. **Extraction Guidelines**:
        - Ensure that the context pertains to **corporate debt obligations** and **corporate credit ratings**.

        - **Extraction Steps**:

            - **forward_guidance**: Capture any guidance regarding future credit ratings, including recommendations or comments made by the rater entity about the ratee entity's situation.

            - **key_drivers**: Extract factors motivating the credit rating or outlook decision, and influencing the credit quality of the ratee entity, these include, but are not limited to, aspects such as:
                - cash flow generation (e.g. earnings, revenues, dividends, assets)
                - insider trading, stock prices, stock picks
                - capital structure changes (e.g. equity actions, acquisitions, mergers)

        - If no explicit information is available for any of the fields mentioned above, return an empty string for that field in the JSON response.

    3. **Response Format**:
        - Your output should be structured as follows:
        {{
            "<id>": {{
                "sentence_key": "<sentence_key as received in the prompt>",
                "forward_guidance": "<forward_guidance_info>",
                "key_drivers": "<list_of_key_drivers or None>"
            }},
            ...
        }}
        - If no information is available for a specific field, ensure that it is represented as an empty string.
""")

extraction_prompt_deprecated = (f"""
    Forget all previous instructions.
    You are assisting a professional analyst in extracting credit ratings and relevant insights associated with the company referred to as "Ratee Entities".

    Your primary task is to extract detailed information regarding credit ratings and associated contextual elements.

    Please follow these guidelines precisely:

    1. **Information Extraction**:
        - Each input consists of a sentence ID, a sentence key, the sentence text, the indication of the rater entities, ratee entities, and unclear entities.
        
        - Ensure that the topic discussed in the text is corporate debt obligations and corporate credit ratings.
        - Focus on the information that is discussed by the rater entities with reference to the ratee entities.
        
        - Extract the following pieces of information:
            - **credit_rating**: The overall credit rating assigned to a Company by a credit rating agency.
                - Focus on actual ratings such as 'AAA', 'AA-', 'BB+', 'investment grade', 'speculative grade', etc., specifically assigned to the company. The overall credit rating may coincide with the long term credit rating or the short term credit rating depending on the sentence.
            -  **credit outlook**: Any mention of the credit rating being assessed in the coming weeks, months or years. Any state such as 'stable', 'positive', 'negative','downgrade', etc. If a prior state is mentioned but is irrelevant due to the current review status changing, it should not be considered. If both short-term and long-term outlooks are mentioned, collect both.
                
            - **short_term_credit_rating**: Extract any credit discussed and specifically referred to a short-term debt instrument, if mentioned. Short-term debt instruments include, among others, Accounts Payable, Short-Term Loans (due to be paid within 12 months or a year), Commercial Paper, Lease Payments, Taxes Due, Salaries and Wages, Stock Dividends (not yet paid). 
                
            - **long_term_credit_rating**: Extract any credit rating discussed and specifically referred to a long-term debt instrument, if mentioned.
            
            - **rationale**: Provide the rationale or reasoning behind the credit rating if it is explicitly mentioned in the text.
            
            - **forward_guidance**: Capture any forward guidance discussed regarding current or future credit ratings, including any potential changes or outlook updates and actions that will be undertaken by the company and discussed by the credit rating agency.
            
            - **debt_instrument**: Identify all debt instruments mentioned, which can be specific types of corporate obligations.
        
        - If no explicit information is available for any of the fields mentioned above, return an empty string for that field in the JSON response.

    2. **Response Format**:
        - Your output should be structured as a JSON object as follows:
        {{
            "<id>": {{
                "sentence_key": "<sentence_key as received in the prompt>",
                "credit_rating": "<credit_rating_extracted>",
                "credit_outlook": "<credit_outlook_extracted>",
                "short_term_credit_rating": "<list_of_short_term_credit_rating_extracted or None>",
                "long_term_credit_rating": "<list_of_long_term_credit_rating_extracted or None>",
                "rationale": "<rationale_for_credit_rating>",
                "forward_guidance": "<forward_guidance_info>",
                "debt_instrument": "<list_of_debt_instrument_under_study or None>"
            }},
            ...
        }}
        - If no information is available for a specific field, ensure that it is represented as an empty string.
""")


def generate_hash(row, columns):
    # This feature is a work in progress
    # Concatenate the rater, ratee, and credit rating as a unique string
    hash_input = "-".join(str(row[col]) for col in columns)
    return hashlib.sha256(hash_input.encode()).hexdigest()  # Create a SHA-256 hash

def generate_novelty(df, columns_to_hash):
    # This feature is a work in progress
    df['novelty_hash'] = df.apply(generate_hash, axis=1, columns=columns_to_hash)
    
    df['date'] = pd.to_datetime(df['date'])

    # Step 2: Sort by hash and date to calculate time difference
    df = df.sort_values(by=['date', 'novelty_hash']).reset_index(drop=True)

    # Step 3: Calculate the time difference between consecutive entries with the same hash
    # Group by the hash and calculate the time difference within each group
    df[f'novelty_{columns_to_hash[-1]}'] = df.groupby('novelty_hash')['date'].diff().dt.days.fillna(365)
    
    #df = df.drop('hash', axis=1)
    
    return df

def clean_dataset(df):
    
    if 'contextualized_chunk_text' in df.columns and 'text' not in df.columns:
        cols_visualize = ['date', 'sentence_id', 'headline','source_name','url', 'contextualized_chunk_text','ratee_entity_rp_entity_id', 'ratee_entity', 'rater_entity',
                          'credit_rating', 'credit_outlook',
       'credit_action', 'credit_status','credit_watchlist','short_term_credit_rating',
       'long_term_credit_rating', 'debt_instrument', 
       'forward_guidance', 'key_drivers']
        
    elif 'text' in df.columns and 'contextualized_chunk_text' not in df.columns:
        
        cols_visualize = ['date', 'sentence_id','headline','source_name','url', 'text','ratee_entity_rp_entity_id', 'ratee_entity', 'rater_entity', 'credit_rating', 'credit_outlook',
       'credit_action', 'credit_status','credit_watchlist','short_term_credit_rating',
       'long_term_credit_rating', 'debt_instrument', 
       'forward_guidance', 'key_drivers']
        
    elif 'text' in df.columns and 'contextualized_chunk_text' in df.columns: 
        
        cols_visualize = ['date', 'sentence_id','headline','source_name','url', 'text', 'contextualized_chunk_text','ratee_entity_rp_entity_id', 'ratee_entity', 'rater_entity',
                          'credit_rating', 'credit_outlook',
       'credit_action', 'credit_status','credit_watchlist','short_term_credit_rating',
       'long_term_credit_rating', 'debt_instrument', 
       'forward_guidance', 'key_drivers']
        
    elif 'text' not in df.columns and 'contextualized_chunk_text' not in df.columns:
        
        cols_visualize = df.columns
        
    df_clean = df.applymap(lambda x: None if x == '' or x == [] or x == 'None' else x)
    df_clean = df_clean[cols_visualize].dropna(subset=['ratee_entity_rp_entity_id']).dropna(subset=['rater_entity']).copy()
    
    return df_clean.reset_index(drop=True)

def group_text_and_labels(df):
    grouped_df = df.groupby(['date', 'sentence_id', 'headline','source_name', 'contextualized_chunk_text', 'url'], dropna=False).apply(
    lambda x: pd.Series({
        'rater_entity': list(x.loc[x['validated_label'] == 'rater', 'entity_name']),
        'ratee_entity': list(x.loc[x['validated_label'] == 'ratee', 'entity_name']),
        'unclear_entities': list(x.loc[x['validated_label'] == 'unclear', 'entity_name']),
    })
).reset_index()
    # Explode both `rater_entities` and `ratee_entities`
    exploded_df = grouped_df.explode('rater_entity', ignore_index=True).explode('ratee_entity', ignore_index=True)
    exploded_df['id'] = range(len(exploded_df))
    return exploded_df

def generate_summary_by_date(
    df,
    date_col='date',
    sentence_id_col='sentence_id',
    fields_for_summary=None
):
    """
    Generates a structured summary input by grouping first by date and sentence ID,
    and then by date to consolidate all summaries for that day.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing rows with multiple dates and sentence IDs.
        date_col (str): The column name representing dates.
        sentence_id_col (str): The column name representing sentence IDs.
        fields_for_summary (list): List of fields to include in the summary string for each sentence.

    Returns:
        pd.DataFrame: DataFrame with one row per date and a 'summary_input' column summarizing the day's grouped data.
    """

    # Ensure NaN values are replaced with empty strings
    df = df.fillna('None').applymap(lambda x: "; ".join(x) if isinstance(x, list) else x)

    # Default fields for summary: include all columns except date and sentence_id
    if fields_for_summary is None:
        fields_for_summary = [col for col in df.columns if col not in [sentence_id_col]]

    # Step 1: Group by `date` and `sentence_id` to consolidate all fields for each sentence
    def aggregate_sentence_fields(group):
        """
        Consolidates all fields for the same `date` and `sentence_id`.
        For example, it combines all raters, ratings, and text into single fields.
        """
        aggregated = {col: "; ".join(filter(None, group[col].unique())) for col in group.columns if col not in [date_col, sentence_id_col]}
        return pd.Series(aggregated)

    sentence_grouped = df.groupby([date_col, sentence_id_col], dropna=False).apply(aggregate_sentence_fields).reset_index()

    # Step 2: Create the `sentence_summary` for each sentence
    def create_sentence_summary(row):
        """
        Constructs a structured string summarizing all the fields for a single sentence ID.
        """
        summary = [f"{field.replace('_', ' ').title()}: {row[field]}" for field in fields_for_summary if row[field]]
        return "\n".join(summary)

    sentence_grouped['sentence_summary'] = sentence_grouped.apply(create_sentence_summary, axis=1)

    # Step 3: Group by `date` to consolidate all sentence summaries and other fields
    def aggregate_date_fields(group):
        """
        Consolidates all `sentence_summary` strings and other fields for the same date.
        """
        aggregated = {col: "; ".join(filter(None, group[col].unique())) for col in group.columns if col not in [date_col, 'sentence_summary']}
        aggregated['summary_input'] = "\n".join(group['sentence_summary'])
        return pd.Series(aggregated)

    date_grouped = sentence_grouped.groupby(date_col, dropna=False).apply(aggregate_date_fields).reset_index()
    date_grouped['id'] = range(len(date_grouped))
    date_grouped['summary_input'] = date_grouped.apply(lambda row: 'Id: ' + str(row.id)+ '\n' + row['summary_input'], axis=1)

    return date_grouped

def group_information_extracted(df,groupby_cols=['date'], include_fields=None):
    # This function is deprecated
    # Fill NaNs and convert lists to strings if needed
    df = df.fillna('').applymap(lambda x: "; ".join(x) if isinstance(x, list) else x)
    
    # Use all columns if include_fields is not specified or is empty
    columns_to_process = include_fields if include_fields else [col for col in df.columns if col not in groupby_cols]
    
    def aggregate_column(x, column_name):
        if column_name == 'key_drivers':  # Customize behavior for specific columns
            return "; ".join(filter(None, x[column_name].unique()))
        elif 'novelty' in column_name:
            return ", ".join(map(str, filter(None, x[column_name].unique())))
        else:
            return ", ".join(filter(None, x[column_name].unique()))
    
    final_df = df.groupby(groupby_cols, dropna=False).apply(
        lambda x: pd.Series({col: aggregate_column(x, col) for col in columns_to_process})
    ).reset_index()
    
    final_df['id'] = range(len(final_df))
    
    return final_df

def clean_ratings_table(df):
    # Step 1: Split the credit_rating column by '/' and explode
    df['credit_rating'] = df['credit_rating'].str.split('/')  # Split into list
    df_chart = df.explode('credit_rating').reset_index(drop=True)   # Explode list to create duplicate rows
    df_chart['rater_entity'] = df_chart['rater_entity'].fillna(df_chart['validated_rater'])
    df_chart['rater_entity'] = df_chart['rater_entity'].fillna('')
    # Apply the function to each row and assign to two new columns
    df_chart[['numeric_rating', 'rating_agency']] = df_chart.apply(map_rating_and_standardize, axis=1, result_type="expand")
    
    return df_chart

def create_text_to_summarize(row, cols):
    # This function is deprecated
    txt_str = '\n'.join([f"{col.replace('_', ' ').title()}: {row[col]}" for col in cols if col not in ["sentence_id", "rp_entity_id", "unclear"]])
    return txt_str
    
def extract_entity_name(entity_key, mapping):
    # Search the dictionary for the matching key
    for name, key in mapping.items():
        if key == entity_key:
            return name
    return None  # Return None if key is not found

def create_consolidated_data_table(text_string, system_prompt,
                    OPENAI_API_KEY, model="gpt-4o-mini-2024-07-18"):
    json_string = extract_data_table(text_string, system_prompt,
                    OPENAI_API_KEY, model)
    
    json_string = re.sub('```', '', json_string)
    json_string = re.sub('json', '', json_string)
    json_string = json.loads(json_string)
    
    data = pd.DataFrame(json_string)
    
    return data

def extract_data_table(text_string, system_prompt, 
                    OPENAI_API_KEY, model="gpt-4o-mini-2024-07-18"):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
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

        # Extracting the summary string from the response
        json_string = response.model_dump()['choices'][0]['message']['content'] 
        return json_string  # Return summary if successful

    except Exception as exception_message:
        error_message = str(exception_message)
        print(f"An error occurred on attempt: {exception_message}")
        return None

extract_data_table_prompt = ("""
Forget all previous instructions.
You are provided with a string containing a detailed credit rating report for a company. Your task is to parse this string and extract specific information to create a structured data table. Each record in this table should be formatted as a JSON object including the following fields:
  
1. **Ratee Entity**: The name of the company being rated, the subject of the report.
2. **Date**: The specific date of the rating event.
3. **Credit Rating**: The explicit credit rating mentioned in the report. Ratings are expressed in letters and numbers such as BBB, Baa1, A-, Prime-2, etc. *Note: Extract exact credit ratings discussed, assigned or placed in review.* If no credit rating is mentioned, write "No Rating Mentioned".
4. **Key Driver**: The primary reason or rationale provided for the assignment of the credit rating.
5. **Rater**: The rating agency that is providing the credit rating.

Create a separate data record for each distinct combination of date, credit rating, rating agency, and key driver. 

To achieve accuracy:
- Parse the report string line-by-line. Lines are identified by line-breaking characters '\n'.
- Generate AT LEAST one record for each date for each distinct combination of date, credit rating, rating agency, and key driver.
- If a line does not explicitly mention an exact credit rating, write "No Credit Rating Mentioned. If a line does not explicitly mention a rating agency, write "No Rating Agency Mentioned".
- Ensure that each extracted credit rating is attributed to the correct date, rating agency, and key driver from the same line.
- Do not include actions like downgrade, upgrade, confirmation, review, or watchlist in the credit rating extracted. This field can only contain exact credit ratings.
- Carefully read the text as credit rating agencies may use similar scales to assign credit ratings.
- Ensure that there is a direct mention of a credit rating from a specific credit rating agency or rater.
- Extract exact ratings related to any debt instrument, e.g. senior unsecured debt, commercial paper, etc., but do not include the credit actions or reviews in the extracted information.
- DO NOT infer the credit rating from descriptions such as "the current rating is at the bottom of the investment-grade scale". The exact rating has to be mentioned.

Input Report Example:
"\n- **2024-01-17**: Boeing Co. senior unsecured debt has been rated Baa2 by Moody's due to concerns over the company's ability to deliver sufficient volumes of its 737 model to enhance free cash flow. S&P has confirmed its BBB- credit rating, justified by concerns over the company's cash flow during the strike.
\n- **2024-04-24**: Boeing Co. has been downgraded BBB- by Fitch with a negative outlook motivated by ongoing cash flow issues and projected annual cash flow insufficient to cover debt obligations. Headwinds in the Commercial Airplanes segment and expectations of new debt issuance are also likely to push the credit rating to a new downgrade in the coming months.
\n- **2024-10-21**: Boeing Co. BBB- rating by Fitch has been placed on review for a downgrade due to cash flow uncertainty, while Moody's also placed their credit rating on review."

Your output should be a JSON array of objects, formatted as follows:
[
  {
    "Ratee Entity": "Boeing Co.",
    "Date": "2024-01-17",
    "Credit Rating": "Baa2",
    "Key Driver": "Concerns over the company's ability to deliver sufficient volumes of its 737 model to enhance free cash flow.",
    "Rater": "Moody's"
  },
  {
    "Ratee Entity": "Boeing Co.",
    "Date": "2024-03-26",
    "Credit Rating": "BBB-",
    "Key Driver": "Concerns over the company's cash flow during the strike.",
    "Rater": "S&P"
  },
  {
    "Ratee Entity": "Boeing Co.",
    "Date": "2024-04-24",
    "Credit Rating": "BBB-",
    "Key Driver": "Negative outlook due to ongoing cash flow issues and projected annual cash flow insufficient to cover debt obligations.",
    "Rater": "Fitch"
  },
  {
    "Ratee Entity": "Boeing Co.",
    "Date": "2024-10-21",
    "Credit Rating": "BBB-",
    "Key Driver": "The rating is in review for a downgrade due to cash flow uncertainty.",
    "Rater": "Fitch"
  },
  {
    "Ratee Entity": "Boeing Co.",
    "Date": "2024-10-21",
    "Credit Rating": "No Rating Mentioned",
    "Key Driver": "The rating is in review for a downgrade due to cash flow uncertainty.",
    "Rater": "Moody's"
  },
  ...
]

Ensure precision by accurately assigning ratings to the date, rater, and key drivers listed in the same line of the report. Do not mix dates, ratings, or drivers across different report entries.
""")