"""
Module for extracting features from credit rating content.

This module handles the extraction of various features from credit rating content,
including entity role detection (rater/ratee), credit ratings, outlooks, etc.
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional, Union

from bigdata_research_tools.labeler.labeler import Labeler, get_prompts_for_labeler, parse_labeling_response
from bigdata_research_tools.llm.base import AsyncLLMEngine
from bigdata_research_tools.llm.utils import run_concurrent_prompts

class FeatureExtractor(Labeler):

    def __init__(
        self,
        llm_model: str = "openai::gpt-4o-mini",
        unknown_label: str = "unclear",
        temperature: float = 0,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the EntityRoleLabeler.

        Args:
            llm_model: Name of the LLM model to use
            unknown_label: Label for unclear classifications
            temperature: Temperature for the LLM
            api_key: API key for the LLM provider (if needed)
        """
        super().__init__(llm_model, unknown_label, temperature)
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Define the labeling prompt for entity role detection
        self.labeling_prompt = """
Forget all previous instructions.

You are assisting a professional analyst in determining the role of a company in news extracts that describe **credit ratings and corporate debt conditions**.

Your primary task is to categorize the role of the company as either a rater, ratee, or unclear based on the provided text.

Please adhere to these guidelines:

1. **Analyze the Sentence**:
    - Each input consists of a sentence ID, a company name ("entity_name"), and the sentence text formatted as:
        - sentence_id: [the id of the text]
        - entity_name: [the name of the company whose role you need to evaluate]
        - headline: [the title of the news article]
        - text: [the sentence requiring analysis]

    - **Identification of Roles**:
        - Determine the role of "entity_name" related to credit ratings and corporate debt:
            - **rater**: entity_name is actively involved in assigning or evaluating credit ratings for others. This typically includes credit rating agencies like Fitch Ratings, Moody's, and S&P, which only serve as raters.
            - **ratee**: entity_name is clearly the subject of a credit rating, i.e., receiving a credit rating or being explicitly assigned a credit rating. For entity_name to be a ratee, it has to be clear that the credit rating discussed is referred to entity_name.
            - **unclear**: Ambiguities in the text or references to non-credit ratings (e.g., stock ratings, general mentions) where the role of entity_name is not explicit.

    - **Role Descriptions and Examples**:
        - **rater**: 
          - Example: "entity_name assigned Other Company a credit rating of AAA."
          - Example: "Other Company has been assigned a credit rating of AAA by entity_name."
          - Example: "entity_name reviewed the BBB- rating of Other Company, maintaining a stable outlook."
        - **ratee**: 
          - Example: "entity_name's credit rating was upgraded to A+ by Another Company."
        - **unclear**: 
          - Example: "entity_name mentions a stable BBB+ rating is crucial, without assigning it themselves."
          - Example: "entity_name provides insights on corporate debt trends without specific rating involvement."

2. **Response Format**:
    - Output the analysis as a properly formatted JSON object:
    {
        "<sentence_id>": {
            "motivation": "<explanation of why the label was selected>",
            "label": "<assigned role: rater, ratee, or unclear>"
        }
    }
    - Start the motivation with entity_name, elaborating why the label was chosen based on sentence content.
    - Ensure accurate JSON syntax: use proper quotes and structure.

3. **Evaluation Instructions**:
    - Assess each sentence independently using provided sentence data.
    - Consider variations and abbreviations of company names:
      - "S&P" for "S&P Global".
      - "Moody's" for "Moody's Corp".
      - "Fitch" for "Fitch Ratings Inc."
      - "Verizon" for "Verizon Communications".
    - Example: "S&P affirmed the AA rating of Other Company." Here, "S&P" stands for "S&P Global" and it clearly acts as the **rater**.
"""

        # Define the validation prompt for entity role validation
        self.validation_prompt = """
Forget all previous instructions.

You are tasked with validating the role assigned to a company in news extracts that describe **credit ratings and corporate debt conditions**.

Your primary task is to assess whether the label assigned to the company (entity_name) is correct or if it needs to be changed based on the provided text and motivation.

Please adhere to these guidelines:

1. **Input Structure**:
    - Each input consists of the following:
        - sentence_id: [the id of the text]
        - entity_name: [the name of the company whose role you need to evaluate]
        - headline: [the title of the news article]
        - text: [the sentence requiring analysis]
        - motivation: [the explanation provided for the assigned label]
        - label: [the current label assigned: 'rater', 'ratee', or 'unclear']

2. **Label Descriptions and Examples**:
    - **'rater'**: If entity_name is involved in assigning a credit rating to another company or evaluating that company's credit rating. Credit rating agencies, such as Fitch Ratings, Moody's, and S&P, always act as raters and never as ratees.
        - *Example*: "entity_name assigned Other Company a credit rating of AAA."
        - *Example*: "Other Company has been assigned a credit rating of AAA by entity_name."
    - **'ratee'**: If a credit rating agency has assigned a credit rating to entity_name. When it is clear that entity_name is receiving a credit rating, it must be labelled as ratee. For entity_name to be considered a ratee, the text has to explicitly mentioned that the credit rating discussed is referred to entity_name.
        - *Example*: "entity_name's credit rating was upgraded to A+ by Other Company."
    - **'unclear'**: If entity_name is mentioned in contexts where it is not explicitly involved in issuing or receiving a credit rating, or when the roles are ambiguous.
        - *Example*: "The investment-grade Other Company credit rating of BBB+ is relevant for entity_name, but does not imply that entity_name is a rater."

3. **Validation Process**:
    - Carefully read the provided Text and Motivation.
    - Based on the context in the Text, determine if the assigned Label accurately reflects the role of entity_name.
    - Maintain the Label if it is correct.
    - If the Label is incorrect, select the appropriate label ('rater', 'ratee', or 'unclear') based on the text's context focusing on credit ratings and corporate debt instruments.
    - If the label is 'unclear', verify if there is clear evidence of entity_name's involvement in rating activities. Correct this to 'rater' or 'ratee' if the Text indicates that a credit rating is explicitly assigned by or to entity_name.
        - Example: "Verizon has a BBB+/Baa1 rating by S&P and Moody's." In this case, both S&P and Moody's should be labeled as 'rater'. Change from 'unclear' to 'rater' if incorrectly labeled.
    - Consider name variations and abbreviations such as "S&P" for "S&P Global", "Moody's" for "Moody's Corp", "Fitch" for "Fitch Ratings Inc.", "Verizon" for "Verizon Communications", etc.
    - Credit rating agencies, such as Fitch Ratings, Moody's, and S&P, always act as raters and never as ratees.

4. **Response Format**:
    - Your output should be structured as a JSON object as follows:
    {
        "<sentence_id>": {
            "validated_label": "<the updated or confirmed label based on your validation>",
            "validated_motivation": "<your rationale for the validated label>"
        }
    }
"""

        # Define the feature extraction prompts
        self.credit_ratings_prompt ="""
    Forget all previous instructions.

You are tasked with extracting credit ratings and relevant insights for a single ratee entity from the provided text, based on a single rater entity.

1. **Input Structure**:
    - Each input consists of:
        - ID: [the unique ID for the input text]
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
        "<sentence_id>": {{
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
"""

        self.debt_instruments_prompt = """
    Forget all previous instructions.
    You are assisting a professional analyst in extracting short-term and long-term credit ratings related to a single ratee entity from the provided text.

    Your primary task is to extract relevant information pertaining specifically to the identified rater entity and ratee entity.

    1. **Input Format**:
        - Each input consists of:
            - ID: [the id of the text]
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
        - Your output should be structured as a JSON object:
        {{
            "<sentence_id>": {{
                "short_term_credit_rating": "<short_term_credit_rating_extracted or None>",
                "long_term_credit_rating": "<long_term_credit_rating_extracted or None>",
                "debt_instrument": "<associated_debt_instrument or None>"
            }},
            ...
        }}
        - If no information is available for a specific field, ensure that it is represented as an empty string.
"""

        self.drivers_guidance_prompt = """
    Forget all previous instructions.
    You are assisting a professional analyst in extracting relevant insights regarding credit ratings and associated contextual elements for a single ratee entity from the provided text.

    Your primary task is to extract detailed information about the rationale, forward guidance, and key drivers impacting the credit rating of the identified ratee.

    1. **Input Format**:
        - Each input consists of the following fields:
            - ID: [the id of the text]
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
        - Your output should be structured as a JSON object:
        {{
            "<sentence_id>": {{
                "forward_guidance": "<forward_guidance_info>",
                "key_drivers": "<list_of_key_drivers or None>"
            }},
            ...
        }}
        - If no information is available for a specific field, ensure that it is represented as an empty string.
"""

    def _add_prompt_fields(self, df_sentences: pd.DataFrame, additional_prompt_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Add additional fields from the DataFrame for the labeling prompt.

        Args:
            df_sentences (DataFrame): The DataFrame containing the search results.
            additional_prompt_fields (Optional[List[str]]): Additional field names to be used in the labeling prompt.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with the additional fields for each row in the DataFrame.
        """
        if additional_prompt_fields:
            missing = set(additional_prompt_fields) - set(df_sentences.columns)
            if missing:
                raise ValueError(f"Columns not found in DataFrame: {missing}")
            else:
                return df_sentences[additional_prompt_fields].to_dict(orient="records")
        else:
            return []

    def assign_entity_roles(
        self, 
        df: pd.DataFrame,
        text_col: str = 'contextualized_chunk_text', 
        additional_prompt_fields: Optional[List[str]] = ['entity_name', 'headline'],
        action_type: str = 'detect',
        system_prompt: Optional[str] = None,
        max_workers: int = 100
    ) -> pd.DataFrame:
        """
        Detect the roles of entities in credit rating content.

        Args:
            df: DataFrame containing the text data
            additional_prompt_fields: Optional list of additional field names to include in prompts
            max_workers: Maximum number of concurrent workers

        Returns:
            DataFrame with entity roles labeled
        """
        if system_prompt is None:
            if action_type == 'detect':
                system_prompt = self.labeling_prompt
            elif action_type == 'validate':
                system_prompt = self.validation_prompt
            else:
                raise ValueError(f"Unknown action type: {action_type}. "
                               "Use 'detect' or 'validate', "
                               "or provide a custom system_prompt.")
        
        # Get text configs using the flexible prompt fields method
        text_configs = self._add_prompt_fields(df, additional_prompt_fields)

        print(text_configs)

        prompts = get_prompts_for_labeler(
            texts=df[text_col].tolist(),
            textsconfig=text_configs
        )
        print(prompts[0])
        print(type(prompts[0]))
        # Run the labeling process
        responses = self._run_labeling_prompts(
            prompts=prompts,
            system_prompt=system_prompt,
            max_workers=max_workers
        )

        responses = [parse_labeling_response(response) for response in responses]
        print(responses)
        
        # Deserialize the responses
        df_labeled = self._deserialize_label_responses(responses)
        if len(df_labeled['motivation'].unique()) == 1 and df_labeled['motivation'].values[0]=='':
            df_labeled = df_labeled.drop(columns=['motivation', 'label'])

        print(df_labeled)
        
        # Merge with the original dataframe #check this because it is wrong
        if 'index' not in df.columns:
            df = df.reset_index()
        df_merged = pd.merge(
            df, ## this creates additional columns if index is already present, it is not very elegant, what about merge(df, df_labels, left_index=True, right_index=True)
            df_labeled.reset_index(),
            on='index',
            how='left'
        ).drop('index', axis=1)

        return df_merged
    
    def extract_features(
        self,
        df: pd.DataFrame,
        text_col: str = 'contextualized_chunk_text',
        additional_prompt_fields: Optional[List[str]] = ['rater_entity', 'ratee_entity', 'unclear_entities'],
        max_workers: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract features from credit rating content.

        Args:
            df: DataFrame containing the text data with entity roles
            text_col: Column name for the text content
            additional_prompt_fields: Optional list of additional field names to include in prompts
            max_workers: Maximum number of concurrent workers

        Returns:
            Dictionary of DataFrames with extracted features
        """
        
        # Get text configs using the flexible prompt fields method
        text_configs = self._add_prompt_fields(df, additional_prompt_fields)
        
        prompts = get_prompts_for_labeler(
            texts=df[text_col].tolist(),
            textsconfig=text_configs
        )
        
        # Extract features using different prompts
        features = {}
        
        # Feature set 1: Credit ratings and outlook
        responses_1 = self._run_labeling_prompts(
            prompts=prompts,
            system_prompt=self.credit_ratings_prompt,
            max_workers=max_workers
        )
        responses_1 = [parse_labeling_response(response) for response in responses_1]
        features['credit_ratings'] = self._deserialize_label_responses(responses_1)
        
        # Feature set 2: Debt instruments
        responses_2 = self._run_labeling_prompts(
            prompts=prompts,
            system_prompt=self.debt_instruments_prompt,
            max_workers=max_workers
        )
        responses_2 = [parse_labeling_response(response) for response in responses_2]
        features['debt_instruments'] = self._deserialize_label_responses(responses_2)
        
        # Feature set 3: Key drivers and forward guidance
        responses_3 = self._run_labeling_prompts(
            prompts=prompts,
            system_prompt=self.drivers_guidance_prompt,
            max_workers=max_workers
        )
        responses_3 = [parse_labeling_response(response) for response in responses_3]
        features['drivers_guidance'] = self._deserialize_label_responses(responses_3)

        result = self._combine_features(df, features)

        return result

    def extract_single_feature(
        self,
        df: pd.DataFrame,
        feature_type: str,
        text_col: str = 'contextualized_chunk_text',
        system_prompt: Optional[str] = None,
        additional_prompt_fields: Optional[List[str]] = ['rater_entity', 'ratee_entity', 'unclear_entities'],
        max_workers: int = 100
    ) -> pd.DataFrame:
        """
        Extract a single feature type from credit rating content.

        Args:
            df: DataFrame containing the text data with entity roles
            feature_type: Type of feature to extract ('credit_ratings', 'debt_instruments', 'drivers_guidance')
            text_col: Column name for the text content
            system_prompt: Custom prompt to use (if not using predefined prompts)
            additional_prompt_fields: Optional list of additional field names to include in prompts
            max_workers: Maximum number of concurrent workers

        Returns:
            DataFrame with extracted feature
        """
        # Determine which prompt to use based on feature_type
        if system_prompt is None:
            if feature_type == 'credit_ratings':
                system_prompt = self.credit_ratings_prompt
            elif feature_type == 'debt_instruments':
                system_prompt = self.debt_instruments_prompt
            elif feature_type == 'drivers_guidance':
                system_prompt = self.drivers_guidance_prompt
            else:
                raise ValueError(f"Unknown feature type: {feature_type}. " 
                               "Use 'credit_ratings', 'debt_instruments', or 'drivers_guidance', "
                               "or provide a custom system_prompt.")
        
        # Get text configs using the flexible prompt fields method
        text_configs = self._add_prompt_fields(df, additional_prompt_fields)
        
        prompts = get_prompts_for_labeler(
            texts=df[text_col].tolist(),
            textsconfig=text_configs
        )
        
        # Extract the requested feature
        responses = self._run_labeling_prompts(
            prompts=prompts,
            system_prompt=system_prompt,
            max_workers=max_workers
        )
        responses = [parse_labeling_response(response) for response in responses]
        feature_df = self._deserialize_label_responses(responses)
        
        return feature_df

    def group_text_and_labels(
        self, 
        df: pd.DataFrame,
        group_columns: List[str] = ['date', 'sentence_id', 'headline', 'source_name', 'contextualized_chunk_text', 'url'],
        role_column: str = 'validated_label',
        entity_column: str = 'entity_name',
    ) -> pd.DataFrame:
        """
        Group entities by their roles and prepare data for feature extraction.
        
        This function takes a DataFrame containing entity role assignments (rater, ratee, unclear)
        and groups them by specified columns. It then creates separate lists for each role type
        and explodes the results to create a Cartesian product of rater and ratee entities.
        
        Args:
            df: DataFrame with validated entity roles
            group_columns: Columns to group by (identifying unique text chunks)
            role_column: Column containing the role labels (rater, ratee, unclear)
            entity_column: Column containing the entity names
            id_column: Name for the ID column to be added to the result
            
        Returns:
            DataFrame with grouped and exploded entities by role
        """
        # Group by specified columns and collect entities by their roles
        grouped_df = df.groupby(group_columns, dropna=False).apply(
            lambda x: pd.Series({
                'rater_entity': list(x.loc[x[role_column] == 'rater', entity_column]),
                'ratee_entity': list(x.loc[x[role_column] == 'ratee', entity_column]),
                'unclear_entities': list(x.loc[x[role_column] == 'unclear', entity_column]),
            })
        ).reset_index()
        
        # Explode both rater_entity and ratee_entity columns to create all combinations
        exploded_df = grouped_df.explode('rater_entity', ignore_index=True).explode('ratee_entity', ignore_index=True)
        
        return exploded_df
        
    def _combine_features(
        self, 
        df: pd.DataFrame, 
        features: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Combine all extracted features into a single DataFrame.

        Args:
            df: Base DataFrame with entity roles
            features: Dictionary of DataFrames with extracted features

        Returns:
            DataFrame with all features combined
        """
        # Create a copy of the base DataFrame
        result_df = df.copy()
        
        # Merge all feature DataFrames
        for feature_name, feature_df in features.items():
            # Merge on index
            feature_df_reset = feature_df.reset_index() ##again you need to make sure that the index is not reset every time
            if 'index' not in result_df.columns:
                result_df=result_df.reset_index()
            result_df = pd.merge(
                result_df,
                feature_df_reset,
                on='index',
                how='left',
            ).drop('index', axis=1)
        
        return result_df


def clean_credit_ratings_features_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the dataset for final output by:
    1. Selecting the appropriate columns based on available data
    2. Converting empty strings, empty lists, and 'None' strings to None
    3. Dropping rows without required entity information
    
    Args:
        df: DataFrame with extracted features
        
    Returns:
        Cleaned DataFrame with standardized columns and values
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Determine which columns to keep based on what's available
    if 'contextualized_chunk_text' in df_copy.columns and 'text' not in df_copy.columns:
        cols_visualize = ['date', 'sentence_id', 'headline', 'source_name', 'url', 'contextualized_chunk_text',
                         'ratee_entity_rp_entity_id', 'ratee_entity', 'rater_entity',
                         'credit_rating', 'credit_outlook', 'credit_action', 'credit_status', 'credit_watchlist',
                         'short_term_credit_rating', 'long_term_credit_rating', 'debt_instrument',
                         'forward_guidance', 'key_drivers']
        
    elif 'text' in df_copy.columns and 'contextualized_chunk_text' not in df_copy.columns:
        cols_visualize = ['date', 'sentence_id', 'headline', 'source_name', 'url', 'text',
                         'ratee_entity_rp_entity_id', 'ratee_entity', 'rater_entity',
                         'credit_rating', 'credit_outlook', 'credit_action', 'credit_status', 'credit_watchlist',
                         'short_term_credit_rating', 'long_term_credit_rating', 'debt_instrument',
                         'forward_guidance', 'key_drivers']
        
    elif 'text' in df_copy.columns and 'contextualized_chunk_text' in df_copy.columns: 
        cols_visualize = ['date', 'sentence_id', 'headline', 'source_name', 'url', 'text', 'contextualized_chunk_text',
                         'ratee_entity_rp_entity_id', 'ratee_entity', 'rater_entity',
                         'credit_rating', 'credit_outlook', 'credit_action', 'credit_status', 'credit_watchlist',
                         'short_term_credit_rating', 'long_term_credit_rating', 'debt_instrument',
                         'forward_guidance', 'key_drivers']
        
    else:
        # If neither text nor contextualized_chunk_text are present, use all columns
        cols_visualize = df_copy.columns
    
    # Clean up empty values: Convert empty strings, empty lists and 'None' strings to None
    df_clean = df_copy.applymap(lambda x: None if x == '' or x == [] or x == 'None' else x)
    
    # Filter to include only the selected columns and drop rows without required entity information
    filtered_cols = [col for col in cols_visualize if col in df_clean.columns]
    
    # Check which required columns exist before filtering
    required_columns = []
    if 'ratee_entity_rp_entity_id' in df_clean.columns:
        required_columns.append('ratee_entity_rp_entity_id')
    if 'rater_entity' in df_clean.columns:
        required_columns.append('rater_entity')
    
    if required_columns:
        df_clean = df_clean[filtered_cols].dropna(subset=required_columns).copy()
    else:
        df_clean = df_clean[filtered_cols].copy()
    
    return df_clean.reset_index(drop=True)
