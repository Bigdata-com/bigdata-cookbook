"""Summary generator using OpenAI (no SDK)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import pandas as pd
from openai import OpenAI

from .constants import DEFAULT_LLM_MODEL
from .openai_sampling import sampling_params_for_model


class SummaryGenerator:
    """
    A class to generate summaries and reports from credit rating data.
    
    This class encapsulates the functionality for processing credit rating data,
    generating summaries, and creating structured reports through OpenAI with
    specialized handling for token limits through text splitting.
    """

    def __init__(
        self,
        llm_model: str = DEFAULT_LLM_MODEL,
        temperature: float = 0,
        max_workers: int = 30,
        api_key: str | None = None,
    ):
        """Initialize the SummaryGenerator with OpenAI client.
        
        Args:
            llm_model: OpenAI model name (e.g., "gpt-4o-mini")
            temperature: Temperature for the model
            max_workers: Maximum number of concurrent workers for batch processing
            api_key: OpenAI API key (defaults to OPENAI_API_KEY)
        """
        self.llm_model = llm_model
        # Plain synchronous OpenAI client (not AsyncOpenAI): notebooks apply
        # nest_asyncio so Jupyter's own event loop can host nested asyncio.run()
        # calls, but that patch breaks AsyncOpenAI's httpx/anyio sniffio-based
        # async-library detection ("unknown async library, or not in async
        # context") both on the main thread and inside worker threads. The sync
        # client sidesteps asyncio entirely, matching the (working) pattern
        # feature_extractor.py already uses for its OpenAI calls.
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.temperature = temperature
        self.max_workers = max_workers
        self.unknown_label = "unclear"
        
        # Set default prompts
        self.credit_ratings_consolidation = """
You are an expert summarizer and assistant. Your task is to merge multiple message contents into a comprehensive timeline report regarding the credit ratings of companies. Do not alter the structure of the inputs and focus on crafting a cohesive output.

Please follow these guidelines while merging:

1. **Consistency**: Ensure the final timeline is coherent, logically structured, and free from contradictions.
2. **Flow**: Maintain a smooth flow between parts. Seamlessly integrate information, especially where overlaps occur.
3. **Preserve Details**: Include all unique and relevant details from all completions, ensuring nothing important is lost.
4. **Avoid Redundancy**: Identify and eliminate redundant or repeated information. Focus on credit rating actions such as upgrades, downgrades, affirmations. If similar events are described in consecutive entries without new actions, merge them as a single entry. If contradictory events are reported in consecutive days, exclude these misreported events from the list.
5. **Structured Events**: Maintain a clear structure, ensuring the timeline reflects all major credit rating actions. Clearly outline each event with the appropriate date and source attribution.
6. **Highlight Novelty**: If information from different sources confirms the same event, consolidate it into one entry, using multiple sources for added credibility only when they offer new insights.

The final report should be a well-organized and accurate consolidation of the various inputs, capturing all significant changes without unnecessary repetition.
"""
        self.daily_credit_ratings_report="""**Task: Report Credit Rating Information with Enhanced Source Tracking**

You are tasked with generating a comprehensive timeline report based on input texts regarding the credit ratings of companies, ensuring the inclusion of news source names and URLs where available. Your output should prioritize data from dates identified as having high novelty, meaning not repeated over a span of a few days, as well as dates indicating changes or developments in credit ratings or outlooks.

**Input Structure:**

- `Ratee Entity`: [Company Name]
- `DateX`: [YYYY-MM-DD hh:mm:ss]
- `TextX`: [Content regarding credit ratings, outlooks, or financial strategies]
- `SourcesX`: [Comma-separated list of source names and URLs]

**Instructions:**

1. **Identify Novelty**:
   - Focus on entries where changes or updates to credit ratings, outlooks, or financial strategies are reported.
   - Do not repeat similar updates if they span multiple consecutive days; consolidate into the most impactful date.
   - If information is repeated over a span of months or years, then report it as happening in their respective dates.
   - Ensure that the information spanning multiple consecutive days is not contradictory. Credit rating agencies are unlikely to announce back-to-back credit rating changes within a few days.

2. **Maintain Temporal Coverage**:
   - Use the Date information wisely and ensure that every date is represented unless the information is repeated.
   - If texts span consecutive days, there is a great possibility that the information is duplicated.
   - If the Dates of two consecutive Texts are very different, then it's likely that the information pertains to separate events.

3. **Data Consolidation**:
   - Only include dates with distinct credit rating or credit outlook updates. Highlight changes in ratings, new credit ratings, outlook revisions, or major financial movements.
   - Prioritize the information that is discussed by more than one source, if available.
   - Ensure that details from multiple sources for the same event are consolidated under one date.

4. **Source Inclusion**:
   - For each credit rating update or outlook change, include all related source names and URLs.
   - Use the format: "[Source Name](URL)" in brackets after each summarized date entry.
   - Exclude entries with contradictory information from different sources.
   - DO NOT infer URLs from outside information. DO NOT use placeholder URLs. Leave the URL blank if no URL is available.
   - Only report Source Names that are given in the text. DO NOT infer sources from outside knowledge.

5. **Content Structure**:
   - Each entry in the summary should contain:
     - **Credit Ratings and Raters**: Summarize all involved raters and their assigned ratings.
     - **Credit Outlooks and Actions**: Emphasize any changes in outlooks or affirmations of ratings.
     - **Key Drivers**: Briefly explain the main factors influencing the rating or outlook decision.

6. **Output Format**:
   - Structure the output as a timeline of key credit rating events.
   - Keep each entry concise, no more than two sentences, while maintaining information clarity.
   - Avoid creating new dates or entries without valid credit data.
   - The structure of your report should be a bulleted list as follows:
        - **<Date>**: Summary of the credit rating event, including all relevant details and sources ([Source1](URL1), [Source2](URL2)).
        - **<Date>**: Summary of another credit rating event, including all relevant details and sources ([Source1](URL1), [Source2](URL2)).
        - **<Date>**: ...

**Example Output**:
### Credit Report
- **2023-01-12**: On January 12, 2023, Moody's placed Boeing Co.'s Baa2 senior unsecured rating and Prime-2 short-term rating on review for a potential downgrade due to concerns over their ability to manage debt and deliver enough 737 models, highlighting production challenges ([NBC San Diego](https://example.com), [Bloomberg Government](https://example.com)).
 - **2023-02-08**: On February 8, 2023, Fitch affirmed Boeing’s BBB- rating but revised the outlook from stable to negative, citing continued supply-chain bottlenecks and slower-than-expected recovery in commercial deliveries ([Reuters](https://example.com), [Yahoo Finance](https://example.com)).
- **2024-03-26**: On March 26, 2024, Moody's placed Boeing Co.'s Baa2 senior unsecured rating and Prime-2 short-term rating on review for a potential downgrade due to concerns over their ability to manage debt and deliver enough 737 models, highlighting production challenges ([NBC San Diego](https://example.com), [Bloomberg Government](https://example.com)).
- **2024-04-24**: On April 24, 2024, Moody’s downgraded Boeing’s senior unsecured rating to Baa3 from Baa2, flagging mounting debt pressures and weaker-than-expected cash flows, while maintaining a negative outlook ([WCVB.com](https://example.com), [BNN Bloomberg](https://example.com)).
- **2025-01-15**: On January 15, 2025, S&P upgraded Boeing’s credit outlook to stable from negative, pointing to improved order flow for the 737 MAX and cost-reduction initiatives starting to materialize ([The Guardian](https://example.com), [MarketWatch](https://example.com)).
- **2025-06-25**: On June 25, 2025, Moody’s upgraded Boeing’s long-term rating to Baa2 from Baa3, citing stronger liquidity and steady delivery momentum across commercial and defense programs ([Benzinga](https://example.com), [The Fly](https://example.com)).
"""
        self.daily_chunk_summarization= """Forget all previous instructions.
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
        - Return a dict containing a <daily_summary_string_generated> with all summarized and consolidated information as detailed above.
        - Your output should be structured as a JSON object with id and the generated string as follows:
        {{
            "<id>": {{
                "daily_summary": "<daily_summary_string_generated>"
        }}
    }}
        
"""
        self.credit_ratings_data_table="""
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
            """
    
    
    def _get_response(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Call OpenAI (sync) and return response text."""
        sampling_kwargs = sampling_params_for_model(
            self.llm_model,
            temperature=self.temperature,
        )
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            **sampling_kwargs,
            **kwargs,
        )
        return response.choices[0].message.content
    
    def get_prompts_for_labeler(
        self,
        texts: list[str],
        textsconfig: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build prompts for labeling."""
        prompts = []
        for idx, text in enumerate(texts):
            prompt_dict = {"index": idx, "text": text}
            if textsconfig and idx < len(textsconfig):
                prompt_dict.update(textsconfig[idx])
            prompts.append(prompt_dict)
        return prompts
    
    def _run_labeling_prompts(
        self,
        prompts: list[dict[str, Any]],
        system_prompt: str,
        max_workers: int = 30,
        timeout: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Run labeling prompts."""
        import concurrent.futures
        
        def label_single(prompt_data: dict[str, Any]) -> dict[str, Any]:
            user_content = f"sentence_id: {prompt_data['index']}\n"
            for key, value in prompt_data.items():
                if key not in ['index']:
                    user_content += f"{key}: {value}\n"
            
            try:
                response = self._get_response([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ], response_format={"type": "json_object"})
                return {"index": prompt_data["index"], "response": response}
            except Exception as e:
                return {"index": prompt_data["index"], "response": json.dumps({"error": str(e)})}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(label_single, prompts))
        
        return results
    
    def deserialize_label_responses_as_df(self, responses: list[dict[str, Any]]) -> pd.DataFrame:
        """Parse JSON responses into a DataFrame."""
        rows = []
        for item in responses:
            try:
                response_data = json.loads(item["response"])
                for sentence_id, data in response_data.items():
                    row = {"index": item["index"]}
                    if isinstance(data, dict):
                        row.update(data)
                    rows.append(row)
            except (json.JSONDecodeError, KeyError):
                rows.append({"index": item["index"], "label": self.unknown_label, "motivation": ""})
        
        return pd.DataFrame(rows).set_index("index")
    
    def _split_text_on_nearest_linebreak(self, text_string: str, num_splits: int) -> list[str]:
        """
        Split text into parts at the nearest line breaks for handling large inputs.
        
        Args:
            text_string: Text to split
            num_splits: Number of splits to create
            
        Returns:
            List of text segments
        """
        split_texts = [text_string]
        
        for _ in range(num_splits - 1):
            new_splits = []
            for text in split_texts:
                mid_index = len(text) // 2
                
                # Find nearby line breaks
                before_split = text.rfind('\n', 0, mid_index)
                after_split = text.find('\n', mid_index)
                
                # Choose split point
                if before_split != -1:
                    split_index = before_split
                elif after_split != -1:
                    split_index = after_split
                else:
                    split_index = mid_index
                
                # Split the text
                string1 = text[:split_index]
                string2 = text[split_index:]
                
                # Add context from first part to second part
                start_of_string1 = string1.split('\n')[:3]
                last_part_of_string1 = string1.split('\n')[-3:]
                
                context = '\n'.join(start_of_string1) + '\n'.join(last_part_of_string1) + '\n'
                string2 = context + string2
                
                new_splits.extend([string1, string2])
            
            split_texts = new_splits
        
        return split_texts
    
    
    def _add_prompt_fields(self, df_sentences: pd.DataFrame, additional_prompt_fields: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """
        Add additional fields from the DataFrame for the labeling prompt.

        Args:
            df_sentences (DataFrame): The DataFrame containing the search results.
            additional_prompt_fields (Optional[List[str]]): Additional field names to be used in the labeling prompt.

        Returns:
            list[dict[str, Any]]: A list of dictionaries with the additional fields for each row in the DataFrame.
        """
        if additional_prompt_fields:
            missing = set(additional_prompt_fields) - set(df_sentences.columns)
            if missing:
                raise ValueError(f"Columns not found in DataFrame: {missing}")
            else:
                return df_sentences[additional_prompt_fields].to_dict(orient="records")
        else:
            return []

    def prepare_daily_summary_input(self, df: pd.DataFrame,
                                    date_col: str = 'date',
                                    sentence_id_col: str = 'sentence_id',
                                    text_col: str = 'text',
                                    summary_input: list[str] | None = None) -> pd.DataFrame:
        """
        Generate summaries grouped by date from a DataFrame.
        
        Args:
            df: Input DataFrame
            date_col: Column name for dates
            sentence_id_col: Column name for sentence IDs
            text_col: Column name for text to summarize
            fields_for_summary: Fields to include in summaries
            
        Returns:
            DataFrame with date-grouped summaries
        """
        # Handle empty values
        df = df.fillna('None').map(lambda x: "; ".join(x) if isinstance(x, list) else x)
        
        # Default fields: all columns except sentence ID
        if summary_input is None:
            fields_for_summary = [text_col]
        else:
            fields_for_summary = summary_input

        if text_col not in fields_for_summary:
            fields_for_summary.append(text_col)

        # Group by date and sentence ID - Aggregate at the chunk level
        def aggregate_sentence_fields(group):
            """Consolidate fields for the same date and sentence ID."""
            aggregated = {col: "; ".join(filter(None, group[col].unique())) 
                         for col in group.columns if col not in [date_col, sentence_id_col]}
            return pd.Series(aggregated)

        sentence_grouped = df.groupby([date_col, sentence_id_col], dropna=False).apply(aggregate_sentence_fields, include_groups=False).reset_index()

        # Create sentence input summaries
        def create_sentence_summary(row):
            """Create structured summary for a sentence."""
            summary = [f"{field.replace('_', ' ').title()}: {row[field]}" 
                      for field in fields_for_summary if row[field]]
            return "\n".join(summary)
            
        sentence_grouped['sentence_summary'] = sentence_grouped.apply(create_sentence_summary, axis=1)
        
        # Group by date to consolidate sentences - Aggregate all chunks from the same day
        def aggregate_date_fields(group):
            """Consolidate sentences for the same date."""
            aggregated = {col: "; ".join(filter(None, group[col].unique())) 
                         for col in group.columns if col not in [date_col, 'sentence_summary']}
            aggregated['summary_input'] = "\n".join(group['sentence_summary'])
            return pd.Series(aggregated)
            
        date_grouped = sentence_grouped.groupby(date_col, dropna=False).apply(aggregate_date_fields,include_groups=False).reset_index()
        # date_grouped['id'] = range(len(date_grouped))
        # date_grouped['summary_input'] = date_grouped.apply(lambda row: 'Id: ' + str(row.id) + '\n' + row['summary_input'], axis=1) not needed if I create the prompts as in the labeler
        
        return date_grouped

    def generate_summaries_df(self, df: pd.DataFrame, summary_input_col: str, system_prompt: Optional[str] = None, additional_prompt_fields: Optional[list] = [], max_workers: int = 30) -> tuple[pd.DataFrame, str]:

        if system_prompt is None:
            system_prompt = self.daily_chunk_summarization
        
        if additional_prompt_fields:
            textsconfig = self._add_prompt_fields(df, additional_prompt_fields=additional_prompt_fields)
        else:
            textsconfig = []

        texts = df[summary_input_col].tolist()

        prompts = self.get_prompts_for_labeler(texts, textsconfig=textsconfig)

        #these prompts should have summary input instead of text as key

        responses = self._run_labeling_prompts(
            prompts, system_prompt, max_workers=max_workers, timeout=None,
        )

        ## add error catching, splits, retries, and consolidation to the failed ones.

        parsed_responses = self.deserialize_label_responses_as_df(responses)
        if (
            'motivation' in parsed_responses.columns
            and len(parsed_responses['motivation'].unique()) == 1
            and parsed_responses['motivation'].values[0] == ''
        ):
            parsed_responses = parsed_responses.drop(columns=['motivation', 'label'])

        if 'index' not in df.columns:
            df = df.reset_index()
        df_merged = pd.merge(
            df,
            parsed_responses.reset_index(),
            on='index',
            how='left'
        ).drop('index', axis=1)

        entity_name = df_merged['ratee_entity'].iloc[0] if 'ratee_entity' in df_merged.columns else 'Unknown Entity'

        # If every row in a date-group failed to parse as JSON (LLM/network flakiness),
        # 'daily_summary' may be entirely absent from df_merged -- fall back to the raw
        # summary_input for that date rather than crashing the whole report generation.
        if 'daily_summary' not in df_merged.columns:
            df_merged['daily_summary'] = df_merged.get(summary_input_col, '')

        report_text_input = f'Ratee Entity: {entity_name}\n' + '\n'.join(
            [f'Date{i}: {str(row.date)}\nText{i}: {row.get("daily_summary") or row.get(summary_input_col, "")}'
             for i, (_, row) in enumerate(df_merged.iterrows())]
        )
        return df_merged, report_text_input

    def summarize_string(self, text: str, system_prompt: Optional[str] = None, max_retries: int = 5,
                      max_split_retries: int = 5) -> str:
        """Summarize a text string with retry and text splitting capabilities."""
        if system_prompt is None:
            system_prompt = self.daily_credit_ratings_report

        # Build chat history
        chat_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        # Try to get response directly
        try:
            return self._get_response(chat_history)
        except Exception as e:
            if 'context_length_exceeded' in str(e) or 'string_above_max_length' in str(e):
                print("Text too long for direct processing, attempting split-and-consolidate approach...")
                
                # Try splitting and processing in chunks
                try:
                    splits = self._split_text_on_nearest_linebreak(text, 2)
                    
                    # Process each split
                    results = []
                    for split_text in splits:
                        split_chat_history = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": split_text}
                        ]
                        response = self._get_response(split_chat_history)
                        results.append(response)
                    
                    # Consolidate results
                    if len(results) > 1:
                        consolidation_prompt = self.credit_ratings_consolidation
                        consolidation_text = '\n\n'.join([f"Completion {i+1}: {comp}" for i, comp in enumerate(results)])
                        
                        consolidation_chat_history = [
                            {"role": "system", "content": consolidation_prompt},
                            {"role": "user", "content": f"Please merge and consolidate these completions into a single response:\n\n{consolidation_text}"}
                        ]
                        
                        return self._get_response(consolidation_chat_history)
                    else:
                        return results[0]
                        
                except Exception as nested_e:
                    print(f"Error during split processing: {str(nested_e)}")
                    return f"Error: Failed to process text after attempts to split. {str(nested_e)}"
            else:
                print(f"Error during summarization: {str(e)}")
                return f"Error: {str(e)}"
    
    def create_consolidated_data_table(self, text: str, system_prompt: Optional[str] = None) -> pd.DataFrame:
        """Extract structured data from text."""
        if system_prompt is None:
            system_prompt = self.credit_ratings_data_table
        
        # Build chat history
        chat_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        try:
            # Get response
            json_string = self._get_response(
                chat_history,
                response_format={"type": "json_object"}
            )
            
            # Clean up JSON string
            json_string = re.sub('```', '', json_string)
            json_string = re.sub('json', '', json_string)

            # Parse JSON into DataFrame. response_format={"type": "json_object"} forces the
            # model to return a top-level JSON *object* even though the prompt asks for a
            # JSON *array* -- the model is free to pick any wrapper key (not always "data"),
            # so look for the first list-valued field instead of assuming one fixed key.
            try:
                parsed = json.loads(json_string)
                if isinstance(parsed, list):
                    records = parsed
                elif isinstance(parsed, dict):
                    if isinstance(parsed.get('data'), list):
                        records = parsed['data']
                    else:
                        list_values = [v for v in parsed.values() if isinstance(v, list)]
                        records = list_values[0] if list_values else [parsed]
                else:
                    records = []
                return pd.DataFrame(records)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}.")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error creating data table: {e}")
            return pd.DataFrame()
    
    def generate_company_report(self, df: pd.DataFrame, entity_id: str,text_col:str='text',
                               fields_for_summary: list[str] | None = None) -> tuple[str, pd.DataFrame]:
        """
        Generate a complete report for a single company.
        
        Args:
            df: Input DataFrame with credit rating data
            entity_id: ID of the entity to report on
            start_date: Start date for the report
            end_date: End date for the report
            fields_for_summary: Fields to include in summaries
            
        Returns:
            Tuple of (report_text, structured_data_df)
        """
        # Filter for the specific entity
        df_summary = df.loc[df.ratee_entity_rp_entity_id.eq(entity_id)].copy().reset_index(drop=True)
        
        # Generate summary by date
        fields = fields_for_summary or ['date', 'ratee_entity', 'headline', 'source_name', 
                                       'url', 'contextualized_chunk_text']
        df_grouped = self.prepare_daily_summary_input(df_summary, text_col=text_col, summary_input=fields)

        if df_grouped.empty:
            return f"No News Found for entity {entity_id}", pd.DataFrame()
        
        else:
            df_summaries, report_text_input = self.generate_summaries_df(df_grouped, summary_input_col='summary_input')
        
            # Generate final report
            print('Generating Company Report...')
            report_text = self.summarize_string(
                report_text_input
            )

            # Extract structured data
            print('Extracting Structured Data Table...')
            structured_data = self.create_consolidated_data_table(report_text)
            
            return report_text, structured_data

    def generate_report_by_entities(self, df: pd.DataFrame, entity_keys: list[str], text_col: str = 'text', 
                              fields_for_summary: list[str] | None = None) -> dict[str, tuple[str, pd.DataFrame]]:
        """
        Process multiple entities in batch.
        
        Args:
            df: Input DataFrame with credit rating data
            entity_keys: entity IDs
            start_date: Start date for the reports
            end_date: End date for the reports
            fields_for_summary: Fields to include in summaries
            
        Returns:
            Dict mapping entity IDs to (report_text, structured_data_df) tuples
        """
        results = {}
        
        # Display progress bar for entity processing
        total_entities = len(entity_keys)
        processed = 0

        for entity_id in entity_keys:
            processed += 1
            print(f"Processing... ({processed}/{total_entities})")
            
            report_text, structured_data = self.generate_company_report(
                df, entity_id,text_col,
                fields_for_summary,
            )
            
            results[entity_id] = (report_text, structured_data)
            
        return results

_initialization_sent = False

def notebook_initialized():
    """Stub for notebook initialization (no SDK tracking)."""
    global _initialization_sent
    if not _initialization_sent:
        _initialization_sent = True
        # Tracking removed - no SDK dependency
        pass

notebook_initialized()