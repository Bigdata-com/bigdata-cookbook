import pandas as pd
import re
import json
import asyncio
from typing import List, Dict, Union, Optional, Tuple, Any

from bigdata_research_tools.llm.base import AsyncLLMEngine
from bigdata_research_tools.labeler.labeler import Labeler, get_prompts_for_labeler, parse_labeling_response


class SummaryGenerator(Labeler):
    """
    A class to generate summaries and reports from credit rating data.
    
    This class encapsulates the functionality for processing credit rating data,
    generating summaries, and creating structured reports through LLM processing.
    Uses the bigdata_research_tools.llm framework for LLM interactions with
    specialized handling for token limits through text splitting.
    """

    def __init__(self, llm_model: str = "openai::gpt-4o-mini", temperature: float = 0, max_workers: int = 30):
        """
        Initialize the SummaryGenerator with LLM configuration.
        
        Args:
            llm_model: LLM model to use in format "provider::model" (e.g., "openai::gpt-4o-mini")
            temperature: Temperature for the model
            max_workers: Maximum number of concurrent workers for batch processing
        """
        self.llm_model = llm_model
        self.llm_engine = AsyncLLMEngine(model=llm_model)
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
    
    def _split_text_on_nearest_linebreak(self, text_string: str, num_splits: int) -> List[str]:
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

    def prepare_daily_summary_input(self, df: pd.DataFrame,
                                    date_col: str = 'date',
                                    sentence_id_col: str = 'sentence_id',
                                    text_col: str = 'text',
                                    summary_input: List[str] = None) -> pd.DataFrame:
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
        df = df.fillna('None').applymap(lambda x: "; ".join(x) if isinstance(x, list) else x)
        
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
            
        sentence_grouped = df.groupby([date_col, sentence_id_col], dropna=False).apply(aggregate_sentence_fields).reset_index()
        
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
            
        date_grouped = sentence_grouped.groupby(date_col, dropna=False).apply(aggregate_date_fields).reset_index()
        # date_grouped['id'] = range(len(date_grouped))
        # date_grouped['summary_input'] = date_grouped.apply(lambda row: 'Id: ' + str(row.id) + '\n' + row['summary_input'], axis=1) not needed if I create the prompts as in the labeler
        
        return date_grouped

    def generate_summaries_df(self, df: pd.DataFrame, summary_input_col: str, system_prompt: Optional[str] = None, additional_prompt_fields: Optional[list] = [], max_workers: int = 30) -> pd.DataFrame:

        if system_prompt is None:
            system_prompt = self.daily_chunk_summarization
        
        if additional_prompt_fields:
            textsconfig = self._add_prompt_fields(df, additional_prompt_fields=additional_prompt_fields)
        else:
            textsconfig = []

        texts = df[summary_input_col].tolist()

        prompts = get_prompts_for_labeler(texts, textsconfig=textsconfig)

        #these prompts should have summary input instead of text as key

        responses = self._run_labeling_prompts(
            prompts, system_prompt, max_workers=max_workers
        )
        responses = [parse_labeling_response(response) for response in responses]

        ## add error catching, splits, retries, and consolidation to the failed ones.

        parsed_responses = self._deserialize_label_responses(responses)
        if len(parsed_responses['motivation'].unique()) == 1 and parsed_responses['motivation'].values[0]=='':
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

        report_text_input = f'Ratee Entity: {entity_name}\n' + '\n'.join(
            [f'Date{i}: {str(row.date)}\nText{i}: {row.daily_summary}' 
             for i, (_, row) in enumerate(df_merged.iterrows())]
        )
        return df_merged, report_text_input

    def summarize_string(self, text: str, system_prompt: Optional[str] = None, max_retries: int = 5,
                      max_split_retries: int = 5) -> str:
        """
        Summarize a text string with retry and text splitting capabilities.
        
        Args:
            text: Text to summarize
            prompt_type: Type of prompt to use
            replacements: Key-value pairs for prompt replacements
            max_retries: Maximum retry attempts
            max_split_retries: Maximum retry attempts with text splitting
            
        Returns:
            Summarized text
        """
        if system_prompt is None:
            system_prompt = self.daily_credit_ratings_report
        
        # Build chat history
        chat_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        # Try to get response directly
        try:
            return asyncio.run(self.llm_engine.get_response(chat_history, temperature=self.temperature))
        except Exception as e:
            if 'context_length_exceeded' in str(e) or 'string_above_max_length' in str(e):
                print("Text too long for direct processing, attempting split-and-consolidate approach...")
                
                # Try splitting and processing in chunks
                try:
                    # Split the text
                    splits = self._split_text_on_nearest_linebreak(text, 2)  # Start with 2 splits
                    
                    # Process each split
                    results = []
                    for split_text in splits:
                        split_chat_history = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": split_text}
                        ]
                        response = asyncio.run(self.llm_engine.get_response(
                            split_chat_history, 
                            temperature=self.temperature
                        ))
                        results.append(response)
                    
                    # Consolidate results
                    if len(results) > 1:
                        consolidation_prompt = self.credit_ratings_consolidation
                        consolidation_text = '\n\n'.join([f"Completion {i+1}: {comp}" for i, comp in enumerate(results)])
                        
                        consolidation_chat_history = [
                            {"role": "system", "content": consolidation_prompt},
                            {"role": "user", "content": f"Please merge and consolidate these completions into a single response:\n\n{consolidation_text}"}
                        ]
                        
                        return asyncio.run(self.llm_engine.get_response(
                            consolidation_chat_history,
                            temperature=self.temperature
                        ))
                    else:
                        return results[0]
                        
                except Exception as nested_e:
                    print(f"Error during split processing: {str(nested_e)}")
                    return f"Error: Failed to process text after attempts to split. {str(nested_e)}"
            else:
                print(f"Error during summarization: {str(e)}")
                return f"Error: {str(e)}"
    
    def create_consolidated_data_table(self, text: str, system_prompt: Optional[str] = None) -> pd.DataFrame:
        """
        Extract structured data from text.
        
        Args:
            text: Text to process
            prompt_type: Type of prompt to use
            
        Returns:
            DataFrame with extracted structured data
        """
        if system_prompt is None:
            system_prompt = self.credit_ratings_data_table
        
        # Build chat history
        chat_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        try:
            # Get response
            json_string = asyncio.run(self.llm_engine.get_response(
                chat_history,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            ))
            
            # Clean up JSON string
            json_string = re.sub('```', '', json_string)
            json_string = re.sub('json', '', json_string)

            # Parse JSON into DataFrame
            try:
                json_string = json.loads(json_string)['data']
                data = pd.DataFrame(json_string)
                return data
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}.")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error creating data table: {e}")
            return pd.DataFrame()
    
    def generate_company_report(self, df: pd.DataFrame, entity_id: str,text_col:str='text',
                               fields_for_summary: List[str] = None,) -> Tuple[str, pd.DataFrame]:
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

    def generate_report_by_entities(self, df: pd.DataFrame, entity_keys: List[str], text_col: str = 'text', 
                              fields_for_summary: List[str] = None) -> Dict[str, Tuple[str, pd.DataFrame]]:
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
