"""
Module for managing labeling operations.

Copyright (C) 2024, RavenPack | Bigdata.com. All rights reserved.
"""

from logging import Logger, getLogger
from typing import Dict, List, Optional, Union

#from risk_labeler import replace_company_placeholders
from pandas import DataFrame

from bigdata_research_tools.labeler.labeler import (
    Labeler,
    parse_labeling_response,
)
from ..prompts.labeler import get_narrative_system_prompt
from ..labeler.labeler import get_prompts_for_labeler
logger: Logger = getLogger(__name__)


class NarrativeLabeler(Labeler):
    """Narrative labeler."""

    def __init__(
        self,
        llm_model: str,
        label_prompt: Optional[str] = None,
        unknown_label: str = "unclear",
        temperature: float = 0,
    ):
        """Initialize narrative labeler.

        Args:
            llm_model: Name of the LLM model to use. Expected format:
                <provider>::<model>, e.g. "openai::gpt-4o-mini"
            label_prompt: Prompt provided by user to label the search result chunks.
                If not provided, then our default labelling prompt is used.
            unknown_label: Label for unclear classifications
            temperature: Temperature to use in the LLM model.
        """
        super().__init__(llm_model, unknown_label, temperature)
        self.label_prompt = label_prompt

    def get_labels(
        self,
        main_theme: str = "",
        theme_labels: List[str] = [],
        texts: List[str] = [],
        titles: Optional[List[str]] = None,
        max_workers: int = 50,
        entity_track: str = "",
        mode: str = "default",
    ) -> DataFrame:
        """
        Process thematic labels for texts.

        Args:
            theme_labels: The main theme to analyze.
            texts: List of texts to label.
            titles: Optional list of titles. If provided, they will be added as "title" field.
            max_workers: Maximum number of concurrent workers.
            entity_track: Entity track for tracking the entity.
            mode: Mode of the labeling.
        Returns:
            DataFrame with schema:
            - index: sentence_id
            - columns:
                - motivation
                - label
        """
        system_prompt = (
            get_narrative_system_prompt(main_theme, theme_labels, mode, entity_track)
            if self.label_prompt is None
            else self.label_prompt
        )

        prompts = get_prompts_for_labeler(texts, titles=titles)


        responses = self._run_labeling_prompts(
            prompts, system_prompt, max_workers=max_workers
        )
        responses = [parse_labeling_response(response) for response in responses]
        return self._deserialize_label_responses(responses)

    def post_process_dataframe(self, df: DataFrame) -> DataFrame:
        """
        Post-process the labeled DataFrame.

        Args:
            df: DataFrame to process. Schema:
                - Index: int
                - Columns:
                    - timestamp_utc: datetime64
                    - document_id: str
                    - sentence_id: str
                    - headline: str
                    - text: str
                    - label: str
                    - motivation: str
        Returns:
            Processed DataFrame. Schema:
            - index: int
            - Columns:
                - Time Period
                - Date
                - Document ID
                - Headline
                - Chunk Text
                - Motivation
                - Label
                - Entity
                - Country Code
                - Entity Type
        """
        # Filter unlabeled sentences
        df = df.loc[df["label"] != self.unknown_label].copy()
        if df.empty:
            logger.warning(f"Empty dataframe: all rows labelled {self.unknown_label}")
            return df

        # Process timestamps
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize(None)

        # Sort and format
        sort_columns = ["timestamp_utc", "label"]
        df = df.sort_values(by=sort_columns).reset_index(drop=True)

        # Add formatted columns
        df["Time Period"] = df["timestamp_utc"].dt.strftime("%b %Y")
        df["Date"] = df["timestamp_utc"].dt.strftime("%Y-%m-%d")

        df = df.rename(
            columns={
                "document_id": "Document ID",
                "sentence_id": "Sentence ID",
                "headline": "Headline",
                "text": "Chunk Text",
                "motivation": "Motivation",
                "label": "Label",
                "entity": "Entity",
                "country_code": "Country Code",
                "entity_type": "Entity Type",
            }
        )

        df = df.explode(["Entity", "Entity Type", "Country Code"], ignore_index=True)

        # Select and order columns
        export_columns = [
            "Time Period",
            "Date",
            "Document ID",
            "Sentence ID",
            "Headline",
            "Chunk Text",
            "Motivation",
            "Label",
            "Entity",
            "Country Code",
            "Entity Type",
        ]

        sort_columns = ["Date", "Time Period", "Document ID", "Headline", "Chunk Text"]
        df = df[export_columns].sort_values(sort_columns).reset_index(drop=True) 
        
        return df



class NarrativeSummarizerFlex(Labeler):
    """Screener summarizer for company-level analysis."""

    def __init__(
        self,
        llm_model: str,
        summary_prompt: Optional[str] = None,
        unknown_label: str = "unclear",
        temperature: float = 0,
    ):
        """
        Args:
            llm_model: Name of the LLM model to use. Expected format:
                <provider>::<model>, e.g. "openai::gpt-4o-mini"
            summary_prompt: Prompt provided by user to summarize the company data.
                If not provided, then our default summarization prompt is used.
            unknown_label: Label for unclear classifications.
            temperature: Temperature to use in the LLM model.
        """
        super().__init__(llm_model, unknown_label, temperature)
        self.summary_prompt = summary_prompt

    def get_summaries(
        self,
        main_theme: str,
        df: DataFrame = None,
        mode: str = "default",
        max_workers: int = 50,
        shift_from: str = "",
        shift_to: str = "",
        entity_track: str = "",
        previous_narrative: Union[str, Dict[str, str]] = "",
        additional_parameters: Dict[str, str] = {},
        data_fields: List[str] = None,
        previous_summary: str = "",
        today_summary: str = "",
    ) -> Union[DataFrame, str]:
        """
        Process entity-level summaries from labeled data.

        Args:
            main_theme: The main theme to analyze.
            df: DataFrame with labeled entity data. Expected columns:
                - Entity: str (entity name)
                - Chunk Text: str (sentence text)
                - Label: str (assigned thematic label)
                - Motivation: str (labeling motivation explaining the theme assignment)
                - Country Code: str (entity country code)
                - Time Period: str (time period info)
                - Date: str (date info)
                - Document ID: str (document identifier)
                - Sentence ID: str (sentence identifier)
                - Headline: str (headline text)
            mode: Mode of the summarization ('default' or 'impact').
            max_workers: Maximum number of concurrent workers.
            shift_from: Source element for shift-based analysis (used in 'impact' mode).
            shift_to: Target element for shift-based analysis (used in 'impact' mode).
            entity_track: Entity track for tracking the entity.
            previous_narrative: Previous narrative for temporal analysis. Can be:
                - str: Single narrative text (legacy format)
                - dict: Dictionary with day keys and narrative values, e.g. {"2024-01-01": "narrative1", "2024-01-02": "narrative2"}
            additional_parameters: Additional parameters to include in the prompt.
            data_fields: List of data fields to include in prompts. Default: ["quotes"].
                Available options: ["quotes", "themes", "motivations"].
                - "quotes" maps to Chunk Text
                - "themes" maps to Label  
                - "motivations" maps to Motivation
        Returns:
            DataFrame with schema:
            - index: entity_name
            - columns:
                - summary: str (comprehensive entity summary)
                - key_points: List[str] (key insights)
                - country: str
                - quote_count: int
        """
        # Special handling for company_narrative_consolidation mode
        if mode == "company_narrative_consolidation":
            return self._handle_narrative_consolidation(
                main_theme, previous_summary, today_summary, max_workers, additional_parameters
            )
        
        # Special handling for companies_daily_highlights_from_daily_key_points mode
        if mode == "companies_daily_highlights_from_daily_key_points":
            return self._handle_companies_daily_highlights(
                main_theme, df, max_workers, additional_parameters
            )
        
        # Special handling for final_summary_general_report mode
        if mode == "final_summary_general_report":
            return self._handle_final_summary_general_report(
                main_theme, df, max_workers, additional_parameters
            )
        
        # Set default data fields if not provided
        if data_fields is None:
            data_fields = ["quotes"]
        
        # Group data by entity
        entity_groups = self._group_data_by_entity(df)
        
        # Generate prompts for each entity
        prompts = self._get_prompts_for_summarizer(entity_groups, data_fields, entity_track, previous_narrative, mode)
        
        # Get system prompt for summarization
        system_prompt = self.summary_prompt or self._get_summarizer_system_prompt(
            main_theme, mode=mode, shift_from=shift_from, shift_to=shift_to, entity_track=entity_track, previous_narrative=previous_narrative, additional_parameters=additional_parameters
        )
        
        # DEBUG: Print system prompt
        print("=" * 80)
        print("DEBUG: SYSTEM PROMPT")
        print("=" * 80)
        print(system_prompt[:1000])
        print("=" * 80)
        import sys
        sys.stdout.flush()

        # DEBUG: Print all prompts
        print("DEBUG: USER PROMPTS (Total: {})".format(len(prompts)))
        print("=" * 80)
        for i, prompt in enumerate(prompts):
            print(f"PROMPT {i+1}:")
            print("-" * 40)
            print(prompt[:1000])
            print("-" * 40)
        print("=" * 80)
        sys.stdout.flush()
        
        # Run summarization prompts using same low-level function
        responses = self._run_labeling_prompts(
            prompts, system_prompt, max_workers=max_workers
        )
        
        # Parse responses (entity-level instead of sentence-level)
        responses = [self._parse_summarization_response(response) for response in responses]
        return self._deserialize_summary_responses(responses, entity_groups)

    def _group_data_by_entity(self, df: DataFrame) -> Dict[str, Dict]:
        """
        Group DataFrame data by entity.
        
        Args:
            df: Input DataFrame with labeled data. Expected columns:
                - Entity: str (entity name)
                - Chunk Text: str (sentence text)
                - Label: str (assigned label)
                - Motivation: str (labeling motivation)
                - Country Code: str (entity country code)
                - Time Period: str (time period info)
                - Date: str (date info)
                - Document ID: str (document identifier)
                - Sentence ID: str (sentence identifier)
                - Headline: str (headline text)
            
        Returns:
            Dictionary with entity names as keys and aggregated data as values
        """
        grouped_data = {}
        
        for entity_name in df['Entity'].unique():
            entity_df = df[df['Entity'] == entity_name]
            
            # Aggregate data for this entity using new column names
            entity_data = {
                'entity_name': entity_name,
                'quotes': entity_df['Chunk Text'].tolist() if 'Chunk Text' in entity_df.columns else [],
                'themes': entity_df['Label'].tolist() if 'Label' in entity_df.columns else [],
                'motivations': entity_df['Motivation'].tolist() if 'Motivation' in entity_df.columns else [],
                'country': entity_df['Country Code'].iloc[0] if 'Country Code' in entity_df.columns else '',
                'quote_count': len(entity_df)
            }
            
            # Add Summary, Key_points, Quotes if they exist (for temporal_company_narrative_from_summaries mode)
            if 'Summary' in entity_df.columns:
                entity_data['summary'] = entity_df['Summary'].iloc[0]  # Take first value since it should be the same for all rows of the same entity
            if 'Key_points' in entity_df.columns:
                entity_data['key_points'] = entity_df['Key_points'].iloc[0]
            if 'Quotes' in entity_df.columns:
                entity_data['quotes'] = entity_df['Quotes'].iloc[0]  # Override the Chunk Text quotes if this exists
                
            # Add Date and Summary lists for final_summary_from_daily_summaries mode
            if 'Date' in entity_df.columns and 'Summary' in entity_df.columns:
                # Sort by date to ensure chronological order
                entity_df_sorted = entity_df.sort_values('Date')
                entity_data['dates'] = entity_df_sorted['Date'].dt.strftime('%Y-%m-%d').tolist()
                entity_data['daily_summaries'] = entity_df_sorted['Summary'].tolist()
                
            grouped_data[entity_name] = entity_data
        
        return grouped_data
    
    def _get_prompts_for_summarizer(self, entity_groups: Dict[str, Dict], data_fields: List[str], entity_track: str = "", previous_narrative: Union[str, Dict[str, str]] = "", mode: str = "default") -> List[str]:
        """
        Generate prompts for entity-level summarization.
        
        Args:
            entity_groups: Dictionary of entity data
            data_fields: List of data fields to include in prompts
            entity_track: Entity track for tracking - if present, don't add Entity field
            previous_narrative: Previous narrative - if present, add before quotes. Can be:
                - str: Single narrative (legacy format)
                - dict: Dictionary with day keys and narrative values
            mode: Mode of summarization to determine prompt structure
            
        Returns:
            List of JSON prompts for summarization with configurable data structure
        """
        from json import dumps
        
        prompts = []
        for entity_name, entity_data in entity_groups.items():
            # Create prompt data - only add Entity name if entity_track is not provided
            prompt_data = {}
            
            # For companies_impact and temporal_company_narrative_from_summaries modes, always add Company field
            if mode == "companies_impact" or mode == "temporal_company_narrative_from_summaries":
                prompt_data['Company'] = entity_name
            elif mode == "final_summary_from_daily_summaries":
                prompt_data['Company or Person'] = entity_name
            elif not entity_track:
                prompt_data['Entity'] = entity_name
            
            # Add previous narrative if provided (before all quotes)
            if previous_narrative:
                if isinstance(previous_narrative, dict):
                    # Filter previous narrative for this specific entity
                    entity_previous = {}
                    
                    for key, narrative in previous_narrative.items():
                        # If key contains entity name, extract the date part
                        if f"{entity_name}_" in key:
                            date_part = key.replace(f"{entity_name}_", "")
                            entity_previous[date_part] = narrative
                        # If key doesn't contain entity name, it's a legacy format, use as is
                        elif "_" not in key:
                            entity_previous[key] = narrative
                    
                    # For temporal_company_narrative_from_summaries mode, use consolidated approach
                    if mode == "temporal_company_narrative_from_summaries":
                        # Consolidate all narratives into a single "Previous Narrative" block
                        if entity_previous:
                            consolidated_narrative = ""
                            for day, narrative in sorted(entity_previous.items()):
                                if narrative:
                                    consolidated_narrative += narrative + "\n\n"
                            
                            if consolidated_narrative.strip():
                                prompt_data['Previous Narrative'] = consolidated_narrative.strip()
                    else:
                        # Original behavior for other modes - separate day entries
                        for day_index, (day, narrative) in enumerate(sorted(entity_previous.items()), 1):
                            prompt_data[f'Narrative day {day_index}'] = narrative
                else:
                    # Legacy string format
                    prompt_data['Previous_Narrative'] = previous_narrative
            
            # Handle different modes for data inclusion
            if mode == "temporal_company_narrative_from_summaries":
                # For this mode, use Summary, Key_points, Quotes directly from the data
                # Add "Today" prefix if there's previous narrative
                summary_key = "Today Summary" if previous_narrative else "Summary"
                key_points_key = "Today Key_points" if previous_narrative else "Key_points"
                quotes_key = "Today Quotes" if previous_narrative else "Quotes"
                
                if 'summary' in entity_data:
                    prompt_data[summary_key] = entity_data['summary']
                if 'key_points' in entity_data:
                    prompt_data[key_points_key] = entity_data['key_points']
                if 'quotes' in entity_data:
                    prompt_data[quotes_key] = entity_data['quotes']
            elif mode == "final_summary_from_daily_summaries":
                # For this mode, structure data with dates and summaries chronologically
                if 'dates' in entity_data and 'daily_summaries' in entity_data:
                    dates = entity_data['dates']
                    summaries = entity_data['daily_summaries']
                    
                    # Create the structured prompt as requested:
                    # Nome Entity, Summary 2024-01-01, Summary 2024-01-02, etc.
                    for date, summary in zip(dates, summaries):
                        prompt_data[f'Summary {date}'] = summary
            else:
                # Original logic for other modes
                # Determine the maximum number of items (based on quotes which should always be present)
                max_items = len(entity_data.get('quotes', []))
                
                # Add data fields based on configuration
                for i in range(max_items):
                    item_index = i + 1
                    
                    if 'quotes' in data_fields and i < len(entity_data.get('quotes', [])):
                        prompt_data[f'Quote_{item_index}'] = entity_data['quotes'][i]
                    
                    if 'themes' in data_fields and i < len(entity_data.get('themes', [])):
                        prompt_data[f'Theme_{item_index}'] = entity_data['themes'][i]
                    
                    if 'motivations' in data_fields and i < len(entity_data.get('motivations', [])):
                        prompt_data[f'Motivation_{item_index}'] = entity_data['motivations'][i]
            
            prompts.append(dumps(prompt_data))
        
        return prompts
    
    def _get_summarizer_system_prompt(
        self, main_theme: str, mode: str = "default", shift_from: str = "", shift_to: str = "", entity_track: str = "", previous_narrative: Union[str, Dict[str, str]] = "",additional_parameters: Dict[str, str] = {}
    ) -> str:
        """
        Get system prompt for summarization.
        
        Args:
            main_theme: The main theme to analyze
            mode: Summarization mode
            shift_from: Source element for shift analysis
            shift_to: Target element for shift analysis
            entity_track: Entity track for tracking the entity
            previous_narrative: Previous narrative for temporal analysis (str or dict format)
            additional_parameters: Additional parameters to include in the prompt.
        Returns:
            System prompt string
        """
        from ..prompts.labeler import get_summarizer_system_prompt
        return get_summarizer_system_prompt(main_theme, mode, shift_from, shift_to, entity_track, previous_narrative, additional_parameters)
    
    def _parse_summarization_response(self, response: str) -> Dict:
        """
        Parse summarization response from LLM.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Parsed response dictionary
        """
        from json import JSONDecodeError, loads
        
        try:
            deserialized_response = loads(response)
        except JSONDecodeError:
            logger.error(f"Error deserializing summarization response: {response}")
            return {}
        
        # The response should be a simple dict with summary
        # e.g., {"summary": "..."}
        if not isinstance(deserialized_response, dict):
            logger.error(f"Expected dict response, got {type(deserialized_response)}")
            return {}
        
        return deserialized_response
    
    def _deserialize_summary_responses(self, responses: List[Dict], entity_groups: Dict[str, Dict]) -> DataFrame:
        """
        Convert parsed responses to DataFrame.
        
        Args:
            responses: List of parsed response dictionaries
            entity_groups: Original entity group data
            
        Returns:
            DataFrame with summarization results
        """
        import pandas as pd
        
        summary_data = []
        entity_names = list(entity_groups.keys())
        
        # Process each response (one per entity)
        for i, response in enumerate(responses):
            if not response:  # Skip empty responses
                continue
            
            # Get the corresponding entity name (responses are in same order as entity_groups)
            if i < len(entity_names):
                entity_name = entity_names[i]
                entity_meta = entity_groups[entity_name]
                
                # Extract summary, bullet points, and quotes from response
                summary_text = response.get('summary', '')
                bullet_points = response.get('bullet_points', [])
                quotes = response.get('quotes', [])
                
                summary_record = {
                    'Entity': entity_name,
                    'summary': summary_text,
                    'key_points': bullet_points,
                    'quotes': quotes,
                    'country': entity_meta.get('country', ''),
                    'quote_count': entity_meta.get('quote_count', 0)
                }
                
                summary_data.append(summary_record)
        
        # Create DataFrame with entity_name as column
        if summary_data:
            df = pd.DataFrame(summary_data)
        else:
            # Return empty DataFrame with expected structure
            df = pd.DataFrame(columns=[
                'Entity', 'summary', 'key_points', 'quotes',
                'country', 'quote_count'
            ])
        
        return df

    def post_process_dataframe(self, df: DataFrame) -> DataFrame:
        """
        Post-process the summarized DataFrame.

        Args:
            df: DataFrame to process. Schema:
                - Index: entity_name
                - Columns:
                    - summary: str
                    - key_points: List[str]
                    - quotes: List[str]
                    - country: str
                    - quote_count: int

        Returns:
            Processed DataFrame with export-ready format.
        """
        if df.empty:
            logger.warning("Empty summarization dataframe")
            return df

        # Reset index to make entity_name a column
        df = df.reset_index()
        
        # Key points not used in simplified format
        # df['key_points_formatted'] = ''
        
        # Sort by entity name
        df = df.sort_values('Entity').reset_index(drop=True)
        
        # Rename columns for export
        df = df.rename(columns={
            'Entity': 'Entity',
            'summary': 'Summary',
            'quotes': 'Quotes',
            'country': 'Country',
            'quote_count': 'Quote Count'
        })
        
        # Select and order columns for export
        export_columns = [
            'Entity',
            'Country',
            'Quote Count',
            'Summary',
            'Quotes'
        ]
        
        # Only include columns that exist in the DataFrame
        available_columns = [col for col in export_columns if col in df.columns]
        df = df[available_columns]
        
        return df
    
    def _handle_narrative_consolidation(
        self, 
        main_theme: str, 
        previous_summary: str, 
        today_summary: str, 
        max_workers: int,
        additional_parameters: Dict[str, str] = {}
    ) -> str:
        """
        Handle narrative consolidation mode that takes two summary strings.
        
        Args:
            main_theme: The main theme to analyze
            previous_summary: Previous summary string
            today_summary: Today's summary string
            max_workers: Maximum number of concurrent workers
            additional_parameters: Additional parameters for the prompt
            
        Returns:
            Consolidated summary as string
        """
        from json import dumps
        
        # Create prompt data with the two summaries
        prompt_data = {
            'Previous Summary': previous_summary,
            'Today Summary': today_summary
        }
        
        # Create single prompt
        prompt = dumps(prompt_data)
        prompts = [prompt]
        
        # Get system prompt for consolidation
        system_prompt = self.summary_prompt or self._get_summarizer_system_prompt(
            main_theme, mode="company_narrative_consolidation", entity_track="", previous_narrative="", additional_parameters=additional_parameters
        )
        
        # DEBUG: Print system prompt
        print("=" * 80)
        print("DEBUG: CONSOLIDATION SYSTEM PROMPT")
        print("=" * 80)
        print(system_prompt)
        print("=" * 80)
        
        # DEBUG: Print prompt
        print("DEBUG: CONSOLIDATION USER PROMPT")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        
        # Run consolidation prompt
        responses = self._run_labeling_prompts(
            prompts, system_prompt, max_workers=max_workers
        )
        
        # Parse response and extract just the summary text
        if responses and len(responses) > 0:
            response = self._parse_summarization_response(responses[0])
            consolidated_summary = response.get('summary', '')
            return consolidated_summary
        else:
            return ""
    
    def _handle_companies_daily_highlights(
        self, 
        main_theme: str, 
        df: DataFrame, 
        max_workers: int,
        additional_parameters: Dict[str, str] = {}
    ) -> DataFrame:
        """
        Handle companies_daily_highlights_from_daily_key_points mode.
        
        Args:
            main_theme: The main theme to analyze
            df: DataFrame with Entity, Date, Key Points columns
            max_workers: Maximum number of concurrent workers
            additional_parameters: Must contain "companies" key with list of companies
            
        Returns:
            DataFrame with highlights results
        """
        from json import dumps
        
        # Get companies list from additional_parameters
        companies = additional_parameters.get("companies", [])
        if not companies:
            logger.warning("No companies provided in additional_parameters")
            return DataFrame()
        
        # Filter DataFrame to only include specified companies
        df_filtered = df[df['Entity'].isin(companies)].copy()
        
        if df_filtered.empty:
            logger.warning(f"No data found for companies: {companies}")
            return DataFrame()
        
        # Group by date and create the structured prompt
        prompt_data = {}
        
        for date in sorted(df_filtered['Date'].dt.strftime('%Y-%m-%d').unique()):
            date_data = df_filtered[df_filtered['Date'].dt.strftime('%Y-%m-%d') == date]
            
            # Create company -> key_points mapping for this date
            company_highlights = {}
            for _, row in date_data.iterrows():
                company = row['Entity']
                key_points = row['Key Points'] if 'Key Points' in row else row.get('Key_points', '')
                company_highlights[company] = key_points
            
            prompt_data[f'Data {date}'] = company_highlights
        
        # Create single prompt
        prompt = dumps(prompt_data)
        prompts = [prompt]
        
        # Get system prompt
        system_prompt = self.summary_prompt or self._get_summarizer_system_prompt(
            main_theme, mode="companies_daily_highlights_from_daily_key_points", 
            entity_track="", previous_narrative="", additional_parameters=additional_parameters
        )
        
        
        # Run prompt
        responses = self._run_labeling_prompts(
            prompts, system_prompt, max_workers=max_workers
        )
        
        # Parse response
        if responses and len(responses) > 0:
            response = self._parse_summarization_response(responses[0])
            
            # The response should be a dict with dates as keys and highlight arrays as values
            # Format: {"Data 2024-01-01": ["highlight1", "highlight2"], "Data 2024-01-02": ["highlight1"]}
            
            # Convert to DataFrame format 
            result_data = []
            for date_key, highlights in response.items():
                # Clean the date key: remove "Data " prefix and keep only YYYY-MM-DD
                clean_date = date_key.replace("Data ", "").strip()
                
                # highlights should be a list of strings
                if isinstance(highlights, list):
                    for highlight in highlights:
                        result_data.append({
                            'Date': clean_date,  # Use cleaned date format (YYYY-MM-DD only)
                            'Highlight': highlight,
                            'Companies': ', '.join(companies)
                        })
                elif isinstance(highlights, str):
                    # In case highlights is a single string, treat it as one highlight
                    result_data.append({
                        'Date': clean_date,
                        'Highlight': highlights,
                        'Companies': ', '.join(companies)
                    })
                else:
                    # Log unexpected format but continue
                    logger.warning(f"Unexpected highlights format for date {date_key}: {type(highlights)}")
            
            if result_data:
                return DataFrame(result_data)
            else:
                # Return empty DataFrame with expected structure if no data
                return DataFrame(columns=['Date', 'Highlight', 'Companies'])
        else:
            return DataFrame(columns=['Date', 'Highlight', 'Companies'])
    
    def _handle_final_summary_general_report(
        self, 
        main_theme: str, 
        df: DataFrame, 
        max_workers: int,
        additional_parameters: Dict[str, str] = {}
    ) -> str:
        """
        Handle final_summary_general_report mode.
        
        Args:
            main_theme: The main theme to analyze
            df: DataFrame with Index(['Entity', 'summary', 'key_points', 'quotes', 'country', 'quote_count'])
            max_workers: Maximum number of concurrent workers
            additional_parameters: Additional parameters for the prompt
            
        Returns:
            String with the final general report
        """
        from json import dumps
        
        # Create the JSON structure: "Entity name": "summary"
        entities_dict = {}
        for _, row in df.iterrows():
            entity_name = row['Entity']
            summary = row['summary']
            entities_dict[entity_name] = summary
        
        # Convert to JSON string for the prompt
        entities_json = dumps(entities_dict, indent=2)
        
        # Create single prompt following the pattern from other methods
        prompt = entities_json
        prompts = [prompt]
        
        # Get system prompt for this mode
        system_prompt = self.summary_prompt or self._get_summarizer_system_prompt(
            main_theme, 
            mode="final_summary_general_report", 
            entity_track="", 
            previous_narrative="", 
            additional_parameters=additional_parameters
        )
        
        # DEBUG: Print prompts (following the same pattern as get_summaries)
        print("=" * 80)
        print("DEBUG: FINAL SUMMARY GENERAL REPORT SYSTEM PROMPT")
        print("=" * 80)
        print(system_prompt[:1000])
        print("=" * 80)
        
        print("DEBUG: FINAL SUMMARY GENERAL REPORT USER PROMPT")
        print("=" * 80)
        print(prompt[:1000])
        print("=" * 80)
        
        # Run prompt using the same method as other handlers
        responses = self._run_labeling_prompts(
            prompts, system_prompt, max_workers=max_workers
        )
        
        # Return the response directly (it's a string, not parsed like normal summaries)
        if responses and len(responses) > 0:
            return responses[0]
        else:
            return "No response generated"