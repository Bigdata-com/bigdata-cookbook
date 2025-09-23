"""
Module for managing labeling operations.

Copyright (C) 2024, RavenPack | Bigdata.com. All rights reserved.
"""

from logging import Logger, getLogger
from typing import Dict, List, Optional

from pandas import DataFrame, Series
from bigdata_research_tools.labeler.risk_labeler import replace_company_placeholders
from bigdata_research_tools.labeler.labeler import (
    Labeler,
    parse_labeling_response,
)
from bigdata_research_tools.prompts.labeler import (
    get_other_entity_placeholder,
    get_target_entity_placeholder,
)

from ..prompts.labeler import get_screener_system_prompt
from ..labeler.labeler import get_prompts_for_labeler



logger: Logger = getLogger(__name__)


class ScreenerLabelerFlex(Labeler):
    """Screener labeler."""

    def __init__(
        self,
        llm_model: str,
        label_prompt: Optional[str] = None,
        unknown_label: str = "unclear",
        temperature: float = 0,
    ):
        """
        Args:
            llm_model: Name of the LLM model to use. Expected format:
                <provider>::<model>, e.g. "openai::gpt-4o-mini"
            label_prompt: Prompt provided by user to label the search result chunks.
                If not provided, then our default labelling prompt is used.
            unknown_label: Label for unclear classifications.
            temperature: Temperature to use in the LLM model.
        """
        super().__init__(llm_model, unknown_label, temperature)
        self.label_prompt = label_prompt

    def get_labels(
        self,
        main_theme: str,
        labels: List[str],
        texts: List[str],
        titles: Optional[List[str]] = None,
        mode: str = "default",
        max_workers: int = 50,
        shift_from: str = "",
        shift_to: str = "",
    ) -> DataFrame:
        """
        Process thematic labels for texts.

        Args:
            main_theme: The main theme to analyze.
            labels: Labels for labelling the chunks.
            texts: List of chunks to label.
            titles: Optional list of article titles for context.
            mode: Mode of the labeling.
            max_workers: Maximum number of concurrent workers.
            shift_from: Source element for shift-based sentiment analysis (used in 'impact' mode).
            shift_to: Target element for shift-based sentiment analysis (used in 'impact' mode).
        Returns:
            DataFrame with schema:
            - index: sentence_id
            - columns:
                - motivation
                - label
        """
        system_prompt = self.label_prompt or get_screener_system_prompt(
            main_theme, labels, unknown_label=self.unknown_label, mode=mode, shift_from=shift_from, shift_to=shift_to
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
                    - entity_id: str
                    - entity_name: str
                    - entity_sector: str
                    - entity_industry: str
                    - entity_country: str
                    - entity_ticker: str
                    - text: str
                    - other_entities: str
                    - entities: List[Dict[str, Any]]
                        - key: str
                        - name: str
                        - ticker: str
                        - start: int
                        - end: int
                    - masked_text: str
                    - other_entities_map: List[Tuple[int, str]]
                    - label: str
                    - motivation: str
        Returns:
            Processed DataFrame. Schema:
            - index: int
            - Columns:
                - Time Period
                - Date
                - Company
                - Sector
                - Industry
                - Country
                - Ticker
                - Document ID
                - Headline
                - Quote
                - Motivation
                - Theme
                - Impact Label
                - Impact Motivation
        """
        # Filter unlabeled sentences
        df = df.loc[df["label"] != self.unknown_label].copy()
        if df.empty:
            logger.warning(f"Empty dataframe: all rows labelled {self.unknown_label}")
            return df

        # Process timestamps
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize(None)

        # Sort and format
        sort_columns = ["entity_name", "timestamp_utc", "label"]
        df = df.sort_values(by=sort_columns).reset_index(drop=True)

        # Replace company placeholders
        df["motivation"] = df.apply(replace_company_placeholders, axis=1, col_name = 'motivation')

        if "Impact Motivation" in df.columns:       
            df["Impact Motivation"] = df.apply(replace_company_placeholders, axis=1, col_name = 'Impact Motivation')

        # Add formatted columns
        df["Time Period"] = df["timestamp_utc"].dt.strftime("%b %Y")
        df["Date"] = df["timestamp_utc"].dt.strftime("%Y-%m-%d")

        df = df.rename(
            columns={
                "document_id": "Document ID",
                "entity_name": "Company",
                "entity_sector": "Sector",
                "entity_industry": "Industry",
                "entity_country": "Country",
                "entity_ticker": "Ticker",
                "headline": "Headline",
                "text": "Quote",
                "motivation": "Motivation",
                "label": "Theme",
            }
        )

        # Select and order columns
        export_columns = [
            "Time Period",
            "Date",
            "Company",
            "Sector",
            "Industry",
            "Country",
            "Ticker",
            "Document ID",
            "Headline",
            "Quote",
            "Motivation",
            "Theme",
            "Impact Motivation",
            "Impact Label",
        ]

        sort_columns = ["Date", "Time Period", "Company", "Document ID", "Headline", "Quote"]
        df = df[export_columns].sort_values(sort_columns).reset_index(drop=True)        
        
        return df


class ScreenerSummarizerFlex(Labeler):
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
        df: DataFrame,
        mode: str = "default",
        max_workers: int = 50,
        shift_from: str = "",
        shift_to: str = "",
    ) -> DataFrame:
        """
        Process company-level summaries from labeled data.

        Args:
            main_theme: The main theme to analyze.
            df: DataFrame with labeled company data. Expected columns:
                - Company: str (company name)
                - Quote: str (sentence text)
                - Theme: str (assigned thematic label)
                - Motivation: str (labeling motivation explaining the theme assignment)
                - Sector: str (company sector)
                - Industry: str (company industry)
                - Country: str (company country)
                - Ticker: str (company ticker)
            mode: Mode of the summarization ('default' or 'impact').
            max_workers: Maximum number of concurrent workers.
            shift_from: Source element for shift-based analysis (used in 'impact' mode).
            shift_to: Target element for shift-based analysis (used in 'impact' mode).

        Returns:
            DataFrame with schema:
            - index: company_name
            - columns:
                - summary: str (comprehensive company summary)
                - key_points: List[str] (key insights)
                - overall_sentiment: str (for impact mode)
                - sector: str
                - industry: str
                - country: str
                - ticker: str
                - quote_count: int
        """
        # Group data by company
        company_groups = self._group_data_by_company(df)
        
        # Generate prompts for each company
        prompts = self._get_prompts_for_summarizer(company_groups)
        
        # Get system prompt for summarization
        system_prompt = self.summary_prompt or self._get_summarizer_system_prompt(
            main_theme, mode=mode, shift_from=shift_from, shift_to=shift_to
        )
        

        
        # Run summarization prompts using same low-level function
        responses = self._run_labeling_prompts(
            prompts, system_prompt, max_workers=max_workers
        )
        
        # Parse responses (company-level instead of sentence-level)
        responses = [self._parse_summarization_response(response) for response in responses]
        return self._deserialize_summary_responses(responses, company_groups)

    def _group_data_by_company(self, df: DataFrame) -> Dict[str, Dict]:
        """
        Group DataFrame data by company.
        
        Args:
            df: Input DataFrame with labeled data. Expected columns:
                - Company: str (company name)
                - Quote: str (sentence text)
                - Theme: str (assigned label)
                - Motivation: str (labeling motivation)
                - Sector: str
                - Industry: str
                - Country: str
                - Ticker: str
            
        Returns:
            Dictionary with company names as keys and aggregated data as values
        """
        grouped_data = {}
        
        for company_name in df['Company'].unique():
            company_df = df[df['Company'] == company_name]
            
            # Aggregate data for this company using new column names
            grouped_data[company_name] = {
                'company_name': company_name,
                'quotes': company_df['Quote'].tolist(),
                'themes': company_df['Theme'].tolist(),
                'motivations': company_df['Motivation'].tolist(),
                'sector': company_df['Sector'].iloc[0] if 'Sector' in company_df.columns else '',
                'industry': company_df['Industry'].iloc[0] if 'Industry' in company_df.columns else '',
                'country': company_df['Country'].iloc[0] if 'Country' in company_df.columns else '',
                'ticker': company_df['Ticker'].iloc[0] if 'Ticker' in company_df.columns else '',
                'quote_count': len(company_df)
            }
        
        return grouped_data
    
    def _get_prompts_for_summarizer(self, company_groups: Dict[str, Dict]) -> List[str]:
        """
        Generate prompts for company-level summarization.
        
        Args:
            company_groups: Dictionary of company data
            
        Returns:
            List of JSON prompts for summarization with Quote/Theme/Motivation structure
        """
        from json import dumps
        
        prompts = []
        for company_name, company_data in company_groups.items():
            # Create simplified prompt with only Company and Quote/Theme/Motivation data
            prompt_data = {
                'Company': company_name
            }
            
            # Add Quote/Theme/Motivation data directly to the main object
            for i, (quote, theme, motivation) in enumerate(zip(
                company_data['quotes'], 
                company_data['themes'], 
                company_data['motivations']
            ), 1):
                prompt_data[f'Quote_{i}'] = quote
                prompt_data[f'Theme_{i}'] = theme
                prompt_data[f'Motivation_{i}'] = motivation
            
            prompts.append(dumps(prompt_data))
        
        return prompts
    
    def _get_summarizer_system_prompt(
        self, main_theme: str, mode: str = "default", shift_from: str = "", shift_to: str = ""
    ) -> str:
        """
        Get system prompt for summarization.
        
        Args:
            main_theme: The main theme to analyze
            mode: Summarization mode
            shift_from: Source element for shift analysis
            shift_to: Target element for shift analysis
            
        Returns:
            System prompt string
        """
        from ..prompts.labeler import get_summarizer_system_prompt
        return get_summarizer_system_prompt(main_theme, mode, shift_from, shift_to)
    
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
    
    def _deserialize_summary_responses(self, responses: List[Dict], company_groups: Dict[str, Dict]) -> DataFrame:
        """
        Convert parsed responses to DataFrame.
        
        Args:
            responses: List of parsed response dictionaries
            company_groups: Original company group data
            
        Returns:
            DataFrame with summarization results
        """
        import pandas as pd
        
        summary_data = []
        company_names = list(company_groups.keys())
        
        # Process each response (one per company)
        for i, response in enumerate(responses):
            if not response:  # Skip empty responses
                continue
            
            # Get the corresponding company name (responses are in same order as company_groups)
            if i < len(company_names):
                company_name = company_names[i]
                company_meta = company_groups[company_name]
                
                # Extract both summary and bullet points from response
                summary_text = response.get('summary', '')
                bullet_points = response.get('bullet_points', [])
                
                summary_record = {
                    'company_name': company_name,
                    'summary': summary_text,
                    'key_points': bullet_points,
                    'overall_sentiment': '',  # Not used in simplified format
                    'sector': company_meta.get('sector', ''),
                    'industry': company_meta.get('industry', ''),
                    'country': company_meta.get('country', ''),
                    'ticker': company_meta.get('ticker', ''),
                    'quote_count': company_meta.get('quote_count', 0)
                }
                
                summary_data.append(summary_record)
        
        # Create DataFrame with company_name as index
        if summary_data:
            df = pd.DataFrame(summary_data)
            df = df.set_index('company_name')
        else:
            # Return empty DataFrame with expected structure
            df = pd.DataFrame(columns=[
                'summary', 'key_points', 'overall_sentiment', 
                'sector', 'industry', 'country', 'ticker', 'quote_count'
            ])
            df.index.name = 'company_name'
        
        return df

    def post_process_dataframe(self, df: DataFrame) -> DataFrame:
        """
        Post-process the summarized DataFrame.

        Args:
            df: DataFrame to process. Schema:
                - Index: company_name
                - Columns:
                    - summary: str
                    - key_points: List[str]
                    - overall_sentiment: str
                    - sector: str
                    - industry: str
                    - country: str
                    - ticker: str
                    - quote_count: int

        Returns:
            Processed DataFrame with export-ready format.
        """
        if df.empty:
            logger.warning("Empty summarization dataframe")
            return df

        # Reset index to make company_name a column
        df = df.reset_index()
        
        # Key points not used in simplified format
        # df['key_points_formatted'] = ''
        
        # Sort by company name
        df = df.sort_values('company_name').reset_index(drop=True)
        
        # Rename columns for export
        df = df.rename(columns={
            'company_name': 'Company',
            'summary': 'Summary',
            'sector': 'Sector',
            'industry': 'Industry',
            'country': 'Country',
            'ticker': 'Ticker',
            'quote_count': 'Quote Count'
        })
        
        # Select and order columns for export
        export_columns = [
            'Company',
            'Sector',
            'Industry', 
            'Country',
            'Ticker',
            'Quote Count',
            'Summary'
        ]
        
        # Only include columns that exist in the DataFrame
        available_columns = [col for col in export_columns if col in df.columns]
        df = df[available_columns]
        
        return df

