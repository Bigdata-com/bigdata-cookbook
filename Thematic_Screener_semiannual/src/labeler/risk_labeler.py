"""
Module for managing labeling operations.

Copyright (C) 2024, RavenPack | Bigdata.com. All rights reserved.
"""

from logging import Logger, getLogger
from typing import List, Optional, Any, Dict

from pandas import DataFrame, Series

from bigdata_research_tools.labeler.labeler import (
    Labeler,
    get_prompts_for_labeler,
    parse_labeling_response,
)
from bigdata_research_tools.prompts.labeler import (
    get_other_entity_placeholder,
    get_target_entity_placeholder,
    get_risk_system_prompt,
)

logger: Logger = getLogger(__name__)


class RiskLabeler(Labeler):
    """Risk labeler."""

    def __init__(
        self,
        llm_model: str,
        label_prompt: Optional[str] = None,
        # TODO (cpinto, 2025.02.07) This value is also in the prompt used.
        #  Changing it here would break the process.
        unknown_label: str = "unclear",
        temperature: float = 0,
    ):
        """
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
        main_theme: str,
        labels: List[str],
        texts: List[str],
        max_workers: int = 50,
        textsconfig: Optional[List[Dict[str, Any]]] = [],
    ) -> DataFrame:
        """
        Process thematic labels for texts.

        Args:
            main_theme: The main theme to analyze.
            labels: Labels for labelling the chunks.
            texts: List of chunks to label.
            max_workers: Maximum number of concurrent workers.

        Returns:
            DataFrame with schema:
            - index: sentence_id
            - columns:
                - motivation
                - label
        """
        system_prompt = (
            get_risk_system_prompt(main_theme, labels)
            if self.label_prompt is None
            else self.label_prompt
        )

        prompts = get_prompts_for_labeler(texts, textsconfig)

        responses = self._run_labeling_prompts(
            prompts, system_prompt, max_workers=max_workers
        )
        responses = [parse_labeling_response(response) for response in responses]

        return self._deserialize_label_responses(responses)

    def post_process_dataframe(self, df: DataFrame, extra_fields: dict, extra_columns: List[str]) -> DataFrame:
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
        df["motivation"] = df.apply(replace_company_placeholders, axis=1)

        # Add formatted columns
        df["Time Period"] = df["timestamp_utc"].dt.strftime("%b %Y")
        df["Date"] = df["timestamp_utc"].dt.strftime("%Y-%m-%d")

        df["Document ID"] = df["document_id"] if "document_id" in df.columns else df["rp_document_id"]
        
        columns_map = {
                "entity_name": "Company",
                "entity_sector": "Sector",
                "entity_industry": "Industry",
                "entity_country": "Country",
                "entity_ticker": "Ticker",
                "headline": "Headline",
                "text": "Quote",
                "motivation": "Motivation",
                "label": "Sub-Scenario"
            }

        if extra_fields:
            columns_map.update(extra_fields)
            if "quotes" in extra_fields.keys():
                if "quotes" in df.columns:
                    df["quotes"] = df.apply(replace_company_placeholders, axis=1, col_name = 'quotes')
                else:
                    print("quotes column not in df")

        df = df.rename(
            columns=columns_map
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
            "Sub-Scenario",
        ]
        
        if extra_columns:
            export_columns += extra_columns

        return df[export_columns]


def replace_company_placeholders(row: Series, col_name: str = 'motivation') -> str:

    """
    Replace company placeholders in text.

    Args:
        row: Row of the DataFrame. Expected columns:
            - motivation: str
            - entity_name: str
            - other_entities_map: List[Tuple[int, str]]
    Returns:
        Text with placeholders replaced.
    """
    text = row[col_name]
    if isinstance(text, str):
        text = text.replace(get_target_entity_placeholder(), row["entity_name"])
        if row.get("other_entities_map"):
            for entity_id, entity_name in row["other_entities_map"]:
                text = text.replace(
                    f"{get_other_entity_placeholder()}_{entity_id}", entity_name)
    
    elif isinstance(text, list):
        text = [t.replace(get_target_entity_placeholder(), row["entity_name"]) for t in text]
        if row.get("other_entities_map"):
            for entity_id, entity_name in row["other_entities_map"]:
                text = [t.replace(f"{get_other_entity_placeholder()}_{entity_id}", entity_name) for t in text]

    return text

# Function to map risk_factor to risk_category
def map_risk_category(risk_factor, mapping):
    return mapping.get(risk_factor, 'Not Applicable')
