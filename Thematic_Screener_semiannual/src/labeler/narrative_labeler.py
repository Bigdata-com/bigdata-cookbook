"""
Module for managing labeling operations.

Copyright (C) 2024, RavenPack | Bigdata.com. All rights reserved.
"""

from logging import Logger, getLogger
from typing import List, Optional
from risk_labeler import replace_company_placeholders
from pandas import DataFrame

from bigdata_research_tools.labeler.labeler import (
    Labeler,
    get_prompts_for_labeler,
    parse_labeling_response,
)
from bigdata_research_tools.prompts.labeler import get_narrative_system_prompt

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
        theme_labels: List[str],
        texts: List[str],
        max_workers: int = 50,
    ) -> DataFrame:
        """
        Process thematic labels for texts.

        Args:
            theme_labels: The main theme to analyze.
            texts: List of texts to label.
            max_workers: Maximum number of concurrent workers.

        Returns:
            DataFrame with schema:
            - index: sentence_id
            - columns:
                - motivation
                - label
        """
        system_prompt = (
            get_narrative_system_prompt(theme_labels)
            if self.label_prompt is None
            else self.label_prompt
        )
        prompts = get_prompts_for_labeler(texts)

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



