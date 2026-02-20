import json
from logging import Logger, getLogger
from typing import Any, Literal

from pandas import DataFrame, Series
from pydantic import BaseModel, field_validator

from bigdata_research_tools.labeler.labeler import Labeler as BaseLabeler
from bigdata_research_tools.llm.base import LLMConfig
from bigdata_research_tools.prompts.labeler import (
    get_other_entity_placeholder,
    get_screener_system_prompt,
    get_target_entity_placeholder,
)

from src.prompts.labeler import get_theme_validation_prompt

logger: Logger = getLogger(__name__)

ImpactType = Literal["Positive", "Negative", "Neutral", "Unclear"]


class ThemeValidationResult(BaseModel):
    """Validated output for a single theme validation (one row)."""

    is_theme_related: bool = False
    impact: ImpactType = "Unclear"
    motivation: str = ""

    @field_validator("is_theme_related", mode="before")
    @classmethod
    def coerce_is_theme_related(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() == "true"
        return False

    @field_validator("impact", mode="before")
    @classmethod
    def normalize_impact(cls, v: Any) -> ImpactType:
        if v is None or v == "":
            return "Unclear"
        s = str(v).strip()
        lower = s.lower()
        if lower in ("positive",):
            return "Positive"
        if lower in ("negative",):
            return "Negative"
        if lower in ("neutral", "clear"):
            return "Neutral"
        if s in ("Positive", "Negative", "Neutral", "Unclear"):
            return s  # type: ignore[return-value]
        return "Unclear"


class Labeler(BaseLabeler):
    """Labeler per validazione tema e impact (theme validation + merge)."""

    def __init__(
        self,
        llm_model_config: str | LLMConfig | dict = "openai::gpt-4o-mini",
        label_prompt: str | None = None,
        unknown_label: str = "unclear",
    ):
        """
        Args:
            llm_model: Name of the LLM model to use. Expected format:
                <provider>::<model>, e.g. "openai::gpt-4o-mini"
            label_prompt: Prompt provided by user to label the search result chunks.
                If not provided, then our default labelling prompt is used.
            unknown_label: Label for unclear classifications.
        """
        super().__init__(llm_model_config, unknown_label)
        self.label_prompt = label_prompt

    def get_labels(
        self,
        main_theme: str,
        labels: list[str],
        texts: list[str],
        timeout: int | None = 20,
        max_workers: int = 55,
    ) -> DataFrame:
        """
        Process thematic labels for texts.

        Args:
            main_theme: The main theme to analyze.
            labels: Labels for labelling the chunks.
            texts: List of chunks to label.
            timeout: Timeout for each LLM request.
            max_workers: Maximum number of concurrent workers.

        Returns:
            DataFrame with schema:
            - index: sentence_id
            - columns:
                - motivation
                - label
        """
        default_prompt = get_screener_system_prompt(
            main_theme, labels, unknown_label=self.unknown_label
        )
        system_prompt = self.label_prompt or default_prompt
        if self.label_prompt:
            try:
                system_prompt = self.label_prompt.format(
                    main_theme=main_theme,
                    label_summaries=labels,
                    unknown_label=self.unknown_label,
                )
            except KeyError:
                pass  # use as-is if template has other placeholders
        prompts = self.get_prompts_for_labeler(texts)

        responses = self._run_labeling_prompts(
            prompts,
            system_prompt,
            max_workers=max_workers,
            timeout=timeout,
            processing_callbacks=[
                self.parse_labeling_response,
                self._deserialize_label_response,
            ],
        )

        return self._convert_to_label_df(responses)

    def get_validation_labels(
        self,
        main_theme: str,
        df_masked: DataFrame,
        timeout: int | None = 20,
        max_workers: int = 55,
    ) -> DataFrame:
        """
        Process theme validation for texts from a DataFrame.

        This method performs a two-step analysis:
        1. Verify if the text is related to the main_theme based on the provided labels/themes
        2. If related, assess the impact on Target Company (Positive/Negative/Clear/Unclear)

        Args:
            main_theme: The main theme to validate against.
            df_masked: DataFrame with columns:
                - masked_text: str (text with company names replaced by placeholders)
                - label: list[str] (labels associated with the text)
                - theme: list[str] (themes associated with the text)
            timeout: Timeout for each LLM request.
            max_workers: Maximum number of concurrent workers.

        Returns:
            DataFrame with schema:
            - index: sentence_id (matching df_masked index)
            - columns:
                - is_theme_related: bool (True if text is related to main_theme)
                - impact: str ("Positive", "Negative", "Clear", "Unclear")
                - motivation: str (explanation of the assessment)
        """
        # Validate required columns
        required_columns = ["masked_text", "label", "theme"]
        missing_columns = [col for col in required_columns if col not in df_masked.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Build prompts with context from each row
        prompts = self._build_validation_prompts(df_masked)

        # Get the system prompt
        system_prompt = get_theme_validation_prompt(main_theme)

        # Run labeling with the validation-specific deserializer
        responses = self._run_labeling_prompts(
            prompts,
            system_prompt,
            max_workers=max_workers,
            timeout=timeout,
            processing_callbacks=[
                self.parse_labeling_response,
                self._deserialize_validation_response,
            ],
        )

        return self._convert_to_validation_df(responses)

    def _build_validation_prompts(self, df_masked: DataFrame) -> list[str]:
        """
        Build prompts including label and theme context for each row.

        Args:
            df_masked: DataFrame with masked_text, label, and theme columns.

        Returns:
            List of JSON-formatted prompts for the LLM.
        """
        prompts = []
        for idx, row in df_masked.iterrows():
            # Ensure labels and themes are lists
            labels = row["label"] if isinstance(row["label"], list) else [row["label"]]
            themes = row["theme"] if isinstance(row["theme"], list) else [row["theme"]]

            prompt = {
                "sentence_id": idx,
                "labels": labels,
                "themes": themes,
                "text": row["masked_text"],
            }
            prompts.append(json.dumps(prompt))
        return prompts

    def _deserialize_validation_response(self, response: str) -> str:
        """
        Deserialize theme validation response using Pydantic validator.

        Args:
            response: JSON string from the LLM containing validation results.

        Returns:
            JSON string with standardized validation response format.
        """
        response_data = json.loads(response)
        response_mapping: dict[int, dict[str, Any]] = {}

        if not response_data or not isinstance(response_data, dict):
            raise ValueError("Response is empty or not a dictionary")

        for k, v in response_data.items():
            try:
                parsed = ThemeValidationResult.model_validate(
                    v if isinstance(v, dict) else {}
                )
                response_mapping[int(k)] = parsed.model_dump()
            except Exception:
                response_mapping[int(k)] = ThemeValidationResult().model_dump()
        return json.dumps(response_mapping)

    def _convert_to_validation_df(self, response_mapping: list[str]) -> DataFrame:
        """
        Convert validation response to a DataFrame.

        Args:
            response_mapping: List of JSON strings with validation results.

        Returns:
            DataFrame with is_theme_related, impact, and motivation columns.
        """
        responses_json = {}
        for response in response_mapping:
            responses_json.update(json.loads(response))
        df_labels = DataFrame.from_dict(responses_json, orient="index")
        df_labels.index = df_labels.index.astype(int)
        df_labels.sort_index(inplace=True)
        return df_labels

    @staticmethod
    def merge_validation_labels(df: DataFrame, labels_df: DataFrame) -> DataFrame:
        """
        Join labels_df onto the original DataFrame (same index).

        Adds columns is_theme_related, impact, motivation to df.

        Args:
            df: DataFrame passed to get_validation_labels (e.g. df_masked or df_subset).
            labels_df: Output of get_validation_labels.

        Returns:
            DataFrame with validation columns added (join on index).
        """
        return df.join(labels_df, how="left")


def merge_validation_labels(df: DataFrame, labels_df: DataFrame) -> DataFrame:
    """
    Join labels_df onto the original DataFrame (same index).
    Wrapper for Labeler.merge_validation_labels.
    """
    return Labeler.merge_validation_labels(df, labels_df)

    def post_process_dataframe(
        self,
        df: DataFrame,
        extra_fields: dict | None = None,
        extra_columns: list[str] | None = None,
    ) -> DataFrame:
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

        columns_map = {
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

        optional_fields = ["topics", "source_name", "source_rank", "url"]
        for field in optional_fields:
            if field in df.columns:
                columns_map[field] = field.replace("_", " ").title()

        if extra_fields:
            columns_map.update(extra_fields)
            if "quotes" in extra_fields.keys():
                if "quotes" in df.columns:
                    df["quotes"] = df.apply(
                        replace_company_placeholders, axis=1, col_name="quotes"
                    )
                else:
                    logger.warning("quotes column not in df")

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
        ]

        if extra_columns:
            export_columns += extra_columns

        for field in optional_fields:
            if field in df.columns:
                export_columns += [field.replace("_", " ").title()]

        df = df.rename(columns=columns_map)

        sort_columns = [
            "Date",
            "Time Period",
            "Company",
            "Document ID",
            "Headline",
            "Quote",
        ]
        df = df[export_columns].sort_values(sort_columns).reset_index(drop=True)

        return df


def replace_company_placeholders(row: Series) -> str:
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
    text = row["motivation"]
    text = text.replace(get_target_entity_placeholder(), row["entity_name"])
    if row.get("other_entities_map"):
        for entity_id, entity_name in row["other_entities_map"]:
            text = text.replace(
                f"{get_other_entity_placeholder()}_{entity_id}", entity_name
            )
    return text
