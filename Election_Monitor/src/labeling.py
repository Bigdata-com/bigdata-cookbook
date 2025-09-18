import asyncio
import os
from typing import Any, Coroutine, Optional, Union, List, Dict, Any
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from bigdata_research_tools.prompts.labeler import(
    get_other_entity_placeholder,
    get_target_entity_placeholder,
)

from bigdata_research_tools.labeler.labeler import (
    Labeler,
    get_prompts_for_labeler,
    parse_labeling_response,
)


# Import necessary libraries for the complete implementation
from httpx import ReadTimeout
from typing import Any, Generator, Iterable, Optional, Union
import openai
from tqdm.asyncio import tqdm as async_tqdm
import hashlib
import pickle
from pathlib import Path

# ==============================================================================
# ORIGINAL IMPLEMENTATION - EXACTLY AS IN THE WORKFLOW
# ==============================================================================

def replace_company_placeholders(row: pd.Series) -> str:
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

class ElectionLabeler(Labeler):
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
        texts: List[str],
        max_workers: int = 50,
    ) -> pd.DataFrame:
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
        system_prompt = self.label_prompt
        prompts = get_prompts_for_labeler(texts)

        responses = self._run_labeling_prompts(
            prompts, system_prompt, max_workers=max_workers
        )
        responses = [parse_labeling_response(response) for response in responses]
        return self._deserialize_label_responses(responses)


DEFAULT_TRUMP_REELECTION_PROMPT = (
    f"""You are a financial analyst identifying companies expressing a view on the possible election of Donald Trump as President of the United States.
Your task is to determine if, based on the provided sentence, the Target Company is expressing a positive or negative view on Trump's election.

Instructions:
1. Label each sentence for the Target Company: 'P' if the company mention that Donald Trump's election will positively affect their business, 'N' if the company will be negatively impacted by Trump's re-election, 'U' if unrelated.
2. Evaluate each sentence individually, focusing on whether the company is mentioning Donald Trump.
3. Use only the information in the sentence; do not infer from outside knowledge.
4. Ensure the text expresses a clear view on the elections and the possible consequences. 
5. Mentioning that the elections are upcoming, or asking a question about the elections does not imply expressing a view.
6. A company that mentions how another company's business may be affected by the elections is not directly related to the elections unless its business is also affected.
7. You will be given a sentence ID, a company name, and the sentence text, for which you must assign the label. Your output should be a JSON object with a very brief motivation for the choice of the label and the label. The motivation must be one short sentence that starts with the company name and should explain why the label is 'Y', 'N', or 'U' for that company, without summarizing the text. Format the JSON like: {{"<ID>": {{"motivation": <motivation>, "label": <label>}}, ...}}.

Example sentences and evaluations:

Example 1:
Sentence: "Target Company executives mentioned that a Trump administration's tax policies would significantly boost our profit margins and allow for expanded operations."
Motivation: Target Company expects their business to benefit from Trump election.
Label: 'P'

Example 2:
Sentence: "Target Company's CEO warned that Trump's trade policies could disrupt our supply chain and increase costs substantially for our manufacturing operations."
Motivation: Target Company expects their business  to be negatively impacted by Trump's election.
Label: 'N'

Example 3:
Sentence: "Target Company analysts noted that the upcoming election between Trump and Biden could create market volatility."
Motivation: Target Company is providing general market commentary without mentioning specific business impact.
Label: 'U'

Example 4:
Sentence: "Target Company reported strong quarterly earnings despite concerns about the presidential election outcome."
Motivation: Target Company's earnings report does not express how Trump's election would affect their business.
Label: 'U'

Example 5:
Sentence: "On the Internet, we see news like the if Trump wins, the trade relations will change and the security situation will change. An of course, the relations with China and Taiwan comes into play. So the security is a big factor."
Motivation: Target_Company's discusses how Trump's election entails a security risk.
Label: 'N'

Example 6:
Sentence: "The former President Trump may win the upcoming presidential election. If that happens, what would be the strategy of Target Company or how will Target Company be prepared for the second Trump administration?"
Motivation: The sentence does not provide information about Target Company's view on the elections.
Label: 'U'
"""
)