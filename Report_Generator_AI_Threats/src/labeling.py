"""Simple OpenAI labeling for report generator (no bigdata-research-tools)."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from openai import OpenAI

from src.openai_utils import completion_token_params_for_model, sampling_params_for_model


class SimpleLabeler:
    """Basic labeler using OpenAI structured outputs."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def get_labels(
        self,
        main_theme: str,
        labels: list[str],
        texts: list[str],
    ) -> pd.DataFrame:
        """Label texts against main theme.

        Args:
            main_theme: The theme to check relevance against
            labels: List of label options (e.g., ['risk'])
            texts: List of text chunks to label

        Returns:
            DataFrame with 'label' column
        """
        results = []
        for text in texts:
            prompt = (
                f"Does the following text discuss '{main_theme}'?\n\n"
                f"Text: {text}\n\n"
                f"If yes, respond with one of: {', '.join(labels)}.\n"
                f"If no or unclear, respond with 'unassigned'."
            )
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    **completion_token_params_for_model(self.model, 50),
                    **sampling_params_for_model(self.model, temperature=0.0),
                )
                label = response.choices[0].message.content.strip().lower()
                if label not in labels and label not in ["unassigned", "unclear"]:
                    label = "unassigned"
            except Exception:
                label = "unassigned"
            results.append({"label": label})
        return pd.DataFrame(results)
