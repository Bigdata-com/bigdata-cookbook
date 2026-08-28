"""Simple OpenAI labeling for report generator (no bigdata-research-tools)."""

from __future__ import annotations

import logging
import os

import pandas as pd
from openai import OpenAI

from .openai_utils import (
    DEFAULT_LLM_MODEL,
    completion_token_params_for_model,
    sampling_params_for_model,
)

logger = logging.getLogger(__name__)


class SimpleLabeler:
    """Basic labeler using OpenAI chat completions."""

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self._logged_errors = 0

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
                    # Luna may consume completion budget on reasoning; keep headroom.
                    **completion_token_params_for_model(self.model, 256),
                    **sampling_params_for_model(
                        self.model,
                        temperature=self.temperature,
                    ),
                )
                content = response.choices[0].message.content or ""
                label = content.strip().lower()
                if label not in labels and label not in ["unassigned", "unclear"]:
                    # Allow exact label match when model returns a multi-word taxonomy key
                    matched = next(
                        (candidate for candidate in labels if candidate.lower() == label),
                        None,
                    )
                    label = matched if matched is not None else "unassigned"
            except Exception as exc:
                if self._logged_errors < 5:
                    logger.error("OpenAI labeling failed for model=%s: %s", self.model, exc)
                    self._logged_errors += 1
                label = "unassigned"
            results.append({"label": label})
        return pd.DataFrame(results)
