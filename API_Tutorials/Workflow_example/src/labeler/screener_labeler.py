"""Minimal OpenAI-based labeler (SDK Labeler / ScreenerLabeler removed).

MIGRATION NOTE:
The original SDK ``Labeler`` (and ``ScreenerLabeler``) classes are gone with
``bigdata-research-tools``. ``Labeler`` below is a small, dependency-light
replacement that calls OpenAI directly per row (parallelized with a thread
pool) and returns the two columns the notebooks need: ``is_theme_related``
and ``impact`` (one of "Positive", "Negative", "Neutral", "Unclear").
Pattern follows ``Report_Generator_AI_Threats/src/labeling.py``.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI

VALID_IMPACTS = ("Positive", "Negative", "Neutral", "Unclear")
DEFAULT_MODEL = "gpt-5.6-luna"


def _sampling_params_for_model(model: str, **params: object) -> dict[str, object]:
    """Luna models only support default sampling — omit temperature/top_p/etc."""
    if "luna" in model.lower():
        return {}
    return {key: value for key, value in params.items() if value is not None}


class Labeler:
    """Validates whether text chunks relate to a theme and their directional impact."""

    def __init__(self, llm_model_config: str = f"openai::{DEFAULT_MODEL}") -> None:
        provider, _, model = llm_model_config.partition("::")
        if provider != "openai":
            raise NotImplementedError(
                f"Unsupported llm_model_config provider: {provider!r} (only 'openai' is supported)"
            )
        self.model = model or DEFAULT_MODEL
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _label_text(self, main_theme: str, text: str) -> dict:
        system_prompt = (
            "You are a financial research analyst validating retrieved text against a theme.\n"
            f"Theme: '{main_theme}'\n"
            "Decide:\n"
            "1. is_theme_related: true if the text is genuinely discussing this theme, else false.\n"
            "2. impact: if is_theme_related is true, classify the directional impact on the "
            "company mentioned as one of Positive, Negative, Neutral, Unclear. "
            "If is_theme_related is false, use 'Unclear'.\n"
            'Respond ONLY with JSON: {"is_theme_related": bool, "impact": str}'
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text or ""},
                ],
                response_format={"type": "json_object"},
                **_sampling_params_for_model(self.model, temperature=0.0),
            )
            data = json.loads(response.choices[0].message.content)
            is_theme_related = bool(data.get("is_theme_related", False))
            impact = str(data.get("impact", "Unclear")).strip().capitalize()
            if impact not in VALID_IMPACTS:
                impact = "Unclear"
        except Exception:
            is_theme_related, impact = False, "Unclear"
        return {"is_theme_related": is_theme_related, "impact": impact}

    def get_validation_labels(
        self,
        main_theme: str,
        df_masked: pd.DataFrame,
        text_column: str = "masked_text",
        timeout: int = 20,
        max_workers: int = 10,
    ) -> pd.DataFrame:
        """Run LLM validation for every row of ``df_masked``, preserving row order."""
        texts = df_masked[text_column].fillna("").tolist()
        results: list[dict] = [None] * len(texts)  # type: ignore[list-item]

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(self._label_text, main_theme, text): i for i, text in enumerate(texts)
            }
            for future in as_completed(future_to_idx, timeout=timeout * max(1, len(texts))):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result(timeout=timeout)
                except Exception:
                    results[idx] = {"is_theme_related": False, "impact": "Unclear"}

        return pd.DataFrame(results, index=df_masked.index)


def merge_validation_labels(df_masked: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate ``labels_df`` (aligned by index) onto ``df_masked``."""
    return pd.concat([df_masked, labels_df], axis=1)
