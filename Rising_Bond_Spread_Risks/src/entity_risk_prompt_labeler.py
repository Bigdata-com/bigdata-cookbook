"""Simplified entity risk prompt labeler (no SDK)."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from .openai_utils import DEFAULT_LLM_MODEL
from .simple_labeler import SimpleLabeler


def label_search_results(
    df_sentences: pd.DataFrame,
    terminal_labels: list[str],
    main_theme: str = "",
    **kwargs,
) -> pd.DataFrame:
    """
    Label the search results with theme labels using SimpleLabeler.

    Args:
        df_sentences: The DataFrame containing the search results.
        terminal_labels: The terminal labels for the risk categories.
        main_theme: The main theme for labeling.

    Returns:
        DataFrame: The labeled DataFrame.
    """
    labeler = SimpleLabeler(
        model=kwargs.get("model", DEFAULT_LLM_MODEL),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    texts = df_sentences.get("text", df_sentences.get("chunk_text", [])).tolist()

    result = labeler.get_labels(
        main_theme=main_theme or "risk",
        labels=terminal_labels,
        texts=texts,
    )

    df_sentences = df_sentences.assign(
        label=result["label"].values, motivation=""
    )

    # Derive a categorical "sentiment" bucket (negative/neutral/positive) from
    # the raw numeric Bigdata sentiment score, if present. The old SDK's
    # RiskLabeler used to emit this; SimpleLabeler only emits a topic label,
    # so we bucket it here to keep downstream code (which filters on
    # `Sentiment == "negative"`) working the same way.
    if "bigdata_sentiment" in df_sentences.columns:
        def _bucket_sentiment(value: Any) -> str:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return "neutral"
            if numeric < -0.05:
                return "negative"
            if numeric > 0.05:
                return "positive"
            return "neutral"

        df_sentences = df_sentences.assign(
            sentiment=df_sentences["bigdata_sentiment"].apply(_bucket_sentiment)
        )

    return df_sentences


def generate_risk_tree(*args, **kwargs) -> dict[str, Any]:
    """Placeholder for risk tree generation (requires custom logic)."""
    raise NotImplementedError(
        "Risk tree generation removed with SDK. "
        "Implement custom logic or use predefined taxonomy."
    )


def get_scored_df(
    df: pd.DataFrame, index_columns: list[str], pivot_column: str
) -> pd.DataFrame:
    """Lightweight replacement for the removed bigdata-research-tools ``get_scored_df``.

    Pivots labeled rows into an entity-level composite score table: one row per
    unique combination of ``index_columns`` (e.g. ``["Entity", "Country"]``),
    one column per distinct value of ``pivot_column`` (e.g. sub-scenario
    labels) counting mentions, plus a trailing ``"Composite Score"`` column
    summing across all sub-scenario columns. Rows labeled "unassigned" /
    "unclear" / empty are excluded before pivoting.

    This matches the column shape expected by ``src/visualization_tool.py``
    (``theme_columns = df.iloc[:, 3:-1]``): ``index_columns`` + a
    "Total Mentions" column + sub-scenario columns + "Composite Score".
    """
    valid = df.loc[~df[pivot_column].isin(["unassigned", "unclear", "", None])].copy()
    if valid.empty:
        return pd.DataFrame(columns=[*index_columns, "Total Mentions", "Composite Score"])

    pivot = (
        valid.groupby(index_columns + [pivot_column])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    theme_columns = [c for c in pivot.columns if c not in index_columns]
    pivot.insert(len(index_columns), "Total Mentions", pivot[theme_columns].sum(axis=1))
    pivot["Composite Score"] = pivot[theme_columns].sum(axis=1)
    pivot = pivot.sort_values("Composite Score", ascending=False).reset_index(drop=True)
    return pivot
