from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.entities import ALL_ENTITIES, EntityDict
from config.topics import TOPICS, TopicDict
from src.utils import console, sanitize_filename, setup_logger

logger = setup_logger(__name__)

RAW_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "raw"
SCORES_OUTPUT = Path(__file__).resolve().parent.parent / "output" / "scores.csv"


def _load_raw_result(entity: EntityDict, topic: TopicDict) -> dict[str, Any] | None:
    entity_slug = sanitize_filename(str(entity["name"]))
    topic_slug = sanitize_filename(str(topic["topic_name"]))
    path = RAW_OUTPUT_DIR / f"{entity_slug}_{topic_slug}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _count_relevant_results(raw: dict[str, Any]) -> int:
    """Count results that actually mention the entity name in headline or content.

    The API always returns up to `limit` documents via semantic search, so raw
    n_results is usually saturated at the limit. Counting entity mentions in the
    returned text gives a true relevance signal that differentiates entities.
    """
    entity_name = raw["entity_name"].lower()
    relevant = 0
    for doc in raw.get("results", []):
        text = (doc.get("headline", "") + " " + doc.get("content", "")).lower()
        if entity_name in text:
            relevant += 1
    return relevant


def _get_applicable_topics(layer: str) -> list[TopicDict]:
    return [t for t in TOPICS if layer in t["applies_to"]]


def compute_scores(
    entities: list[EntityDict] | None = None,
) -> pd.DataFrame:
    """Aggregate raw search results into entity-level scores.

    Returns a DataFrame with per-entity scores and per-topic mention counts.
    """
    target_entities = entities or ALL_ENTITIES
    rows: list[dict[str, Any]] = []

    for entity in target_entities:
        layer = str(entity["layer"])
        applicable_topics = _get_applicable_topics(layer)

        positive_count = 0
        negative_count = 0
        topic_counts: dict[str, int] = {}
        top_positive_topic = ""
        top_positive_count = 0
        top_negative_topic = ""
        top_negative_count = 0

        for topic in applicable_topics:
            result = _load_raw_result(entity, topic)
            count = _count_relevant_results(result) if result else 0
            topic_name = str(topic["topic_name"])
            polarity = str(topic["polarity"])

            topic_counts[topic_name] = count

            if polarity == "positive":
                positive_count += count
                if count > top_positive_count:
                    top_positive_count = count
                    top_positive_topic = topic_name
            else:
                negative_count += count
                if count > top_negative_count:
                    top_negative_count = count
                    top_negative_topic = topic_name

        terms_power_score = (
            positive_count / (positive_count + negative_count + 1) * 100
        )
        stress_score = 100 - terms_power_score

        row: dict[str, Any] = {
            "entity_name": entity["name"],
            "layer": layer,
            "ticker": entity["ticker"],
            "positive_count": positive_count,
            "negative_count": negative_count,
            "terms_power_score": round(terms_power_score, 2),
            "stress_score": round(stress_score, 2),
            "top_positive_topic": top_positive_topic,
            "top_negative_topic": top_negative_topic,
        }
        row.update(topic_counts)
        rows.append(row)

    df = pd.DataFrame(rows)

    SCORES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORES_OUTPUT, index=False)
    logger.info("Scores written to %s", SCORES_OUTPUT)

    console.rule("[bold cyan]Scoring Summary")
    for layer_name in ("lender", "borrower", "bank"):
        layer_df = df[df["layer"] == layer_name].copy()
        if layer_df.empty:
            continue

        if layer_name == "lender":
            layer_df = layer_df.sort_values("terms_power_score", ascending=False)
            console.print(f"\n[bold]Lender Ranking[/bold] (by Terms Power Score):")
        elif layer_name == "borrower":
            layer_df = layer_df.sort_values("stress_score", ascending=False)
            console.print(f"\n[bold]Borrower Distress[/bold] (by Stress Score):")
        else:
            _g = layer_df["bank_market_share_gain"].fillna(0) if "bank_market_share_gain" in layer_df.columns else 0
            _p = layer_df["bank_credit_pullback"].fillna(0) if "bank_credit_pullback" in layer_df.columns else 0
            layer_df["net_position"] = _g - _p
            layer_df = layer_df.sort_values("net_position", ascending=False)
            console.print(f"\n[bold]Bank Contagion[/bold] (by Net Position):")

        for _, row_data in layer_df.iterrows():
            score_col = (
                "terms_power_score"
                if layer_name == "lender"
                else "stress_score"
                if layer_name == "borrower"
                else "net_position"
            )
            score_val = row_data.get(score_col, "N/A")
            console.print(
                f"  {row_data['entity_name']:<30} {score_col}={score_val}"
            )

    return df


if __name__ == "__main__":
    compute_scores()
