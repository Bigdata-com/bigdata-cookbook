"""Labeling wrappers for baseline chunk vs extracted sentence comparison."""

from __future__ import annotations

from typing import Any

from src import screener


def _resolve_company_names(
    records: list[dict[str, Any]],
    universe_df,
) -> list[dict[str, Any]]:
    id_column = screener.UNIVERSE_ID_COLUMN
    name_column = screener.UNIVERSE_NAME_COLUMN
    lookup = {
        str(row[id_column]): str(row[name_column])
        for _, row in universe_df.iterrows()
        if str(row[id_column]).strip()
    }
    enriched: list[dict[str, Any]] = []
    for record in records:
        entity_id = record.get("entity_id")
        company_name = lookup.get(str(entity_id), "") if entity_id else ""
        enriched.append({**record, "company_name": company_name})
    return enriched


def _labeling_sentences(
    records: list[dict[str, Any]],
    *,
    text_field: str,
) -> list[dict[str, Any]]:
    sentences: list[dict[str, Any]] = []
    for record in records:
        sentences.append(
            {
                "sentence_id": record["record_id"],
                "text": record[text_field],
                "company_name": record.get("company_name") or "",
            }
        )
    return sentences


def run_labeling_passes(
    records: list[dict[str, Any]],
    *,
    universe_df,
    main_theme: str,
    analyst_focus: str,
    labels: list[str],
    model: str,
    requests_per_minute: int,
    max_concurrent_requests: int,
) -> list[dict[str, Any]]:
    """Run baseline (chunk) and experiment (sentence) labeling passes."""
    enriched = _resolve_company_names(records, universe_df)

    baseline_sentences = _labeling_sentences(enriched, text_field="chunk_text")
    experiment_sentences = _labeling_sentences(enriched, text_field="extracted_sentence")

    baseline_labels = screener.label_sentences(
        baseline_sentences,
        main_theme=main_theme,
        labels=labels,
        analyst_focus=analyst_focus,
        model=model,
        requests_per_minute=requests_per_minute,
        max_concurrent_requests=max_concurrent_requests,
    )
    experiment_labels = screener.label_sentences(
        experiment_sentences,
        main_theme=main_theme,
        labels=labels,
        analyst_focus=analyst_focus,
        model=model,
        requests_per_minute=requests_per_minute,
        max_concurrent_requests=max_concurrent_requests,
    )

    merged: list[dict[str, Any]] = []
    for record in enriched:
        record_id = str(record["record_id"])
        baseline = baseline_labels.get(record_id, {})
        experiment = experiment_labels.get(record_id, {})
        merged.append(
            {
                **record,
                "baseline_label": baseline.get("label", ""),
                "baseline_materiality": baseline.get("materiality", ""),
                "baseline_motivation": baseline.get("motivation", ""),
                "experiment_label": experiment.get("label", ""),
                "experiment_materiality": experiment.get("materiality", ""),
                "experiment_motivation": experiment.get("motivation", ""),
            }
        )
    return merged
