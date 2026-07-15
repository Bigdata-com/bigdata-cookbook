"""Assess ranking signals and labeling against the provenance-locked golden set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _precision_at_k(y_true: pd.Series, scores: pd.Series, k: int) -> float:
    if k <= 0 or y_true.empty:
        return 0.0
    top = scores.sort_values(ascending=False).head(k).index
    hits = int(y_true.loc[top].sum())
    return round(hits / k, 4)


def _recall_at_k(y_true: pd.Series, scores: pd.Series, k: int) -> float:
    positives = int(y_true.sum())
    if positives == 0 or k <= 0:
        return 0.0
    top = scores.sort_values(ascending=False).head(k).index
    hits = int(y_true.loc[top].sum())
    return round(hits / positives, 4)


def _average_precision(y_true: pd.Series, scores: pd.Series) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    ordered = scores.sort_values(ascending=False)
    hits = 0
    total = 0.0
    for rank, idx in enumerate(ordered.index, start=1):
        if bool(y_true.loc[idx]):
            hits += 1
            total += hits / rank
    return round(total / positives, 4)


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    true_pos = int(((y_true) & (y_pred)).sum())
    false_pos = int((~y_true & y_pred).sum())
    false_neg = int((y_true & ~y_pred).sum())
    true_neg = int((~y_true & ~y_pred).sum())
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (true_pos + true_neg) / len(y_true) if len(y_true) else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": true_pos,
        "false_positives": false_pos,
        "false_negatives": false_neg,
        "true_negatives": true_neg,
    }


def _collapse_records_to_chunks(records_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate entity-sentence records to one row per chunk."""
    grouped = records_df.groupby("chunk_index", as_index=False).agg(
        search_relevance=("search_relevance", "max"),
        embedding_relevance=("embedding_relevance", "max"),
        evidence_score=("evidence_score", "max"),
        leaf_label=("leaf_label", "first"),
        company_name=("company_name", "first"),
        plan_file=("plan_file", "first"),
    )
    if "baseline_label" in records_df.columns:
        baseline = (
            records_df.groupby("chunk_index")["baseline_label"]
            .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0])
            .rename("baseline_label")
        )
        grouped = grouped.merge(baseline, on="chunk_index", how="left")
    if "experiment_label" in records_df.columns:
        experiment = (
            records_df.groupby("chunk_index")["experiment_label"]
            .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0])
            .rename("experiment_label")
        )
        grouped = grouped.merge(experiment, on="chunk_index", how="left")
    for column in ("prov_baseline_label", "prov_experiment_label"):
        if column in records_df.columns:
            collapsed = (
                records_df.groupby("chunk_index")[column]
                .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0])
                .rename(column)
            )
            grouped = grouped.merge(collapsed, on="chunk_index", how="left")
    grouped = _attach_trigger_sentence_scores(records_df, grouped)
    return grouped


def _record_is_relevant(row: pd.Series, label_column: str) -> bool:
    return _predict_relevant(str(row.get(label_column) or ""), str(row.get("leaf_label") or ""))


def _attach_trigger_sentence_scores(
    records_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach embed/search scores for the sentence that triggered relevance."""
    sentence_label_column = (
        "prov_experiment_label"
        if "prov_experiment_label" in records_df.columns
        else "experiment_label"
        if "experiment_label" in records_df.columns
        else None
    )
    if sentence_label_column is None:
        return chunk_df

    relevant_records = records_df[
        records_df.apply(lambda row: _record_is_relevant(row, sentence_label_column), axis=1)
    ]
    if relevant_records.empty:
        enriched = chunk_df.copy()
        enriched["sentence_embedding_relevance"] = float("nan")
        enriched["sentence_search_relevance"] = float("nan")
        return enriched

    sentence_embed = relevant_records.groupby("chunk_index")["embedding_relevance"].max()
    sentence_search = relevant_records.groupby("chunk_index")["search_relevance"].max()
    enriched = chunk_df.copy()
    enriched["sentence_embedding_relevance"] = enriched["chunk_index"].map(sentence_embed)
    enriched["sentence_search_relevance"] = enriched["chunk_index"].map(sentence_search)
    return enriched


def _sentence_label_column(chunk_df: pd.DataFrame) -> str | None:
    if "prov_experiment_label" in chunk_df.columns:
        return "prov_experiment_label"
    if "experiment_label" in chunk_df.columns:
        return "experiment_label"
    return None


def apply_embed_threshold_gate(
    chunk_df: pd.DataFrame,
    *,
    embed_threshold: float,
    label_column: str | None = None,
) -> pd.Series:
    """Return chunk predictions: sentence relevant AND embed >= threshold."""
    label_col = label_column or _sentence_label_column(chunk_df)
    if label_col is None:
        raise ValueError("No sentence label column found for embed threshold gate.")
    sentence_relevant = chunk_df.apply(
        lambda row: _predict_relevant(row[label_col], row["leaf_label"]),
        axis=1,
    )
    embed_scores = chunk_df["sentence_embedding_relevance"].fillna(-1.0).astype(float)
    return sentence_relevant & (embed_scores >= embed_threshold)


DEFAULT_EMBED_THRESHOLDS: tuple[float, ...] = (
    0.0,
    0.35,
    0.40,
    0.45,
    0.50,
    0.52,
    0.55,
    0.58,
    0.60,
)


def assess_embed_threshold_grid(
    chunk_df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_EMBED_THRESHOLDS,
    label_column: str | None = None,
) -> list[dict[str, Any]]:
    """Sweep embed thresholds on top of provenance-locked sentence labels."""
    label_col = label_column or _sentence_label_column(chunk_df)
    if label_col is None or "sentence_embedding_relevance" not in chunk_df.columns:
        return []

    y_true = chunk_df["gold_relevant"].astype(bool)
    baseline_pred = chunk_df.apply(
        lambda row: _predict_relevant(row[label_col], row["leaf_label"]),
        axis=1,
    )
    baseline_stats = _binary_metrics(y_true, baseline_pred)
    rows: list[dict[str, Any]] = [
        {
            "embed_threshold": None,
            "method": "sentence_label_only",
            **baseline_stats,
        }
    ]
    for threshold in thresholds:
        y_pred = apply_embed_threshold_gate(
            chunk_df,
            embed_threshold=threshold,
            label_column=label_col,
        )
        stats = _binary_metrics(y_true, y_pred)
        rows.append({"embed_threshold": threshold, "method": "sentence_plus_embed", **stats})
    return rows


def _predict_relevant(label: str, leaf_label: str) -> bool:
    normalized = str(label or "").strip()
    pathway = str(leaf_label or "").strip()
    return bool(normalized and pathway and normalized == pathway and normalized != "unclear")


def assess_ranking(chunk_df: pd.DataFrame) -> dict[str, Any]:
    """Compute ranking metrics for each score column against gold relevance."""
    y_true = chunk_df["gold_relevant"].astype(bool)
    positive_count = int(y_true.sum())
    ks = sorted({positive_count, 5, 10, 20, min(50, len(chunk_df))})
    metrics: dict[str, Any] = {"gold_positives": positive_count}
    for column in ("search_relevance", "embedding_relevance", "evidence_score"):
        if column not in chunk_df.columns:
            continue
        scores = chunk_df[column].astype(float)
        column_metrics: dict[str, Any] = {
            "average_precision": _average_precision(y_true, scores),
        }
        for k in ks:
            column_metrics[f"precision_at_{k}"] = _precision_at_k(y_true, scores, k)
            column_metrics[f"recall_at_{k}"] = _recall_at_k(y_true, scores, k)
        metrics[column] = column_metrics
    return metrics


def assess_labeling(chunk_df: pd.DataFrame) -> dict[str, Any]:
    """Compare labeling passes against provenance-locked gold."""
    y_true = chunk_df["gold_relevant"].astype(bool)
    results: dict[str, Any] = {}
    candidates = (
        ("baseline_chunk", "baseline_label"),
        ("experiment_sentence", "experiment_label"),
        ("prov_baseline_chunk", "prov_baseline_label"),
        ("prov_experiment_sentence", "prov_experiment_label"),
    )
    for method, column in candidates:
        if column not in chunk_df.columns:
            continue
        y_pred = chunk_df.apply(
            lambda row: _predict_relevant(row[column], row["leaf_label"]),
            axis=1,
        )
        results[method] = _binary_metrics(y_true, y_pred)
    return results


def assess_gold_spot_check(gold_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize gold-relevant rows for manual validation notes."""
    relevant = gold_df[gold_df["gold_relevant"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for _, row in relevant.iterrows():
        rows.append(
            {
                "chunk_index": int(row["chunk_index"]),
                "company_name": row.get("company_name"),
                "leaf_label": row.get("leaf_label"),
                "gold_materiality": row.get("gold_materiality"),
                "gold_evidence_quality": row.get("gold_evidence_quality"),
                "search_relevance": round(float(row.get("search_relevance") or 0), 4),
                "motivation_preview": str(row.get("gold_motivation") or "")[:160],
            }
        )
    return rows


def run_assessment(
    *,
    golden_path: Path,
    records_path: Path,
    output_dir: Path,
    embed_threshold: float | None = None,
) -> dict[str, Any]:
    """Load inputs, compute metrics, and write assessment artifacts."""
    gold_df = pd.read_csv(golden_path)
    records_df = pd.read_csv(records_path)
    chunk_df = _collapse_records_to_chunks(records_df)
    merged = chunk_df.merge(
        gold_df[
            [
                "chunk_index",
                "gold_relevant",
                "gold_label",
                "gold_materiality",
                "gold_evidence_quality",
            ]
        ],
        on="chunk_index",
        how="inner",
    )
    if len(merged) != len(gold_df):
        raise ValueError(
            f"Expected {len(gold_df)} merged chunks, got {len(merged)}. "
            "Check chunk_index alignment between golden set and records."
        )

    embed_grid = assess_embed_threshold_grid(merged)
    embed_gated: dict[str, Any] | None = None
    if embed_threshold is not None:
        y_true = merged["gold_relevant"].astype(bool)
        y_pred = apply_embed_threshold_gate(merged, embed_threshold=embed_threshold)
        embed_gated = {
            "embed_threshold": embed_threshold,
            **_binary_metrics(y_true, y_pred),
        }

    assessment: dict[str, Any] = {
        "chunk_count": int(len(merged)),
        "gold_positives": int(merged["gold_relevant"].astype(bool).sum()),
        "gold_negatives": int((~merged["gold_relevant"].astype(bool)).sum()),
        "ranking": assess_ranking(merged),
        "labeling": assess_labeling(merged),
        "embed_threshold_grid": embed_grid,
        "embed_threshold_gate": embed_gated,
        "gold_spot_check": assess_gold_spot_check(gold_df),
        "plan_breakdown": merged.groupby("leaf_label")["gold_relevant"]
        .agg(relevant="sum", total="count")
        .reset_index()
        .to_dict(orient="records"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "assessment_merged.csv", index=False)
    if embed_grid:
        pd.DataFrame(embed_grid).to_csv(output_dir / "embed_threshold_grid.csv", index=False)
    (output_dir / "assessment.json").write_text(
        json.dumps(assessment, indent=2),
        encoding="utf-8",
    )

    md_lines = [
        "# Golden Set Assessment",
        "",
        f"- Chunks evaluated: **{assessment['chunk_count']}**",
        f"- Gold relevant: **{assessment['gold_positives']}**",
        f"- Gold unclear: **{assessment['gold_negatives']}**",
        "",
        "## Ranking vs gold relevance",
        "",
    ]
    for signal, stats in assessment["ranking"].items():
        if signal == "gold_positives":
            continue
        md_lines.append(f"### {signal}")
        md_lines.append(f"- Average precision: **{stats['average_precision']}**")
        for key, value in stats.items():
            if key == "average_precision":
                continue
            md_lines.append(f"- {key}: **{value}**")
        md_lines.append("")

    if assessment["labeling"]:
        md_lines.extend(["## Labeling vs gold (pathway match)", ""])
        for method, stats in assessment["labeling"].items():
            md_lines.append(f"### {method}")
            md_lines.append(
                f"- Accuracy **{stats['accuracy']:.1%}** | "
                f"Precision **{stats['precision']:.1%}** | "
                f"Recall **{stats['recall']:.1%}** | "
                f"F1 **{stats['f1']:.3f}**"
            )
            md_lines.append(
                f"- TP {stats['true_positives']} | FP {stats['false_positives']} | "
                f"FN {stats['false_negatives']} | TN {stats['true_negatives']}"
            )
            md_lines.append("")

    if embed_grid:
        md_lines.extend(["## Sentence label + embed threshold grid", ""])
        md_lines.append("| threshold | flagged | TP | FP | FN | precision | recall |")
        md_lines.append("|-----------|---------|----|----|-----|-----------|--------|")
        for row in embed_grid:
            threshold = row["embed_threshold"]
            threshold_label = "none" if threshold is None else f"{threshold:.2f}"
            flagged = row["true_positives"] + row["false_positives"]
            md_lines.append(
                f"| {threshold_label} | {flagged} | {row['true_positives']} | "
                f"{row['false_positives']} | {row['false_negatives']} | "
                f"{row['precision']:.1%} | {row['recall']:.1%} |"
            )
        md_lines.append("")

    if embed_gated is not None:
        md_lines.extend(
            [
                "## Selected embed threshold gate",
                "",
                f"- Threshold: **{embed_gated['embed_threshold']}**",
                f"- Precision **{embed_gated['precision']:.1%}** | "
                f"Recall **{embed_gated['recall']:.1%}** | "
                f"TP {embed_gated['true_positives']} | FP {embed_gated['false_positives']} | "
                f"FN {embed_gated['false_negatives']}",
                "",
            ]
        )

    (output_dir / "assessment.md").write_text("\n".join(md_lines), encoding="utf-8")
    return assessment
