"""Comparison metrics and report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _distribution_stats(series: pd.Series) -> dict[str, float]:
    clean = series.dropna().astype(float)
    if clean.empty:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    return {
        "mean": round(float(clean.mean()), 4),
        "median": round(float(clean.median()), 4),
        "p25": round(float(clean.quantile(0.25)), 4),
        "p75": round(float(clean.quantile(0.75)), 4),
    }


def _rank(series: pd.Series) -> pd.Series:
    return series.rank(method="average", ascending=False)


def _top_k_overlap(left: pd.Series, right: pd.Series, k: int) -> float:
    left_top = set(left.sort_values(ascending=False).head(k).index)
    right_top = set(right.sort_values(ascending=False).head(k).index)
    if not left_top:
        return 0.0
    return round(len(left_top & right_top) / len(left_top), 4)


def _rescue_counts(df: pd.DataFrame) -> dict[str, int]:
    midpoint = len(df) // 2
    search_rank = _rank(df["search_relevance"])
    embed_rank = _rank(df["embedding_relevance"])
    search_bottom = set(search_rank[search_rank > midpoint].index)
    search_top = set(search_rank[search_rank <= midpoint].index)
    embed_bottom = set(embed_rank[embed_rank > midpoint].index)
    embed_top = set(embed_rank[embed_rank <= midpoint].index)
    return {
        "search_bottom_embed_top": len(search_bottom & embed_top),
        "search_top_embed_bottom": len(search_top & embed_bottom),
    }


def build_review_queue(df: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    search_rank = _rank(df["search_relevance"])
    embed_rank = _rank(df["embedding_relevance"])
    rank_gap = (search_rank - embed_rank).abs()
    label_disagree = pd.Series(False, index=df.index)
    if "baseline_label" in df.columns and "experiment_label" in df.columns:
        label_disagree = df["baseline_label"].fillna("") != df["experiment_label"].fillna("")
    queue = df[(label_disagree) | (rank_gap >= 30)].copy()
    queue["rank_gap"] = rank_gap.loc[queue.index]
    queue = queue.sort_values("rank_gap", ascending=False).head(limit)
    return queue


def generate_report(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    skip_labeling: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_dir / "scored_records.csv", index=False)

    search_stats = _distribution_stats(df["search_relevance"])
    evidence_stats = _distribution_stats(df["evidence_score"])
    embed_stats = _distribution_stats(df["embedding_relevance"])
    spearman = float(
        df["search_relevance"]
        .rank(method="average")
        .corr(df["embedding_relevance"].rank(method="average"), method="pearson")
    )
    top20 = _top_k_overlap(df["search_relevance"], df["embedding_relevance"], 20)
    top50 = _top_k_overlap(df["search_relevance"], df["embedding_relevance"], 50)

    extraction_ok_rate = (
        round(float(df["extraction_ok"].mean()), 4) if "extraction_ok" in df else 0.0
    )
    compression = (
        round(float((df["sentence_char_len"] / df["chunk_char_len"].replace(0, pd.NA)).median()), 4)
        if {"sentence_char_len", "chunk_char_len"}.issubset(df.columns)
        else 0.0
    )

    labeling_summary: dict[str, Any] = {}
    label_pairs = (
        ("baseline_label", "experiment_label"),
        ("prov_baseline_label", "prov_experiment_label"),
    )
    for left_col, right_col in label_pairs:
        if skip_labeling or left_col not in df.columns:
            continue
        agreement = float((df[left_col] == df[right_col]).mean())
        prefix = "prov_" if left_col.startswith("prov_") else ""
        labeling_summary[f"{prefix}label_agreement_rate"] = round(agreement, 4)
        labeling_summary[f"{prefix}baseline_unclear_rate"] = round(
            float((df[left_col] == "unclear").mean()),
            4,
        )
        labeling_summary[f"{prefix}experiment_unclear_rate"] = round(
            float((df[right_col] == "unclear").mean()),
            4,
        )
    if not skip_labeling and "baseline_label" in df.columns:
        df.to_csv(output_dir / "labeled_comparison.csv", index=False)
        review_queue = build_review_queue(df)
        review_queue.to_csv(output_dir / "review_queue.csv", index=False)

    summary: dict[str, Any] = {
        "record_count": int(len(df)),
        "unique_chunks": int(df["chunk_index"].nunique()) if "chunk_index" in df else int(len(df)),
        "score_distributions": {
            "search_relevance": search_stats,
            "evidence_score": evidence_stats,
            "embedding_relevance": embed_stats,
        },
        "spearman_search_vs_embedding": round(spearman, 4),
        "top20_overlap": top20,
        "top50_overlap": top50,
        "rescue_analysis": _rescue_counts(df),
        "extraction_ok_rate": extraction_ok_rate,
        "median_sentence_to_chunk_ratio": compression,
        "labeling": labeling_summary,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(7, 6))
    plt.scatter(df["search_relevance"], df["embedding_relevance"], alpha=0.5, s=20)
    plt.xlabel("Search relevance (baseline)")
    plt.ylabel("Embedding relevance (experiment)")
    plt.title("Search vs embedding relevance")
    plt.tight_layout()
    plt.savefig(output_dir / "relevance_scatter.png", dpi=150)
    plt.close()

    md_lines = [
        "# Entity-Sentence Rerank Experiment Summary",
        "",
        f"- Records (entity–sentence–chunk): **{summary['record_count']}**",
        f"- Unique chunks sampled: **{summary['unique_chunks']}**",
        f"- Extraction success rate: **{summary['extraction_ok_rate']:.1%}**",
        f"- Median sentence/chunk length ratio: **{summary['median_sentence_to_chunk_ratio']}**",
        "",
        "## Score distributions",
        "",
        f"- Search relevance: {search_stats}",
        f"- Evidence score (relevance × |sentiment|): {evidence_stats}",
        f"- Embedding relevance: {embed_stats}",
        "",
        "## Ranking comparison",
        "",
        f"- Spearman(search, embedding): **{summary['spearman_search_vs_embedding']}**",
        f"- Top-20 overlap: **{summary['top20_overlap']}**",
        f"- Top-50 overlap: **{summary['top50_overlap']}**",
        f"- Rescue (search bottom / embed top): "
        f"**{summary['rescue_analysis']['search_bottom_embed_top']}**",
        f"- Rescue (search top / embed bottom): "
        f"**{summary['rescue_analysis']['search_top_embed_bottom']}**",
        "",
    ]
    if labeling_summary:
        md_lines.extend(
            [
                "## Labeling comparison",
                "",
            ]
        )
        for key, value in labeling_summary.items():
            if key.endswith("_rate"):
                md_lines.append(f"- {key}: **{value:.1%}**")
        md_lines.append("")
    (output_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    return summary
