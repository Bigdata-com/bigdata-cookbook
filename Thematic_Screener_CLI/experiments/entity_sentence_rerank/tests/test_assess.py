"""Tests for golden set assessment helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from assess import (  # noqa: E402
    _average_precision,
    _binary_metrics,
    _collapse_records_to_chunks,
    _predict_relevant,
    _precision_at_k,
    _recall_at_k,
    apply_embed_threshold_gate,
)


def test_predict_relevant_requires_pathway_match() -> None:
    assert _predict_relevant("Cryogenics and specialized components", "Cryogenics and specialized components")
    assert not _predict_relevant("unclear", "Cryogenics and specialized components")
    assert not _predict_relevant("Quantum hardware developers", "Cryogenics and specialized components")


def test_ranking_metrics() -> None:
    y_true = pd.Series([True, False, True, False], index=[0, 1, 2, 3])
    scores = pd.Series([0.9, 0.8, 0.7, 0.1], index=[0, 1, 2, 3])
    assert _precision_at_k(y_true, scores, 2) == 0.5
    assert _recall_at_k(y_true, scores, 2) == 0.5
    assert _average_precision(y_true, scores) == 0.8333


def test_binary_metrics() -> None:
    y_true = pd.Series([True, True, False, False])
    y_pred = pd.Series([True, False, True, False])
    metrics = _binary_metrics(y_true, y_pred)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_negatives"] == 1


def test_collapse_records_to_chunks() -> None:
    records = pd.DataFrame(
        {
            "chunk_index": [0, 0, 1],
            "search_relevance": [0.2, 0.9, 0.4],
            "embedding_relevance": [0.3, 0.8, 0.5],
            "evidence_score": [0.1, 0.7, 0.2],
            "leaf_label": ["A", "A", "B"],
            "company_name": ["Co", "Co", "Other"],
            "plan_file": ["p1", "p1", "p2"],
            "baseline_label": ["unclear", "A", "B"],
            "experiment_label": ["unclear", "unclear", "B"],
        }
    )
    collapsed = _collapse_records_to_chunks(records)
    assert len(collapsed) == 2
    assert collapsed.loc[collapsed["chunk_index"] == 0, "search_relevance"].iloc[0] == 0.9
    assert collapsed.loc[collapsed["chunk_index"] == 0, "baseline_label"].iloc[0] == "A"


def test_embed_threshold_gate_requires_label_and_embed() -> None:
    chunk_df = pd.DataFrame(
        {
            "chunk_index": [0, 1, 2],
            "leaf_label": ["A", "A", "A"],
            "prov_experiment_label": ["A", "unclear", "A"],
            "gold_relevant": [True, False, False],
            "sentence_embedding_relevance": [0.60, 0.80, 0.40],
        }
    )
    gated = apply_embed_threshold_gate(chunk_df, embed_threshold=0.52)
    assert gated.tolist() == [True, False, False]
