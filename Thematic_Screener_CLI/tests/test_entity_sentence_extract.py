"""Tests for entity sentence extraction."""

from __future__ import annotations

import sys
from pathlib import Path

EXTRACT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "entity_sentence_rerank"
sys.path.insert(0, str(EXTRACT_DIR))

from extract import extract_entity_sentence_records  # noqa: E402


def test_extracts_one_record_per_query_entity_detection() -> None:
    chunk_text = (
        "NVIDIA reported strong AI revenue growth in the quarter. "
        "NVIDIA also expanded its data center partnerships."
    )
    rows = [
        {
            "chunk_text": chunk_text,
            "plan_entity_ids": ["253C3A"],
            "detections": [
                {"id": "253C3A", "start": 0, "end": 6, "type": "entity"},
                {"id": "253C3A", "start": 58, "end": 64, "type": "entity"},
                {"id": "OTHER1", "start": 80, "end": 86, "type": "entity"},
            ],
        }
    ]
    records = extract_entity_sentence_records(rows)
    assert len(records) == 2
    assert all(record["entity_id"] == "253C3A" for record in records)
    assert all(record["extraction_ok"] for record in records)
    assert records[0]["extracted_sentence"] != records[1]["extracted_sentence"]


def test_ignores_detections_outside_search_query_universe() -> None:
    rows = [
        {
            "chunk_text": "NVIDIA reported strong revenue. AMD also grew.",
            "plan_entity_ids": ["253C3A"],
            "detections": [
                {"id": "253C3A", "start": 0, "end": 6, "type": "entity"},
                {"id": "AMD001", "start": 32, "end": 35, "type": "entity"},
            ],
        }
    ]
    records = extract_entity_sentence_records(rows)
    assert len(records) == 1
    assert records[0]["entity_id"] == "253C3A"
    assert "NVIDIA" in records[0]["extracted_sentence"]
    assert "AMD" not in records[0]["extracted_sentence"]


def test_fallback_when_no_query_entity_detections() -> None:
    rows = [
        {
            "chunk_text": "Generic industry commentary without target entities.",
            "plan_entity_ids": ["253C3A"],
            "detections": [{"id": "OTHER1", "start": 0, "end": 7, "type": "entity"}],
        }
    ]
    records = extract_entity_sentence_records(rows)
    assert len(records) == 1
    assert records[0]["extraction_ok"] is False
    assert records[0]["extraction_method"] == "fallback_no_query_entity_detection"
