"""Extract entity-local sentences using detection coordinates."""

from __future__ import annotations

import re
from typing import Any

SENTENCE_END_PATTERN = re.compile(r"(?<=[.!?])\s+")
ABBREVIATION_GUARD = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|U\.S|U\.K|Inc|Ltd|Corp|Co)\.\s*$",
    re.IGNORECASE,
)


def _target_entity_ids(row: dict[str, Any]) -> set[str]:
    """Entities we care about: those in the plan/search basket for this query."""
    plan_ids = row.get("plan_entity_ids") or []
    return {str(entity_id) for entity_id in plan_ids}


def _entity_detections(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Detections for entities targeted by the search query only."""
    allowed = _target_entity_ids(row)
    detections: list[dict[str, Any]] = []
    for detection in row.get("detections") or []:
        if str(detection.get("type") or "") != "entity":
            continue
        entity_id = str(detection.get("id") or "")
        if entity_id not in allowed:
            continue
        start = detection.get("start")
        end = detection.get("end")
        if start is None or end is None:
            continue
        detections.append(
            {
                "entity_id": entity_id,
                "start": int(start),
                "end": int(end),
            }
        )
    detections.sort(key=lambda item: (item["start"], item["end"]))
    return detections


def _sentence_bounds(text: str, mention_start: int, mention_end: int) -> tuple[int, int]:
    """Return char offsets for the sentence containing the entity mention."""
    if not text:
        return 0, 0

    left = text.rfind(". ", 0, mention_start)
    left_exclaim = text.rfind("! ", 0, mention_start)
    left_question = text.rfind("? ", 0, mention_start)
    left_newline = text.rfind("\n", 0, mention_start)
    sentence_start = max(left, left_exclaim, left_question, left_newline) + 1
    if sentence_start < 0:
        sentence_start = 0

    # Walk back if we stopped on an abbreviation (e.g. "Inc.")
    prefix = text[sentence_start:mention_start]
    while ABBREVIATION_GUARD.search(prefix):
        prev = text.rfind(". ", 0, sentence_start - 1)
        prev_exclaim = text.rfind("! ", 0, sentence_start - 1)
        prev_question = text.rfind("? ", 0, sentence_start - 1)
        sentence_start = max(prev, prev_exclaim, prev_question) + 1
        if sentence_start < 0:
            sentence_start = 0
            break
        prefix = text[sentence_start:mention_start]

    right_candidates = [
        text.find(". ", mention_end),
        text.find("! ", mention_end),
        text.find("? ", mention_end),
        text.find("\n", mention_end),
    ]
    positive = [candidate for candidate in right_candidates if candidate >= 0]
    if positive:
        sentence_end = min(positive) + 1
    else:
        sentence_end = len(text)

    return sentence_start, sentence_end


def extract_entity_sentence_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand chunks into entity–sentence–chunk records using detection coordinates."""
    records: list[dict[str, Any]] = []
    record_index = 0

    for chunk_index, row in enumerate(rows):
        text = str(row.get("chunk_text") or "")
        detections = _entity_detections(row)
        chunk_char_len = len(text)

        if not detections:
            records.append(
                {
                    **row,
                    "chunk_index": chunk_index,
                    "record_id": record_index,
                    "entity_id": None,
                    "entity_span": "",
                    "detection_start": None,
                    "detection_end": None,
                    "sentence_start": None,
                    "sentence_end": None,
                    "extracted_sentence": text,
                    "extraction_method": "fallback_no_query_entity_detection",
                    "extraction_ok": False,
                    "chunk_char_len": chunk_char_len,
                    "sentence_char_len": chunk_char_len,
                }
            )
            record_index += 1
            continue

        seen_spans: set[tuple[str, int, int]] = set()
        for detection in detections:
            entity_id = detection["entity_id"]
            start = detection["start"]
            end = detection["end"]
            span_key = (entity_id, start, end)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)

            sentence_start, sentence_end = _sentence_bounds(text, start, end)
            extracted = text[sentence_start:sentence_end].strip()
            entity_span = text[start:end]

            records.append(
                {
                    **row,
                    "chunk_index": chunk_index,
                    "record_id": record_index,
                    "entity_id": entity_id,
                    "entity_span": entity_span,
                    "detection_start": start,
                    "detection_end": end,
                    "sentence_start": sentence_start,
                    "sentence_end": sentence_end,
                    "extracted_sentence": extracted,
                    "extraction_method": "detection_coordinates",
                    "extraction_ok": True,
                    "chunk_char_len": chunk_char_len,
                    "sentence_char_len": len(extracted),
                }
            )
            record_index += 1

    return records
