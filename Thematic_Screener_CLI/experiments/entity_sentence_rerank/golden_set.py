"""Build a provenance-locked golden set for chunk relevance evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src import screener
from src.openai_parallel import ChatRequest, RateLimitConfig, run_chat_requests_parallel

DEFAULT_GOLDEN_MODEL = "gpt-5.5"

GOLDEN_SYSTEM_PROMPT = """Forget all previous prompts.
You are creating a gold-standard evaluation set for a thematic company screener.

Theme:
{main_theme}

Analyst focus (mandatory scope):
{analyst_focus}

Retrieval pathway label (fixed):
{leaf_label}

Search query that retrieved this evidence:
{search_query}

Your task is to judge whether a retrieved text chunk provides evidence that the
target company has economic exposure along THIS SPECIFIC retrieval pathway only.

Rules:
1. The assigned label must be exactly "{leaf_label}" if relevant, otherwise "unclear".
2. Do NOT assign any other taxonomy label, even if the text fits a different pathway.
3. A chunk is relevant only if it connects the target company to the pathway described
   by the search query (correct value-chain role: supplier vs operator vs customer).
4. Assign "unclear" for name-drops, vendor PR about other companies, logo-only text,
   market backdrop, peers, or wrong role (e.g., customer listed as hardware seller).
5. Apply analyst focus scope strictly (geography, segment, mechanism).
6. Use only the chunk text and company name provided.

Return valid JSON only with these fields:
- relevant (boolean)
- label (string: "{leaf_label}" or "unclear")
- materiality ("high", "medium", "low", or "unclear")
- motivation (string; begin with "Target Company")
- evidence_quality ("strong", "weak", or "noise")
"""

GOLDEN_USER_TEMPLATE = """Judge this retrieved chunk:
{payload}"""


def _target_entity_id(row: dict[str, Any]) -> str | None:
    plan_ids = {str(entity_id) for entity_id in row.get("plan_entity_ids") or []}
    primary = row.get("primary_entity_id")
    if primary and str(primary) in plan_ids:
        return str(primary)
    for entity_id in row.get("entity_ids") or []:
        if str(entity_id) in plan_ids:
            return str(entity_id)
    if primary:
        return str(primary)
    entity_ids = row.get("entity_ids") or []
    return str(entity_ids[0]) if entity_ids else None


def _company_name(universe_df: pd.DataFrame, entity_id: str | None) -> str:
    if not entity_id:
        return ""
    id_column = screener.UNIVERSE_ID_COLUMN
    name_column = screener.UNIVERSE_NAME_COLUMN
    matches = universe_df[universe_df[id_column].astype(str) == str(entity_id)]
    if matches.empty:
        return ""
    return str(matches.iloc[0][name_column])


def prepare_golden_chunks(
    chunk_rows: list[dict[str, Any]],
    *,
    universe_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Attach chunk_index, target entity, and company_name for annotation."""
    prepared: list[dict[str, Any]] = []
    for chunk_index, row in enumerate(chunk_rows):
        entity_id = _target_entity_id(row)
        prepared.append(
            {
                **row,
                "chunk_index": chunk_index,
                "target_entity_id": entity_id,
                "company_name": _company_name(universe_df, entity_id),
            }
        )
    return prepared


def _parse_golden_response(raw: str, *, leaf_label: str) -> dict[str, str | bool]:
    payload = json.loads(raw)
    relevant = bool(payload.get("relevant", False))
    label = str(payload.get("label", "unclear")).strip()
    materiality = str(payload.get("materiality", "unclear")).strip()
    motivation = str(payload.get("motivation", "")).strip()
    evidence_quality = str(payload.get("evidence_quality", "noise")).strip()

    if relevant:
        if label != leaf_label:
            label = leaf_label
    else:
        label = "unclear"
        if materiality not in {"high", "medium", "low"}:
            materiality = "unclear"

    if materiality not in {"high", "medium", "low", "unclear"}:
        materiality = "unclear"
    if evidence_quality not in {"strong", "weak", "noise"}:
        evidence_quality = "noise"

    return {
        "gold_relevant": relevant,
        "gold_label": label,
        "gold_materiality": materiality,
        "gold_motivation": motivation,
        "gold_evidence_quality": evidence_quality,
    }


def annotate_golden_chunks(
    chunks: list[dict[str, Any]],
    *,
    main_theme: str,
    analyst_focus: str,
    model: str = DEFAULT_GOLDEN_MODEL,
    requests_per_minute: int = 500,
    max_concurrent_requests: int = 20,
) -> list[dict[str, Any]]:
    """Call the golden model once per chunk with provenance-locked instructions."""
    requests: list[ChatRequest] = []
    for row in chunks:
        leaf_label = str(row.get("leaf_label") or "")
        search_query = str(row.get("search_query") or "")
        system_prompt = GOLDEN_SYSTEM_PROMPT.format(
            main_theme=main_theme,
            analyst_focus=analyst_focus,
            leaf_label=leaf_label,
            search_query=search_query,
        )
        user_payload = {
            "chunk_index": row["chunk_index"],
            "company_name": row.get("company_name") or "",
            "target_entity_id": row.get("target_entity_id"),
            "leaf_label": leaf_label,
            "search_query": search_query,
            "text": row.get("chunk_text") or "",
        }
        requests.append(
            ChatRequest(
                request_id=str(row["chunk_index"]),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": GOLDEN_USER_TEMPLATE.format(
                            payload=json.dumps(user_payload, ensure_ascii=False)
                        ),
                    },
                ],
                model=model,
                temperature=1.0,
                response_format={"type": "json_object"},
            )
        )

    responses = run_chat_requests_parallel(
        requests,
        rate_limit=RateLimitConfig(
            requests_per_minute=requests_per_minute,
            max_concurrent_requests=max_concurrent_requests,
        ),
        show_progress=True,
    )

    by_id = {response.request_id: response for response in responses}
    annotated: list[dict[str, Any]] = []
    for row in chunks:
        chunk_index = str(row["chunk_index"])
        response = by_id.get(chunk_index)
        leaf_label = str(row.get("leaf_label") or "")
        merged = {**row}
        if response is None or not response.succeeded or not response.content:
            merged.update(
                {
                    "gold_relevant": False,
                    "gold_label": "unclear",
                    "gold_materiality": "unclear",
                    "gold_motivation": "",
                    "gold_evidence_quality": "noise",
                    "gold_error": str(response.error) if response and response.error else "missing response",
                }
            )
        else:
            try:
                parsed = _parse_golden_response(response.content, leaf_label=leaf_label)
                merged.update(parsed)
                merged["gold_error"] = ""
            except json.JSONDecodeError as exc:
                merged.update(
                    {
                        "gold_relevant": False,
                        "gold_label": "unclear",
                        "gold_materiality": "unclear",
                        "gold_motivation": "",
                        "gold_evidence_quality": "noise",
                        "gold_error": f"json parse: {exc}",
                    }
                )
        annotated.append(merged)
    return annotated


def write_golden_set(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    model: str,
    source_path: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "golden_set.csv"
    jsonl_path = output_dir / "golden_set.jsonl"
    manifest_path = output_dir / "golden_set_manifest.json"

    export_rows: list[dict[str, Any]] = []
    for row in rows:
        export_rows.append(
            {
                "chunk_index": row.get("chunk_index"),
                "plan_file": row.get("plan_file"),
                "leaf_label": row.get("leaf_label"),
                "search_query": row.get("search_query"),
                "target_entity_id": row.get("target_entity_id"),
                "company_name": row.get("company_name"),
                "document_id": row.get("document_id"),
                "headline": row.get("headline"),
                "search_relevance": row.get("search_relevance"),
                "sentiment": row.get("sentiment"),
                "evidence_score": float(row.get("search_relevance") or 0)
                * abs(float(row.get("sentiment") or 0)),
                "chunk_text": row.get("chunk_text"),
                "gold_relevant": row.get("gold_relevant"),
                "gold_label": row.get("gold_label"),
                "gold_materiality": row.get("gold_materiality"),
                "gold_evidence_quality": row.get("gold_evidence_quality"),
                "gold_motivation": row.get("gold_motivation"),
                "gold_error": row.get("gold_error", ""),
            }
        )

    df = pd.DataFrame(export_rows)
    df.to_csv(csv_path, index=False)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for export_row in export_rows:
            handle.write(json.dumps(export_row, ensure_ascii=False) + "\n")

    manifest = {
        "model": model,
        "source": str(source_path),
        "row_count": len(export_rows),
        "relevant_count": int(sum(1 for row in export_rows if row.get("gold_relevant"))),
        "unclear_count": int(sum(1 for row in export_rows if not row.get("gold_relevant"))),
        "labeling_mode": "provenance_locked",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return csv_path


def summarize_golden_set(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return quick distribution stats for the golden set."""
    relevant = sum(1 for row in rows if row.get("gold_relevant"))
    quality_counts: dict[str, int] = {}
    materiality_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    for row in rows:
        quality = str(row.get("gold_evidence_quality") or "noise")
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
        materiality = str(row.get("gold_materiality") or "unclear")
        materiality_counts[materiality] = materiality_counts.get(materiality, 0) + 1
        label = str(row.get("gold_label") or "unclear")
        label_counts[label] = label_counts.get(label, 0) + 1
    errors = sum(1 for row in rows if row.get("gold_error"))
    return {
        "row_count": len(rows),
        "gold_relevant_count": relevant,
        "gold_irrelevant_count": len(rows) - relevant,
        "gold_evidence_quality": quality_counts,
        "gold_materiality": materiality_counts,
        "gold_label": label_counts,
        "gold_errors": errors,
    }
