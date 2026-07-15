"""Provenance-locked labeling for baseline chunk vs experiment sentence comparison."""

from __future__ import annotations

import json
from typing import Any

from src.openai_parallel import ChatRequest, RateLimitConfig, run_chat_requests_parallel

PROVENANCE_SYSTEM_PROMPT = """Forget all previous prompts.
You are assisting a professional analyst with a thematic company screener.

Theme:
{main_theme}

Analyst focus (mandatory scope):
{analyst_focus}

Retrieval pathway label (fixed):
{leaf_label}

Search query that retrieved this evidence:
{search_query}

Your task is to judge whether the input text provides evidence that the target company
has economic exposure along THIS SPECIFIC retrieval pathway only.

Rules:
1. The assigned label must be exactly "{leaf_label}" if relevant, otherwise "unclear".
2. Do NOT assign any other taxonomy label, even if the text fits a different pathway.
3. Text is relevant only if it connects the target company to the pathway described
   by the search query (correct value-chain role: supplier vs operator vs customer).
4. Assign "unclear" for name-drops, vendor PR about other companies, logo-only text,
   market backdrop, peers, or wrong role.
5. Apply analyst focus scope strictly.
6. Use only the provided text and company name.

Return valid JSON only with these fields:
- label (string: "{leaf_label}" or "unclear")
- materiality ("high", "medium", "low", or "unclear")
- motivation (string; begin with "Target Company")
"""


def _build_requests(
    records: list[dict[str, Any]],
    *,
    main_theme: str,
    analyst_focus: str,
    text_field: str,
    request_suffix: str,
    model: str,
) -> list[ChatRequest]:
    requests: list[ChatRequest] = []
    for record in records:
        leaf_label = str(record.get("leaf_label") or "")
        search_query = str(record.get("search_query") or "")
        system_prompt = PROVENANCE_SYSTEM_PROMPT.format(
            main_theme=main_theme,
            analyst_focus=analyst_focus,
            leaf_label=leaf_label,
            search_query=search_query,
        )
        payload = {
            "record_id": record.get("record_id"),
            "company_name": record.get("company_name") or "",
            "leaf_label": leaf_label,
            "search_query": search_query,
            "text": record.get(text_field) or "",
        }
        requests.append(
            ChatRequest(
                request_id=f"{record['record_id']}:{request_suffix}",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                model=model,
                temperature=1.0,
                response_format={"type": "json_object"},
            )
        )
    return requests


def _parse_label_response(raw: str, *, leaf_label: str) -> dict[str, str]:
    payload = json.loads(raw)
    label = str(payload.get("label", "unclear")).strip()
    materiality = str(payload.get("materiality", "unclear")).strip()
    motivation = str(payload.get("motivation", "")).strip()
    if label != leaf_label:
        label = "unclear"
        if materiality not in {"high", "medium", "low"}:
            materiality = "unclear"
    if materiality not in {"high", "medium", "low", "unclear"}:
        materiality = "unclear"
    return {"label": label, "materiality": materiality, "motivation": motivation}


def run_provenance_labeling_passes(
    records: list[dict[str, Any]],
    *,
    main_theme: str,
    analyst_focus: str,
    model: str,
    requests_per_minute: int,
    max_concurrent_requests: int,
) -> list[dict[str, Any]]:
    """Run provenance-locked chunk and sentence labeling passes."""
    baseline_requests = _build_requests(
        records,
        main_theme=main_theme,
        analyst_focus=analyst_focus,
        text_field="chunk_text",
        request_suffix="baseline",
        model=model,
    )
    experiment_requests = _build_requests(
        records,
        main_theme=main_theme,
        analyst_focus=analyst_focus,
        text_field="extracted_sentence",
        request_suffix="experiment",
        model=model,
    )
    responses = run_chat_requests_parallel(
        baseline_requests + experiment_requests,
        rate_limit=RateLimitConfig(
            requests_per_minute=requests_per_minute,
            max_concurrent_requests=max_concurrent_requests,
        ),
        show_progress=True,
    )
    by_id = {response.request_id: response for response in responses}

    merged: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record["record_id"])
        leaf_label = str(record.get("leaf_label") or "")
        row = {**record}
        for suffix, prefix in (("baseline", "prov_baseline"), ("experiment", "prov_experiment")):
            response = by_id.get(f"{record_id}:{suffix}")
            if response is None or not response.succeeded or not response.content:
                row[f"{prefix}_label"] = "unclear"
                row[f"{prefix}_materiality"] = "unclear"
                row[f"{prefix}_motivation"] = ""
                row[f"{prefix}_error"] = (
                    str(response.error) if response and response.error else "missing response"
                )
                continue
            try:
                parsed = _parse_label_response(response.content, leaf_label=leaf_label)
                row[f"{prefix}_label"] = parsed["label"]
                row[f"{prefix}_materiality"] = parsed["materiality"]
                row[f"{prefix}_motivation"] = parsed["motivation"]
                row[f"{prefix}_error"] = ""
            except json.JSONDecodeError as exc:
                row[f"{prefix}_label"] = "unclear"
                row[f"{prefix}_materiality"] = "unclear"
                row[f"{prefix}_motivation"] = ""
                row[f"{prefix}_error"] = f"json parse: {exc}"
        merged.append(row)
    return merged
