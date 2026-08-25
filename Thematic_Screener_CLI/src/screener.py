"""Pipeline functions for the thematic screener.

Pure building blocks with no CLI parsing or run-directory logic.
Orchestration and persistence are handled by :mod:`src.cli` together with
:class:`src.run_context.RunContext`.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from bigdata_smart_batching import (
    convert_to_dataframe,  # noqa: F401  (kept available for downstream use)
    deduplicate_documents,
    execute_search,
    load_plan,
    plan_search,
    save_plan,
)
from openai import OpenAI
from pydantic import BaseModel, Field

from src.helpers import (
    build_leaf_ancestry,
    get_leaf_label_summary_options,
    get_leaf_labels,
    get_leaf_search_queries,
    get_leaf_summaries,
    print_tree,
)
from src.modes import AnalysisMode, LeafField, ModeProfile, get_profile
from src.openai_parallel import (
    ChatRequest,
    RateLimitConfig,
    run_chat_requests_parallel,
    sampling_params_for_model,
)
from src.prompts import (
    SYSTEM_MESSAGE_LABELS_DERIVATIVES,
    TAXONOMY_STYLE_DERIVATIVES,
    TAXONOMY_STYLE_EXPOSURE,
)

logger = logging.getLogger(__name__)

_THEMATIC_PROFILE = get_profile(AnalysisMode.THEMATIC_SCREENER)

DEFAULT_MAIN_THEME = _THEMATIC_PROFILE.default_main_theme
DEFAULT_ANALYST_FOCUS = _THEMATIC_PROFILE.default_analyst_focus
DEFAULT_START_DATE = "2025-06-01"
DEFAULT_END_DATE = "2026-06-09"
DEFAULT_LABELS_MODEL = "gpt-5.6-luna"
DEFAULT_LABELING_MODEL = "gpt-5.6-luna"
DEFAULT_SUMMARY_MODEL = "gpt-5.6-luna"
DEFAULT_MAX_LEAF_LABELS = 15
DEFAULT_CHUNK_PERCENTAGE = 0.02
DEFAULT_REQUESTS_PER_MINUTE = 350
DEFAULT_LABELING_REQUESTS_PER_MINUTE = 10_000
DEFAULT_LABELING_MAX_CONCURRENT = 80
DEFAULT_SUMMARY_MAX_CONCURRENT = 40
DEFAULT_RERANK_THRESHOLD = 0.0
DEFAULT_SEARCH_CATEGORY: dict[str, Any] = {
    "mode": "INCLUDE",
    "values": ["news_premium", "transcripts", "filings"],
}

UNIVERSE_ID_COLUMN = "RP_ENTITY_ID"
UNIVERSE_ID_ALIASES: tuple[str, ...] = ("RP_ENTITY_ID", "RP_COMPANY_ID")
UNIVERSE_NAME_COLUMN = "COMPANY_NAME"

# Optional universe columns used to enrich the risk-analyzer export. When absent
# the corresponding fields fall back to "Unknown" (sector/industry) or None.
UNIVERSE_TICKER_COLUMN = "TICKER"
UNIVERSE_SECTOR_COLUMN = "SECTOR"
UNIVERSE_INDUSTRY_COLUMN = "INDUSTRY"
UNIVERSE_COUNTRY_COLUMN = "COUNTRY"

UNKNOWN_VALUE = "Unknown"

MAX_MOTIVATIONS_CHARS = 120_000


class Node(BaseModel):
    """Recursive node of the sub-theme taxonomy returned by the LLM."""

    node: int
    label: str
    summary: str
    search_query: str = ""
    children: list[Node] = Field(default_factory=list)


Node.model_rebuild()


class CompanySummary(BaseModel):
    """Structured company-level summary returned by the LLM."""

    summary: str


def normalize_max_leaf_labels(max_leaf_labels: int | None) -> int | None:
    """Return ``None`` when ``max_leaf_labels`` is unset or ``0`` (no cap)."""
    if max_leaf_labels is None or max_leaf_labels == 0:
        return None
    if max_leaf_labels < 0:
        raise ValueError("max_leaf_labels must be zero or positive")
    return max_leaf_labels


def analyst_focus_with_leaf_cap(analyst_focus: str, max_leaf_labels: int | None) -> str:
    """Append a leaf-count instruction for taxonomy generation when capped."""
    cap = normalize_max_leaf_labels(max_leaf_labels)
    if cap is None:
        return analyst_focus
    return f"{analyst_focus}\nLimit the final tree to at most {cap} leaf nodes."


def generate_taxonomy(
    main_theme: str,
    analyst_focus: str,
    model: str = DEFAULT_LABELS_MODEL,
    profile: ModeProfile | None = None,
    client: OpenAI | None = None,
    print_taxonomy: bool = True,
    max_leaf_labels: int | None = DEFAULT_MAX_LEAF_LABELS,
    taxonomy_style: str = TAXONOMY_STYLE_EXPOSURE,
    extra_instruction: str = "",
    grounding_brief: str = "",
) -> Node:
    """Generate the sub-concept taxonomy tree for ``main_theme`` via the LLM.

    The mode ``profile`` selects the prompts so the same call produces a theme
    taxonomy (thematic-screener) or a risk taxonomy (risk-analyzer).
    ``taxonomy_style='derivatives'`` is thematic-only and forces a 1st/2nd/3rd
    hop tree. ``grounding_brief`` is injected into that prompt when provided.
    """
    active_profile = profile if profile is not None else _THEMATIC_PROFILE
    openai_client = client if client is not None else OpenAI()
    resolved_style = (
        taxonomy_style
        if active_profile.mode is AnalysisMode.THEMATIC_SCREENER
        else TAXONOMY_STYLE_EXPOSURE
    )
    focus_text = (
        f"{analyst_focus_with_leaf_cap(analyst_focus, max_leaf_labels)}"
        f"{extra_instruction}"
    )
    if resolved_style == TAXONOMY_STYLE_DERIVATIVES:
        brief = grounding_brief.strip() or (
            "No Bigdata.com briefing was provided. "
            "Infer derivative pathways from economic reasoning."
        )
        system_prompt = SYSTEM_MESSAGE_LABELS_DERIVATIVES.format(
            main_theme=main_theme,
            analyst_focus=focus_text,
            grounding_brief=brief,
        )
    else:
        system_prompt = active_profile.labels_system_prompt.format(
            main_theme=main_theme,
            analyst_focus=focus_text,
        )
    completion = openai_client.chat.completions.parse(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": active_profile.labels_user_prompt.format(main_theme=main_theme),
            },
        ],
        **sampling_params_for_model(
            model,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=42,
        ),
    )

    content = completion.choices[0].message.content
    root = Node.model_validate_json(content)

    if print_taxonomy:
        print_tree(root)

    return root


def leaf_labels(root: Node, profile: ModeProfile | None = None) -> list[str]:
    """Extract the working label set from the taxonomy per the mode profile.

    Thematic mode uses leaf ``summary`` values when configured; risk mode uses
    leaf ``label`` values so they line up with the taxonomy and scoring.
    """
    active_profile = profile if profile is not None else _THEMATIC_PROFILE
    if active_profile.leaf_field is LeafField.LABEL:
        return get_leaf_labels(root)
    return get_leaf_summaries(root)


def write_taxonomy_artifacts(
    root: Node,
    themes_path: Path,
    search_queries_path: Path,
    taxonomy_tree_path: Path | None = None,
    profile: ModeProfile | None = None,
) -> tuple[list[str], list[str]]:
    """Persist taxonomy tree, leaf labels, and leaf search queries."""
    labels = leaf_labels(root, profile)
    search_queries = get_leaf_search_queries(root)
    if len(labels) != len(search_queries):
        raise ValueError("Leaf label and search_query counts must match")

    themes_path.parent.mkdir(parents=True, exist_ok=True)
    themes_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    search_queries_path.write_text("\n".join(search_queries) + "\n", encoding="utf-8")
    if taxonomy_tree_path is not None:
        taxonomy_tree_path.write_text(root.model_dump_json(indent=2), encoding="utf-8")
    return labels, search_queries


def generate_labels(
    main_theme: str,
    analyst_focus: str,
    model: str = DEFAULT_LABELS_MODEL,
    profile: ModeProfile | None = None,
    client: OpenAI | None = None,
    print_taxonomy: bool = True,
    max_leaf_labels: int | None = DEFAULT_MAX_LEAF_LABELS,
    taxonomy_style: str = TAXONOMY_STYLE_EXPOSURE,
    extra_instruction: str = "",
    grounding_brief: str = "",
) -> list[str]:
    """Generate the leaf-level working label set for ``main_theme``.

    Thin wrapper over :func:`generate_taxonomy` and :func:`leaf_labels`; kept
    for backward compatibility.
    """
    root = generate_taxonomy(
        main_theme=main_theme,
        analyst_focus=analyst_focus,
        model=model,
        profile=profile,
        client=client,
        print_taxonomy=print_taxonomy,
        max_leaf_labels=max_leaf_labels,
        taxonomy_style=taxonomy_style,
        extra_instruction=extra_instruction,
        grounding_brief=grounding_brief,
    )
    return leaf_labels(root, profile)


def plan_filename(label: str) -> str:
    """Return the plan JSON filename for a label."""
    safe_label = "".join(character if character.isalnum() else "_" for character in label)
    safe_label = "_".join(part for part in safe_label.split("_") if part)
    return f"{safe_label}.json"


def _find_universe_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first matching column name, case-insensitive."""
    normalized = {str(column).upper(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.upper() in normalized:
            return normalized[candidate.upper()]
    return None


def load_universe(universe_path: str | Path) -> pd.DataFrame:
    """Load the company universe CSV.

    Accepts ``RP_ENTITY_ID`` (preferred) or legacy ``RP_COMPANY_ID``, plus
    ``COMPANY_NAME`` (or ``NAME`` / ``COMPANY`` aliases).
    """
    raw_df = pd.read_csv(universe_path)
    id_column = _find_universe_column(raw_df.columns, UNIVERSE_ID_ALIASES)
    if id_column is None:
        raise ValueError(
            f"universe file {universe_path} is missing required ID column "
            f"(expected one of: {', '.join(UNIVERSE_ID_ALIASES)})"
        )
    name_column = _find_universe_column(
        raw_df.columns, (UNIVERSE_NAME_COLUMN, "NAME", "COMPANY")
    )
    if name_column is None:
        raise ValueError(
            f"universe file {universe_path} is missing required name column "
            f"(expected one of: {UNIVERSE_NAME_COLUMN}, NAME, COMPANY)"
        )
    universe_df = pd.DataFrame(
        {
            UNIVERSE_ID_COLUMN: raw_df[id_column].astype(str).str.strip(),
            UNIVERSE_NAME_COLUMN: raw_df[name_column].astype(str).str.strip(),
        }
    )
    return universe_df.reset_index(drop=True)


def _remove_stale_plans(plans_path: Path, current_paths: list[Path]) -> list[Path]:
    """Delete plan files in ``plans_path`` that the current taxonomy did not produce.

    Re-running a run directory after the taxonomy changed would otherwise leave
    plans for retired labels behind, and ``run_search`` executes every plan file
    it finds.
    """
    current_names = {path.name for path in current_paths}
    stale = [path for path in sorted(plans_path.glob("*.json")) if path.name not in current_names]
    for path in stale:
        path.unlink()
    return stale


def build_plans(
    labels: list[str],
    search_queries: list[str],
    company_ids: list[str],
    plans_dir: str | Path,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    category: dict[str, Any] | None = None,
) -> list[Path]:
    """Create and persist one search plan per label/search-query pair.

    Plans left over from an earlier taxonomy in the same directory are removed
    once the current set has been built.
    """
    if len(labels) != len(search_queries):
        raise ValueError("labels and search_queries must have the same length")

    plans_path = Path(plans_dir)
    plans_path.mkdir(parents=True, exist_ok=True)
    search_category = category if category is not None else DEFAULT_SEARCH_CATEGORY

    saved_paths: list[Path] = []
    for label, search_query in zip(labels, search_queries, strict=True):
        plan = plan_search(
            universe=company_ids,
            start_date=start_date,
            end_date=end_date,
            volume_query_mode="iterative",
            text=search_query,
            category=search_category,
        )
        plan_path = plans_path / plan_filename(label)
        save_plan(plan, str(plan_path))
        saved_paths.append(plan_path)

    stale_paths = _remove_stale_plans(plans_path, saved_paths)
    if stale_paths:
        logger.info(
            "Removed %d stale plan file(s) from %s: %s",
            len(stale_paths),
            plans_path,
            ", ".join(path.name for path in stale_paths),
        )

    return saved_paths


def _summarize_single_plan(plan_path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Extract chunk count from one loaded search plan."""
    baskets: list[dict[str, Any]] = plan.get("baskets", [])

    theme = ""
    if baskets:
        theme = baskets[0].get("query", {}).get("text", "")

    chunks = sum(int(basket.get("expected_chunks", 0) or 0) for basket in baskets)

    return {
        "plan_file": plan_path.name,
        "theme": theme,
        "chunks": chunks,
    }


def summarize_plans(plans_dir: str | Path) -> pd.DataFrame:
    """Summarize chunk counts for every search plan JSON file in a directory.

    Returns one row per plan, sorted by chunk count descending.
    """
    plans_path = Path(plans_dir)
    plan_files = sorted(plans_path.glob("*.json"))
    if not plan_files:
        raise FileNotFoundError(f"no plan files found in {plans_path}")

    rows: list[dict[str, Any]] = []
    for plan_file in plan_files:
        plan = load_plan(str(plan_file))
        rows.append(_summarize_single_plan(plan_file, plan))

    summary_df = pd.DataFrame(rows)
    return summary_df.sort_values("chunks", ascending=False).reset_index(drop=True)


def format_plans_summary(summary_df: pd.DataFrame) -> str:
    """Build a text summary of per-plan chunk counts and the total."""
    lines = ["Per-plan chunks:"]
    for row in summary_df.itertuples(index=False):
        label = row.theme or row.plan_file
        lines.append(f"  - {label}: {row.chunks:,}")
    lines.append(f"Total: {summary_df['chunks'].sum():,}")
    return "\n".join(lines)


def run_search(
    plans_dir: str | Path,
    chunk_percentage: float = DEFAULT_CHUNK_PERCENTAGE,
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
) -> list[dict[str, Any]]:
    """Execute every plan in ``plans_dir`` and return deduplicated documents."""
    plans_path = Path(plans_dir)
    plan_files = sorted(plans_path.glob("*.json"))
    if not plan_files:
        raise FileNotFoundError(
            f"no plan files found in {plans_path}. Please run the 'plans' command first."
        )

    results: list[dict[str, Any]] = []
    for plan_file in plan_files:
        logger.info("Executing plan %s", plan_file.name)
        plan = load_plan(str(plan_file))
        results_raw = execute_search(
            search_plan=plan,
            chunk_percentage=chunk_percentage,
            requests_per_minute=requests_per_minute,
            basket_filtered_entities=True,
        )
        results.extend(results_raw)

    return deduplicate_documents(results)


def extract_sentences(
    results: list[dict[str, Any]],
    company_df: pd.DataFrame,
    rerank_threshold: float = DEFAULT_RERANK_THRESHOLD,
) -> list[dict[str, Any]]:
    """Flatten retrieved documents into labeled sentence records.

    Each chunk becomes a sentence with an id, text, and resolved company
    name (looked up from the universe DataFrame). Chunks whose ``relevance``
    score is below ``rerank_threshold`` are dropped (a threshold of ``0.0``
    keeps every chunk).
    """
    sentences: list[dict[str, Any]] = []
    idx = 0
    for document in results:
        document_id = document.get("id")
        headline = document.get("headline")
        timestamp = document.get("timestamp")
        url = document.get("url")
        for chunk in document.get("chunks", []):
            relevance = chunk.get("relevance")
            if (
                rerank_threshold
                and relevance is not None
                and relevance < rerank_threshold
            ):
                continue
            sentences.append(
                {
                    "sentence_id": idx,
                    "text": chunk.get("text"),
                    "entity": chunk.get("entity_ids"),
                    "document_id": document_id,
                    "headline": headline,
                    "timestamp": timestamp,
                    "url": url,
                    "sentiment": chunk.get("sentiment"),
                    "relevance": relevance,
                }
            )
            idx += 1

    id_to_name = dict(
        zip(company_df[UNIVERSE_ID_COLUMN], company_df[UNIVERSE_NAME_COLUMN])
    )
    for sentence in sentences:
        entity = sentence.get("entity")
        if entity:
            first_entity = entity[0]
            company_name = id_to_name.get(first_entity)
            if pd.isna(company_name) or not str(company_name).strip():
                company_name = first_entity
            sentence["company_name"] = company_name
        sentence.pop("entity", None)

    return sentences


_LABELING_PAYLOAD_FIELDS = ("sentence_id", "text", "company_name")


def _labeling_payload(sentence: dict[str, Any]) -> dict[str, Any]:
    """Project a sentence onto the fields shown to the labeling model."""
    return {key: sentence[key] for key in _LABELING_PAYLOAD_FIELDS if key in sentence}


@dataclass(frozen=True, slots=True)
class LabelingLatencyStats:
    """Per-request latency percentiles for one labeling batch."""

    p50: float | None
    p95: float | None
    p99: float | None
    max_seconds: float | None


def _labeling_latency_stats(latencies: list[float]) -> LabelingLatencyStats:
    """Return p50/p95/p99/max latencies from per-request elapsed times."""
    if not latencies:
        return LabelingLatencyStats(p50=None, p95=None, p99=None, max_seconds=None)
    ordered = sorted(latencies)
    p50 = ordered[len(ordered) // 2]
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    p99_index = max(0, math.ceil(0.99 * len(ordered)) - 1)
    return LabelingLatencyStats(
        p50=p50,
        p95=ordered[p95_index],
        p99=ordered[p99_index],
        max_seconds=ordered[-1],
    )


def _labeling_system_prompt(
    profile: ModeProfile,
    main_theme: str,
    labels: list[str],
    analyst_focus: str,
    *,
    prompt_labels: list[str] | None = None,
) -> str:
    """Format the mode-specific labeling system prompt."""
    labels_for_prompt = prompt_labels if prompt_labels is not None else labels
    if profile.mode is AnalysisMode.THEMATIC_SCREENER:
        return profile.labeling_system_prompt.format(
            main_theme=main_theme,
            analyst_focus=analyst_focus,
            labels=str(labels_for_prompt),
        )
    return profile.labeling_system_prompt.format(
        main_theme=main_theme,
        labels=str(labels_for_prompt),
    )


def _labeling_prompt_labels(
    profile: ModeProfile,
    labels: list[str],
    taxonomy_root: Node | None,
) -> list[str]:
    """Build label strings injected into the labeling prompt."""
    if profile.mode is AnalysisMode.RISK_ANALYZER and taxonomy_root is not None:
        return get_leaf_label_summary_options(taxonomy_root)
    return labels


def label_sentences(
    sentences: list[dict[str, Any]],
    main_theme: str,
    labels: list[str],
    analyst_focus: str = DEFAULT_ANALYST_FOCUS,
    model: str = DEFAULT_LABELING_MODEL,
    profile: ModeProfile | None = None,
    taxonomy_root: Node | None = None,
    requests_per_minute: int = DEFAULT_LABELING_REQUESTS_PER_MINUTE,
    max_concurrent_requests: int = DEFAULT_LABELING_MAX_CONCURRENT,
    metrics_out: dict[str, float | int | None] | None = None,
) -> dict[str, dict[str, str]]:
    """Label each sentence against the theme/risk and label set.

    Returns a flat mapping of ``sentence_id`` -> label fields.
    """
    active_profile = profile if profile is not None else _THEMATIC_PROFILE
    prompt_labels = _labeling_prompt_labels(active_profile, labels, taxonomy_root)
    system_prompt = _labeling_system_prompt(
        active_profile,
        main_theme,
        labels,
        analyst_focus,
        prompt_labels=prompt_labels,
    )
    requests = [
        ChatRequest(
            request_id=str(sentence["sentence_id"]),
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": str(_labeling_payload(sentence))},
            ],
            model=model,
            response_format={"type": "json_object"},
        )
        for sentence in sentences
    ]

    started_at = time.perf_counter()
    responses = run_chat_requests_parallel(
        requests,
        rate_limit=RateLimitConfig(
            requests_per_minute=requests_per_minute,
            max_concurrent_requests=max_concurrent_requests,
        ),
        max_retries=5,
    )
    elapsed_seconds = time.perf_counter() - started_at

    parsed_responses: dict[str, dict[str, str]] = {}
    for response in responses:
        if not response.succeeded:
            logger.warning("Labeling request %s failed: %s", response.request_id, response.error)
            continue
        try:
            payload = json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning("Could not parse labeling response for %s", response.request_id)
            continue
        if {"motivation", "label"}.issubset(payload):
            parsed_responses[response.request_id] = payload
            continue
        if len(payload) == 1 and response.request_id not in payload:
            only_fields = next(iter(payload.values()))
            if isinstance(only_fields, dict) and {"motivation", "label"}.issubset(only_fields):
                parsed_responses[response.request_id] = only_fields
                continue
        for sentence_id, fields in payload.items():
            if isinstance(fields, dict):
                parsed_responses[str(sentence_id)] = fields

    if metrics_out is not None:
        latencies = [response.elapsed_seconds for response in responses if response.succeeded]
        stats = _labeling_latency_stats(latencies)
        metrics_out.clear()
        metrics_out.update(
            {
                "elapsed_seconds": round(elapsed_seconds, 2),
                "request_count": len(requests),
                "succeeded_count": len(latencies),
                "parsed_count": len(parsed_responses),
                "requests_per_second": round(len(latencies) / elapsed_seconds, 2)
                if elapsed_seconds > 0
                else 0.0,
                "latency_p50_seconds": round(stats.p50, 2) if stats.p50 is not None else None,
                "latency_p95_seconds": round(stats.p95, 2) if stats.p95 is not None else None,
                "latency_p99_seconds": round(stats.p99, 2) if stats.p99 is not None else None,
                "latency_max_seconds": round(stats.max_seconds, 2)
                if stats.max_seconds is not None
                else None,
            }
        )

    return parsed_responses


def build_labeled_dataframe(
    sentences: list[dict[str, Any]],
    parsed_responses: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Merge sentences with their labels and drop unclear or unknown rows."""
    sentences_df = pd.DataFrame(sentences)
    sentences_df["sentence_id"] = sentences_df["sentence_id"].astype(str)

    responses_df = pd.DataFrame.from_dict(parsed_responses, orient="index").reset_index(
        names="sentence_id"
    )

    merged_df = sentences_df.merge(responses_df, on="sentence_id", how="left")
    if "label" not in merged_df.columns:
        merged_df["label"] = pd.NA
    if "materiality" not in merged_df.columns:
        merged_df["materiality"] = pd.NA

    merged_df = merged_df[
        (~merged_df["company_name"].isna()) & (merged_df["label"] != "unclear")
    ]
    return merged_df.reset_index(drop=True)


def _company_summary_system_prompt(main_theme: str, profile: ModeProfile | None = None) -> str:
    active_profile = profile if profile is not None else _THEMATIC_PROFILE
    return active_profile.company_summary_template.format(main_theme=main_theme)


def _company_evidence_block(rows: pd.DataFrame) -> str:
    evidence_lines: list[str] = []
    for row in rows.fillna("").itertuples(index=False):
        materiality = getattr(row, "materiality", "")
        label = getattr(row, "label", "")
        revenue = getattr(row, "revenue_generation", "")
        cost = getattr(row, "cost_efficiency", "")
        motivation = getattr(row, "motivation", "")
        if not str(motivation).strip():
            continue
        evidence_lines.append(

                f"- materiality={materiality}; label={label}; "
                f"revenue_generation={revenue}; cost_efficiency={cost}; "
                f"motivation={motivation}"

        )
    lines = evidence_lines
    block = "\n".join(lines)
    if len(block) > MAX_MOTIVATIONS_CHARS:
        block = (
            block[:MAX_MOTIVATIONS_CHARS]
            + "\n\n[Truncated: additional motivations omitted due to length.]"
        )
    return block


def summarize_companies(
    merged_df: pd.DataFrame,
    main_theme: str,
    model: str = DEFAULT_SUMMARY_MODEL,
    profile: ModeProfile | None = None,
    requests_per_minute: int = DEFAULT_LABELING_REQUESTS_PER_MINUTE,
    max_concurrent_requests: int = DEFAULT_SUMMARY_MAX_CONCURRENT,
) -> pd.DataFrame:
    """Produce one cohesive summary per company."""
    if "motivation" not in merged_df.columns:
        return pd.DataFrame(columns=["company_name", "summary"])

    company_motivation_rows = [
        {"company_name": company_name, "motivations_text": _company_evidence_block(group)}
        for company_name, group in merged_df.groupby("company_name", sort=True)
    ]
    company_motivations = pd.DataFrame(company_motivation_rows)
    company_motivations = company_motivations[
        company_motivations["motivations_text"].str.len() > 0
    ]

    system_prompt = _company_summary_system_prompt(main_theme, profile)
    summary_requests = [
        ChatRequest(
            request_id=row.company_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Company: {row.company_name}\n\n"
                        f"Motivations ({main_theme}):\n{row.motivations_text}"
                    ),
                },
            ],
            model=model,
            temperature=0.0,
            seed=42,
            response_format={"type": "json_object"},
        )
        for row in company_motivations.itertuples(index=False)
    ]

    summary_responses = run_chat_requests_parallel(
        summary_requests,
        rate_limit=RateLimitConfig(
            requests_per_minute=requests_per_minute,
            max_concurrent_requests=max_concurrent_requests,
        ),
        max_retries=5,
        show_progress=True,
    )

    company_summaries: list[dict[str, str]] = []
    for response in summary_responses:
        if not response.succeeded:
            logger.warning("Summary request %s failed: %s", response.request_id, response.error)
            continue
        try:
            payload = json.loads(response.content)
            summary = CompanySummary.model_validate(payload).summary
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Summary parse error for %s: %s", response.request_id, exc)
            continue
        company_summaries.append({"company_name": response.request_id, "summary": summary})

    if not company_summaries:
        return pd.DataFrame(columns=["company_name", "summary"])

    return (
        pd.DataFrame(company_summaries)
        .sort_values("company_name")
        .reset_index(drop=True)
    )


def build_screener_dataframe(
    merged_df: pd.DataFrame,
    company_summaries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join company summaries onto the labeled sentences."""
    if company_summaries_df.empty:
        result = merged_df.copy()
        result["summary"] = pd.NA
        return result
    return merged_df.merge(company_summaries_df, on="company_name", how="left")


# ---------------------------------------------------------------------------
# Risk-analyzer exports
#
# These builders convert the labeled-sentence artifacts into the JSON schema
# consumed by the Bigdata Risk Analyzer app (``risk_scoring`` / ``risk_taxonomy``
# / ``content``) and into a multi-sheet Excel workbook.
# ---------------------------------------------------------------------------


def _clean_scalar(value: Any) -> Any:
    """Return ``None`` for missing/NaN scalars, otherwise the value itself."""
    try:
        if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return value
    return value


def _split_timestamp(timestamp: Any) -> tuple[str, str]:
    """Split an ISO timestamp into ``(date, time_period)`` strings.

    ``date`` is ``YYYY-MM-DD`` and ``time_period`` is ``Mon YYYY`` (e.g. ``Jun 2026``).
    Returns empty strings when the timestamp is missing or unparsable.
    """
    if not isinstance(timestamp, str) or not timestamp:
        return "", ""
    date_part = timestamp[:10]
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return date_part, ""
    return date_part, parsed.strftime("%b %Y")


def _company_metadata_lookup(universe_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Map company name to ticker/sector/industry/country from the universe.

    Optional columns are used when present; otherwise sector/industry default to
    ``"Unknown"`` and ticker/country to ``None``.
    """
    columns = set(universe_df.columns)
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in universe_df.iterrows():
        name = row[UNIVERSE_NAME_COLUMN]
        lookup[name] = {
            "ticker": _clean_scalar(row[UNIVERSE_TICKER_COLUMN])
            if UNIVERSE_TICKER_COLUMN in columns
            else None,
            "sector": _clean_scalar(row[UNIVERSE_SECTOR_COLUMN]) or UNKNOWN_VALUE
            if UNIVERSE_SECTOR_COLUMN in columns
            else UNKNOWN_VALUE,
            "industry": _clean_scalar(row[UNIVERSE_INDUSTRY_COLUMN]) or UNKNOWN_VALUE
            if UNIVERSE_INDUSTRY_COLUMN in columns
            else UNKNOWN_VALUE,
            "country": _clean_scalar(row[UNIVERSE_COUNTRY_COLUMN])
            if UNIVERSE_COUNTRY_COLUMN in columns
            else None,
        }
    return lookup


def build_risk_taxonomy(root: Node) -> dict[str, Any]:
    """Serialize the taxonomy tree to the app's ``risk_taxonomy`` shape."""
    return root.model_dump()


def _risk_factor_channel(label: str, ancestry: dict[str, list[str]]) -> tuple[str, str]:
    """Derive ``(risk_factor, risk_channel)`` for a leaf sub-scenario.

    ``risk_factor`` is the immediate parent label and ``risk_channel`` the
    grandparent, falling back gracefully for shallow trees.
    """
    chain = ancestry.get(label, [])
    risk_factor = chain[-1] if chain else label
    risk_channel = chain[-2] if len(chain) >= 2 else risk_factor
    return risk_factor, risk_channel


def build_content_chunks(
    screener_df: pd.DataFrame,
    root: Node,
    universe_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build the ``content`` array of labeled chunks for the app JSON."""
    ancestry = build_leaf_ancestry(root)
    metadata = _company_metadata_lookup(universe_df)

    chunks: list[dict[str, Any]] = []
    for _, row in screener_df.iterrows():
        label = _clean_scalar(row.get("label")) or ""
        company = _clean_scalar(row.get("company_name")) or ""
        date, time_period = _split_timestamp(row.get("timestamp"))
        risk_factor, risk_channel = _risk_factor_channel(label, ancestry)
        company_meta = metadata.get(company, {})

        chunks.append(
            {
                "time_period": time_period,
                "date": date,
                "company": company,
                "sector": company_meta.get("sector", UNKNOWN_VALUE),
                "industry": company_meta.get("industry", UNKNOWN_VALUE),
                "country": company_meta.get("country"),
                "ticker": company_meta.get("ticker"),
                "document_id": _clean_scalar(row.get("document_id")) or "",
                "headline": _clean_scalar(row.get("headline")) or "",
                "quote": _clean_scalar(row.get("text")) or "",
                "motivation": _clean_scalar(row.get("motivation")) or "",
                "sub_scenario": label,
                "risk_channel": risk_channel,
                "risk_factor": risk_factor,
                "highlights": [],
            }
        )
    return chunks


def build_risk_scoring(
    screener_df: pd.DataFrame,
    universe_df: pd.DataFrame,
) -> dict[str, Any]:
    """Aggregate per-company risk scoring (counts) for the app JSON."""
    metadata = _company_metadata_lookup(universe_df)
    scoring: dict[str, Any] = {}

    summaries: dict[str, Any] = {}
    if "summary" in screener_df.columns:
        for company, group in screener_df.groupby("company_name", sort=True):
            summaries[company] = _clean_scalar(group["summary"].iloc[0])

    for company, group in screener_df.groupby("company_name", sort=True):
        counts = group["label"].value_counts()
        risks = {str(label): int(count) for label, count in counts.items()}
        company_meta = metadata.get(company, {})
        scoring[str(company)] = {
            "ticker": company_meta.get("ticker"),
            "sector": company_meta.get("sector", UNKNOWN_VALUE),
            "industry": company_meta.get("industry", UNKNOWN_VALUE),
            "composite_score": int(sum(risks.values())),
            "motivation": summaries.get(company),
            "risks": risks,
        }
    return scoring


def build_risk_analysis_json(
    screener_df: pd.DataFrame,
    root: Node,
    universe_df: pd.DataFrame,
) -> dict[str, Any]:
    """Assemble the full risk-format payload (``risk_scoring`` / ``risk_taxonomy`` / ``content``)."""
    return {
        "risk_scoring": build_risk_scoring(screener_df, universe_df),
        "risk_taxonomy": build_risk_taxonomy(root),
        "content": build_content_chunks(screener_df, root, universe_df),
    }


def build_theme_scoring(
    screener_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    all_labels: list[str],
) -> dict[str, Any]:
    """Aggregate per-company theme scoring for the theme-format app JSON.

    Every company's ``themes`` map lists the full leaf-label set (0 when a
    label is absent) and ``composite_score`` is the total count of relevant
    quotes.
    """
    metadata = _company_metadata_lookup(universe_df)

    summaries: dict[str, Any] = {}
    if "summary" in screener_df.columns:
        for company, group in screener_df.groupby("company_name", sort=True):
            summaries[company] = _clean_scalar(group["summary"].iloc[0])

    scoring: dict[str, Any] = {}
    for company, group in screener_df.groupby("company_name", sort=True):
        counts = group["label"].value_counts().to_dict()
        themes = {str(label): int(counts.get(label, 0)) for label in all_labels}
        company_meta = metadata.get(company, {})
        scoring[str(company)] = {
            "ticker": company_meta.get("ticker"),
            "industry": company_meta.get("industry", UNKNOWN_VALUE),
            "motivation": summaries.get(company),
            "composite_score": int(sum(themes.values())),
            "themes": themes,
        }
    return scoring


def build_theme_content(
    screener_df: pd.DataFrame,
    universe_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build the ``content`` array of labeled chunks for the theme-format JSON."""
    metadata = _company_metadata_lookup(universe_df)

    chunks: list[dict[str, Any]] = []
    for _, row in screener_df.iterrows():
        company = _clean_scalar(row.get("company_name")) or ""
        date, time_period = _split_timestamp(row.get("timestamp"))
        company_meta = metadata.get(company, {})

        chunks.append(
            {
                "time_period": time_period,
                "date": date,
                "company": company,
                "sector": company_meta.get("sector", UNKNOWN_VALUE),
                "industry": company_meta.get("industry", UNKNOWN_VALUE),
                "country": company_meta.get("country"),
                "ticker": company_meta.get("ticker"),
                "document_id": _clean_scalar(row.get("document_id")) or "",
                "headline": _clean_scalar(row.get("headline")) or "",
                "quote": _clean_scalar(row.get("text")) or "",
                "motivation": _clean_scalar(row.get("motivation")) or "",
                "theme": _clean_scalar(row.get("label")) or "",
            }
        )
    return chunks


def build_theme_report(
    screener_df: pd.DataFrame,
    root: Node,
    universe_df: pd.DataFrame,
) -> dict[str, Any]:
    """Assemble the full theme-format payload (``theme_scoring`` / ``theme_taxonomy`` / ``content``)."""
    all_labels = get_leaf_labels(root)
    return {
        "theme_scoring": build_theme_scoring(screener_df, universe_df, all_labels),
        "theme_taxonomy": build_risk_taxonomy(root),
        "content": build_theme_content(screener_df, universe_df),
    }


def build_report_json(
    screener_df: pd.DataFrame,
    root: Node,
    universe_df: pd.DataFrame,
    mode: AnalysisMode | str,
) -> dict[str, Any]:
    """Build the app-uploadable report JSON in the shape matching ``mode``.

    Risk mode emits the ``risk_scoring`` schema; thematic mode emits the
    ``theme_scoring`` schema.
    """
    resolved = AnalysisMode(mode) if not isinstance(mode, AnalysisMode) else mode
    if resolved is AnalysisMode.RISK_ANALYZER:
        return build_risk_analysis_json(screener_df, root, universe_df)
    return build_theme_report(screener_df, root, universe_df)


def build_company_scoring_df(screener_df: pd.DataFrame) -> pd.DataFrame:
    """Build a company x label count matrix with a composite-score column."""
    if screener_df.empty or "label" not in screener_df.columns:
        return pd.DataFrame(columns=["company_name", "Composite Score"])

    counts = (
        screener_df.groupby(["company_name", "label"]).size().reset_index(name="count")
    )
    pivot = (
        counts.pivot(index="company_name", columns="label", values="count")
        .fillna(0)
        .astype(int)
    )
    pivot["Composite Score"] = pivot.sum(axis=1)
    return pivot.sort_values("Composite Score", ascending=False).reset_index()


def build_taxonomy_df(root: Node) -> pd.DataFrame:
    """Flatten the taxonomy tree into rows (node, label, summary, parent, depth)."""
    rows: list[dict[str, Any]] = []

    def _walk(node: Node, parent: str | None, depth: int) -> None:
        rows.append(
            {
                "node": node.node,
                "label": node.label,
                "summary": node.summary,
                "parent": parent,
                "depth": depth,
            }
        )
        for child in node.children:
            _walk(child, node.label, depth + 1)

    _walk(root, None, 0)
    return pd.DataFrame(rows)


EXCEL_MAX_CELL_CHARS = 32767
EXCEL_TRUNCATION_NOTICE = " [...truncated for Excel; full text in the CSV and JSON exports]"


def _fit_excel_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim text cells that exceed Excel's per-cell character limit.

    A single retrieved passage can run past the limit, in which case Excel
    truncates it silently. Marking the cut keeps the workbook honest about
    where the full text still lives.
    """
    result = frame.copy()
    budget = EXCEL_MAX_CELL_CHARS - len(EXCEL_TRUNCATION_NOTICE)
    for column in result.columns:
        if not (
            pd.api.types.is_string_dtype(result[column])
            or pd.api.types.is_object_dtype(result[column])
        ):
            continue
        values = result[column].astype("string")
        oversized = values.str.len() > EXCEL_MAX_CELL_CHARS
        if not oversized.any():
            continue
        result[column] = values.mask(
            oversized, values.str.slice(0, budget) + EXCEL_TRUNCATION_NOTICE
        )
    return result


def export_excel(
    screener_df: pd.DataFrame,
    company_summaries_df: pd.DataFrame,
    root: Node,
    path: str | Path,
) -> Path:
    """Write a multi-sheet Excel workbook summarizing the run."""
    output_path = Path(path)
    scoring_df = build_company_scoring_df(screener_df)
    taxonomy_df = build_taxonomy_df(root)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        _fit_excel_cells(screener_df).to_excel(writer, sheet_name="Results", index=False)
        if not company_summaries_df.empty:
            _fit_excel_cells(company_summaries_df).to_excel(
                writer, sheet_name="Company Summaries", index=False
            )
        scoring_df.to_excel(writer, sheet_name="Company Scoring", index=False)
        _fit_excel_cells(taxonomy_df).to_excel(writer, sheet_name="Taxonomy", index=False)

    return output_path
