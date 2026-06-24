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
from pathlib import Path
from typing import Any

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
    get_leaf_labels,
    get_leaf_search_queries,
    print_tree,
)
from src.openai_parallel import (
    ChatRequest,
    RateLimitConfig,
    run_chat_requests_parallel,
)
from src.prompts import (
    SYSTEM_MESSAGE_LABELS,
    SYSTEM_PROMPT_LABELING,
    USER_MESSAGE_LABELS,
)

logger = logging.getLogger(__name__)

DEFAULT_MAIN_THEME = "AI disruption in product development"
DEFAULT_ANALYST_FOCUS = "How companies are including AI in their development cycle"
DEFAULT_START_DATE = "2025-06-01"
DEFAULT_END_DATE = "2026-06-09"
DEFAULT_LABELS_MODEL = "gpt-5.4-nano"
DEFAULT_LABELING_MODEL = "gpt-5.4-nano"
DEFAULT_SUMMARY_MODEL = "gpt-5.4-nano"
DEFAULT_CHUNK_PERCENTAGE = 0.02
DEFAULT_REQUESTS_PER_MINUTE = 350
DEFAULT_SEARCH_CATEGORY: dict[str, Any] = {
    "mode": "INCLUDE",
    "values": ["news_premium", "transcripts", "filings"],
}

UNIVERSE_ID_COLUMN = "RP_COMPANY_ID"
UNIVERSE_NAME_COLUMN = "COMPANY_NAME"

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


def generate_labels(
    main_theme: str,
    analyst_focus: str,
    model: str = DEFAULT_LABELS_MODEL,
    client: OpenAI | None = None,
    print_taxonomy: bool = True,
) -> Node:
    """Generate a taxonomy tree for ``main_theme`` via the LLM."""
    openai_client = client if client is not None else OpenAI()
    completion = openai_client.chat.completions.parse(
        model=model,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=42,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": SYSTEM_MESSAGE_LABELS.format(
                    main_theme=main_theme, analyst_focus=analyst_focus
                ),
            },
            {
                "role": "user",
                "content": USER_MESSAGE_LABELS.format(main_theme=main_theme),
            },
        ],
    )

    content = completion.choices[0].message.content
    root = Node.model_validate_json(content)

    if print_taxonomy:
        print_tree(root)

    return root


def write_taxonomy_artifacts(
    root: Node,
    themes_path: Path,
    search_queries_path: Path,
    taxonomy_tree_path: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Persist taxonomy tree, leaf labels, and leaf search queries."""
    labels = get_leaf_labels(root)
    search_queries = get_leaf_search_queries(root)
    if len(labels) != len(search_queries):
        raise ValueError("Leaf label and search_query counts must match")

    themes_path.parent.mkdir(parents=True, exist_ok=True)
    themes_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    search_queries_path.write_text("\n".join(search_queries) + "\n", encoding="utf-8")
    if taxonomy_tree_path is not None:
        taxonomy_tree_path.write_text(root.model_dump_json(indent=2), encoding="utf-8")
    return labels, search_queries


def plan_filename(label: str) -> str:
    """Return the plan JSON filename for a label."""
    safe_label = "".join(character if character.isalnum() else "_" for character in label)
    safe_label = "_".join(part for part in safe_label.split("_") if part)
    return f"{safe_label}.json"


def load_universe(universe_path: str | Path) -> pd.DataFrame:
    """Load the company universe CSV.

    The CSV is expected to expose ``RP_COMPANY_ID`` and ``COMPANY_NAME``
    columns (as in ``XNAS_companies.csv``).
    """
    universe_df = pd.read_csv(universe_path)
    missing = {UNIVERSE_ID_COLUMN, UNIVERSE_NAME_COLUMN} - set(universe_df.columns)
    if missing:
        raise ValueError(
            f"universe file {universe_path} is missing required columns: {sorted(missing)}"
        )
    return universe_df.reset_index(drop=True)


def build_plans(
    labels: list[str],
    search_queries: list[str],
    company_ids: list[str],
    plans_dir: str | Path,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    category: dict[str, Any] | None = None,
) -> list[Path]:
    """Create and persist one search plan per label/search-query pair."""
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
) -> list[dict[str, Any]]:
    """Flatten retrieved documents into labeled sentence records.

    Each chunk becomes a sentence with an id, text, and resolved company
    name (looked up from the universe DataFrame).
    """
    sentences: list[dict[str, Any]] = []
    idx = 0
    for document in results:
        for chunk in document.get("chunks", []):
            sentences.append(
                {
                    "sentence_id": idx,
                    "text": chunk.get("text"),
                    "entity": chunk.get("entity_ids"),
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


def label_sentences(
    sentences: list[dict[str, Any]],
    main_theme: str,
    labels: list[str],
    analyst_focus: str = DEFAULT_ANALYST_FOCUS,
    model: str = DEFAULT_LABELING_MODEL,
    requests_per_minute: int = 10000,
    max_concurrent_requests: int = 40,
    metrics_out: dict[str, float | int | None] | None = None,
) -> dict[str, dict[str, str]]:
    """Label each sentence against the theme, analyst focus, and label set.

    Returns a flat mapping of ``sentence_id`` -> label fields.
    """
    system_prompt = SYSTEM_PROMPT_LABELING.format(
        main_theme=main_theme,
        analyst_focus=analyst_focus,
        labels=str(labels),
    )
    requests = [
        ChatRequest(
            request_id=str(sentence["sentence_id"]),
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": str(sentence)},
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


def _company_summary_system_prompt(main_theme: str) -> str:
    return f"""You are assisting a professional analyst evaluating how the theme
"{main_theme}" affects companies.

You will receive a company name and a list of analyst motivations. Each motivation explains why a
specific sentence was labeled in the context of the theme.

Write one cohesive company-level summary that:
- Synthesizes the main themes and business exposures implied by the motivations
- Prioritizes high-materiality evidence over medium and low materiality evidence
- Highlights the most important products, markets, revenue/cost drivers, and risks when mentioned
- Avoids repeating the same point; merge overlapping motivations
- Uses clear, professional prose (1 short paragraph)
- Does not invent facts beyond what the motivations support

Return JSON only: {{"summary": "<your summary>"}}"""


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
    requests_per_minute: int = 10000,
    max_concurrent_requests: int = 20,
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

    system_prompt = _company_summary_system_prompt(main_theme)
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
