"""Pipeline functions for the thematic screener.

Pure building blocks with no CLI parsing or run-directory logic.
Orchestration and persistence are handled by :mod:`src.cli` together with
:class:`src.run_context.RunContext`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from bigdata_smart_batching import (
    convert_to_dataframe,  # noqa: F401  (kept available for downstream use)
    deduplicate_documents,
    execute_search,
    load_plan,
    plan_search,
    save_plan,
)

from src.helpers import get_leaf_summaries, print_tree
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
DEFAULT_LABELS_MODEL = "gpt-4o"
DEFAULT_LABELING_MODEL = "gpt-4o-mini"
DEFAULT_SUMMARY_MODEL = "gpt-4o-mini"
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
    children: list["Node"] = Field(default_factory=list)


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
) -> list[str]:
    """Generate leaf-level theme labels for ``main_theme`` via the LLM.

    Prompts the model for a taxonomy tree, then extracts the summaries of
    all leaf nodes as the working label set.
    """
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

    return get_leaf_summaries(root)


def plan_filename(label: str) -> str:
    """Return the plan JSON filename for a label."""
    return f"{label.replace(' ', '_').replace('.', '')}.json"


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
    company_ids: list[str],
    plans_dir: str | Path,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    category: dict[str, Any] | None = None,
) -> list[Path]:
    """Create and persist one search plan per label."""
    plans_path = Path(plans_dir)
    plans_path.mkdir(parents=True, exist_ok=True)
    search_category = category if category is not None else DEFAULT_SEARCH_CATEGORY

    saved_paths: list[Path] = []
    for label in labels:
        plan = plan_search(
            universe=company_ids,
            start_date=start_date,
            end_date=end_date,
            volume_query_mode="iterative",
            text=label,
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
        raise FileNotFoundError(f"no plan files found in {plans_path}. Please run the 'plans' command first.")

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
            sentence["company_name"] = id_to_name.get(first_entity)
        sentence.pop("entity", None)

    return sentences


def label_sentences(
    sentences: list[dict[str, Any]],
    main_theme: str,
    labels: list[str],
    model: str = DEFAULT_LABELING_MODEL,
    requests_per_minute: int = 10000,
    max_concurrent_requests: int = 20,
) -> dict[str, dict[str, str]]:
    """Label each sentence against the theme and label set.

    Returns a flat mapping of ``sentence_id`` -> label fields.
    """
    requests = [
        ChatRequest(
            request_id=str(sentence["sentence_id"]),
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_LABELING.format(
                        main_theme=main_theme, labels=str(labels)
                    ),
                },
                {"role": "user", "content": str(sentence)},
            ],
            model=model,
            response_format={"type": "json_object"},
        )
        for sentence in sentences
    ]

    responses = run_chat_requests_parallel(
        requests,
        rate_limit=RateLimitConfig(
            requests_per_minute=requests_per_minute,
            max_concurrent_requests=max_concurrent_requests,
        ),
        max_retries=5,
    )

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
        for sentence_id, fields in payload.items():
            parsed_responses[str(sentence_id)] = fields

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
- Highlights the most important products, markets, and revenue/cost drivers when mentioned
- Avoids repeating the same point; merge overlapping motivations
- Uses clear, professional prose (1 short paragraph)
- Does not invent facts beyond what the motivations support

Return JSON only: {{"summary": "<your summary>"}}"""


def _motivations_block(motivations: pd.Series) -> str:
    lines = [f"- {text.strip()}" for text in motivations.dropna().astype(str) if text.strip()]
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

    company_motivations = (
        merged_df.groupby("company_name", sort=True)["motivation"]
        .apply(_motivations_block)
        .reset_index(name="motivations_text")
    )
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
