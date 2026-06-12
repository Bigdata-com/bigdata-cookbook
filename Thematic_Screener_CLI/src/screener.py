"""Pipeline functions for the thematic screener.

Pure building blocks with no CLI parsing or run-directory logic.
Orchestration and persistence are handled by :mod:`src.cli` together with
:class:`src.run_context.RunContext`.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
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

from src.helpers import (
    build_leaf_ancestry,
    get_leaf_labels,
    get_leaf_summaries,
    print_tree,
)
from src.modes import AnalysisMode, LeafField, ModeProfile, get_profile
from src.openai_parallel import (
    ChatRequest,
    RateLimitConfig,
    run_chat_requests_parallel,
)

logger = logging.getLogger(__name__)

_THEMATIC_PROFILE = get_profile(AnalysisMode.THEMATIC_SCREENER)

DEFAULT_MAIN_THEME = _THEMATIC_PROFILE.default_main_theme
DEFAULT_ANALYST_FOCUS = _THEMATIC_PROFILE.default_analyst_focus
DEFAULT_START_DATE = "2025-06-01"
DEFAULT_END_DATE = "2026-06-09"
DEFAULT_LABELS_MODEL = "gpt-4o"
DEFAULT_LABELING_MODEL = "gpt-4o-mini"
DEFAULT_SUMMARY_MODEL = "gpt-4o-mini"
DEFAULT_CHUNK_PERCENTAGE = 0.02
DEFAULT_REQUESTS_PER_MINUTE = 350
DEFAULT_RERANK_THRESHOLD = 0.0
DEFAULT_SEARCH_CATEGORY: dict[str, Any] = {
    "mode": "INCLUDE",
    "values": ["news_premium", "transcripts", "filings"],
}

UNIVERSE_ID_COLUMN = "RP_COMPANY_ID"
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
    children: list["Node"] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


Node.model_rebuild()


class CompanySummary(BaseModel):
    """Structured company-level summary returned by the LLM."""

    summary: str


def generate_taxonomy(
    main_theme: str,
    analyst_focus: str,
    model: str = DEFAULT_LABELS_MODEL,
    profile: ModeProfile | None = None,
    client: OpenAI | None = None,
    print_taxonomy: bool = True,
) -> Node:
    """Generate the sub-concept taxonomy tree for ``main_theme`` via the LLM.

    The mode ``profile`` selects the prompts so the same call produces a theme
    taxonomy (thematic-screener) or a risk taxonomy (risk-analyzer).
    """
    active_profile = profile if profile is not None else _THEMATIC_PROFILE
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
                "content": active_profile.labels_system_prompt.format(
                    main_theme=main_theme, analyst_focus=analyst_focus
                ),
            },
            {
                "role": "user",
                "content": active_profile.labels_user_prompt.format(main_theme=main_theme),
            },
        ],
    )

    content = completion.choices[0].message.content
    root = Node.model_validate_json(content)

    if print_taxonomy:
        print_tree(root)

    return root


def leaf_labels(root: Node, profile: ModeProfile | None = None) -> list[str]:
    """Extract the working label set from the taxonomy per the mode profile.

    Thematic mode uses leaf ``summary`` values (historical behavior); risk mode
    uses leaf ``label`` values so they line up with the taxonomy and scoring.
    """
    active_profile = profile if profile is not None else _THEMATIC_PROFILE
    if active_profile.leaf_field is LeafField.LABEL:
        return get_leaf_labels(root)
    return get_leaf_summaries(root)


def generate_labels(
    main_theme: str,
    analyst_focus: str,
    model: str = DEFAULT_LABELS_MODEL,
    profile: ModeProfile | None = None,
    client: OpenAI | None = None,
    print_taxonomy: bool = True,
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
    )
    return leaf_labels(root, profile)


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
            sentence["company_name"] = id_to_name.get(first_entity)
        sentence.pop("entity", None)

    return sentences


_LABELING_PAYLOAD_FIELDS = ("sentence_id", "text", "company_name")


def _labeling_payload(sentence: dict[str, Any]) -> dict[str, Any]:
    """Project a sentence onto the fields shown to the labeling model.

    Document metadata kept for downstream export (headline, timestamp, ...) is
    intentionally excluded so the labeling prompt is identical to its historical
    form.
    """
    return {key: sentence[key] for key in _LABELING_PAYLOAD_FIELDS if key in sentence}


def label_sentences(
    sentences: list[dict[str, Any]],
    main_theme: str,
    labels: list[str],
    model: str = DEFAULT_LABELING_MODEL,
    profile: ModeProfile | None = None,
    requests_per_minute: int = 10000,
    max_concurrent_requests: int = 20,
) -> dict[str, dict[str, str]]:
    """Label each sentence against the theme/risk and label set.

    Returns a flat mapping of ``sentence_id`` -> label fields.
    """
    active_profile = profile if profile is not None else _THEMATIC_PROFILE
    requests = [
        ChatRequest(
            request_id=str(sentence["sentence_id"]),
            messages=[
                {
                    "role": "system",
                    "content": active_profile.labeling_system_prompt.format(
                        main_theme=main_theme, labels=str(labels)
                    ),
                },
                {"role": "user", "content": str(_labeling_payload(sentence))},
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


def _company_summary_system_prompt(main_theme: str, profile: ModeProfile | None = None) -> str:
    active_profile = profile if profile is not None else _THEMATIC_PROFILE
    return active_profile.company_summary_template.format(main_theme=main_theme)


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
    profile: ModeProfile | None = None,
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
        screener_df.to_excel(writer, sheet_name="Results", index=False)
        if not company_summaries_df.empty:
            company_summaries_df.to_excel(writer, sheet_name="Company Summaries", index=False)
        scoring_df.to_excel(writer, sheet_name="Company Scoring", index=False)
        taxonomy_df.to_excel(writer, sheet_name="Taxonomy", index=False)

    return output_path
