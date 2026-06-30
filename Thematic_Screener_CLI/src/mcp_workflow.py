"""Workflow services backing the Thematic Screener MCP server."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src import screener
from src.prompts import SYSTEM_MESSAGE_LABELS, SYSTEM_PROMPT_LABELING, USER_MESSAGE_LABELS
from src.run_context import RunContext
from src.screener import _company_evidence_block, _company_summary_system_prompt
from src.search_query import has_exposure_meta_language

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_GLOBAL_UNIVERSE = PROJECT_ROOT / "global_all_caps.csv"
DEFAULT_EUROPE_ML_CAPS_UNIVERSE = PROJECT_ROOT / "europe_ml_caps.csv"
SAMPLE_XNAS_UNIVERSE = PROJECT_ROOT / "XNAS_companies.csv"

RETRIEVAL_CHUNKS_PER_COST_UNIT = 10
RETRIEVAL_COST_USD_PER_UNIT = 0.015
DEFAULT_ENRICHMENT_REQUESTS_PER_MINUTE = 10_000
DEFAULT_ENRICHMENT_MAX_CONCURRENT = 40
CHARS_PER_TOKEN_ESTIMATE = 4
LABELING_OUTPUT_TOKENS_ESTIMATE = 180
SUMMARY_OUTPUT_TOKENS_ESTIMATE = 120
SENTENCE_RETENTION_FACTOR = 0.45
SUMMARY_COMPANY_RETENTION_FACTOR = 0.55
ESTIMATED_MOTIVATION_CHARS_PER_SENTENCE = 250
DEFAULT_LABELING_INPUT_USD_PER_MTOK = 0.10
DEFAULT_LABELING_OUTPUT_USD_PER_MTOK = 0.40
DEFAULT_SUMMARY_INPUT_USD_PER_MTOK = 0.10
DEFAULT_SUMMARY_OUTPUT_USD_PER_MTOK = 0.40
ENRICHMENT_COST_SAFETY_MARGIN = 1.5
MIN_ESTIMATED_COST_USD = 0.01
ENRICHMENT_COST_DECIMALS = 3
# Claude Desktop MCP tool calls commonly time out around 4 minutes.
MCP_CLIENT_TOOL_TIMEOUT_SECONDS = 240
MCP_ENRICHMENT_SPLIT_THRESHOLD_SECONDS = 180
# Conservative MCP batch budget sized for straggler/tail latency (not average throughput).
MCP_LABELING_BATCH_SAFE_SECONDS = 90
LABELING_BATCH_TARGET_SECONDS = MCP_LABELING_BATCH_SAFE_SECONDS
MAX_LABELING_BATCH_SIZE = 120
# Measured on gpt-5.4-nano labeling (Jun 2026): ~4s average per wave; ~30s straggler per wave
# (p99/max; one 150-batch run hit 58s wall with conc=40).
MCP_LABELING_WAVE_SECONDS = 4.0
MCP_LABELING_WAVE_SECONDS_STRAGGLER = 30.0
MCP_LABELING_STRAGGLER_SAFETY_MARGIN = 1.25
# Backward-compatible aliases used by older docs/tests.
MCP_LABELING_WAVE_SECONDS_P95 = MCP_LABELING_WAVE_SECONDS_STRAGGLER
MCP_LABELING_SECONDS_PER_REQUEST = MCP_LABELING_WAVE_SECONDS_STRAGGLER
LABELING_PROGRESS_FILENAME = "labeling_progress.json"
LABELING_RESPONSES_FILENAME = "labeling_responses.json"
LABELING_RESPONSES_JSONL_FILENAME = "labeling_responses.jsonl"
MIN_LABELING_BATCH_SIZE = 25
AVERAGE_LABELING_REQUEST_SECONDS = MCP_LABELING_WAVE_SECONDS
BASE_SUMMARY_REQUEST_SECONDS = 3.0
SUMMARY_SECONDS_PER_1K_INPUT_CHARS = 0.15
# Deprecated flat-rate constants kept for backward-compatible overrides in tests/docs.
COST_PER_LABELING_REQUEST_USD = 0.001
COST_PER_SUMMARY_REQUEST_USD = 0.002
AVERAGE_OPENAI_REQUEST_SECONDS = AVERAGE_LABELING_REQUEST_SECONDS
DEFAULT_PRESETS: tuple[dict[str, float | str], ...] = (
    {"name": "quick_scan", "chunk_percentage": 0.005},
    {"name": "balanced", "chunk_percentage": 0.02},
    {"name": "deep_dive", "chunk_percentage": 0.05},
)

UNIVERSE_FILENAME = "universe.csv"
UNIVERSE_MANIFEST_FILENAME = "universe_manifest.json"
TAXONOMY_TREE_FILENAME = "taxonomy_tree.json"
MINDMAP_VALIDATION_FILENAME = "mindmap_validation.json"
BUDGET_SUMMARY_FILENAME = "budget_summary.json"
BUDGET_APPROVAL_FILENAME = "budget_approval.json"
ENRICHMENT_SUMMARY_FILENAME = "enrichment_summary.json"
ENRICHMENT_APPROVAL_FILENAME = "enrichment_approval.json"
EVIDENCE_DIGEST_FILENAME = "evidence_digest.json"
EVIDENCE_PREVIEW_FILENAME = "evidence_preview.json"
ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"

ENTITY_ID_ALIASES = ("RP_ENTITY_ID", screener.UNIVERSE_ID_COLUMN)
OPTIONAL_NAME_ALIASES = (screener.UNIVERSE_NAME_COLUMN, "NAME", "COMPANY")
DEFAULT_SEARCH_CATEGORY: dict[str, Any] = screener.DEFAULT_SEARCH_CATEGORY
TEXT_PREVIEW_CHARS = 12_000
CSV_PREVIEW_ROWS = 20
CSV_PREVIEW_COLUMNS = 30
JSON_PREVIEW_ITEMS = 20
REPRESENTATIVE_CHUNK_TEXT_CHARS = 900
QUERY_TEXT_PREVIEW_CHARS = 900
LARGE_ARTIFACT_THRESHOLD_BYTES = 1_048_576
MAX_TOOL_RESPONSE_BYTES = 900_000
STOP_WORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "been",
    "being",
    "but",
    "can",
    "company",
    "from",
    "have",
    "into",
    "its",
    "more",
    "not",
    "our",
    "their",
    "the",
    "this",
    "that",
    "they",
    "with",
    "will",
    "would",
}

StageStatus = Literal["created", "pending_approval", "running", "completed", "failed"]


class McpWorkflowError(Exception):
    """Raised when an MCP workflow stage cannot proceed."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _context(run_id: str | None) -> RunContext:
    context = RunContext.create(DEFAULT_RUNS_ROOT, run_id)
    context.ensure_run_dir()
    return context


def _run_relative(context: RunContext, path: Path) -> str:
    return str(path.relative_to(context.run_dir))


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _read_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload: dict[str, Any] | list[dict[str, Any]] = json.load(handle)
    return payload


def _artifact_handle(
    artifact_id: str,
    path: str,
    preview_available: bool = True,
    queryable: bool = False,
    description: str | None = None,
) -> dict[str, Any]:
    handle: dict[str, Any] = {
        "artifact_id": artifact_id,
        "path": path,
        "preview_available": preview_available,
        "queryable": queryable,
    }
    if description:
        handle["description"] = description
    return handle


def _response(
    run_id: str,
    stage: str,
    status: StageStatus,
    summary: str,
    artifacts: list[dict[str, Any]] | None = None,
    next_actions: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "stage": stage,
        "status": status,
        "summary": summary,
    }
    if artifacts is not None:
        payload["artifacts"] = artifacts
    if next_actions is not None:
        payload["next_actions"] = next_actions
    payload.update(extra)
    return _cap_tool_response(payload)


def _payload_size_bytes(payload: Any) -> int:
    return len(json.dumps(payload, default=str).encode("utf-8"))


def _cap_tool_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep MCP tool payloads under Claude's practical inline size limit."""
    if _payload_size_bytes(payload) <= MAX_TOOL_RESPONSE_BYTES:
        return payload

    capped = dict(payload)
    for key in (
        "preview",
        "rows",
        "artifacts",
        "representative_chunks",
        "digest_preview",
        "mindmap_preview",
        "review_pack_preview",
        "heatmap_preview",
    ):
        value = capped.get(key)
        if not isinstance(value, list):
            continue
        trimmed = list(value)
        while trimmed and _payload_size_bytes({**capped, key: trimmed}) > MAX_TOOL_RESPONSE_BYTES:
            trimmed = trimmed[:-1]
        capped[key] = trimmed
        capped["response_truncated"] = True

    if _payload_size_bytes(capped) > MAX_TOOL_RESPONSE_BYTES:
        capped["response_truncated"] = True
        capped["access_note"] = (
            "Response exceeded the 1MB MCP inline limit. Use smaller limits, "
            "query_artifact with filters, or export_artifact for the file path."
        )
    return capped


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[: max_chars - 3].rstrip()}..."


def _compact_query_row(row: dict[str, Any]) -> dict[str, Any]:
    compact = dict(row)
    if "text" in compact:
        compact["text"] = _truncate_text(str(compact["text"]), QUERY_TEXT_PREVIEW_CHARS)
    if "motivation" in compact:
        compact["motivation"] = _truncate_text(str(compact["motivation"]), 600)
    return compact


def _artifact_path(context: RunContext, artifact_id: str) -> Path:
    paths = {
        "run_config": context.config_path,
        "universe_manifest": context.run_dir / UNIVERSE_MANIFEST_FILENAME,
        "normalized_universe": context.run_dir / UNIVERSE_FILENAME,
        "taxonomy_tree": context.run_dir / TAXONOMY_TREE_FILENAME,
        "themes": context.themes_path,
        "search_queries": context.search_queries_path,
        "mindmap_validation": context.run_dir / MINDMAP_VALIDATION_FILENAME,
        "plans": context.plans_dir,
        "budget_summary": context.run_dir / BUDGET_SUMMARY_FILENAME,
        "budget_approval": context.run_dir / BUDGET_APPROVAL_FILENAME,
        "enrichment_summary": context.run_dir / ENRICHMENT_SUMMARY_FILENAME,
        "enrichment_approval": context.run_dir / ENRICHMENT_APPROVAL_FILENAME,
        "results": context.results_path,
        "evidence_digest": context.run_dir / EVIDENCE_DIGEST_FILENAME,
        "evidence_preview": context.run_dir / EVIDENCE_PREVIEW_FILENAME,
        "labeled_sentences": context.labeled_sentences_path,
        "company_summaries": context.company_summaries_path,
        "screener_results": context.screener_results_path,
        "artifact_manifest": context.run_dir / ARTIFACT_MANIFEST_FILENAME,
    }
    if artifact_id not in paths:
        raise McpWorkflowError(f"Unknown artifact_id: {artifact_id}")
    return paths[artifact_id]


def _path_content_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".csv":
        return "text/csv"
    if path.suffix == ".txt":
        return "text/plain"
    if path.suffix == ".png":
        return "image/png"
    if path.is_dir():
        return "inode/directory"
    return "application/octet-stream"


def _record_for_artifact(context: RunContext, artifact_id: str) -> dict[str, Any] | None:
    path = _artifact_path(context, artifact_id)
    if not path.exists():
        return None

    record: dict[str, Any] = {
        "artifact_id": artifact_id,
        "run_id": context.run_name,
        "stage": _stage_for_artifact(artifact_id),
        "path": _run_relative(context, path),
        "content_type": _path_content_type(path),
        "created_at": _utc_now(),
        "preview_available": artifact_id != "plans" or path.is_dir(),
        "queryable": artifact_id
        in {
            "results",
            "evidence_digest",
            "labeled_sentences",
            "company_summaries",
            "screener_results",
        },
    }

    if path.is_file():
        size_bytes = path.stat().st_size
        record["size_bytes"] = size_bytes
        if size_bytes > LARGE_ARTIFACT_THRESHOLD_BYTES:
            record["large_artifact"] = True
            record["access_note"] = (
                "Artifact exceeds 1MB. Use query_artifact with filters or export_artifact; "
                "do not request full inline previews."
            )
        if path.suffix == ".csv":
            try:
                record["row_count"] = int(sum(1 for _line in path.open(encoding="utf-8")) - 1)
            except OSError:
                record["row_count"] = None
        if artifact_id == "results":
            results = _load_results(context)
            record["document_count"] = len(results)
            record["chunk_count"] = len(_flatten_result_chunks(results))
    if path.is_dir():
        record["file_count"] = len(
            [file_path for file_path in path.glob("*") if file_path.is_file()]
        )

    return record


def _stage_for_artifact(artifact_id: str) -> str:
    stage_by_artifact = {
        "run_config": "intent_capture",
        "universe_manifest": "intent_capture",
        "normalized_universe": "intent_capture",
        "taxonomy_tree": "mindmap_creation",
        "themes": "mindmap_creation",
        "search_queries": "mindmap_creation",
        "mindmap_validation": "mindmap_validation",
        "plans": "planning",
        "budget_summary": "budget_preview",
        "budget_approval": "budget_approval",
        "enrichment_summary": "enrichment_preview",
        "enrichment_approval": "enrichment_approval",
        "results": "retrieval",
        "evidence_digest": "evidence_digest",
        "evidence_preview": "evidence_digest",
        "labeled_sentences": "labeling",
        "company_summaries": "company_summaries",
        "screener_results": "company_summaries",
        "artifact_manifest": "artifact_access",
    }
    return stage_by_artifact.get(artifact_id, "artifact_access")


def _refresh_artifact_manifest(context: RunContext) -> list[dict[str, Any]]:
    artifact_ids = [
        "run_config",
        "universe_manifest",
        "normalized_universe",
        "taxonomy_tree",
        "themes",
        "search_queries",
        "mindmap_validation",
        "plans",
        "budget_summary",
        "budget_approval",
        "enrichment_summary",
        "enrichment_approval",
        "results",
        "evidence_digest",
        "evidence_preview",
        "labeled_sentences",
        "company_summaries",
        "screener_results",
    ]
    records = [
        record
        for artifact_id in artifact_ids
        if (record := _record_for_artifact(context, artifact_id)) is not None
    ]
    manifest = {"run_id": context.run_name, "artifacts": records}
    _write_json(context.run_dir / ARTIFACT_MANIFEST_FILENAME, manifest)
    return records


def _universe_path_for_mode(universe: dict[str, Any]) -> Path:
    mode = str(universe.get("mode", "default_global_all_caps"))
    if mode == "default_global_all_caps":
        return DEFAULT_GLOBAL_UNIVERSE
    if mode == "default_europe_ml_caps":
        return DEFAULT_EUROPE_ML_CAPS_UNIVERSE
    if mode == "sample_xnas":
        return SAMPLE_XNAS_UNIVERSE
    if mode in {"csv_path", "uploaded_csv"}:
        path_value = universe.get("path") or universe.get("upload_id")
        if not path_value:
            raise McpWorkflowError(f"Universe mode {mode} requires a path or upload_id")
        return Path(str(path_value)).expanduser()
    raise McpWorkflowError(f"Universe mode {mode} does not resolve to a CSV path")


def _find_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {column.upper(): column for column in columns}
    for candidate in candidates:
        if candidate.upper() in normalized:
            return normalized[candidate.upper()]
    return None


def _normalize_universe_dataframe(universe: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    mode = str(universe.get("mode", "default_global_all_caps"))
    warnings: list[str] = []

    if mode == "inline_entity_ids":
        entity_ids = [str(entity_id).strip() for entity_id in universe.get("entity_ids", [])]
        entity_ids = [entity_id for entity_id in entity_ids if entity_id]
        if not entity_ids:
            raise McpWorkflowError("inline_entity_ids requires at least one RP entity ID")
        universe_df = pd.DataFrame(
            {
                screener.UNIVERSE_ID_COLUMN: entity_ids,
                screener.UNIVERSE_NAME_COLUMN: entity_ids,
            }
        )
        source_id_column = "RP_ENTITY_ID"
        source_name_column: str | None = None
    else:
        csv_path = _universe_path_for_mode(universe)
        if not csv_path.exists():
            raise FileNotFoundError(f"Universe CSV not found: {csv_path}")
        raw_df = pd.read_csv(csv_path)
        id_column = str(
            universe.get("id_column") or _find_column(raw_df.columns, ENTITY_ID_ALIASES) or ""
        )
        if not id_column:
            raise McpWorkflowError(
                f"Universe CSV must contain one of these columns: {', '.join(ENTITY_ID_ALIASES)}"
            )
        name_column = universe.get("name_column") or _find_column(
            raw_df.columns, OPTIONAL_NAME_ALIASES
        )
        source_name_column = str(name_column) if name_column in raw_df.columns else None
        if source_name_column is None:
            warnings.append("No company name column found; COMPANY_NAME will be blank.")

        universe_df = pd.DataFrame(
            {
                screener.UNIVERSE_ID_COLUMN: raw_df[id_column].astype(str).str.strip(),
                screener.UNIVERSE_NAME_COLUMN: (
                    raw_df[source_name_column].astype(str).str.strip()
                    if source_name_column is not None
                    else ""
                ),
            }
        )
        source_id_column = id_column

    before_dedupe = len(universe_df)
    universe_df = (
        universe_df[universe_df[screener.UNIVERSE_ID_COLUMN].astype(str).str.len() > 0]
        .drop_duplicates(subset=[screener.UNIVERSE_ID_COLUMN])
        .reset_index(drop=True)
    )
    removed = before_dedupe - len(universe_df)
    if removed:
        warnings.append(f"Removed {removed} blank or duplicate entity IDs.")

    manifest = {
        "mode": mode,
        "row_count": int(len(universe_df)),
        "id_column": source_id_column,
        "normalized_id_column": screener.UNIVERSE_ID_COLUMN,
        "name_column": source_name_column,
        "normalized_name_column": screener.UNIVERSE_NAME_COLUMN,
        "warnings": warnings,
        "sample": universe_df.head(5).to_dict(orient="records"),
    }
    return universe_df, manifest


def validate_universe(run_id: str, universe: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate and persist a normalized universe for a run."""
    context = _context(run_id)
    config = context.load_config()
    selected_universe = (
        universe or config.get("universe_input") or {"mode": "default_global_all_caps"}
    )
    universe_df, manifest = _normalize_universe_dataframe(selected_universe)

    universe_path = context.run_dir / UNIVERSE_FILENAME
    universe_df.to_csv(universe_path, index=False)
    _write_json(context.run_dir / UNIVERSE_MANIFEST_FILENAME, manifest)
    context.save_config(
        {
            "universe": str(universe_path),
            "universe_input": selected_universe,
            "universe_mode": manifest["mode"],
        }
    )
    _refresh_artifact_manifest(context)

    return _response(
        run_id=context.run_name,
        stage="intent_capture",
        status="completed",
        summary=(
            f"Validated {manifest['row_count']:,} unique companies/entities for the run universe."
        ),
        universe_summary=manifest,
        artifacts=[
            _artifact_handle("universe_manifest", UNIVERSE_MANIFEST_FILENAME),
            _artifact_handle("normalized_universe", UNIVERSE_FILENAME, queryable=True),
        ],
        next_actions=["generate_mindmap", "build_search_plans"],
    )


def create_run(
    main_theme: str,
    analyst_focus: str,
    run_id: str | None = None,
    universe: dict[str, Any] | None = None,
    start_date: str = screener.DEFAULT_START_DATE,
    end_date: str = screener.DEFAULT_END_DATE,
    output_goal: str | None = None,
) -> dict[str, Any]:
    """Create a run and validate its initial universe."""
    context = _context(run_id)
    selected_universe = universe or {"mode": "default_global_all_caps"}
    context.save_config(
        {
            "main_theme": main_theme,
            "analyst_focus": analyst_focus,
            "start_date": start_date,
            "end_date": end_date,
            "output_goal": output_goal,
            "universe_input": selected_universe,
        }
    )
    universe_response = validate_universe(context.run_name, selected_universe)
    return _response(
        run_id=context.run_name,
        stage="intent_capture",
        status="created",
        summary=(
            f"Run created for {main_theme} from {start_date} to {end_date}; "
            f"validated {universe_response['universe_summary']['row_count']:,} universe rows."
        ),
        universe_summary=universe_response["universe_summary"],
        artifacts=universe_response["artifacts"],
        next_actions=["generate_mindmap"],
    )


def _flatten_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
    children = node.get("children") or []
    if not children:
        return [node]
    leaves: list[dict[str, Any]] = []
    for child in children:
        if isinstance(child, dict):
            leaves.extend(_flatten_nodes(child))
    return leaves


def _node_preview(root: dict[str, Any]) -> dict[str, Any]:
    children = root.get("children") or []
    leaves = _flatten_nodes(root)
    return {
        "root_label": root.get("label", ""),
        "branch_count": len(children),
        "leaf_count": len(leaves),
        "leaf_labels": [str(leaf.get("label") or leaf.get("summary") or "") for leaf in leaves],
        "leaves": [
            {
                "label": str(leaf.get("label") or ""),
                "summary": str(leaf.get("summary") or ""),
                "search_query": str(leaf.get("search_query") or ""),
            }
            for leaf in leaves
        ],
    }


def _write_taxonomy(context: RunContext, taxonomy: dict[str, Any]) -> tuple[list[str], list[str]]:
    root = screener.Node.model_validate(taxonomy)
    labels, search_queries = screener.write_taxonomy_artifacts(
        root,
        themes_path=context.themes_path,
        search_queries_path=context.search_queries_path,
        taxonomy_tree_path=context.taxonomy_tree_path,
    )
    _refresh_artifact_manifest(context)
    return labels, search_queries


def _load_taxonomy(context: RunContext) -> dict[str, Any]:
    path = context.run_dir / TAXONOMY_TREE_FILENAME
    if path.exists():
        payload = _read_json(path)
        if isinstance(payload, dict):
            return payload
    labels = context.read_themes()
    return {
        "node": 1,
        "label": "Theme labels",
        "summary": "Leaf-only taxonomy reconstructed from themes.txt",
        "search_query": "",
        "children": [
            {
                "node": index + 2,
                "label": label,
                "summary": label,
                "search_query": "",
                "children": [],
            }
            for index, label in enumerate(labels)
        ],
    }


def generate_mindmap(
    run_id: str,
    max_leaf_labels: int | None = None,
    model: str = screener.DEFAULT_LABELS_MODEL,
) -> dict[str, Any]:
    """Generate and persist a taxonomy tree with leaf labels."""
    _load_environment()
    context = _context(run_id)
    config = context.load_config()
    main_theme = str(config.get("main_theme", screener.DEFAULT_MAIN_THEME))
    analyst_focus = str(config.get("analyst_focus", screener.DEFAULT_ANALYST_FOCUS))

    client = OpenAI()
    extra_instruction = ""
    if max_leaf_labels is not None:
        extra_instruction = f"\nLimit the final tree to at most {max_leaf_labels} leaf nodes."
    completion = client.chat.completions.create(
        model=model,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": SYSTEM_MESSAGE_LABELS.format(
                    main_theme=main_theme,
                    analyst_focus=f"{analyst_focus}{extra_instruction}",
                ),
            },
            {"role": "user", "content": USER_MESSAGE_LABELS.format(main_theme=main_theme)},
        ],
    )
    content = completion.choices[0].message.content
    if content is None:
        raise McpWorkflowError("OpenAI returned an empty taxonomy response")
    root = screener.Node.model_validate_json(content)
    labels, _search_queries = _write_taxonomy(context, root.model_dump())
    context.save_config({"labels_model": model})

    return _response(
        run_id=context.run_name,
        stage="mindmap_creation",
        status="completed",
        summary=f"Generated a taxonomy with {len(labels)} leaf labels.",
        mindmap_preview=_node_preview(root.model_dump()),
        artifacts=[
            _artifact_handle("taxonomy_tree", TAXONOMY_TREE_FILENAME, queryable=True),
            _artifact_handle("themes", "themes.txt"),
            _artifact_handle("search_queries", "search_queries.txt"),
        ],
        next_actions=["validate_mindmap", "update_mindmap", "build_search_plans"],
    )


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", text.lower())
        if token not in STOP_WORDS and len(token) > 2
    }


def validate_mindmap(run_id: str) -> dict[str, Any]:
    """Run lightweight taxonomy quality checks."""
    context = _context(run_id)
    config = context.load_config()
    taxonomy = _load_taxonomy(context)
    leaves = _flatten_nodes(taxonomy)
    focus_tokens = _tokenize(
        f"{config.get('main_theme', screener.DEFAULT_MAIN_THEME)} "
        f"{config.get('analyst_focus', screener.DEFAULT_ANALYST_FOCUS)}"
    )
    findings: list[dict[str, Any]] = []
    labels: list[str] = []
    search_queries: list[str] = []

    for leaf in leaves:
        label = str(leaf.get("label") or leaf.get("summary") or "").strip()
        summary = str(leaf.get("summary") or "").strip()
        search_query = str(leaf.get("search_query") or "").strip()
        labels.append(label)
        search_queries.append(search_query)

        if len(_tokenize(label)) < 2:
            findings.append(
                {
                    "severity": "medium",
                    "check": "label_clarity",
                    "message": f"Leaf label is too short or vague: {label}",
                    "suggested_action": "Use a concise role-oriented exposure label.",
                }
            )
        if not search_query:
            findings.append(
                {
                    "severity": "high",
                    "check": "missing_search_query",
                    "message": f"Leaf is missing search_query text: {label}",
                    "suggested_action": "Add document-voice retrieval text for Bigdata search.",
                }
            )
        elif has_exposure_meta_language(search_query):
            findings.append(
                {
                    "severity": "medium",
                    "check": "search_query_meta_language",
                    "message": f"Search query uses analyst exposure framing: {search_query}",
                    "suggested_action": "Rewrite as company operational/disclosure language.",
                }
            )
        elif len(_tokenize(search_query)) < 3:
            findings.append(
                {
                    "severity": "medium",
                    "check": "search_query_searchability",
                    "message": f"Search query is too short or vague: {search_query}",
                    "suggested_action": "Expand into a concrete product/service/customer phrase.",
                }
            )
        elif summary and search_query.lower() == summary.lower():
            findings.append(
                {
                    "severity": "low",
                    "check": "search_query_equals_summary",
                    "message": f"Search query duplicates analyst summary for: {label}",
                    "suggested_action": "Rewrite search_query in document voice.",
                }
            )

    for index, label in enumerate(labels):
        tokens = _tokenize(label)
        if focus_tokens and tokens and len(tokens & focus_tokens) == 0:
            findings.append(
                {
                    "severity": "low",
                    "check": "focus_alignment",
                    "message": f"Leaf may be weakly aligned with analyst focus: {label}",
                    "suggested_action": "Check whether this leaf supports the requested screen.",
                }
            )
        for later_label in labels[index + 1 :]:
            later_tokens = _tokenize(later_label)
            union = tokens | later_tokens
            if union and len(tokens & later_tokens) / len(union) >= 0.65:
                findings.append(
                    {
                        "severity": "medium",
                        "check": "overlap",
                        "message": f"Potential overlap between '{label}' and '{later_label}'.",
                        "suggested_action": "Merge or sharpen one of these leaves.",
                    }
                )

    if len(labels) > 15:
        findings.append(
            {
                "severity": "medium",
                "check": "budget_risk",
                "message": f"Taxonomy has {len(labels)} leaves, which may inflate retrieval cost.",
                "suggested_action": "Consider reducing to 8-12 high-signal leaves.",
            }
        )

    payload = {
        "run_id": context.run_name,
        "created_at": _utc_now(),
        "leaf_count": len(labels),
        "findings": findings,
    }
    _write_json(context.run_dir / MINDMAP_VALIDATION_FILENAME, payload)
    _refresh_artifact_manifest(context)

    summary = (
        "Mindmap validation passed without findings."
        if not findings
        else f"Mindmap validation found {len(findings)} potential issues."
    )
    return _response(
        run_id=context.run_name,
        stage="mindmap_validation",
        status="completed",
        summary=summary,
        findings=findings,
        artifacts=[_artifact_handle("mindmap_validation", MINDMAP_VALIDATION_FILENAME)],
        next_actions=["update_mindmap", "build_search_plans"],
    )


def update_mindmap(
    run_id: str,
    instructions: str,
    model: str = screener.DEFAULT_LABELS_MODEL,
) -> dict[str, Any]:
    """Use the main LLM to revise the taxonomy according to user instructions."""
    _load_environment()
    context = _context(run_id)
    taxonomy = _load_taxonomy(context)
    config = context.load_config()
    main_theme = str(config.get("main_theme", screener.DEFAULT_MAIN_THEME))
    analyst_focus = str(config.get("analyst_focus", screener.DEFAULT_ANALYST_FOCUS))
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You revise analyst screening taxonomies. Return only valid JSON matching "
                    "this recursive shape: node integer, label string, summary string, "
                    "search_query string, children array. Branch nodes use an empty search_query. "
                    "Leaf search_query values must use document/disclosure voice for Bigdata "
                    "retrieval and must not copy summary verbatim."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Main theme: {main_theme}\nAnalyst focus: {analyst_focus}\n\n"
                    f"Current taxonomy:\n{json.dumps(taxonomy, indent=2)}\n\n"
                    f"Revision instructions: {instructions}"
                ),
            },
        ],
    )
    content = completion.choices[0].message.content
    if content is None:
        raise McpWorkflowError("OpenAI returned an empty revised taxonomy response")
    root = screener.Node.model_validate_json(content)
    labels, _search_queries = _write_taxonomy(context, root.model_dump())

    return _response(
        run_id=context.run_name,
        stage="mindmap_revision",
        status="completed",
        summary=f"Updated taxonomy now has {len(labels)} leaf labels.",
        mindmap_preview=_node_preview(root.model_dump()),
        artifacts=[
            _artifact_handle("taxonomy_tree", TAXONOMY_TREE_FILENAME, queryable=True),
            _artifact_handle("themes", "themes.txt"),
            _artifact_handle("search_queries", "search_queries.txt"),
        ],
        next_actions=["validate_mindmap", "build_search_plans"],
    )


def build_search_plans(
    run_id: str,
    category: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build Bigdata search plans without retrieval."""
    _load_environment()
    context = _context(run_id)
    config = context.load_config()
    labels = context.read_themes()
    search_queries = context.read_search_queries()
    universe_path = Path(str(config.get("universe", context.run_dir / UNIVERSE_FILENAME)))
    if not universe_path.exists():
        validate_universe(context.run_name, config.get("universe_input"))
    universe_df = screener.load_universe(universe_path)
    company_ids = universe_df[screener.UNIVERSE_ID_COLUMN].astype(str).tolist()
    start_date = str(config.get("start_date", screener.DEFAULT_START_DATE))
    end_date = str(config.get("end_date", screener.DEFAULT_END_DATE))
    context.ensure_plans_dir()
    saved_paths = screener.build_plans(
        labels=labels,
        search_queries=search_queries,
        company_ids=company_ids,
        plans_dir=context.plans_dir,
        start_date=start_date,
        end_date=end_date,
        category=category or DEFAULT_SEARCH_CATEGORY,
    )
    context.save_config({"start_date": start_date, "end_date": end_date})
    _refresh_artifact_manifest(context)

    return _response(
        run_id=context.run_name,
        stage="planning",
        status="completed",
        summary=f"Built {len(saved_paths)} search plans. Retrieval has not started.",
        artifacts=[_artifact_handle("plans", "plans/", queryable=True)],
        next_actions=["estimate_retrieval_budget"],
    )


def _estimate_retrieval_cost_usd(
    selected_chunks: int,
    *,
    cost_usd_per_unit: float = RETRIEVAL_COST_USD_PER_UNIT,
    chunks_per_unit: int = RETRIEVAL_CHUNKS_PER_COST_UNIT,
) -> float:
    """Estimate retrieval spend from chunk count ($0.015 per 10 chunks by default)."""
    if selected_chunks <= 0:
        return 0.0
    return round((selected_chunks / chunks_per_unit) * cost_usd_per_unit, 2)


def _max_chunks_for_retrieval_budget(
    max_cost_usd: float,
    *,
    cost_usd_per_unit: float = RETRIEVAL_COST_USD_PER_UNIT,
    chunks_per_unit: int = RETRIEVAL_CHUNKS_PER_COST_UNIT,
) -> int:
    """Return the maximum chunk count affordable under a dollar cap."""
    if max_cost_usd <= 0 or cost_usd_per_unit <= 0:
        return 1
    return max(1, math.floor(max_cost_usd * chunks_per_unit / cost_usd_per_unit))


def _retrieval_pricing_from_budget(budget_payload: dict[str, Any]) -> tuple[float, int]:
    """Read retrieval pricing from a budget payload, including legacy per-chunk keys."""
    if "retrieval_cost_usd_per_10_chunks" in budget_payload:
        return (
            float(budget_payload["retrieval_cost_usd_per_10_chunks"]),
            int(budget_payload.get("retrieval_chunks_per_cost_unit", RETRIEVAL_CHUNKS_PER_COST_UNIT)),
        )
    legacy_per_chunk = budget_payload.get("cost_per_chunk_usd")
    if legacy_per_chunk is not None:
        # Older runs stored 0.015 as if it were per chunk; pricing is per 10 chunks.
        return (RETRIEVAL_COST_USD_PER_UNIT, RETRIEVAL_CHUNKS_PER_COST_UNIT)
    return (RETRIEVAL_COST_USD_PER_UNIT, RETRIEVAL_CHUNKS_PER_COST_UNIT)


def estimate_retrieval_budget(
    run_id: str,
    retrieval_cost_usd_per_10_chunks: float = RETRIEVAL_COST_USD_PER_UNIT,
    presets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Estimate chunks and dollar cost from saved plans."""
    context = _context(run_id)
    summary_df = screener.summarize_plans(context.plans_dir)
    total_expected_chunks = int(summary_df["chunks"].sum())
    selected_presets = presets or list(DEFAULT_PRESETS)
    preset_rows: list[dict[str, Any]] = []
    for preset in selected_presets:
        chunk_percentage = float(preset["chunk_percentage"])
        selected_chunks = round(total_expected_chunks * chunk_percentage)
        preset_rows.append(
            {
                "name": str(preset["name"]),
                "chunk_percentage": chunk_percentage,
                "selected_chunks": selected_chunks,
                "estimated_cost_usd": _estimate_retrieval_cost_usd(
                    selected_chunks,
                    cost_usd_per_unit=retrieval_cost_usd_per_10_chunks,
                ),
            }
        )

    per_plan = summary_df.to_dict(orient="records")
    payload = {
        "run_id": context.run_name,
        "created_at": _utc_now(),
        "total_expected_chunks": total_expected_chunks,
        "retrieval_cost_usd_per_10_chunks": retrieval_cost_usd_per_10_chunks,
        "retrieval_chunks_per_cost_unit": RETRIEVAL_CHUNKS_PER_COST_UNIT,
        "presets": preset_rows,
        "per_plan": per_plan,
    }
    _write_json(context.run_dir / BUDGET_SUMMARY_FILENAME, payload)
    _refresh_artifact_manifest(context)

    return _response(
        run_id=context.run_name,
        stage="budget_preview",
        status="pending_approval",
        summary=f"Planning found {total_expected_chunks:,} expected chunks before sampling.",
        total_expected_chunks=total_expected_chunks,
        retrieval_cost_usd_per_10_chunks=retrieval_cost_usd_per_10_chunks,
        retrieval_chunks_per_cost_unit=RETRIEVAL_CHUNKS_PER_COST_UNIT,
        presets=preset_rows,
        artifacts=[_artifact_handle("budget_summary", BUDGET_SUMMARY_FILENAME)],
        next_actions=["approve_budget"],
    )


def approve_budget(run_id: str, selection: dict[str, Any]) -> dict[str, Any]:
    """Persist the user's approved retrieval budget."""
    context = _context(run_id)
    budget_payload = _read_json(context.run_dir / BUDGET_SUMMARY_FILENAME)
    if not isinstance(budget_payload, dict):
        raise McpWorkflowError("Invalid budget_summary.json payload")
    total_expected_chunks = int(budget_payload["total_expected_chunks"])
    cost_usd_per_unit, chunks_per_unit = _retrieval_pricing_from_budget(budget_payload)

    approved = _resolve_budget_selection(
        selection,
        budget_payload,
        total_expected_chunks,
        cost_usd_per_unit=cost_usd_per_unit,
        chunks_per_unit=chunks_per_unit,
    )
    approved["estimated_cost_usd"] = _estimate_retrieval_cost_usd(
        int(approved["selected_chunks"]),
        cost_usd_per_unit=cost_usd_per_unit,
        chunks_per_unit=chunks_per_unit,
    )
    approved["retrieval_cost_usd_per_10_chunks"] = cost_usd_per_unit
    approved["retrieval_chunks_per_cost_unit"] = chunks_per_unit
    approved["created_at"] = _utc_now()
    _write_json(context.run_dir / BUDGET_APPROVAL_FILENAME, approved)
    context.save_config({"chunk_percentage": approved["chunk_percentage"]})
    _refresh_artifact_manifest(context)

    return _response(
        run_id=context.run_name,
        stage="budget_approval",
        status="completed",
        summary=(
            f"Approved retrieval: about {approved['selected_chunks']:,} chunks and "
            f"${approved['estimated_cost_usd']:,.2f}."
        ),
        approved_budget=approved,
        artifacts=[_artifact_handle("budget_approval", BUDGET_APPROVAL_FILENAME)],
        next_actions=["run_retrieval"],
    )


def _resolve_budget_selection(
    selection: dict[str, Any],
    budget_payload: dict[str, Any],
    total_expected_chunks: int,
    *,
    cost_usd_per_unit: float = RETRIEVAL_COST_USD_PER_UNIT,
    chunks_per_unit: int = RETRIEVAL_CHUNKS_PER_COST_UNIT,
) -> dict[str, Any]:
    if "preset" in selection:
        preset_name = str(selection["preset"])
        for preset in budget_payload.get("presets", []):
            if str(preset["name"]) == preset_name:
                return {
                    "selection": {"preset": preset_name},
                    "chunk_percentage": float(preset["chunk_percentage"]),
                    "selected_chunks": int(preset["selected_chunks"]),
                }
        raise McpWorkflowError(f"Unknown budget preset: {preset_name}")
    if "chunk_percentage" in selection:
        chunk_percentage = float(selection["chunk_percentage"])
        selected_chunks = round(total_expected_chunks * chunk_percentage)
        return {
            "selection": {"chunk_percentage": chunk_percentage},
            "chunk_percentage": chunk_percentage,
            "selected_chunks": selected_chunks,
        }
    if "selected_chunks" in selection:
        selected_chunks = int(selection["selected_chunks"])
        return {
            "selection": {"selected_chunks": selected_chunks},
            "chunk_percentage": selected_chunks / total_expected_chunks,
            "selected_chunks": selected_chunks,
        }
    if "max_cost_usd" in selection:
        max_cost_usd = float(selection["max_cost_usd"])
        selected_chunks = _max_chunks_for_retrieval_budget(
            max_cost_usd,
            cost_usd_per_unit=cost_usd_per_unit,
            chunks_per_unit=chunks_per_unit,
        )
        return {
            "selection": {"max_cost_usd": max_cost_usd},
            "chunk_percentage": selected_chunks / total_expected_chunks,
            "selected_chunks": selected_chunks,
        }
    raise McpWorkflowError(
        "Budget selection requires preset, chunk_percentage, selected_chunks, or max_cost_usd"
    )


def _format_duration(total_seconds: int) -> str:
    """Return a human-readable duration estimate."""
    if total_seconds <= 0:
        return "under 1 minute"
    if total_seconds < 60:
        return f"about {total_seconds} seconds"
    minutes = math.ceil(total_seconds / 60)
    if minutes < 60:
        return f"about {minutes} minutes"
    hours = minutes / 60
    return f"about {hours:.1f} hours"


def _approx_tokens(text: str) -> int:
    """Approximate token count from text length."""
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def _format_enrichment_cost(amount_usd: float) -> str:
    """Format enrichment cost without rounding small approvals to zero."""
    bounded = max(MIN_ESTIMATED_COST_USD, amount_usd)
    return f"${bounded:,.{ENRICHMENT_COST_DECIMALS}f}"


def _enrichment_execution_mode(
    estimated_total_seconds: int,
    estimated_labeling_seconds: int,
) -> str:
    """Recommend combined or split enrichment execution for MCP clients."""
    if estimated_total_seconds > MCP_ENRICHMENT_SPLIT_THRESHOLD_SECONDS:
        return "split"
    if estimated_labeling_seconds > MCP_ENRICHMENT_SPLIT_THRESHOLD_SECONDS:
        return "split"
    return "combined"


def _enrichment_execution_next_actions(execution_mode: str) -> list[str]:
    """Return MCP next actions for enrichment after budget approval."""
    if execution_mode == "split":
        return ["run_labeling", "run_company_summaries"]
    return ["run_enrichment", "run_labeling", "run_company_summaries"]


def _enrichment_execution_note(execution_mode: str) -> str:
    """Explain why split enrichment is recommended for MCP hosts."""
    if execution_mode != "split":
        return ""
    return (
        "Estimated enrichment latency exceeds the Claude Desktop MCP tool timeout "
        f"({MCP_CLIENT_TOOL_TIMEOUT_SECONDS // 60} minutes). Call `run_labeling` in batches "
        "(repeat until complete), then `run_company_summaries`."
    )


def _labeling_wave_count(request_count: int, max_concurrent_requests: int) -> int:
    """Return the number of concurrency-limited waves for a batch."""
    if request_count <= 0 or max_concurrent_requests <= 0:
        return 0
    return math.ceil(request_count / max_concurrent_requests)


def _resolve_labeling_straggler_seconds(progress: dict[str, Any]) -> float:
    """Pick a per-wave straggler estimate from defaults and observed batch tails."""
    default = MCP_LABELING_WAVE_SECONDS_STRAGGLER
    elapsed = progress.get("last_batch_elapsed_seconds")
    batch_size = progress.get("batch_size")
    concurrency = progress.get("max_concurrent_requests")
    if elapsed is None or not batch_size or not concurrency:
        return default

    waves = _labeling_wave_count(int(batch_size), int(concurrency))
    if waves <= 0:
        return default

    calibrated = (float(elapsed) / waves) * MCP_LABELING_STRAGGLER_SAFETY_MARGIN
    latency_max = progress.get("last_batch_latency_max_seconds")
    if latency_max is not None:
        calibrated = max(calibrated, float(latency_max))
    latency_p99 = progress.get("last_batch_latency_p99_seconds")
    if latency_p99 is not None:
        calibrated = max(calibrated, float(latency_p99))
    return max(default, calibrated)


def _compute_labeling_batch_size(
    sentence_count: int,
    requests_per_minute: int,
    max_concurrent_requests: int,
    target_seconds: int = MCP_LABELING_BATCH_SAFE_SECONDS,
    straggler_seconds: float = MCP_LABELING_WAVE_SECONDS_STRAGGLER,
) -> int:
    """Pick a labeling batch size that should finish within the MCP-safe budget."""
    if sentence_count <= 0:
        return 0
    if sentence_count <= MIN_LABELING_BATCH_SIZE:
        return sentence_count
    low = MIN_LABELING_BATCH_SIZE
    high = min(sentence_count, MAX_LABELING_BATCH_SIZE)
    best = low
    while low <= high:
        mid = (low + high) // 2
        seconds = _estimate_mcp_labeling_batch_seconds(
            mid,
            requests_per_minute,
            max_concurrent_requests,
            straggler_seconds=straggler_seconds,
        )
        if seconds <= target_seconds:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def _labeling_progress_path(context: RunContext) -> Path:
    return context.run_dir / LABELING_PROGRESS_FILENAME


def _labeling_responses_path(context: RunContext) -> Path:
    return context.run_dir / LABELING_RESPONSES_FILENAME


def _labeling_responses_jsonl_path(context: RunContext) -> Path:
    return context.run_dir / LABELING_RESPONSES_JSONL_FILENAME


def _load_labeling_responses(context: RunContext) -> dict[str, dict[str, str]]:
    responses: dict[str, dict[str, str]] = {}

    legacy_path = _labeling_responses_path(context)
    if legacy_path.exists():
        payload = _read_json(legacy_path)
        if isinstance(payload, dict):
            responses.update(
                {
                    str(sentence_id): fields
                    for sentence_id, fields in payload.items()
                    if isinstance(fields, dict)
                }
            )

    jsonl_path = _labeling_responses_jsonl_path(context)
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    continue
                sentence_id = str(payload.get("sentence_id", ""))
                fields = payload.get("fields")
                if sentence_id and isinstance(fields, dict):
                    responses[sentence_id] = fields
    return responses


def _append_labeling_responses(
    context: RunContext,
    batch_responses: dict[str, dict[str, str]],
) -> None:
    """Append one batch of labeling responses without rewriting prior batches."""
    if not batch_responses:
        return
    jsonl_path = _labeling_responses_jsonl_path(context)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for sentence_id, fields in batch_responses.items():
            handle.write(
                json.dumps({"sentence_id": str(sentence_id), "fields": fields}, ensure_ascii=False)
            )
            handle.write("\n")


def _load_labeling_progress(context: RunContext) -> dict[str, Any]:
    path = _labeling_progress_path(context)
    if not path.exists():
        return {}
    payload = _read_json(path)
    return payload if isinstance(payload, dict) else {}


def _save_labeling_progress(context: RunContext, progress: dict[str, Any]) -> None:
    _write_json(_labeling_progress_path(context), progress)


def _extract_labeling_sentences(context: RunContext) -> list[dict[str, Any]]:
    config = context.load_config()
    results = _load_results(context)
    universe_path = Path(str(config.get("universe", context.run_dir / UNIVERSE_FILENAME)))
    universe_df = screener.load_universe(universe_path)
    sentences = screener.extract_sentences(results, universe_df)
    if not sentences:
        raise McpWorkflowError("No sentences found in retrieval results")
    return sentences


def _enrichment_status(context: RunContext) -> dict[str, Any]:
    """Summarize labeling and summary artifact state for MCP recovery."""
    progress = _load_labeling_progress(context)
    processed_sentences = int(progress.get("processed_sentences", 0) or 0)
    total_sentences = int(progress.get("total_sentences", 0) or 0)
    labeling_complete = bool(progress.get("complete"))
    labeled_row_count = 0
    if context.labeled_sentences_path.exists() and labeling_complete:
        labeled_row_count = max(0, sum(1 for _ in context.labeled_sentences_path.open()) - 1)
    summaries_complete = context.company_summaries_path.exists()
    screener_complete = context.screener_results_path.exists()
    status: dict[str, Any] = {
        "labeling_complete": labeling_complete,
        "processed_sentences": processed_sentences,
        "total_sentences": total_sentences,
        "labeled_row_count": labeled_row_count,
        "labeling_progress": progress,
        "company_summaries_complete": summaries_complete,
        "screener_results_complete": screener_complete,
    }
    if processed_sentences > 0 and not labeling_complete:
        status["recovery_note"] = (
            f"Labeling in progress: {processed_sentences:,}"
            f"{f' of {total_sentences:,}' if total_sentences else ''} sentences stored. "
            "Call `run_labeling` again to continue. Ignore MCP timeout errors if progress advances."
        )
    elif labeling_complete and not summaries_complete:
        status["recovery_note"] = (
            "Labeling complete. Call `run_company_summaries` next."
        )
    elif labeling_complete and summaries_complete:
        status["recovery_note"] = (
            "Enrichment artifacts are present. If an MCP tool timed out, the work may still "
            "have completed server-side."
        )
    return status


def _apply_enrichment_cost(
    input_tokens: int,
    output_tokens: int,
    input_usd_per_mtok: float,
    output_usd_per_mtok: float,
    safety_margin: float,
) -> float:
    """Convert token estimates to a safety-adjusted USD cost."""
    raw_cost = (
        input_tokens * input_usd_per_mtok + output_tokens * output_usd_per_mtok
    ) / 1_000_000
    return round(raw_cost * safety_margin, ENRICHMENT_COST_DECIMALS)


def _estimate_openai_batch_seconds(
    request_count: int,
    requests_per_minute: int,
    max_concurrent_requests: int,
    seconds_per_wave: float = AVERAGE_LABELING_REQUEST_SECONDS,
) -> int:
    """Estimate wall-clock seconds for a parallel OpenAI batch.

    ``seconds_per_wave`` is the expected duration of one concurrency-limited wave
    (``ceil(request_count / max_concurrent)`` waves), not per-request serial time.
    """
    if request_count <= 0:
        return 0
    rpm_seconds = math.ceil(request_count / requests_per_minute * 60)
    concurrency_seconds = math.ceil(
        request_count / max_concurrent_requests
    ) * seconds_per_wave
    return int(max(rpm_seconds, concurrency_seconds))


def _estimate_mcp_labeling_batch_seconds(
    request_count: int,
    requests_per_minute: int,
    max_concurrent_requests: int,
    straggler_seconds: float = MCP_LABELING_WAVE_SECONDS_STRAGGLER,
) -> int:
    """Estimate one MCP labeling batch using straggler latency per concurrency wave."""
    return _estimate_openai_batch_seconds(
        request_count,
        requests_per_minute,
        max_concurrent_requests,
        straggler_seconds,
    )


def _estimate_summary_batch_seconds(
    summary_requests: list[dict[str, int | float]],
    requests_per_minute: int,
    max_concurrent_requests: int,
) -> int:
    """Estimate summary latency using payload size and concurrency."""
    if not summary_requests:
        return 0
    per_request_seconds = [
        BASE_SUMMARY_REQUEST_SECONDS
        + (float(request["input_chars"]) / 1000.0) * SUMMARY_SECONDS_PER_1K_INPUT_CHARS
        for request in summary_requests
    ]
    average_seconds = sum(per_request_seconds) / len(per_request_seconds)
    return _estimate_openai_batch_seconds(
        len(summary_requests),
        requests_per_minute,
        max_concurrent_requests,
        average_seconds,
    )


def _load_enrichment_inputs(context: RunContext) -> dict[str, Any]:
    """Load sentences, labels, and optional prior labeled output for estimation."""
    config = context.load_config()
    results = _load_results(context)
    universe_path = Path(str(config.get("universe", context.run_dir / UNIVERSE_FILENAME)))
    universe_df = screener.load_universe(universe_path)
    sentences = screener.extract_sentences(results, universe_df)
    main_theme = str(config.get("main_theme", screener.DEFAULT_MAIN_THEME))
    analyst_focus = str(config.get("analyst_focus", screener.DEFAULT_ANALYST_FOCUS))
    labels = context.read_themes()
    labeled_df: pd.DataFrame | None = None
    if context.labeled_sentences_path.exists():
        labeled_df = pd.read_csv(context.labeled_sentences_path)
    return {
        "sentences": sentences,
        "main_theme": main_theme,
        "analyst_focus": analyst_focus,
        "labels": labels,
        "labeled_df": labeled_df,
    }


def _estimate_labeling_usage(
    sentences: list[dict[str, Any]],
    main_theme: str,
    labels: list[str],
    analyst_focus: str,
) -> dict[str, int]:
    """Estimate labeling token usage from extracted sentences."""
    system_prompt = SYSTEM_PROMPT_LABELING.format(
        main_theme=main_theme,
        analyst_focus=analyst_focus,
        labels=str(labels),
    )
    system_tokens = _approx_tokens(system_prompt)
    input_tokens = 0
    for sentence in sentences:
        input_tokens += system_tokens + _approx_tokens(str(sentence))
    output_tokens = len(sentences) * LABELING_OUTPUT_TOKENS_ESTIMATE
    return {
        "sentence_count": len(sentences),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _estimate_summary_usage(
    sentences: list[dict[str, Any]],
    main_theme: str,
    labeled_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """Estimate summary request count, payload size, and token usage."""
    system_prompt = _company_summary_system_prompt(main_theme)
    summary_requests: list[dict[str, int | str]] = []

    if labeled_df is not None and not labeled_df.empty and "company_name" in labeled_df.columns:
        grouped = labeled_df.groupby("company_name", sort=True)
        for company_name, group in grouped:
            motivations = _company_evidence_block(group)
            if not str(motivations).strip():
                continue
            user_content = (
                f"Company: {company_name}\n\nMotivations ({main_theme}):\n{motivations}"
            )
            summary_requests.append(
                {
                    "company_name": str(company_name),
                    "input_chars": len(system_prompt) + len(user_content),
                    "user_content": user_content,
                }
            )
        input_tokens = sum(
            _approx_tokens(system_prompt) + _approx_tokens(str(request["user_content"]))
            for request in summary_requests
        )
        return {
            "raw_company_count": int(labeled_df["company_name"].nunique()),
            "estimated_summary_company_count": len(summary_requests),
            "summary_requests": summary_requests,
            "input_tokens": input_tokens,
            "output_tokens": len(summary_requests) * SUMMARY_OUTPUT_TOKENS_ESTIMATE,
        }

    companies: dict[str, list[dict[str, Any]]] = {}
    for sentence in sentences:
        company_name = str(sentence.get("company_name", "")).strip()
        if not company_name:
            continue
        companies.setdefault(company_name, []).append(sentence)

    raw_company_count = len(companies)
    ranked_companies = sorted(
        companies.items(),
        key=lambda item: (len(item[1]), sum(len(str(s.get("text", ""))) for s in item[1])),
        reverse=True,
    )
    estimated_summary_company_count = 0
    if raw_company_count > 0:
        estimated_summary_company_count = max(
            1,
            min(
                raw_company_count,
                round(raw_company_count * SUMMARY_COMPANY_RETENTION_FACTOR),
            ),
        )

    for company_name, company_sentences in ranked_companies[:estimated_summary_company_count]:
        retained_sentences = max(
            1,
            round(len(company_sentences) * SENTENCE_RETENTION_FACTOR),
        )
        motivation_chars = min(
            retained_sentences * ESTIMATED_MOTIVATION_CHARS_PER_SENTENCE,
            screener.MAX_MOTIVATIONS_CHARS,
        )
        user_content = (
            f"Company: {company_name}\n\nMotivations ({main_theme}):\n"
            f"{'-' * motivation_chars}"
        )
        summary_requests.append(
            {
                "company_name": company_name,
                "input_chars": len(system_prompt) + len(user_content),
                "user_content": user_content,
            }
        )

    input_tokens = sum(
        _approx_tokens(system_prompt) + _approx_tokens(str(request["user_content"]))
        for request in summary_requests
    )
    return {
        "raw_company_count": raw_company_count,
        "estimated_summary_company_count": estimated_summary_company_count,
        "summary_requests": summary_requests,
        "input_tokens": input_tokens,
        "output_tokens": estimated_summary_company_count * SUMMARY_OUTPUT_TOKENS_ESTIMATE,
    }


def _estimate_enrichment_plan(
    context: RunContext,
    labeling_input_usd_per_mtok: float = DEFAULT_LABELING_INPUT_USD_PER_MTOK,
    labeling_output_usd_per_mtok: float = DEFAULT_LABELING_OUTPUT_USD_PER_MTOK,
    summary_input_usd_per_mtok: float = DEFAULT_SUMMARY_INPUT_USD_PER_MTOK,
    summary_output_usd_per_mtok: float = DEFAULT_SUMMARY_OUTPUT_USD_PER_MTOK,
    requests_per_minute: int = DEFAULT_ENRICHMENT_REQUESTS_PER_MINUTE,
    max_concurrent_requests: int = DEFAULT_ENRICHMENT_MAX_CONCURRENT,
    safety_margin: float = ENRICHMENT_COST_SAFETY_MARGIN,
) -> dict[str, Any]:
    """Build token- and payload-aware enrichment cost/latency estimates."""
    inputs = _load_enrichment_inputs(context)
    sentences = cast(list[dict[str, Any]], inputs["sentences"])
    main_theme = str(inputs["main_theme"])
    analyst_focus = str(inputs["analyst_focus"])
    labels = cast(list[str], inputs["labels"])
    labeled_df = cast(pd.DataFrame | None, inputs["labeled_df"])

    labeling_usage = _estimate_labeling_usage(sentences, main_theme, labels, analyst_focus)
    summary_usage = _estimate_summary_usage(sentences, main_theme, labeled_df)

    labeling_cost = _apply_enrichment_cost(
        labeling_usage["input_tokens"],
        labeling_usage["output_tokens"],
        labeling_input_usd_per_mtok,
        labeling_output_usd_per_mtok,
        safety_margin,
    )
    summary_cost = _apply_enrichment_cost(
        summary_usage["input_tokens"],
        summary_usage["output_tokens"],
        summary_input_usd_per_mtok,
        summary_output_usd_per_mtok,
        safety_margin,
    )
    total_cost = round(labeling_cost + summary_cost, ENRICHMENT_COST_DECIMALS)
    total_cost = max(MIN_ESTIMATED_COST_USD, total_cost)

    labeling_seconds = _estimate_openai_batch_seconds(
        labeling_usage["sentence_count"],
        requests_per_minute,
        max_concurrent_requests,
        AVERAGE_LABELING_REQUEST_SECONDS,
    )
    summary_seconds = _estimate_summary_batch_seconds(
        cast(list[dict[str, int | float]], summary_usage["summary_requests"]),
        requests_per_minute,
        max_concurrent_requests,
    )
    total_seconds = labeling_seconds + summary_seconds
    execution_mode = _enrichment_execution_mode(total_seconds, labeling_seconds)
    execution_note = _enrichment_execution_note(execution_mode)
    sample_batch_size = _compute_labeling_batch_size(
        int(labeling_usage["sentence_count"]),
        requests_per_minute,
        max_concurrent_requests,
    )
    estimated_mcp_batch_seconds = _estimate_mcp_labeling_batch_seconds(
        sample_batch_size,
        requests_per_minute,
        max_concurrent_requests,
    )

    return {
        "sentence_count": labeling_usage["sentence_count"],
        "raw_company_count": summary_usage["raw_company_count"],
        "estimated_summary_company_count": summary_usage["estimated_summary_company_count"],
        "labeling_input_tokens": labeling_usage["input_tokens"],
        "labeling_output_tokens": labeling_usage["output_tokens"],
        "summary_input_tokens": summary_usage["input_tokens"],
        "summary_output_tokens": summary_usage["output_tokens"],
        "estimated_labeling_cost_usd": labeling_cost,
        "estimated_summary_cost_usd": summary_cost,
        "estimated_total_cost_usd": total_cost,
        "estimated_labeling_seconds": labeling_seconds,
        "estimated_summary_seconds": summary_seconds,
        "estimated_total_seconds": total_seconds,
        "estimated_latency_human": _format_duration(total_seconds),
        "recommended_execution_mode": execution_mode,
        "recommended_labeling_batch_size": sample_batch_size,
        "estimated_mcp_labeling_batch_seconds": estimated_mcp_batch_seconds,
        "labeling_wave_seconds_average": MCP_LABELING_WAVE_SECONDS,
        "labeling_wave_seconds_straggler": MCP_LABELING_WAVE_SECONDS_STRAGGLER,
        "mcp_labeling_batch_safe_seconds": MCP_LABELING_BATCH_SAFE_SECONDS,
        "labeling_wave_seconds_p95": MCP_LABELING_WAVE_SECONDS_STRAGGLER,
        "mcp_execution_note": execution_note,
        "mcp_client_tool_timeout_seconds": MCP_CLIENT_TOOL_TIMEOUT_SECONDS,
        "estimation_method": "token_based_with_safety_margin",
        "safety_margin": safety_margin,
        "labeling_input_usd_per_mtok": labeling_input_usd_per_mtok,
        "labeling_output_usd_per_mtok": labeling_output_usd_per_mtok,
        "summary_input_usd_per_mtok": summary_input_usd_per_mtok,
        "summary_output_usd_per_mtok": summary_output_usd_per_mtok,
        "requests_per_minute": requests_per_minute,
        "max_concurrent_requests": max_concurrent_requests,
        "uses_prior_labels": labeled_df is not None and not labeled_df.empty,
    }


def _enrichment_counts(context: RunContext) -> dict[str, int]:
    """Count sentences and distinct companies available for enrichment."""
    plan = _estimate_enrichment_plan(context)
    return {
        "sentence_count": int(plan["sentence_count"]),
        "company_count": int(plan["estimated_summary_company_count"]),
        "raw_company_count": int(plan["raw_company_count"]),
    }


def _require_enrichment_approval(context: RunContext) -> dict[str, Any]:
    """Load and validate enrichment approval artifact."""
    approval_path = context.run_dir / ENRICHMENT_APPROVAL_FILENAME
    if not approval_path.exists():
        raise McpWorkflowError(
            "Enrichment must be approved first. Call estimate_enrichment_budget, "
            "then approve_enrichment before run_enrichment."
        )
    approval = _read_json(approval_path)
    if not isinstance(approval, dict) or not approval.get("approved"):
        raise McpWorkflowError("Invalid or missing enrichment approval")
    return approval


def run_retrieval(
    run_id: str,
    requests_per_minute: int = screener.DEFAULT_REQUESTS_PER_MINUTE,
) -> dict[str, Any]:
    """Execute approved search plans."""
    _load_environment()
    context = _context(run_id)
    approval_path = context.run_dir / BUDGET_APPROVAL_FILENAME
    if not approval_path.exists():
        raise McpWorkflowError("Budget must be approved before retrieval")
    approval = _read_json(approval_path)
    if not isinstance(approval, dict):
        raise McpWorkflowError("Invalid budget_approval.json payload")
    chunk_percentage = float(approval["chunk_percentage"])
    results = screener.run_search(
        plans_dir=context.plans_dir,
        chunk_percentage=chunk_percentage,
        requests_per_minute=requests_per_minute,
    )
    context.ensure_results_dir()
    _write_json(context.results_path, results)
    context.save_config(
        {
            "chunk_percentage": chunk_percentage,
            "requests_per_minute": requests_per_minute,
        }
    )
    chunks = _flatten_result_chunks(results)
    _refresh_artifact_manifest(context)

    return _response(
        run_id=context.run_name,
        stage="retrieval",
        status="completed",
        summary=f"Retrieved {len(chunks):,} chunks across {len(results):,} deduplicated documents.",
        retrieval_stats={"document_count": len(results), "chunk_count": len(chunks)},
        artifacts=[_artifact_handle("results", "results/results.json", queryable=True)],
        next_actions=["summarize_retrieval"],
    )


def _load_results(context: RunContext) -> list[dict[str, Any]]:
    if not context.results_path.exists():
        raise FileNotFoundError(f"results not found at {context.results_path}")
    payload = _read_json(context.results_path)
    if not isinstance(payload, list):
        raise McpWorkflowError("results artifact must contain a JSON list")
    return payload


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _chunk_evidence_score(relevance: Any, sentiment: Any) -> float:
    """Score chunk importance as relevance multiplied by absolute sentiment."""
    try:
        relevance_value = float(relevance)
        sentiment_value = float(sentiment)
    except (TypeError, ValueError):
        return 0.0
    return relevance_value * abs(sentiment_value)


def _flatten_result_chunks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document_index, document in enumerate(results):
        document_id = str(
            _first_present(document, ("id", "document_id", "source_id", "doc_id"))
            or f"doc_{document_index:06d}"
        )
        source_category = str(
            _first_present(document, ("category", "source_category", "document_type", "source"))
            or "unknown"
        )
        document_date = _first_present(document, ("date", "published_at", "publication_date"))
        chunks = document.get("chunks", [])
        if not isinstance(chunks, list):
            continue
        for chunk_index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            chunk_mapping = cast(dict[str, Any], chunk)
            chunk_id = str(
                _first_present(chunk_mapping, ("id", "chunk_id"))
                or f"{document_id}_chunk_{chunk_index:04d}"
            )
            entity_ids = chunk_mapping.get("entity_ids") or document.get("entity_ids") or []
            if not isinstance(entity_ids, list):
                entity_ids = [entity_ids]
            relevance = chunk_mapping.get("relevance")
            sentiment = chunk_mapping.get("sentiment")
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "source_category": source_category,
                    "document_date": document_date,
                    "company_id": str(entity_ids[0]) if entity_ids else None,
                    "text": str(chunk_mapping.get("text") or ""),
                    "relevance": relevance,
                    "sentiment": sentiment,
                    "score": _chunk_evidence_score(relevance, sentiment),
                }
            )
    return rows


def _company_name_lookup(context: RunContext) -> dict[str, str]:
    config = context.load_config()
    universe_path = Path(str(config.get("universe", context.run_dir / UNIVERSE_FILENAME)))
    if not universe_path.exists():
        return {}
    universe_df = pd.read_csv(universe_path)
    if screener.UNIVERSE_ID_COLUMN not in universe_df.columns:
        return {}
    if screener.UNIVERSE_NAME_COLUMN not in universe_df.columns:
        return {}
    return {
        str(row[screener.UNIVERSE_ID_COLUMN]): str(row[screener.UNIVERSE_NAME_COLUMN])
        for row in universe_df.to_dict(orient="records")
    }


def _enrich_chunk_rows(context: RunContext, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    id_to_name = _company_name_lookup(context)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        company_id = row.get("company_id")
        company_name = id_to_name.get(str(company_id), "") if company_id else ""
        enriched.append({**row, "company_name": company_name})
    return enriched


def _top_terms(texts: Iterable[str], limit: int) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = [token for token in _tokenize(text) if token not in STOP_WORDS]
        counter.update(tokens)
    return [{"term": term, "count": count} for term, count in counter.most_common(limit)]


def summarize_retrieval(
    run_id: str,
    max_representative_chunks: int = 30,
) -> dict[str, Any]:
    """Create a deterministic evidence digest from retrieved chunks."""
    context = _context(run_id)
    results = _load_results(context)
    chunk_rows = _enrich_chunk_rows(context, _flatten_result_chunks(results))
    chunks_df = pd.DataFrame(chunk_rows)
    if chunks_df.empty:
        digest = {
            "run_id": context.run_name,
            "created_at": _utc_now(),
            "retrieval_stats": {
                "document_count": len(results),
                "chunk_count": 0,
                "company_count": 0,
            },
            "source_mix": [],
            "top_signals": [],
            "top_companies": [],
            "representative_chunks": [],
        }
    else:
        source_mix = (
            chunks_df.groupby("source_category", dropna=False)
            .agg(document_count=("document_id", "nunique"), chunk_count=("chunk_id", "count"))
            .reset_index()
            .sort_values(["chunk_count", "document_count"], ascending=False)
            .to_dict(orient="records")
        )
        company_rows = (
            chunks_df[chunks_df["company_name"].astype(str).str.len() > 0]
            .groupby("company_name", dropna=False)
            .agg(
                evidence_count=("chunk_id", "count"),
                document_count=("document_id", "nunique"),
                relevance_score=("score", "sum"),
            )
            .reset_index()
            .sort_values(
                ["relevance_score", "evidence_count", "document_count"],
                ascending=False,
            )
            .head(20)
        )
        representative_chunks = (
            chunks_df.sort_values(["score", "chunk_id"], ascending=[False, True])
            .head(max_representative_chunks)
            .assign(
                text=lambda df: df["text"].astype(str).str.slice(0, REPRESENTATIVE_CHUNK_TEXT_CHARS)
            )
            .to_dict(orient="records")
        )
        top_terms = _top_terms(chunks_df["text"].dropna().astype(str), 15)
        top_signals = [
            {
                "signal": f"High-frequency evidence term: {term['term']}",
                "summary": f"The term appears across {term['count']} retrieved chunks.",
                "supporting_chunk_count": term["count"],
                "representative_chunk_ids": [
                    str(row["chunk_id"])
                    for row in representative_chunks
                    if term["term"] in str(row.get("text", "")).lower()
                ][:5],
            }
            for term in top_terms[:8]
        ]
        digest = {
            "run_id": context.run_name,
            "created_at": _utc_now(),
            "retrieval_stats": {
                "document_count": len(results),
                "chunk_count": len(chunk_rows),
                "company_count": int(chunks_df["company_name"].replace("", pd.NA).nunique()),
            },
            "source_mix": source_mix,
            "top_signals": top_signals,
            "top_companies": company_rows.to_dict(orient="records"),
            "representative_chunks": representative_chunks,
        }

    _write_json(context.run_dir / EVIDENCE_DIGEST_FILENAME, digest)
    preview = {
        "top_signals": digest["top_signals"][:5],
        "top_companies": digest["top_companies"][:10],
        "representative_chunks": digest["representative_chunks"][:10],
    }
    _write_json(context.run_dir / EVIDENCE_PREVIEW_FILENAME, preview)
    _refresh_artifact_manifest(context)

    return _response(
        run_id=context.run_name,
        stage="evidence_digest",
        status="completed",
        summary="Evidence digest created with top signals, representative chunks, and source mix.",
        digest_preview=preview,
        artifacts=[
            _artifact_handle("evidence_digest", EVIDENCE_DIGEST_FILENAME, queryable=True),
            _artifact_handle("evidence_preview", EVIDENCE_PREVIEW_FILENAME),
        ],
        next_actions=["get_run_summary", "estimate_enrichment_budget"],
    )


def estimate_enrichment_budget(
    run_id: str,
    labeling_model: str = screener.DEFAULT_LABELING_MODEL,
    summary_model: str = screener.DEFAULT_SUMMARY_MODEL,
    requests_per_minute: int = DEFAULT_ENRICHMENT_REQUESTS_PER_MINUTE,
    max_concurrent_requests: int = DEFAULT_ENRICHMENT_MAX_CONCURRENT,
    labeling_input_usd_per_mtok: float = DEFAULT_LABELING_INPUT_USD_PER_MTOK,
    labeling_output_usd_per_mtok: float = DEFAULT_LABELING_OUTPUT_USD_PER_MTOK,
    summary_input_usd_per_mtok: float = DEFAULT_SUMMARY_INPUT_USD_PER_MTOK,
    summary_output_usd_per_mtok: float = DEFAULT_SUMMARY_OUTPUT_USD_PER_MTOK,
    safety_margin: float = ENRICHMENT_COST_SAFETY_MARGIN,
) -> dict[str, Any]:
    """Estimate OpenAI cost and latency for labeling plus company summaries."""
    context = _context(run_id)
    if not context.results_path.exists():
        raise McpWorkflowError("Run retrieval before estimating enrichment budget")
    plan = _estimate_enrichment_plan(
        context,
        labeling_input_usd_per_mtok=labeling_input_usd_per_mtok,
        labeling_output_usd_per_mtok=labeling_output_usd_per_mtok,
        summary_input_usd_per_mtok=summary_input_usd_per_mtok,
        summary_output_usd_per_mtok=summary_output_usd_per_mtok,
        requests_per_minute=requests_per_minute,
        max_concurrent_requests=max_concurrent_requests,
        safety_margin=safety_margin,
    )
    sentence_count = int(plan["sentence_count"])
    raw_company_count = int(plan["raw_company_count"])
    summary_company_count = int(plan["estimated_summary_company_count"])
    total_cost = float(plan["estimated_total_cost_usd"])
    latency_human = str(plan["estimated_latency_human"])
    execution_mode = str(plan.get("recommended_execution_mode", "combined"))
    execution_note = str(plan.get("mcp_execution_note", ""))
    summary_text = (
        f"Enrichment covers {sentence_count:,} labeling requests and about "
        f"{summary_company_count:,} company summaries "
        f"({raw_company_count:,} companies in retrieved evidence). "
        f"Estimated cost {_format_enrichment_cost(total_cost)}; latency {latency_human}."
    )
    if execution_note:
        summary_text = f"{summary_text} {execution_note}"

    payload = {
        "run_id": context.run_name,
        "created_at": _utc_now(),
        "sentence_count": sentence_count,
        "raw_company_count": raw_company_count,
        "estimated_summary_company_count": summary_company_count,
        "company_count": summary_company_count,
        "labeling_model": labeling_model,
        "summary_model": summary_model,
        **plan,
        "includes": ["labeling", "company_summaries", "screener_results"],
    }
    _write_json(context.run_dir / ENRICHMENT_SUMMARY_FILENAME, payload)
    _refresh_artifact_manifest(context)

    return _response(
        run_id=context.run_name,
        stage="enrichment_preview",
        status="pending_approval",
        summary=summary_text,
        enrichment_budget=payload,
        recommended_execution_mode=execution_mode,
        artifacts=[_artifact_handle("enrichment_summary", ENRICHMENT_SUMMARY_FILENAME)],
        next_actions=["approve_enrichment"],
    )


def approve_enrichment(
    run_id: str,
    approved: bool = True,
    labeling_model: str | None = None,
    summary_model: str | None = None,
) -> dict[str, Any]:
    """Persist user approval for the enrichment budget."""
    if not approved:
        raise McpWorkflowError("Enrichment was not approved")
    context = _context(run_id)
    summary_path = context.run_dir / ENRICHMENT_SUMMARY_FILENAME
    if not summary_path.exists():
        raise McpWorkflowError("Call estimate_enrichment_budget before approve_enrichment")
    summary = _read_json(summary_path)
    if not isinstance(summary, dict):
        raise McpWorkflowError("Invalid enrichment_summary.json payload")
    approved_payload = {
        **summary,
        "approved": True,
        "approved_at": _utc_now(),
    }
    if labeling_model is not None:
        approved_payload["labeling_model"] = labeling_model
    if summary_model is not None:
        approved_payload["summary_model"] = summary_model
    _write_json(context.run_dir / ENRICHMENT_APPROVAL_FILENAME, approved_payload)
    _refresh_artifact_manifest(context)

    total_cost = float(approved_payload.get("estimated_total_cost_usd", 0))
    latency_human = str(approved_payload.get("estimated_latency_human", ""))
    execution_mode = str(
        approved_payload.get("recommended_execution_mode", "combined"),
    )
    execution_note = _enrichment_execution_note(execution_mode)
    approval_summary = (
        f"Approved enrichment: {_format_enrichment_cost(total_cost)} estimated cost, "
        f"{latency_human} estimated latency."
    )
    if execution_note:
        approval_summary = f"{approval_summary} {execution_note}"
    return _response(
        run_id=context.run_name,
        stage="enrichment_approval",
        status="completed",
        summary=approval_summary,
        approved_enrichment=approved_payload,
        recommended_execution_mode=execution_mode,
        artifacts=[_artifact_handle("enrichment_approval", ENRICHMENT_APPROVAL_FILENAME)],
        next_actions=_enrichment_execution_next_actions(execution_mode),
    )


def _finalize_labeled_dataframe(
    context: RunContext,
    sentences: list[dict[str, Any]],
    parsed_responses: dict[str, dict[str, str]],
    model: str,
) -> pd.DataFrame:
    labeled_df = screener.build_labeled_dataframe(sentences, parsed_responses)
    labeled_df.to_csv(context.labeled_sentences_path, index=False)
    context.save_config({"labeling_model": model})
    _refresh_artifact_manifest(context)
    return labeled_df


def _execute_labeling(
    context: RunContext,
    model: str,
    requests_per_minute: int,
    max_concurrent_requests: int,
    batch_size: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Label retrieved sentences in MCP-safe batches with resume support."""
    config = context.load_config()
    sentences = _extract_labeling_sentences(context)
    total_sentences = len(sentences)
    progress = _load_labeling_progress(context)
    if progress.get("complete") and context.labeled_sentences_path.exists():
        labeled_df = pd.read_csv(context.labeled_sentences_path)
        return labeled_df, {
            "sentence_count": len(labeled_df),
            "labeled_path": str(context.labeled_sentences_path),
            "total_sentences": total_sentences,
            "processed_sentences": int(progress.get("processed_sentences", total_sentences)),
            "batch_complete": True,
            "already_complete": True,
        }

    stored_total = int(progress.get("total_sentences", 0) or 0)
    if stored_total not in {0, total_sentences}:
        progress = {}
        jsonl_path = _labeling_responses_jsonl_path(context)
        if jsonl_path.exists():
            jsonl_path.unlink()

    next_index = int(progress.get("next_index", 0) or 0)
    straggler_seconds = _resolve_labeling_straggler_seconds(progress)
    resolved_batch_size = batch_size
    if resolved_batch_size is None:
        remaining = max(total_sentences - next_index, 0)
        resolved_batch_size = _compute_labeling_batch_size(
            remaining,
            requests_per_minute,
            max_concurrent_requests,
            straggler_seconds=straggler_seconds,
        )
    resolved_batch_size = max(1, min(resolved_batch_size, total_sentences - next_index))

    parsed_responses = _load_labeling_responses(context)
    batch = sentences[next_index : next_index + resolved_batch_size]
    batch_metrics: dict[str, float | int | None] = {}
    if batch:
        batch_responses = screener.label_sentences(
            sentences=batch,
            main_theme=str(config.get("main_theme", screener.DEFAULT_MAIN_THEME)),
            analyst_focus=str(config.get("analyst_focus", screener.DEFAULT_ANALYST_FOCUS)),
            labels=context.read_themes(),
            model=model,
            requests_per_minute=requests_per_minute,
            max_concurrent_requests=max_concurrent_requests,
            metrics_out=batch_metrics,
        )
        parsed_responses.update(batch_responses)
        _append_labeling_responses(context, batch_responses)
        next_index += len(batch)

    batch_complete = next_index >= total_sentences
    progress_payload: dict[str, Any] = {
        "total_sentences": total_sentences,
        "processed_sentences": next_index,
        "next_index": next_index,
        "batch_size": resolved_batch_size,
        "max_concurrent_requests": max_concurrent_requests,
        "labeling_straggler_seconds": round(straggler_seconds, 2),
        "estimated_batch_safe_seconds": _estimate_mcp_labeling_batch_seconds(
            len(batch),
            requests_per_minute,
            max_concurrent_requests,
            straggler_seconds=straggler_seconds,
        )
        if batch
        else 0,
        "complete": batch_complete,
        "updated_at": _utc_now(),
    }
    if batch_metrics:
        progress_payload.update(
            {
                "last_batch_elapsed_seconds": batch_metrics.get("elapsed_seconds"),
                "last_batch_requests_per_second": batch_metrics.get("requests_per_second"),
                "last_batch_request_count": batch_metrics.get("request_count"),
                "last_batch_succeeded_count": batch_metrics.get("succeeded_count"),
                "last_batch_parsed_count": batch_metrics.get("parsed_count"),
                "last_batch_latency_p50_seconds": batch_metrics.get("latency_p50_seconds"),
                "last_batch_latency_p95_seconds": batch_metrics.get("latency_p95_seconds"),
                "last_batch_latency_p99_seconds": batch_metrics.get("latency_p99_seconds"),
                "last_batch_latency_max_seconds": batch_metrics.get("latency_max_seconds"),
            }
        )
    _save_labeling_progress(context, progress_payload)

    if batch_complete:
        labeled_df = _finalize_labeled_dataframe(context, sentences, parsed_responses, model)
    else:
        labeled_df = pd.DataFrame()

    metrics: dict[str, Any] = {
        "sentence_count": len(labeled_df) if batch_complete else 0,
        "stored_responses": len(parsed_responses),
        "labeled_path": str(context.labeled_sentences_path),
        "total_sentences": total_sentences,
        "processed_sentences": next_index,
        "batch_size": resolved_batch_size,
        "batch_complete": batch_complete,
        "already_complete": False,
        "labeled_csv_ready": batch_complete,
        "max_concurrent_requests": max_concurrent_requests,
    }
    if batch_metrics:
        metrics["last_batch_elapsed_seconds"] = batch_metrics.get("elapsed_seconds")
        metrics["last_batch_requests_per_second"] = batch_metrics.get("requests_per_second")
        metrics["last_batch_latency_p50_seconds"] = batch_metrics.get("latency_p50_seconds")
        metrics["last_batch_latency_p95_seconds"] = batch_metrics.get("latency_p95_seconds")
        metrics["last_batch_latency_p99_seconds"] = batch_metrics.get("latency_p99_seconds")
        metrics["last_batch_latency_max_seconds"] = batch_metrics.get("latency_max_seconds")
        metrics["labeling_straggler_seconds"] = round(straggler_seconds, 2)
    return labeled_df, metrics


def _execute_company_summaries(
    context: RunContext,
    labeled_df: pd.DataFrame,
    model: str,
    requests_per_minute: int,
    max_concurrent_requests: int,
) -> dict[str, Any]:
    """Run company summaries from labeled sentences and return metrics."""
    config = context.load_config()
    company_summaries_df = screener.summarize_companies(
        merged_df=labeled_df,
        main_theme=str(config.get("main_theme", screener.DEFAULT_MAIN_THEME)),
        model=model,
        requests_per_minute=requests_per_minute,
        max_concurrent_requests=max_concurrent_requests,
    )
    company_summaries_df.to_csv(context.company_summaries_path, index=False)
    screener_df = screener.build_screener_dataframe(labeled_df, company_summaries_df)
    screener_df.to_csv(context.screener_results_path, index=False)
    context.save_config({"summary_model": model})
    return {
        "company_count": len(company_summaries_df),
        "summaries_path": str(context.company_summaries_path),
        "screener_results_path": str(context.screener_results_path),
    }


def run_enrichment(
    run_id: str,
    requests_per_minute: int = DEFAULT_ENRICHMENT_REQUESTS_PER_MINUTE,
    max_concurrent_requests: int = DEFAULT_ENRICHMENT_MAX_CONCURRENT,
) -> dict[str, Any]:
    """Run labeling and company summaries after enrichment approval."""
    _load_environment()
    context = _context(run_id)
    approval = _require_enrichment_approval(context)
    labeling_model = str(approval.get("labeling_model", screener.DEFAULT_LABELING_MODEL))
    summary_model = str(approval.get("summary_model", screener.DEFAULT_SUMMARY_MODEL))
    rpm = int(approval.get("requests_per_minute", requests_per_minute))
    max_concurrent = int(approval.get("max_concurrent_requests", max_concurrent_requests))

    labeled_df: pd.DataFrame | None = None
    labeling_metrics: dict[str, Any] = {}
    while True:
        labeled_df, labeling_metrics = _execute_labeling(
            context,
            model=labeling_model,
            requests_per_minute=rpm,
            max_concurrent_requests=max_concurrent,
        )
        if labeling_metrics.get("batch_complete") or labeling_metrics.get("already_complete"):
            break
    if labeled_df is None:
        raise McpWorkflowError("Labeling did not produce a labeled dataframe")
    summary_metrics = _execute_company_summaries(
        context,
        labeled_df,
        model=summary_model,
        requests_per_minute=rpm,
        max_concurrent_requests=max_concurrent,
    )
    _refresh_artifact_manifest(context)

    return _response(
        run_id=context.run_name,
        stage="enrichment",
        status="completed",
        summary=(
            f"Labeled {labeling_metrics['sentence_count']:,} sentences and generated "
            f"{summary_metrics['company_count']:,} company summaries."
        ),
        labeling=labeling_metrics,
        company_summaries=summary_metrics,
        artifacts=[
            _artifact_handle("labeled_sentences", "labeled_sentences.csv", queryable=True),
            _artifact_handle("company_summaries", "company_summaries.csv", queryable=True),
            _artifact_handle("screener_results", "screener_results.csv", queryable=True),
        ],
        next_actions=["get_run_summary", "query_artifact", "export_artifact"],
    )


def run_labeling(
    run_id: str,
    model: str = screener.DEFAULT_LABELING_MODEL,
    requests_per_minute: int = DEFAULT_ENRICHMENT_REQUESTS_PER_MINUTE,
    max_concurrent_requests: int = DEFAULT_ENRICHMENT_MAX_CONCURRENT,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Label retrieved chunks after enrichment approval (batched for MCP timeouts)."""
    _load_environment()
    context = _context(run_id)
    approval = _require_enrichment_approval(context)
    resolved_model = str(approval.get("labeling_model", model))
    rpm = int(approval.get("requests_per_minute", requests_per_minute))
    max_concurrent = int(approval.get("max_concurrent_requests", max_concurrent_requests))
    labeled_df, metrics = _execute_labeling(
        context,
        model=resolved_model,
        requests_per_minute=rpm,
        max_concurrent_requests=max_concurrent,
        batch_size=batch_size,
    )

    if metrics.get("already_complete"):
        summary = (
            f"Labeling already complete with {metrics['sentence_count']:,} evidence rows "
            "in labeled_sentences.csv."
        )
        next_actions = ["run_company_summaries", "get_run_summary"]
        status: StageStatus = "completed"
    elif metrics.get("batch_complete"):
        summary = (
            f"Labeled {metrics['sentence_count']:,} evidence rows from "
            f"{metrics['processed_sentences']:,} retrieved sentences."
        )
        next_actions = ["run_company_summaries", "get_run_summary"]
        status = "completed"
    else:
        summary = (
            f"Stored labeling batch through sentence {metrics['processed_sentences']:,} of "
            f"{metrics['total_sentences']:,} "
            f"({metrics['stored_responses']:,} responses saved). "
            "Call `run_labeling` again to continue; labeled_sentences.csv is written at the end."
        )
        next_actions = ["run_labeling", "get_run_summary"]
        status = "running"

    return _response(
        run_id=context.run_name,
        stage="labeling",
        status=status,
        summary=summary,
        labeling_progress=metrics,
        enrichment_status=_enrichment_status(context),
        artifacts=[_artifact_handle("labeled_sentences", "labeled_sentences.csv", queryable=True)],
        next_actions=next_actions,
    )


def run_company_summaries(
    run_id: str,
    model: str = screener.DEFAULT_SUMMARY_MODEL,
    source: str = "labeled_sentences",
) -> dict[str, Any]:
    """Generate company summaries after labeling (legacy split step)."""
    _load_environment()
    context = _context(run_id)
    approval = _require_enrichment_approval(context)
    if source != "labeled_sentences":
        raise McpWorkflowError(
            "Company summaries require source='labeled_sentences'. "
            "Run labeling first or use run_enrichment for the combined step."
        )
    if not context.labeled_sentences_path.exists():
        raise McpWorkflowError("Run labeling before company summaries")

    merged_df = pd.read_csv(context.labeled_sentences_path)
    resolved_model = str(approval.get("summary_model", model))
    rpm = int(approval.get("requests_per_minute", DEFAULT_ENRICHMENT_REQUESTS_PER_MINUTE))
    max_concurrent = int(
        approval.get("max_concurrent_requests", DEFAULT_ENRICHMENT_MAX_CONCURRENT)
    )
    metrics = _execute_company_summaries(
        context,
        labeled_df=merged_df,
        model=resolved_model,
        requests_per_minute=rpm,
        max_concurrent_requests=max_concurrent,
    )
    _refresh_artifact_manifest(context)

    return _response(
        run_id=context.run_name,
        stage="company_summaries",
        status="completed",
        summary=(
            f"Generated {metrics['company_count']:,} company summaries and final screener CSV."
        ),
        artifacts=[
            _artifact_handle("company_summaries", "company_summaries.csv", queryable=True),
            _artifact_handle("screener_results", "screener_results.csv", queryable=True),
        ],
        next_actions=["get_run_summary", "query_artifact", "export_artifact"],
    )


def get_run_summary(run_id: str) -> dict[str, Any]:
    """Return a compact summary of run state and key artifacts."""
    context = _context(run_id)
    config = context.load_config()
    artifacts = _refresh_artifact_manifest(context)
    digest_path = context.run_dir / EVIDENCE_DIGEST_FILENAME
    digest_preview: dict[str, Any] | None = None
    if digest_path.exists():
        digest = _read_json(digest_path)
        if isinstance(digest, dict):
            digest_preview = {
                "retrieval_stats": digest.get("retrieval_stats", {}),
                "top_signals": digest.get("top_signals", [])[:5],
                "top_companies": digest.get("top_companies", [])[:10],
            }
    return _response(
        run_id=context.run_name,
        stage="artifact_access",
        status="completed",
        summary=f"Run has {len(artifacts)} artifacts available.",
        run_config=config,
        digest_preview=digest_preview,
        enrichment_status=_enrichment_status(context),
        artifacts=artifacts,
        next_actions=[
            "list_artifacts",
            "get_artifact_preview",
            "query_artifact",
            "export_artifact",
        ],
    )


def list_artifacts(run_id: str) -> dict[str, Any]:
    """List existing run artifacts."""
    context = _context(run_id)
    artifacts = _refresh_artifact_manifest(context)
    return _response(
        run_id=context.run_name,
        stage="artifact_access",
        status="completed",
        summary=f"Found {len(artifacts)} artifacts for this run.",
        artifacts=artifacts,
        next_actions=["get_artifact_preview", "query_artifact", "export_artifact"],
    )


def get_artifact_preview(
    run_id: str,
    artifact_id: str,
    limit: int = JSON_PREVIEW_ITEMS,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Return a bounded artifact preview."""
    context = _context(run_id)
    path = _artifact_path(context, artifact_id)
    offset = _cursor_to_offset(cursor)
    preview, next_cursor, truncated = _preview_path(context, artifact_id, path, limit, offset)
    size_bytes = path.stat().st_size if path.is_file() else None
    return _response(
        run_id=context.run_name,
        stage="artifact_access",
        status="completed",
        summary=f"Previewing artifact {artifact_id}.",
        artifact_id=artifact_id,
        preview=preview,
        truncated=truncated,
        next_cursor=next_cursor,
        size_bytes=size_bytes,
        large_artifact=bool(size_bytes and size_bytes > LARGE_ARTIFACT_THRESHOLD_BYTES),
    )


def _cursor_to_offset(cursor: str | None) -> int:
    if cursor is None:
        return 0
    if cursor.startswith("offset:"):
        return int(cursor.split(":", maxsplit=1)[1])
    return 0


def _preview_path(
    context: RunContext,
    artifact_id: str,
    path: Path,
    limit: int,
    offset: int,
) -> tuple[Any, str | None, bool]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    if artifact_id == "results":
        rows = _enrich_chunk_rows(context, _flatten_result_chunks(_load_results(context)))
        page = [_compact_query_row(row) for row in rows[offset : offset + limit]]
        next_cursor = f"offset:{offset + limit}" if offset + limit < len(rows) else None
        return page, next_cursor, next_cursor is not None

    if path.is_dir():
        files = sorted(file_path.name for file_path in path.glob("*"))
        rows = files[offset : offset + limit]
        next_cursor = f"offset:{offset + limit}" if offset + limit < len(files) else None
        return rows, next_cursor, next_cursor is not None

    file_size = path.stat().st_size if path.is_file() else 0
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        preview_df = df.iloc[offset : offset + limit, :CSV_PREVIEW_COLUMNS]
        records = [_compact_query_row(row) for row in preview_df.to_dict(orient="records")]
        next_cursor = f"offset:{offset + limit}" if offset + limit < len(df) else None
        return records, next_cursor, next_cursor is not None

    if path.suffix == ".json":
        if file_size > LARGE_ARTIFACT_THRESHOLD_BYTES:
            return (
                {
                    "preview_type": "metadata_only",
                    "size_bytes": file_size,
                    "message": (
                        "Artifact exceeds 1MB. Use query_artifact with filters or "
                        "export_artifact for the local file path."
                    ),
                },
                None,
                True,
            )
        payload = _read_json(path)
        if isinstance(payload, list):
            rows = payload[offset : offset + limit]
            next_cursor = f"offset:{offset + limit}" if offset + limit < len(payload) else None
            compact_rows = [
                _compact_query_row(row) if isinstance(row, dict) else row for row in rows
            ]
            return compact_rows, next_cursor, next_cursor is not None
        if isinstance(payload, dict):
            preview_text = json.dumps(payload, default=str)
            if len(preview_text) > TEXT_PREVIEW_CHARS:
                return (
                    {
                        "preview_type": "truncated_json",
                        "keys": list(payload.keys())[:JSON_PREVIEW_ITEMS],
                        "preview_text": preview_text[:TEXT_PREVIEW_CHARS],
                    },
                    None,
                    True,
                )
        return payload, None, False

    text = path.read_text(encoding="utf-8")
    truncated = len(text) > TEXT_PREVIEW_CHARS
    return text[:TEXT_PREVIEW_CHARS], None, truncated


def query_artifact(
    run_id: str,
    artifact_id: str,
    filters: dict[str, Any] | None = None,
    text_query: str | None = None,
    limit: int = 25,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Filter a queryable artifact."""
    context = _context(run_id)
    path = _artifact_path(context, artifact_id)
    offset = _cursor_to_offset(cursor)
    if artifact_id == "results":
        page, total_matches = _query_result_rows(context, filters or {}, text_query, offset, limit)
    else:
        rows = _artifact_rows(context, artifact_id, path)
        filtered = _filter_rows(rows, filters or {}, text_query)
        total_matches = len(filtered)
        page = [_compact_query_row(row) for row in filtered[offset : offset + limit]]
    next_cursor = f"offset:{offset + limit}" if offset + limit < total_matches else None
    return _response(
        run_id=context.run_name,
        stage="artifact_access",
        status="completed",
        summary=f"Query returned {total_matches:,} matching rows from {artifact_id}.",
        artifact_id=artifact_id,
        rows=page,
        total_matches=total_matches,
        next_cursor=next_cursor,
    )


def _query_result_rows(
    context: RunContext,
    filters: dict[str, Any],
    text_query: str | None,
    offset: int,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    matches: list[dict[str, Any]] = []
    total_matches = 0
    for row in _enrich_chunk_rows(context, _flatten_result_chunks(_load_results(context))):
        if not _row_matches(row, filters, text_query):
            continue
        if total_matches >= offset and len(matches) < limit:
            matches.append(_compact_query_row(row))
        total_matches += 1
    return matches, total_matches


def _row_matches(row: dict[str, Any], filters: dict[str, Any], text_query: str | None) -> bool:
    for key, expected in filters.items():
        expected_values = expected if isinstance(expected, list) else [expected]
        row_value = str(row.get(key, ""))
        if all(row_value != str(value) for value in expected_values):
            return False
    if text_query and text_query.lower() not in json.dumps(row, default=str).lower():
        return False
    return True


def _artifact_rows(context: RunContext, artifact_id: str, path: Path) -> list[dict[str, Any]]:
    if artifact_id == "results":
        return _enrich_chunk_rows(context, _flatten_result_chunks(_load_results(context)))
    if path.suffix == ".csv":
        return pd.read_csv(path).fillna("").to_dict(orient="records")
    if artifact_id == "evidence_digest":
        payload = _read_json(path)
        if not isinstance(payload, dict):
            return []
        rows: list[dict[str, Any]] = []
        for key in ("top_signals", "top_companies", "representative_chunks", "source_mix"):
            values = payload.get(key, [])
            if isinstance(values, list):
                rows.extend([{**row, "section": key} for row in values if isinstance(row, dict)])
        return rows
    if path.suffix == ".json":
        payload = _read_json(path)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            return [payload]
    return []


def _filter_rows(
    rows: list[dict[str, Any]],
    filters: dict[str, Any],
    text_query: str | None,
) -> list[dict[str, Any]]:
    return [row for row in rows if _row_matches(row, filters, text_query)]


def export_artifact(run_id: str, artifact_id: str) -> dict[str, Any]:
    """Return an artifact path for host-side access."""
    context = _context(run_id)
    path = _artifact_path(context, artifact_id)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    size_bytes = path.stat().st_size if path.is_file() else None
    access_note = None
    if size_bytes and size_bytes > LARGE_ARTIFACT_THRESHOLD_BYTES:
        access_note = (
            "Artifact exceeds 1MB. Read it from the exported local path; do not attempt "
            "to load the full file inline in chat."
        )
    return _response(
        run_id=context.run_name,
        stage="artifact_access",
        status="completed",
        summary=f"Artifact {artifact_id} is available at {path}.",
        artifact_id=artifact_id,
        export_path=str(path),
        content_type=_path_content_type(path),
        size_bytes=size_bytes,
        large_artifact=bool(size_bytes and size_bytes > LARGE_ARTIFACT_THRESHOLD_BYTES),
        access_note=access_note,
    )
