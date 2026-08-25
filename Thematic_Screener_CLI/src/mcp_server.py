"""FastMCP server for the Thematic Screener workflow."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from typing import Any

from mcp.server.fastmcp import FastMCP

from src import mcp_workflow

SERVER_NAME = "thematic-screener"

logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)

mcp = FastMCP(SERVER_NAME)


def _normalize_tool_universe(universe: dict[str, Any] | str | None) -> dict[str, Any] | None:
    """Coerce MCP client universe payloads before workflow handling."""
    return mcp_workflow._coerce_universe_input(universe)


def _run_workflow(func: Callable[..., dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Keep third-party progress prints off MCP stdout."""
    with redirect_stdout(sys.stderr):
        return func(**kwargs)


@mcp.tool()
def create_run(
    main_theme: str,
    analyst_focus: str,
    run_id: str | None = None,
    universe: dict[str, Any] | str | None = None,
    start_date: str = mcp_workflow.screener.DEFAULT_START_DATE,
    end_date: str = mcp_workflow.screener.DEFAULT_END_DATE,
    output_goal: str | None = None,
    taxonomy_style: str = "exposure",
    ground_with_bigdata: bool = False,
) -> dict[str, Any]:
    """Create a run, persist the run brief, and validate the universe."""
    return _run_workflow(
        mcp_workflow.create_run,
        main_theme=main_theme,
        analyst_focus=analyst_focus,
        run_id=run_id,
        universe=_normalize_tool_universe(universe),
        start_date=start_date,
        end_date=end_date,
        output_goal=output_goal,
        taxonomy_style=taxonomy_style,
        ground_with_bigdata=ground_with_bigdata,
    )


@mcp.tool()
def validate_universe(
    run_id: str,
    universe: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Validate and normalize a universe CSV or inline RP entity ID list."""
    return _run_workflow(
        mcp_workflow.validate_universe,
        run_id=run_id,
        universe=_normalize_tool_universe(universe),
    )


@mcp.tool()
def generate_mindmap(
    run_id: str,
    max_leaf_labels: int | None = None,
    model: str = mcp_workflow.screener.DEFAULT_LABELS_MODEL,
    taxonomy_style: str | None = None,
    ground_with_bigdata: bool | None = None,
) -> dict[str, Any]:
    """Generate and store a taxonomy tree for the run theme."""
    return _run_workflow(
        mcp_workflow.generate_mindmap,
        run_id=run_id,
        max_leaf_labels=max_leaf_labels,
        model=model,
        taxonomy_style=taxonomy_style,
        ground_with_bigdata=ground_with_bigdata,
    )


@mcp.tool()
def validate_mindmap(run_id: str) -> dict[str, Any]:
    """Validate taxonomy quality before search planning."""
    return _run_workflow(mcp_workflow.validate_mindmap, run_id=run_id)


@mcp.tool()
def update_mindmap(
    run_id: str,
    instructions: str,
    model: str = mcp_workflow.screener.DEFAULT_LABELS_MODEL,
) -> dict[str, Any]:
    """Apply user revision instructions to the stored taxonomy."""
    return _run_workflow(
        mcp_workflow.update_mindmap,
        run_id=run_id,
        instructions=instructions,
        model=model,
    )


@mcp.tool()
def build_search_plans(
    run_id: str,
    category: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build Bigdata search plans without retrieving documents."""
    return _run_workflow(mcp_workflow.build_search_plans, run_id=run_id, category=category)


@mcp.tool()
def estimate_retrieval_budget(
    run_id: str,
    retrieval_cost_usd_per_10_chunks: float = mcp_workflow.RETRIEVAL_COST_USD_PER_UNIT,
    presets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Estimate chunk counts and dollar cost before retrieval."""
    return _run_workflow(
        mcp_workflow.estimate_retrieval_budget,
        run_id=run_id,
        retrieval_cost_usd_per_10_chunks=retrieval_cost_usd_per_10_chunks,
        presets=presets,
    )


@mcp.tool()
def approve_budget(run_id: str, selection: dict[str, Any]) -> dict[str, Any]:
    """Persist an approved retrieval budget."""
    return _run_workflow(mcp_workflow.approve_budget, run_id=run_id, selection=selection)


@mcp.tool()
def run_retrieval(
    run_id: str,
    requests_per_minute: int = mcp_workflow.screener.DEFAULT_REQUESTS_PER_MINUTE,
) -> dict[str, Any]:
    """Run retrieval after budget approval."""
    return _run_workflow(
        mcp_workflow.run_retrieval,
        run_id=run_id,
        requests_per_minute=requests_per_minute,
    )


@mcp.tool()
def summarize_retrieval(
    run_id: str,
    max_representative_chunks: int = 30,
) -> dict[str, Any]:
    """Create an evidence digest over retrieved chunks."""
    return _run_workflow(
        mcp_workflow.summarize_retrieval,
        run_id=run_id,
        max_representative_chunks=max_representative_chunks,
    )


@mcp.tool()
def estimate_enrichment_budget(
    run_id: str,
    labeling_model: str = mcp_workflow.screener.DEFAULT_LABELING_MODEL,
    summary_model: str = mcp_workflow.screener.DEFAULT_SUMMARY_MODEL,
    requests_per_minute: int = mcp_workflow.DEFAULT_ENRICHMENT_REQUESTS_PER_MINUTE,
    max_concurrent_requests: int = mcp_workflow.DEFAULT_ENRICHMENT_MAX_CONCURRENT,
    labeling_input_usd_per_mtok: float = mcp_workflow.DEFAULT_LABELING_INPUT_USD_PER_MTOK,
    labeling_output_usd_per_mtok: float = mcp_workflow.DEFAULT_LABELING_OUTPUT_USD_PER_MTOK,
    summary_input_usd_per_mtok: float = mcp_workflow.DEFAULT_SUMMARY_INPUT_USD_PER_MTOK,
    summary_output_usd_per_mtok: float = mcp_workflow.DEFAULT_SUMMARY_OUTPUT_USD_PER_MTOK,
    safety_margin: float = mcp_workflow.ENRICHMENT_COST_SAFETY_MARGIN,
) -> dict[str, Any]:
    """Estimate OpenAI cost and latency for labeling plus company summaries."""
    return _run_workflow(
        mcp_workflow.estimate_enrichment_budget,
        run_id=run_id,
        labeling_model=labeling_model,
        summary_model=summary_model,
        requests_per_minute=requests_per_minute,
        max_concurrent_requests=max_concurrent_requests,
        labeling_input_usd_per_mtok=labeling_input_usd_per_mtok,
        labeling_output_usd_per_mtok=labeling_output_usd_per_mtok,
        summary_input_usd_per_mtok=summary_input_usd_per_mtok,
        summary_output_usd_per_mtok=summary_output_usd_per_mtok,
        safety_margin=safety_margin,
    )


@mcp.tool()
def approve_enrichment(
    run_id: str,
    approved: bool = True,
    labeling_model: str | None = None,
    summary_model: str | None = None,
) -> dict[str, Any]:
    """Persist user approval for the enrichment budget."""
    return _run_workflow(
        mcp_workflow.approve_enrichment,
        run_id=run_id,
        approved=approved,
        labeling_model=labeling_model,
        summary_model=summary_model,
    )


@mcp.tool()
def run_enrichment(
    run_id: str,
    requests_per_minute: int = mcp_workflow.DEFAULT_ENRICHMENT_REQUESTS_PER_MINUTE,
    max_concurrent_requests: int = mcp_workflow.DEFAULT_ENRICHMENT_MAX_CONCURRENT,
) -> dict[str, Any]:
    """Run labeling and company summaries after enrichment approval."""
    return _run_workflow(
        mcp_workflow.run_enrichment,
        run_id=run_id,
        requests_per_minute=requests_per_minute,
        max_concurrent_requests=max_concurrent_requests,
    )


@mcp.tool()
def run_labeling(
    run_id: str,
    model: str = mcp_workflow.screener.DEFAULT_LABELING_MODEL,
    requests_per_minute: int = mcp_workflow.DEFAULT_ENRICHMENT_REQUESTS_PER_MINUTE,
    max_concurrent_requests: int = mcp_workflow.DEFAULT_ENRICHMENT_MAX_CONCURRENT,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Label retrieved chunks in MCP-safe batches after enrichment approval."""
    return _run_workflow(
        mcp_workflow.run_labeling,
        run_id=run_id,
        model=model,
        requests_per_minute=requests_per_minute,
        max_concurrent_requests=max_concurrent_requests,
        batch_size=batch_size,
    )


@mcp.tool()
def run_company_summaries(
    run_id: str,
    model: str = mcp_workflow.screener.DEFAULT_SUMMARY_MODEL,
    source: str = "labeled_sentences",
) -> dict[str, Any]:
    """Optionally generate company summaries and final screener output."""
    return _run_workflow(
        mcp_workflow.run_company_summaries,
        run_id=run_id,
        model=model,
        source=source,
    )


@mcp.tool()
def get_run_summary(run_id: str) -> dict[str, Any]:
    """Return a compact summary of run state, artifacts, and key findings."""
    return _run_workflow(mcp_workflow.get_run_summary, run_id=run_id)


@mcp.tool()
def list_artifacts(run_id: str) -> dict[str, Any]:
    """List artifacts available for a run."""
    return _run_workflow(mcp_workflow.list_artifacts, run_id=run_id)


@mcp.tool()
def get_artifact_preview(
    run_id: str,
    artifact_id: str,
    limit: int = mcp_workflow.JSON_PREVIEW_ITEMS,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Return a bounded preview for an artifact."""
    return _run_workflow(
        mcp_workflow.get_artifact_preview,
        run_id=run_id,
        artifact_id=artifact_id,
        limit=limit,
        cursor=cursor,
    )


@mcp.tool()
def query_artifact(
    run_id: str,
    artifact_id: str,
    filters: dict[str, Any] | None = None,
    text_query: str | None = None,
    limit: int = 25,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Filter rows/chunks inside a queryable artifact."""
    return _run_workflow(
        mcp_workflow.query_artifact,
        run_id=run_id,
        artifact_id=artifact_id,
        filters=filters,
        text_query=text_query,
        limit=limit,
        cursor=cursor,
    )


@mcp.tool()
def export_artifact(run_id: str, artifact_id: str) -> dict[str, Any]:
    """Return the local path for a full artifact."""
    return _run_workflow(mcp_workflow.export_artifact, run_id=run_id, artifact_id=artifact_id)


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
