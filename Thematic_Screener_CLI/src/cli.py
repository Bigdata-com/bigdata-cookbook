"""Command-line interface for the thematic screener pipeline.

Subcommands mirror the four pipeline stages and each writes its artifacts
into an isolated run directory (``runs/<run_name>/``):

    generate-labels  Generate theme labels from a main theme + analyst focus.
    plans            Build search plans for a company universe CSV.
    search           Execute the plans and store deduplicated results.
    label-sentences  Label sentences, summarize companies, export the screener CSV.
    summarize-plans  Summarize all search plans in a plans folder.
    run-all          Run every stage in sequence within one isolated run.

Run with, e.g.::

    python -m src.cli run-all --run-name my_run
    python -m src.cli generate-labels --main-theme "..." --analyst-focus "..."
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src import screener
from src.run_context import RunContext

logger = logging.getLogger("screener.cli")

DEFAULT_RUNS_ROOT = "runs"
DEFAULT_UNIVERSE = "XNAS_companies.csv"


def _load_environment() -> None:
    """Load .env and validate the required API keys."""
    env_path = Path.cwd() / ".env"
    load_dotenv(env_path, override=False)

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set; add it to your .env or environment")
    if not os.getenv("BIGDATA_API_KEY"):
        raise SystemExit("BIGDATA_API_KEY is not set; add it to your .env or environment")

    logger.info("Loaded environment from %s", env_path)


def _make_context(args: argparse.Namespace) -> RunContext:
    context = RunContext.create(args.runs_root, args.run_name)
    context.ensure_run_dir()
    logger.info("Using run directory: %s", context.run_dir)
    return context


def _resolve(args: argparse.Namespace, config: dict[str, Any], key: str, default: Any) -> Any:
    """Pick an explicit CLI flag, else the persisted config, else a built-in default."""
    if key in vars(args):
        return getattr(args, key)
    if config.get(key) is not None:
        return config[key]
    return default


def _from_config(config: dict[str, Any], key: str, default: Any) -> Any:
    """Pick a persisted config value, else a built-in default."""
    if config.get(key) is not None:
        return config[key]
    return default


def run_labels(context: RunContext, args: argparse.Namespace) -> list[str]:
    config = context.load_config()
    main_theme = _resolve(args, config, "main_theme", screener.DEFAULT_MAIN_THEME)
    analyst_focus = _resolve(args, config, "analyst_focus", screener.DEFAULT_ANALYST_FOCUS)
    model = _resolve(args, config, "labels_model", screener.DEFAULT_LABELS_MODEL)

    logger.info("Generating labels for theme: %s", main_theme)
    labels = screener.generate_labels(
        main_theme=main_theme,
        analyst_focus=analyst_focus,
        model=model,
    )

    context.write_themes(labels)
    context.save_config(
        {
            "main_theme": main_theme,
            "analyst_focus": analyst_focus,
            "labels_model": model,
        }
    )
    logger.info("Wrote %d labels to %s", len(labels), context.themes_path)
    return labels


def run_plans(context: RunContext, args: argparse.Namespace) -> list[Path]:
    config = context.load_config()
    labels = context.read_themes()

    universe_path = _resolve(args, config, "universe", DEFAULT_UNIVERSE)
    start_date = _resolve(args, config, "start_date", screener.DEFAULT_START_DATE)
    end_date = _resolve(args, config, "end_date", screener.DEFAULT_END_DATE)

    universe_df = screener.load_universe(universe_path)
    company_ids = universe_df[screener.UNIVERSE_ID_COLUMN].tolist()
    logger.info(
        "Planning %d labels over %d companies from %s",
        len(labels),
        len(company_ids),
        universe_path,
    )

    context.ensure_plans_dir()
    saved = screener.build_plans(
        labels=labels,
        company_ids=company_ids,
        plans_dir=context.plans_dir,
        start_date=start_date,
        end_date=end_date,
    )

    context.save_config(
        {
            "universe": str(universe_path),
            "start_date": start_date,
            "end_date": end_date,
        }
    )
    logger.info("Saved %d plans to %s", len(saved), context.plans_dir)
    return saved


def run_search(context: RunContext, args: argparse.Namespace) -> list[dict[str, Any]]:
    config = context.load_config()
    chunk_percentage = _resolve(args, config, "chunk_percentage", screener.DEFAULT_CHUNK_PERCENTAGE)
    requests_per_minute = _resolve(
        args, config, "requests_per_minute", screener.DEFAULT_REQUESTS_PER_MINUTE
    )

    results = screener.run_search(
        plans_dir=context.plans_dir,
        chunk_percentage=chunk_percentage,
        requests_per_minute=requests_per_minute,
    )

    context.ensure_results_dir()
    with context.results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, default=str)

    context.save_config(
        {
            "chunk_percentage": chunk_percentage,
            "requests_per_minute": requests_per_minute,
        }
    )
    logger.info("Retrieved %d deduplicated documents -> %s", len(results), context.results_path)
    return results


def _load_results(context: RunContext) -> list[dict[str, Any]]:
    if not context.results_path.exists():
        raise FileNotFoundError(
            f"results not found at {context.results_path}; run the 'search' step first"
        )
    with context.results_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run_label(
    context: RunContext,
    args: argparse.Namespace,
    results: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    config = context.load_config()
    main_theme = _from_config(config, "main_theme", screener.DEFAULT_MAIN_THEME)
    labeling_model = _resolve(args, config, "labeling_model", screener.DEFAULT_LABELING_MODEL)
    summary_model = _resolve(args, config, "summary_model", screener.DEFAULT_SUMMARY_MODEL)
    universe_path = _from_config(config, "universe", DEFAULT_UNIVERSE)

    labels = context.read_themes()
    if results is None:
        results = _load_results(context)

    universe_df = screener.load_universe(universe_path)

    sentences = screener.extract_sentences(results, universe_df)
    logger.info("Extracted %d sentences", len(sentences))

    parsed_responses = screener.label_sentences(
        sentences=sentences,
        main_theme=main_theme,
        labels=labels,
        model=labeling_model,
    )
    merged_df = screener.build_labeled_dataframe(sentences, parsed_responses)
    merged_df.to_csv(context.labeled_sentences_path, index=False)
    logger.info("Wrote %d labeled sentences to %s", len(merged_df), context.labeled_sentences_path)

    company_summaries_df = screener.summarize_companies(
        merged_df=merged_df,
        main_theme=main_theme,
        model=summary_model,
    )
    company_summaries_df.to_csv(context.company_summaries_path, index=False)
    logger.info(
        "Wrote %d company summaries to %s",
        len(company_summaries_df),
        context.company_summaries_path,
    )

    screener_df = screener.build_screener_dataframe(merged_df, company_summaries_df)
    screener_df.to_csv(context.screener_results_path, index=False)

    context.save_config(
        {
            "labeling_model": labeling_model,
            "summary_model": summary_model,
        }
    )
    logger.info(
        "Wrote final screener results (%d rows) to %s",
        len(screener_df),
        context.screener_results_path,
    )
    return screener_df


def run_all(context: RunContext, args: argparse.Namespace) -> None:
    run_labels(context, args)
    run_plans(context, args)
    results = run_search(context, args)
    run_label(context, args, results=results)
    logger.info("run-all complete for %s", context.run_dir)


def _cmd_labels(args: argparse.Namespace) -> None:
    run_labels(_make_context(args), args)


def _cmd_plans(args: argparse.Namespace) -> None:
    run_plans(_make_context(args), args)


def _cmd_search(args: argparse.Namespace) -> None:
    run_search(_make_context(args), args)


def _cmd_label(args: argparse.Namespace) -> None:
    run_label(_make_context(args), args)


def _cmd_run_all(args: argparse.Namespace) -> None:
    run_all(_make_context(args), args)


def run_summarize_plans(args: argparse.Namespace) -> None:
    if getattr(args, "plans_dir", None):
        plans_dir = Path(args.plans_dir)
    else:
        plans_dir = _make_context(args).plans_dir

    summary_df = screener.summarize_plans(plans_dir)
    print(screener.format_plans_summary(summary_df))
    logger.info("Summarized %d plans from %s", len(summary_df), plans_dir)


def _cmd_summarize_plans(args: argparse.Namespace) -> None:
    run_summarize_plans(args)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-name",
        default=None,
        help="Unique run name (defaults to a UTC timestamp).",
    )
    parser.add_argument(
        "--runs-root",
        default=DEFAULT_RUNS_ROOT,
        help="Parent directory that holds all runs (default: runs).",
    )


def _add_labels_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--main-theme", action="store", default=argparse.SUPPRESS, help="Main screening theme.")
    parser.add_argument(
        "--analyst-focus", action="store", default=argparse.SUPPRESS, help="Analyst focus for the taxonomy."
    )
    parser.add_argument(
        "--labels-model", action="store", default=argparse.SUPPRESS, help="Model used to generate labels."
    )


def _add_plans_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--universe", action="store", default=argparse.SUPPRESS, help="Path to the company universe CSV.")
    parser.add_argument(
        "--start-date", action="store", default=argparse.SUPPRESS, help="Search start date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end-date", action="store", default=argparse.SUPPRESS, help="Search end date (YYYY-MM-DD)."
    )


def _add_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--chunk-percentage",
        type=float,
        action="store",
        default=argparse.SUPPRESS,
        help="Fraction of chunks to retrieve per basket (default: 0.02).",
    )
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        action="store",
        default=argparse.SUPPRESS,
        help="Requests-per-minute cap for the search (default: 350).",
    )


def _add_label_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--labeling-model", action="store", default=argparse.SUPPRESS, help="Model used to label sentences."
    )
    parser.add_argument(
        "--summary-model", action="store", default=argparse.SUPPRESS, help="Model used to summarize companies."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="screener",
        description="Thematic screener pipeline CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    labels_parser = subparsers.add_parser("generate-labels", help="Generate theme labels.")
    _add_common_args(labels_parser)
    _add_labels_args(labels_parser)
    labels_parser.set_defaults(func=_cmd_labels, requires_api_keys=True)

    plans_parser = subparsers.add_parser("plans", help="Build search plans.")
    _add_common_args(plans_parser)
    _add_plans_args(plans_parser)
    plans_parser.set_defaults(func=_cmd_plans, requires_api_keys=True)

    search_parser = subparsers.add_parser("search", help="Execute search plans.")
    _add_common_args(search_parser)
    _add_search_args(search_parser)
    search_parser.set_defaults(func=_cmd_search, requires_api_keys=True)

    label_parser = subparsers.add_parser("label-sentences", help="Label sentences and summarize.")
    _add_common_args(label_parser)
    _add_label_args(label_parser)
    label_parser.set_defaults(func=_cmd_label, requires_api_keys=True)

    run_all_parser = subparsers.add_parser("run-all", help="Run every stage in sequence.")
    _add_common_args(run_all_parser)
    _add_labels_args(run_all_parser)
    _add_plans_args(run_all_parser)
    _add_search_args(run_all_parser)
    run_all_parser.add_argument(
        "--labeling-model",
        action="store",
        default=argparse.SUPPRESS,
        help="Model used to label sentences.",
    )
    run_all_parser.add_argument(
        "--summary-model",
        action="store",
        default=argparse.SUPPRESS,
        help="Model used to summarize companies.",
    )
    run_all_parser.set_defaults(func=_cmd_run_all, requires_api_keys=True)

    summarize_plans_parser = subparsers.add_parser(
        "summarize-plans",
        help="Summarize all search plans in a plans folder.",
    )
    _add_common_args(summarize_plans_parser)
    summarize_plans_parser.add_argument(
        "--plans-dir",
        action="store",
        default=argparse.SUPPRESS,
        help="Path to a plans folder (default: runs/<run_name>/plans).",
    )
    summarize_plans_parser.set_defaults(func=_cmd_summarize_plans, requires_api_keys=False)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "requires_api_keys", True):
        _load_environment()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
