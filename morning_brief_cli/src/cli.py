"""Command-line interface for the morning brief pipeline.

Subcommands mirror the three pipeline stages and each writes its artifacts
into an isolated run directory (``runs/<run_name>/``):

    plan        Build smart-batching search plans for each research topic.
    search      Execute the plans and store per-topic results.
    compile     Summarise results and render the brief (Markdown and/or HTML).
    generate    Run every stage in sequence (plan → search → compile).
    show-plans  Print per-topic chunk estimates without making API calls.

Run with, e.g.::

    python -m src.cli generate --run-name mag7_brief
    python -m src.cli generate --portfolio my_portfolio.csv --format html
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src import brief as brief_module
from src.brief import BriefData, PortfolioCompany
from src.renderer import write_brief
from src.run_context import RunContext
from src.topics import TOPICS

logger = logging.getLogger("morning_brief.cli")

DEFAULT_RUNS_ROOT = "runs"
DEFAULT_PORTFOLIO = "mag7_companies.csv"
DEFAULT_FORMAT = "both"


# ──────────────────────────────────────────────── environment ─────────────────


def _load_environment() -> None:
    """Load .env and validate the required API keys."""
    env_path = Path.cwd() / ".env"
    load_dotenv(env_path, override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set — add it to your .env or environment")
    if not os.getenv("BIGDATA_API_KEY"):
        raise SystemExit("BIGDATA_API_KEY is not set — add it to your .env or environment")
    logger.info("Loaded environment from %s", env_path)


# ──────────────────────────────────────────────── helpers ─────────────────────


def _make_context(args: argparse.Namespace) -> RunContext:
    context = RunContext.create(args.runs_root, args.run_name)
    context.ensure_run_dir()
    logger.info("Run directory: %s", context.run_dir)
    return context


def _resolve(args: argparse.Namespace, config: dict[str, Any], key: str, default: Any) -> Any:
    """Precedence: explicit CLI flag > persisted config > built-in default."""
    if key in vars(args) and getattr(args, key) is not None:
        return getattr(args, key)
    if config.get(key) is not None:
        return config[key]
    return default


def _from_config(config: dict[str, Any], key: str, default: Any) -> Any:
    return config.get(key) if config.get(key) is not None else default


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _thirty_days_ago() -> str:
    return (datetime.now(UTC) - timedelta(days=30)).strftime("%Y-%m-%d")


def _load_portfolio(
    portfolio_spec: str, context: RunContext
) -> list[PortfolioCompany]:
    """Load portfolio from a CSV path; copy it into the run directory."""
    path = Path(portfolio_spec)
    if not path.exists():
        raise SystemExit(f"Portfolio file not found: {portfolio_spec}")
    companies = brief_module.load_portfolio(path)
    if not companies:
        raise SystemExit("Portfolio is empty — add at least one company row.")
    shutil.copy(path, context.portfolio_path)
    logger.info("Loaded %d companies from %s", len(companies), path)
    return companies


def _load_portfolio_for_compile(
    context: RunContext, config: dict[str, Any]
) -> list[PortfolioCompany]:
    """Load portfolio, preferring the snapshot saved during the plan step."""
    if context.portfolio_path.exists():
        return brief_module.load_portfolio(context.portfolio_path)
    portfolio_spec = _from_config(config, "portfolio", DEFAULT_PORTFOLIO)
    return brief_module.load_portfolio(Path(portfolio_spec))


# ──────────────────────────────────────────────── stage: plan ─────────────────


def run_plan(context: RunContext, args: argparse.Namespace) -> list[Path]:
    config = context.load_config()
    portfolio_spec = _resolve(args, config, "portfolio", DEFAULT_PORTFOLIO)
    start_date = _resolve(args, config, "start_date", _thirty_days_ago())
    end_date = _resolve(args, config, "end_date", _today())

    companies = _load_portfolio(portfolio_spec, context)
    company_ids = [c.company_id for c in companies]

    logger.info(
        "Planning %d topics over %d companies  [%s → %s]",
        len(TOPICS),
        len(company_ids),
        start_date,
        end_date,
    )

    context.ensure_plans_dir()
    saved = brief_module.build_plans(
        topics=TOPICS,
        company_ids=company_ids,
        plans_dir=context.plans_dir,
        start_date=start_date,
        end_date=end_date,
    )
    context.save_config({
        "portfolio": str(portfolio_spec),
        "start_date": start_date,
        "end_date": end_date,
    })
    logger.info("Saved %d plans to %s", len(saved), context.plans_dir)
    return saved


# ──────────────────────────────────────────────── stage: search ───────────────


def run_search(
    context: RunContext, args: argparse.Namespace
) -> dict[str, list[dict[str, Any]]]:
    config = context.load_config()
    chunk_pct = _resolve(args, config, "chunk_percentage", brief_module.DEFAULT_CHUNK_PERCENTAGE)
    rpm = _resolve(args, config, "requests_per_minute", brief_module.DEFAULT_REQUESTS_PER_MINUTE)

    context.ensure_results_dir()
    results = brief_module.run_search_per_topic(
        topics=TOPICS,
        plans_dir=context.plans_dir,
        results_dir=context.results_dir,
        chunk_percentage=chunk_pct,
        requests_per_minute=rpm,
    )
    context.save_config({"chunk_percentage": chunk_pct, "requests_per_minute": rpm})
    return results


# ──────────────────────────────────────────────── stage: compile ──────────────


def run_compile(
    context: RunContext,
    args: argparse.Namespace,
    topic_results: dict[str, list[dict[str, Any]]] | None = None,
) -> BriefData:
    config = context.load_config()
    model = _resolve(args, config, "model", brief_module.DEFAULT_SUMMARY_MODEL)
    fmt = _resolve(args, config, "format", DEFAULT_FORMAT)
    end_date = _from_config(config, "end_date", _today())

    companies = _load_portfolio_for_compile(context, config)

    if topic_results is None:
        topic_results = brief_module.load_topic_results(TOPICS, context.results_dir)

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    brief_data = brief_module.compile_brief(
        topics=TOPICS,
        portfolio=companies,
        topic_results=topic_results,
        brief_date=end_date,
        generated_at=generated_at,
        model=model,
    )

    context.ensure_briefs_dir()
    written = write_brief(brief_data, context.briefs_dir, formats=[fmt])
    for fmt_key, out_path in written.items():
        logger.info("Wrote %s brief -> %s", fmt_key.upper(), out_path)
        print(f"  {fmt_key.upper()}: {out_path}")

    context.save_config({"model": model, "format": fmt})
    return brief_data


# ──────────────────────────────────────────────── stage: generate (run-all) ───


def run_generate(context: RunContext, args: argparse.Namespace) -> None:
    run_plan(context, args)
    results = run_search(context, args)
    run_compile(context, args, topic_results=results)
    logger.info("generate complete — run dir: %s", context.run_dir)


# ──────────────────────────────────────────────── show-plans ──────────────────


def run_show_plans(context: RunContext, args: argparse.Namespace) -> None:
    plans_dir_override = getattr(args, "plans_dir", None)
    plans_dir = Path(plans_dir_override) if plans_dir_override else context.plans_dir

    summary_df = brief_module.summarize_plans(TOPICS, plans_dir)
    if summary_df.empty:
        print("No plans found in", plans_dir, "— run 'plan' step first.")
        return

    print(f"\nPer-topic chunk estimates  ({plans_dir})")
    print("-" * 55)
    for row in summary_df.itertuples(index=False):
        print(f"  {row.topic:<32} {row.chunks:>10,}")
    print("-" * 55)
    print(f"  {'TOTAL':<32} {summary_df['chunks'].sum():>10,}")
    print()


# ──────────────────────────────────────────────── command dispatchers ─────────


def _cmd_plan(args: argparse.Namespace) -> None:
    run_plan(_make_context(args), args)


def _cmd_search(args: argparse.Namespace) -> None:
    run_search(_make_context(args), args)


def _cmd_compile(args: argparse.Namespace) -> None:
    run_compile(_make_context(args), args)


def _cmd_generate(args: argparse.Namespace) -> None:
    run_generate(_make_context(args), args)


def _cmd_show_plans(args: argparse.Namespace) -> None:
    run_show_plans(_make_context(args), args)


# ──────────────────────────────────────────────── argument parser ─────────────


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-name", default=None, help="Unique run name (defaults to UTC timestamp).")
    p.add_argument(
        "--runs-root",
        default=DEFAULT_RUNS_ROOT,
        help="Parent directory for all runs (default: runs).",
    )


def _add_portfolio_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--portfolio",
        default=argparse.SUPPRESS,
        metavar="PATH",
        help=(
            f"Portfolio CSV with RP_COMPANY_ID + COMPANY_NAME columns"
            f" (default: {DEFAULT_PORTFOLIO})."
        ),
    )


def _add_date_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--start-date", default=argparse.SUPPRESS, help="Search window start (YYYY-MM-DD)."
    )
    p.add_argument(
        "--end-date", default=argparse.SUPPRESS, help="Search window end (YYYY-MM-DD)."
    )


def _add_search_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--chunk-percentage",
        type=float,
        default=argparse.SUPPRESS,
        help=(
            f"Fraction of chunks to retrieve per topic"
            f" (default: {brief_module.DEFAULT_CHUNK_PERCENTAGE})."
        ),
    )
    p.add_argument(
        "--requests-per-minute",
        type=int,
        default=argparse.SUPPRESS,
        help=f"Bigdata API rate cap (default: {brief_module.DEFAULT_REQUESTS_PER_MINUTE}).",
    )


def _add_compile_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        default=argparse.SUPPRESS,
        help=f"OpenAI model for section summaries (default: {brief_module.DEFAULT_SUMMARY_MODEL}).",
    )
    p.add_argument(
        "--format",
        choices=["md", "html", "both"],
        default=argparse.SUPPRESS,
        dest="format",
        help="Output format: md, html, or both (default: both).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="morning-brief",
        description=(
            "Generate a daily institutional morning brief for an equity portfolio.\n\n"
            "Topics covered per company: Earnings & Guidance, Macro & Policy, "
            "Analyst & Sentiment, M&A & Corporate, Supply Chain & Ops."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subs = parser.add_subparsers(dest="command", required=True)

    # generate (recommended entry point)
    gen = subs.add_parser(
        "generate",
        help="Full pipeline: plan → search → compile.",
        description="Run the full morning brief pipeline in one command.",
    )
    _add_common_args(gen)
    _add_portfolio_arg(gen)
    _add_date_args(gen)
    _add_search_args(gen)
    _add_compile_args(gen)
    gen.set_defaults(func=_cmd_generate, requires_api_keys=True)

    # plan
    plan_p = subs.add_parser(
        "plan", help="Build smart-batching search plans (call before 'search')."
    )
    _add_common_args(plan_p)
    _add_portfolio_arg(plan_p)
    _add_date_args(plan_p)
    plan_p.set_defaults(func=_cmd_plan, requires_api_keys=True)

    # search
    search_p = subs.add_parser("search", help="Execute search plans and store per-topic results.")
    _add_common_args(search_p)
    _add_search_args(search_p)
    search_p.set_defaults(func=_cmd_search, requires_api_keys=True)

    # compile
    compile_p = subs.add_parser("compile", help="Summarise results and render the brief.")
    _add_common_args(compile_p)
    _add_compile_args(compile_p)
    compile_p.set_defaults(func=_cmd_compile, requires_api_keys=True)

    # show-plans
    show_p = subs.add_parser(
        "show-plans", help="Print per-topic chunk estimates without API calls."
    )
    _add_common_args(show_p)
    show_p.add_argument(
        "--plans-dir",
        default=argparse.SUPPRESS,
        help="Path to a specific plans directory (default: runs/<run_name>/plans).",
    )
    show_p.set_defaults(func=_cmd_show_plans, requires_api_keys=False)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "requires_api_keys", True):
        _load_environment()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
