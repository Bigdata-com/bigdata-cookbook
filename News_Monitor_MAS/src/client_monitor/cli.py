"""CLI entry point for the client news monitor PoC."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.client_monitor.config import DEFAULT_CATEGORY_PROFILE
from src.client_monitor.pipeline import (
    build_config,
    default_output_dir,
    load_environment,
    run_monitor,
)

logger = logging.getLogger("client_monitor.cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Client news monitor PoC — retrieval-only screening over fixed monitor topics.",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=Path("us_sml.csv"),
        help="Universe CSV with RP_ENTITY_ID and COMPANY_NAME (default: us_sml.csv)",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("taxonomy.csv"),
        help="Taxonomy CSV for curated topic filters (default: taxonomy.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: runs/client_poc_<timestamp>)",
    )
    parser.add_argument(
        "--window-end",
        type=str,
        default=None,
        help="Window end ISO timestamp (default: now UTC)",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=15,
        help="Window length in minutes (default: 15)",
    )
    parser.add_argument(
        "--search-mode",
        type=str,
        default="text+topic",
        choices=["text", "topic", "text+topic", "entity_only"],
        help="Search composition mode (default: text+topic; entity_only = no taxonomy filter)",
    )
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="Run text, topic, and text+topic sequentially",
    )
    parser.add_argument(
        "--category-profile",
        type=str,
        default=DEFAULT_CATEGORY_PROFILE,
        choices=["news_premium", "news"],
        help=(
            "Bigdata document category filter (default: news_premium; "
            "news includes broader wire coverage, more noise)"
        ),
    )
    parser.add_argument(
        "--chunk-percentage",
        type=float,
        default=0.5,
        help="Proportional chunk sampling per basket (default: 0.5)",
    )
    parser.add_argument(
        "--limit-entities",
        type=int,
        default=0,
        help="Limit universe to first N entities (0 = all, default: 0)",
    )
    parser.add_argument(
        "--max-chunks-per-basket",
        type=int,
        default=None,
        help="Optional hard cap on max_chunks per basket",
    )
    parser.add_argument(
        "--seen-headlines-db",
        type=Path,
        default=None,
        help="Optional SQLite path for cross-run headline dedup",
    )
    parser.add_argument(
        "--force-baseline-refresh",
        action="store_true",
        help="Force refresh of 30-day MAS baselines",
    )
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=350,
        help="Bigdata search rate limit (default: 350)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the client news monitor."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    load_environment()
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir()

    config = build_config(
        universe_path=args.universe,
        taxonomy_path=args.taxonomy,
        output_dir=output_dir,
        window_end=args.window_end,
        window_minutes=args.window_minutes,
        search_mode=args.search_mode,
        compare_modes=args.compare_modes,
        category_profile=args.category_profile,
        chunk_percentage=args.chunk_percentage,
        limit_entities=args.limit_entities,
        max_chunks_per_basket=args.max_chunks_per_basket,
        seen_headlines_db=args.seen_headlines_db,
        force_baseline_refresh=args.force_baseline_refresh,
        requests_per_minute=args.requests_per_minute,
    )

    logger.info("Output directory: %s", config.output_dir)
    summary = run_monitor(config)
    logger.info(
        "Done — %d chunk rows, %d alerts, %d with stories (est. cost $%.4f)",
        summary["retrieval"]["chunk_rows"],
        summary["mas"]["alert_count"],
        summary["mas"]["alerts_with_stories_count"],
        summary["estimated_cost_usd"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
