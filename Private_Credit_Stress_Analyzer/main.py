"""Private Credit Stress Analyzer — Full Pipeline.

Usage:
  uv run python main.py                        # full run
  uv run python main.py --skip-search          # use cached results
  uv run python main.py --clear-cache          # delete raw cache, then run pipeline
  uv run python main.py --layer lender         # run only lender layer
  uv run python main.py --entity "Blue Owl Capital"  # single entity
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.search import RAW_OUTPUT_DIR, run_all_searches
from src.scorer import compute_scores
from src.reporter import generate_reports
from src.utils import console, format_elapsed


def clear_cache() -> int:
    """Remove all cached raw search results (output/raw/*.json). Returns count deleted."""
    if not RAW_OUTPUT_DIR.exists():
        return 0
    deleted = 0
    for f in RAW_OUTPUT_DIR.glob("*.json"):
        f.unlink()
        deleted += 1
    return deleted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Private Credit Stress Analyzer — end-to-end pipeline"
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip search phase and use cached raw results",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete all cached raw results in output/raw/ before running (forces fresh search)",
    )
    parser.add_argument(
        "--layer",
        type=str,
        choices=["lender", "borrower", "bank"],
        default=None,
        help="Run only a specific entity layer",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Run only a single entity by name",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Max parallel search workers (default: 5)",
    )
    args = parser.parse_args()

    start = time.time()
    console.rule("[bold magenta]Private Credit Stress Analyzer")

    if args.clear_cache:
        n = clear_cache()
        console.print(f"\n[bold]Cache cleared:[/bold] removed {n} file(s) from output/raw/")

    # Phase 1: Search
    if not args.skip_search:
        console.print("\n[bold]Phase 1:[/bold] Running searches...")
        run_all_searches(
            layer_filter=args.layer,
            entity_filter=args.entity,
            max_workers=args.max_workers,
        )
    else:
        console.print("\n[bold]Phase 1:[/bold] [yellow]Skipped (--skip-search)[/yellow]")

    # Phase 2: Scoring
    console.print("\n[bold]Phase 2:[/bold] Computing scores...")
    df = compute_scores()

    # Phase 3: Reports
    console.print("\n[bold]Phase 3:[/bold] Generating reports...")
    generate_reports(df)

    elapsed = (time.time() - start) * 1000
    console.rule(f"[bold green]Pipeline complete in {format_elapsed(elapsed)}")


if __name__ == "__main__":
    main()
