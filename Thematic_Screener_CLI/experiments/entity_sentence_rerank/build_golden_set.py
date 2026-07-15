#!/usr/bin/env python3
"""Build a 100-chunk golden set with GPT-5.5 (provenance-locked labels)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from golden_set import (  # noqa: E402
    DEFAULT_GOLDEN_MODEL,
    annotate_golden_chunks,
    prepare_golden_chunks,
    summarize_golden_set,
    write_golden_set,
)
from retrieve import read_jsonl  # noqa: E402
from run_experiment import DEFAULT_SOURCE_RUN, _load_run_config  # noqa: E402

from src import screener  # noqa: E402

DEFAULT_CHUNKS_PATH = EXPERIMENT_DIR / "outputs" / "retrieved_chunks.jsonl"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build golden set from retrieved chunks.")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_GOLDEN_MODEL)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--requests-per-minute", type=int, default=500)
    parser.add_argument("--max-concurrent", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    if not args.chunks_path.exists():
        msg = f"Chunks file not found: {args.chunks_path}. Run run_experiment.py first."
        raise SystemExit(msg)

    run_config = _load_run_config(args.source_run)
    universe_df = screener.load_universe(Path(str(run_config["universe_path"])))
    chunk_rows = read_jsonl(args.chunks_path)[: args.limit]
    prepared = prepare_golden_chunks(chunk_rows, universe_df=universe_df)

    annotated = annotate_golden_chunks(
        prepared,
        main_theme=str(run_config["main_theme"]),
        analyst_focus=str(run_config["analyst_focus"]),
        model=args.model,
        requests_per_minute=args.requests_per_minute,
        max_concurrent_requests=args.max_concurrent,
    )

    csv_path = write_golden_set(
        annotated,
        args.output_dir,
        model=args.model,
        source_path=args.chunks_path,
    )
    summary = summarize_golden_set(annotated)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote golden set to {csv_path}")


if __name__ == "__main__":
    main()
