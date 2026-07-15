#!/usr/bin/env python3
"""CLI to assess experiment outputs against the golden set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from assess import run_assessment  # noqa: E402

DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
DEFAULT_GOLDEN = DEFAULT_OUTPUT_DIR / "golden_set.csv"
DEFAULT_RECORDS = DEFAULT_OUTPUT_DIR / "labeled_comparison.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assess experiment against golden set.")
    parser.add_argument("--golden-path", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--records-path", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--embed-threshold",
        type=float,
        default=None,
        help="Evaluate sentence label + embedding cosine threshold gate (e.g. 0.52).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.golden_path.exists():
        raise SystemExit(f"Golden set not found: {args.golden_path}")
    records_path = args.records_path
    if not records_path.exists():
        fallback = args.output_dir / "scored_records.csv"
        if fallback.exists():
            records_path = fallback
        else:
            raise SystemExit(
                f"Records file not found: {args.records_path}. "
                "Run run_experiment.py first."
            )

    assessment = run_assessment(
        golden_path=args.golden_path,
        records_path=records_path,
        output_dir=args.output_dir,
        embed_threshold=args.embed_threshold,
    )
    print(json.dumps(assessment, indent=2))
    print(f"\nWrote assessment to {args.output_dir / 'assessment.json'}")


if __name__ == "__main__":
    main()
