#!/usr/bin/env python3
"""Entity-sentence rerank experiment CLI."""

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

from compare import generate_report  # noqa: E402
from embed import add_embedding_scores  # noqa: E402
from extract import extract_entity_sentence_records  # noqa: E402
from label import _resolve_company_names, run_labeling_passes  # noqa: E402
from label_provenance import run_provenance_labeling_passes  # noqa: E402
from retrieve import read_jsonl, retrieve_tagged_chunks, write_jsonl  # noqa: E402

from src import screener  # noqa: E402

DEFAULT_SOURCE_RUN = PROJECT_ROOT / "runs" / "run_20260629_081049"
DEFAULT_PLANS = (
    "Quantum_hardware_developers.json,"
    "Cryogenics_and_specialized_components.json"
)
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
DEFAULT_OUTPUT_DIR_500 = EXPERIMENT_DIR / "outputs_500"


def _list_plan_files(plans_dir: Path) -> list[str]:
    return sorted(path.name for path in plans_dir.glob("*.json"))


def _load_run_config(source_run: Path) -> dict[str, str | list[str]]:
    config_path = source_run / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    themes_path = source_run / "themes.txt"
    labels = [
        line.strip()
        for line in themes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    universe_path = Path(str(config["universe"]))
    chunk_percentage = float(config.get("chunk_percentage", screener.DEFAULT_CHUNK_PERCENTAGE))
    return {
        "main_theme": str(config.get("main_theme", screener.DEFAULT_MAIN_THEME)),
        "analyst_focus": str(config.get("analyst_focus", screener.DEFAULT_ANALYST_FOCUS)),
        "labels": labels,
        "labeling_model": str(config.get("labeling_model", screener.DEFAULT_LABELING_MODEL)),
        "requests_per_minute": int(
            config.get("requests_per_minute", screener.DEFAULT_REQUESTS_PER_MINUTE)
        ),
        "chunk_percentage": chunk_percentage,
        "universe_path": str(universe_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run entity-sentence rerank experiment.")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--plans", default=DEFAULT_PLANS)
    parser.add_argument(
        "--all-plans",
        action="store_true",
        help="Use every plan JSON in the source run plans/ directory.",
    )
    parser.add_argument("--chunk-percentage", type=float, default=None)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-labeling", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument(
        "--provenance-locked",
        action="store_true",
        help="Label with fixed leaf_label per retrieval plan (matches golden set rules).",
    )
    parser.add_argument(
        "--label-model",
        default=None,
        help="Override labeling model (defaults to source run config).",
    )
    parser.add_argument("--label-concurrency", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    run_config = _load_run_config(args.source_run)
    chunk_percentage = (
        args.chunk_percentage
        if args.chunk_percentage is not None
        else float(run_config["chunk_percentage"])
    )
    plans_dir = args.source_run / "plans"
    if args.all_plans:
        plan_files = _list_plan_files(plans_dir)
    else:
        plan_files = [part.strip() for part in args.plans.split(",") if part.strip()]
    if not plan_files:
        raise SystemExit(f"No plan files found under {plans_dir}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieved_path = output_dir / "retrieved_chunks.jsonl"
    if args.skip_retrieval and retrieved_path.exists():
        chunk_rows = read_jsonl(retrieved_path)
    else:
        chunk_rows = retrieve_tagged_chunks(
            plans_dir=plans_dir,
            plan_files=plan_files,
            chunk_percentage=chunk_percentage,
            requests_per_minute=int(run_config["requests_per_minute"]),
            sample_size=args.sample_size,
            seed=args.seed,
        )
        write_jsonl(chunk_rows, retrieved_path)

    entity_records = extract_entity_sentence_records(chunk_rows)
    extracted_path = output_dir / "entity_sentence_records.jsonl"
    write_jsonl(entity_records, extracted_path)

    scored_records = add_embedding_scores(entity_records)

    if not args.skip_labeling:
        universe_df = screener.load_universe(Path(str(run_config["universe_path"])))
        label_model = args.label_model or str(run_config["labeling_model"])
        if args.provenance_locked:
            enriched = _resolve_company_names(scored_records, universe_df)
            scored_records = run_provenance_labeling_passes(
                enriched,
                main_theme=str(run_config["main_theme"]),
                analyst_focus=str(run_config["analyst_focus"]),
                model=label_model,
                requests_per_minute=int(run_config["requests_per_minute"]),
                max_concurrent_requests=args.label_concurrency,
            )
        else:
            scored_records = run_labeling_passes(
                scored_records,
                universe_df=universe_df,
                main_theme=str(run_config["main_theme"]),
                analyst_focus=str(run_config["analyst_focus"]),
                labels=list(run_config["labels"]),
                model=label_model,
                requests_per_minute=int(run_config["requests_per_minute"]),
                max_concurrent_requests=args.label_concurrency,
            )

    summary = generate_report(
        scored_records,
        output_dir,
        skip_labeling=args.skip_labeling,
    )
    print(json.dumps(summary, indent=2))
    print(f"\nWrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
