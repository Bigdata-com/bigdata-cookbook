"""
Benchmark script for Smart Batching search pipeline.

Measures execution time and chunk retrieval counts across multiple
test configurations (texts, date ranges, chunk percentages).

Results are stored as chunk-level DataFrames (parquet) for flexible
downstream analysis — no pre-aggregated timeseries.

Usage:
    python benchmark.py --mode full_grid
    python benchmark.py --mode smart --chunk-percentage 0.01
"""

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from dotenv import load_dotenv


sys.path.insert(0, str(Path.cwd()))

from src import (
    execute_search,
    load_universe_from_csv,
    plan_search,
    execute_full_grid_search,
)

env_path = Path.cwd() / ".env"
if env_path.exists():
    load_dotenv(env_path)

API_BASE_URL = os.getenv("BIGDATA_API_BASE_URL", "https://api.bigdata.com")
os.environ["BIGDATA_API_BASE_URL"] = API_BASE_URL
API_KEY = os.getenv("BIGDATA_API_KEY")

if not API_KEY:
    raise ValueError(
        "BIGDATA_API_KEY not set. Set it in .env or as an environment variable."
    )


class BenchmarkCase(TypedDict):
    id: str
    text: str
    start_date: str
    end_date: str
    chunk_percentage: float


UNIVERSE_CSV = "id_name_mapping_us_top_3000.csv"
#UNIVERSE_CSV = "small_test.csv"
#UNIVERSE_CSV = "id_name_mapping_us_top_3000_reduced.csv"
#UNIVERSE_CSV = "00067A.csv"
REQUESTS_PER_MINUTE = 450
VOLUME_QUERY_MODE = "iterative"
MAX_ITERATIONS_PER_BATCH = 10
APPLY_VOLUME_SPLITS = True
MIN_PERIOD_DAYS = 30

BATCH_SIZE = 10
MAX_CHUNKS_PER_REQUEST = 1000
WINDOW_DAYS = 90

# All benchmark artifacts (parquet + json) are written under this directory.
BENCHMARK_DIR = Path("benchmark")

BENCHMARK_CASES: list[BenchmarkCase] = [
    {
        "id": "tariffs_china_2025",
        "text": "The company is affected by US import tariffs against China",
        "start_date": "2025-01-01",
        "end_date": "2025-06-30",
        "chunk_percentage": 100,
    },
    {
        "id": "leadership_2023",
        "text": "The company has faced leadership changes and executive appointments",
        "start_date": "2023-09-01",
        "end_date": "2023-12-31",
        "chunk_percentage": 100,
    },    
    {
        "id": "confidence_decline_2021",
        "text": "Decline in customer confidence in the company",
        "start_date": "2021-01-01",
        "end_date": "2021-06-30",
        "chunk_percentage": 100,
    }, 
    
]
if True:
    BENCHMARK_CASES: list[BenchmarkCase] = [  
        {
            "id": "confidence_decline_2010",
            "text": "Decline in customer confidence in the company",
            "start_date": "2010-01-01",
            "end_date": "2010-12-31",
            "chunk_percentage": 100,
        }, 
        
    ]



class BenchmarkResult(TypedDict):
    case_index: int
    case_id: str
    text: str
    start_date: str
    end_date: str
    chunk_percentage: float
    num_companies: int
    plan_time_s: float
    num_baskets: int
    total_expected_chunks: int
    total_chunk_budget: int
    total_chunks_requested: int
    execute_time_s: float
    num_results: int
    total_chunks_retrieved: int


class FullGridBenchmarkResult(TypedDict):
    case_index: int
    case_id: str
    text: str
    start_date: str
    end_date: str
    num_companies: int
    num_batches: int
    num_windows: int
    max_chunks_per_request: int
    total_chunk_budget: int
    execute_time_s: float
    num_results: int
    total_chunks_retrieved: int


# ---------------------------------------------------------------------------
# Chunk DataFrame helpers
# ---------------------------------------------------------------------------

def _timestamp_to_date(ts: Any) -> str | None:
    """Convert timestamp to date string YYYY-MM-DD. Accepts ISO datetime or date-only strings."""
    if ts is None:
        return None
    if isinstance(ts, str):
        if re.match(r"^\d{4}-\d{2}-\d{2}$", ts.strip()):
            return ts.strip()[:10]
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            if len(ts) >= 10 and ts[4] == "-" and ts[7] == "-":
                return ts[:10]
            return None
    if hasattr(ts, "strftime"):
        return ts.strftime("%Y-%m-%d")
    return None


def _extract_entity_ids(chunk: dict[str, Any]) -> list[str]:
    """Extract entity IDs from a chunk, trying enriched field then raw detections."""
    ids = chunk.get("entity_ids", [])
    if ids:
        return [str(eid) for eid in ids if eid is not None]

    detections = chunk.get("detections", [])
    if detections:
        return [
            str(d["id"])
            for d in detections
            if isinstance(d, dict) and d.get("id") is not None
        ]

    inner = chunk.get("chunk")
    if isinstance(inner, dict):
        detections = inner.get("detections", [])
        if detections:
            return [
                str(d["id"])
                for d in detections
                if isinstance(d, dict) and d.get("id") is not None
            ]

    return []


_CHUNKS_DF_COLUMNS = [
    "date",
    "timestamp",
    "doc_id",
    "headline",
    "source_id",
    "source_name",
    "chunk_index",
    "chunk_text",
    "relevance",
    "sentiment",
    "entity_ids",
    "url",
]


def results_to_chunks_df(
    results_raw: list[dict[str, Any]],
    *,
    universe_entity_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Convert raw search results into a chunk-level DataFrame.

    Each row represents one chunk with document metadata, relevance,
    sentiment, and entity IDs.  When *universe_entity_ids* is provided,
    entity IDs are filtered to only those in the universe.
    """
    rows: list[dict[str, Any]] = []
    for doc in results_raw:
        doc_id = doc.get("id", "")
        timestamp = doc.get("timestamp", "")
        date = _timestamp_to_date(timestamp)
        headline = doc.get("headline", "")
        url = doc.get("url", "")
        source = doc.get("source", {}) or {}
        source_id = source.get("id", "") if isinstance(source, dict) else ""
        source_name = source.get("name", "") if isinstance(source, dict) else ""

        chunks = doc.get("chunks", [])
        if not chunks:
            continue

        for chunk in chunks:
            entity_ids = _extract_entity_ids(chunk)
            if universe_entity_ids is not None:
                entity_ids = [eid for eid in entity_ids if eid in universe_entity_ids]

            rows.append({
                "date": date,
                "timestamp": timestamp,
                "doc_id": doc_id,
                "headline": headline,
                "source_id": source_id,
                "source_name": source_name,
                "chunk_index": chunk.get("cnum"),
                "chunk_text": chunk.get("text", ""),
                "relevance": chunk.get("relevance"),
                "sentiment": chunk.get("sentiment"),
                "entity_ids": entity_ids,
                "url": url,
            })

    if not rows:
        return pd.DataFrame(columns=_CHUNKS_DF_COLUMNS)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def _compute_num_windows(start_date: str, end_date: str, window_days: int) -> int:
    """Compute the number of non-overlapping time windows that cover the date range."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days + 1
    return math.ceil(total_days / window_days)


def _compute_smart_chunks_requested(
    baskets: list[dict[str, Any]],
    chunk_percentage: float,
) -> int:
    """Compute the exact total chunks that will be requested given the plan baskets and chunk_percentage."""
    total = 0
    for basket in baskets:
        expected = basket.get("expected_chunks", 0)
        if expected > 0:
            total += min(max(1, int(expected * chunk_percentage)), 1000)
        else:
            total += basket.get("query", {}).get("max_chunks", 100)
    return total


# ---------------------------------------------------------------------------
# Benchmark runners — each returns (summary_dict, chunks_df)
# ---------------------------------------------------------------------------

def run_benchmark_case_full_grid(
    case: BenchmarkCase,
    case_index: int,
) -> tuple[FullGridBenchmarkResult, pd.DataFrame]:
    """Run a single benchmark case using full grid search (no Smart Batching)."""
    text = case["text"]
    start_date = case["start_date"]
    end_date = case["end_date"]

    companies = load_universe_from_csv(UNIVERSE_CSV)
    num_batches = math.ceil(len(companies) / BATCH_SIZE)
    total_chunk_budget = num_batches * MAX_CHUNKS_PER_REQUEST

    print(
        f"  Chunk budget: {num_batches} batches x {MAX_CHUNKS_PER_REQUEST}"
        f" max_chunks = {total_chunk_budget:,}"
    )

    exec_start = time.perf_counter()
    results_raw = execute_full_grid_search(
        text=text,
        universe_csv_path=UNIVERSE_CSV,
        start_date=start_date,
        end_date=end_date,
        batch_size=BATCH_SIZE,
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        max_chunks_per_request=MAX_CHUNKS_PER_REQUEST,
    )
    exec_time = time.perf_counter() - exec_start

    chunks_df = results_to_chunks_df(results_raw, universe_entity_ids=set(companies))

    result = FullGridBenchmarkResult(
        case_index=case_index,
        case_id=case["id"],
        text=text,
        start_date=start_date,
        end_date=end_date,
        num_companies=len(companies),
        num_batches=num_batches,
        num_windows=1,
        max_chunks_per_request=MAX_CHUNKS_PER_REQUEST,
        total_chunk_budget=total_chunk_budget,
        execute_time_s=round(exec_time, 2),
        num_results=len(results_raw),
        total_chunks_retrieved=len(chunks_df),
    )
    return result, chunks_df


def run_benchmark_case_full_grid_split_by_window(
    case: BenchmarkCase,
    case_index: int,
    *,
    total_chunk_budget: int | None = None,
) -> tuple[FullGridBenchmarkResult, pd.DataFrame]:
    """Run a full grid search split by time windows, with max_chunks adjusted to match the budget."""
    text = case["text"]
    start_date = case["start_date"]
    end_date = case["end_date"]

    companies = load_universe_from_csv(UNIVERSE_CSV)
    num_batches = math.ceil(len(companies) / BATCH_SIZE)
    num_windows = _compute_num_windows(start_date, end_date, WINDOW_DAYS)

    if total_chunk_budget is not None:
        adjusted_max_chunks = max(1, total_chunk_budget // (num_batches * num_windows))     
        print(
            f"  Chunk budget: {total_chunk_budget:,} / ({num_batches} batches x {num_windows} windows)"
            f" = {adjusted_max_chunks} max_chunks/request"
        )        
    else:
        adjusted_max_chunks = MAX_CHUNKS_PER_REQUEST



    exec_start = time.perf_counter()
    results_raw = execute_full_grid_search(
        text=text,
        universe_csv_path=UNIVERSE_CSV,
        start_date=start_date,
        end_date=end_date,
        batch_size=BATCH_SIZE,
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        max_chunks_per_request=adjusted_max_chunks,
        window_days=WINDOW_DAYS,
    )
    exec_time = time.perf_counter() - exec_start

    chunks_df = results_to_chunks_df(results_raw, universe_entity_ids=set(companies))

    result = FullGridBenchmarkResult(
        case_index=case_index,
        case_id=case["id"],
        text=text,
        start_date=start_date,
        end_date=end_date,
        num_companies=len(companies),
        num_batches=num_batches,
        num_windows=num_windows,
        max_chunks_per_request=adjusted_max_chunks,
        total_chunk_budget=total_chunk_budget,
        execute_time_s=round(exec_time, 2),
        num_results=len(results_raw),
        total_chunks_retrieved=len(chunks_df),
    )
    return result, chunks_df


def run_benchmark_case(
    case: BenchmarkCase,
    case_index: int,
    *,
    total_chunk_budget: int | None = None,
) -> tuple[BenchmarkResult, pd.DataFrame]:
    """Run a single benchmark case (Smart Batching: plan + execute)."""
    text = case["text"]
    start_date = case["start_date"]
    end_date = case["end_date"]

    companies = load_universe_from_csv(UNIVERSE_CSV)

    plan_start = time.perf_counter()
    plan = plan_search(
        text=text,
        universe_csv_path=UNIVERSE_CSV,
        start_date=start_date,
        end_date=end_date,
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        volume_query_mode=VOLUME_QUERY_MODE,
        max_iterations_per_batch=MAX_ITERATIONS_PER_BATCH,
        apply_volume_splits=APPLY_VOLUME_SPLITS,
        min_period_days=MIN_PERIOD_DAYS,
        min_entities_per_basket=BATCH_SIZE,
    )
    plan_time = time.perf_counter() - plan_start

    baskets = plan.get("baskets", [])
    num_baskets = len(baskets)
    total_expected = plan.get("total_expected_chunks", 0)

    if total_expected > 0 and total_chunk_budget is not None:
        computed_chunk_pct = total_chunk_budget / total_expected
    else:
        computed_chunk_pct = case["chunk_percentage"]

    total_chunks_requested = _compute_smart_chunks_requested(baskets, computed_chunk_pct)

    # Make sure computed_chunk_pct is not greater than 1.0
    computed_chunk_pct = min(computed_chunk_pct, 1.0)

    if total_chunk_budget is not None:
        print(
            f"  Chunk budget: {total_chunk_budget:,} | plan expected: {total_expected:,}"
            f" | computed chunk%%: {computed_chunk_pct*100:.2f}%"
            f" | actual requested: {total_chunks_requested:,}"
        )

    exec_start = time.perf_counter()
    results_raw = execute_search(
        search_plan=plan,
        chunk_percentage=computed_chunk_pct,
        requests_per_minute=REQUESTS_PER_MINUTE,
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        second_pass=False,
    )
    exec_time = time.perf_counter() - exec_start

    chunks_df = results_to_chunks_df(results_raw, universe_entity_ids=set(companies))

    result = BenchmarkResult(
        case_index=case_index,
        case_id=case["id"],
        text=text,
        start_date=start_date,
        end_date=end_date,
        chunk_percentage=computed_chunk_pct,
        num_companies=len(companies),
        plan_time_s=round(plan_time, 2),
        num_baskets=num_baskets,
        total_expected_chunks=total_expected,
        total_chunk_budget=total_chunk_budget,
        total_chunks_requested=total_chunks_requested,
        execute_time_s=round(exec_time, 2),
        num_results=len(results_raw),
        total_chunks_retrieved=len(chunks_df),
    )
    return result, chunks_df


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a summary table of all Smart Batching benchmark results."""
    print(f"\n{'='*80}")
    print("SMART BATCHING BENCHMARK SUMMARY")
    print(f"{'='*80}")

    header = (
        f"{'#':>3}  {'Plan(s)':>8}  {'Exec(s)':>8}  {'Total(s)':>8}  "
        f"{'Baskets':>8}  {'Expected':>10}  {'Budget':>10}  {'Requested':>10}  "
        f"{'Retrieved':>10}  {'Chunk%':>8}  Text"
    )
    print(header)
    print("-" * len(header) + "-" * 30)

    total_plan = 0.0
    total_exec = 0.0
    total_chunks = 0

    for r in results:
        total_time = r["plan_time_s"] + r["execute_time_s"]
        total_plan += r["plan_time_s"]
        total_exec += r["execute_time_s"]
        total_chunks += r["total_chunks_retrieved"]
        short_text = r["text"][:35] + "..." if len(r["text"]) > 35 else r["text"]
        print(
            f"{r['case_index']+1:>3}  {r['plan_time_s']:>8.2f}  {r['execute_time_s']:>8.2f}  "
            f"{total_time:>8.2f}  {r['num_baskets']:>8}  "
            f"{r['total_expected_chunks']:>10,}"
            f"{r['total_chunks_retrieved']:>10,}  "
            f"{r['chunk_percentage']*100:>7.2f}%  {short_text}"
        )

    grand_total = total_plan + total_exec
    print("-" * len(header) + "-" * 30)
    print(
        f"{'TOT':>3}  {total_plan:>8.2f}  {total_exec:>8.2f}  "
        f"{grand_total:>8.2f}  {'':>8}  {'':>10}  {'':>10}  {'':>10}  {total_chunks:>10,}"
    )


def print_summary_full_grid(
    results: list[FullGridBenchmarkResult],
    title: str = "FULL GRID BENCHMARK SUMMARY",
) -> None:
    """Print a summary table of all full-grid benchmark results."""
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")

    header = (
        f"{'#':>3}  {'Exec(s)':>8}  {'Batches':>8}  {'Windows':>8}  "
        f"{'MaxChk':>8}  {'Budget':>10}  {'Retrieved':>10}  Text"
    )
    print(header)
    print("-" * len(header) + "-" * 30)

    total_exec = 0.0
    total_chunks = 0

    for r in results:
        total_exec += r["execute_time_s"]
        total_chunks += r["total_chunks_retrieved"]
        short_text = r["text"][:40] + "..." if len(r["text"]) > 40 else r["text"]
        print(
            f"{r['case_index']+1:>3}  {r['execute_time_s']:>8.2f}  "
            f"{r['num_batches']:>8}  {r['num_windows']:>8}  "
            f"{r['max_chunks_per_request']:>8}"
            f"{r['total_chunks_retrieved']:>10,}  {short_text}"
        )

    print("-" * len(header) + "-" * 30)
    print(
        f"{'TOT':>3}  {total_exec:>8.2f}  {'':>8}  {'':>8}  "
        f"{'':>8}  {'':>10}  {total_chunks:>10,}"
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_benchmark_results(
    base_path: Path | str,
    *,
    chunks_df: pd.DataFrame,
    full_grid_results: list[FullGridBenchmarkResult] | None = None,
    full_grid_results_split_by_window: list[FullGridBenchmarkResult] | None = None,
    smart_batching_results: list[BenchmarkResult] | None = None,
) -> None:
    """Save benchmark results: chunk data to parquet, summaries to JSON.

    Args:
        base_path: Base file path (without extension). Two files are created:
            ``<base_path>.parquet`` — chunk-level data with *method* and *case_index* columns.
            ``<base_path>.json``    — lightweight summary with metadata and per-case stats.
        chunks_df: Combined chunk DataFrame (all methods / cases).
        full_grid_results: Summary dicts for full-grid runs.
        full_grid_results_split_by_window: Summary dicts for full-grid-split runs.
        smart_batching_results: Summary dicts for smart-batching runs.
    """
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    parquet_path = base.with_suffix(".parquet")
    chunks_df.to_parquet(parquet_path, index=False)
    print(f"Chunk data saved to {parquet_path}  ({len(chunks_df):,} rows)")

    payload: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "universe_csv": UNIVERSE_CSV,
            "num_cases": max(
                len(full_grid_results or []),
                len(smart_batching_results or []),
            ),
            "total_chunks": len(chunks_df),
        }
    }
    if full_grid_results is not None:
        payload["full_grid_results"] = full_grid_results
    if full_grid_results_split_by_window is not None:
        payload["full_grid_results_split_by_window"] = full_grid_results_split_by_window
    if smart_batching_results is not None:
        payload["smart_batching_results"] = smart_batching_results

    json_path = base.with_suffix(".json")
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Summary saved to {json_path}")


def save_single_benchmark_case(
    base_path: Path | str,
    *,
    case_index: int,
    case_id: str,
    chunks_df: pd.DataFrame,
    full_grid_split_result: FullGridBenchmarkResult | None = None,
    smart_batching_result: BenchmarkResult | None = None,
) -> None:
    """Save one benchmark case to its own parquet + JSON under *base_path*."""
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    parquet_path = base.with_suffix(".parquet")
    chunks_df.to_parquet(parquet_path, index=False)
    print(f"  [{case_id}] chunk data → {parquet_path}  ({len(chunks_df):,} rows)")

    payload: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "universe_csv": UNIVERSE_CSV,
            "case_index": case_index,
            "case_id": case_id,
            "num_cases": 1,
            "total_chunks": len(chunks_df),
        }
    }
    if full_grid_split_result is not None:
        payload["full_grid_results_split_by_window"] = [full_grid_split_result]
    if smart_batching_result is not None:
        payload["smart_batching_results"] = [smart_batching_result]

    json_path = base.with_suffix(".json")
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [{case_id}] summary → {json_path}")


def append_chunks_to_case_parquet(
    parquet_path: Path,
    new_chunks: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """Append new chunk rows to an existing case parquet, replacing any rows with the same method.

    If the parquet file does not exist yet, creates it from *new_chunks*.
    Returns the combined DataFrame that was written.
    """
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    new_chunks = new_chunks.assign(method=method)

    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        existing = existing[existing["method"] != method]
        combined = pd.concat([existing, new_chunks], ignore_index=True)
    else:
        combined = new_chunks

    combined.to_parquet(parquet_path, index=False)
    print(f"  Saved {len(new_chunks):,} rows (method={method}) → {parquet_path}  (total {len(combined):,} rows)")
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Smart Batching vs Full Grid search.",
    )
    parser.add_argument(
        "--mode",
        choices=["full_grid", "smart"],
        required=True,
        help="Search strategy: 'full_grid' or 'smart' (Smart Batching).",
    )
    parser.add_argument(
        "--chunk-percentage",
        type=float,
        default=None,
        help="Chunk percentage for Smart Batching (required when --mode=smart).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.mode == "smart" and args.chunk_percentage is None:
        parser.error("--chunk-percentage is required when --mode=smart")

    if args.mode == "full_grid":
        method_label = "full_grid"
    else:
        method_label = f"smart_batching_{args.chunk_percentage}"

    print(f"Mode           : {args.mode}")
    if args.mode == "smart":
        print(f"Chunk %%        : {args.chunk_percentage}")
    print(f"Method label   : {method_label}")
    print(f"API Base URL   : {API_BASE_URL}")
    print(f"Universe CSV   : {UNIVERSE_CSV}")
    print(f"Benchmark cases: {len(BENCHMARK_CASES)}")

    companies = load_universe_from_csv(UNIVERSE_CSV)
    num_batches = math.ceil(len(companies) / BATCH_SIZE)

    print(f"Batch size     : {BATCH_SIZE}")
    print(f"Num batches    : {num_batches}")
    print(f"Max chunks/req : {MAX_CHUNKS_PER_REQUEST}")

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "full_grid":
        print(f"\n{'#'*80}")
        print("# FULL GRID SEARCH BENCHMARK")
        print(f"{'#'*80}")

        fg_results: list[FullGridBenchmarkResult] = []
        for idx, case in enumerate(BENCHMARK_CASES):
            print(f"\n>>> Running full grid case {idx + 1}/{len(BENCHMARK_CASES)}: {case['id']}")
            result, chunks_df = run_benchmark_case_full_grid_split_by_window(
                case, idx, total_chunk_budget=None,
            )
            fg_results.append(result)

            parquet_path = BENCHMARK_DIR / f"benchmark_{case['id']}.parquet"
            append_chunks_to_case_parquet(parquet_path, chunks_df, method=method_label)

        print_summary_full_grid(fg_results)

    else:
        print(f"\n{'#'*80}")
        print("# SMART BATCHING BENCHMARK")
        print(f"{'#'*80}")

        for case in BENCHMARK_CASES:
            case["chunk_percentage"] = args.chunk_percentage

        sm_results: list[BenchmarkResult] = []
        for idx, case in enumerate(BENCHMARK_CASES):
            print(f"\n>>> Running smart batching case {idx + 1}/{len(BENCHMARK_CASES)}: {case['id']}")
            result, chunks_df = run_benchmark_case(case, idx, total_chunk_budget=None)
            sm_results.append(result)

            parquet_path = BENCHMARK_DIR / f"benchmark_{case['id']}.parquet"
            append_chunks_to_case_parquet(parquet_path, chunks_df, method=method_label)

        print_summary(sm_results)


if __name__ == "__main__":
    main()
