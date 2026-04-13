"""
Backtest script for Smart Batching vs Full Grid retrieval.

Runs chunk retrieval using three approaches for a given query:
  1. Full grid search (one request per entity batch, no time splitting)
  2. Full grid search with time-window splitting
  3. Smart Batching (plan + execute with volume-aware splitting)

Results are stored as chunk-level DataFrames (parquet) for downstream analysis.
"""

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path.cwd()))

from src import (
    execute_full_grid_search,
    execute_search,
    load_universe_from_csv,
    plan_search,
    portfolio_backtesting_pipeline,
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

# ---------------------------------------------------------------------------
# Backtest configuration
# ---------------------------------------------------------------------------

TEXT = "Companies will face production delays due to semiconductor supply shortages disrupting global production."
START_DATE = "2021-01-01"
END_DATE = "2021-06-30"

UNIVERSE_CSV = "id_name_mapping_us_top_3000.csv"
#UNIVERSE_CSV = "00067A.csv"

BATCH_SIZE = 10
MAX_CHUNKS_PER_REQUEST = 1000
WINDOW_DAYS = 90

REQUESTS_PER_MINUTE = 450
VOLUME_QUERY_MODE = "iterative"
MAX_ITERATIONS_PER_BATCH = 10
APPLY_VOLUME_SPLITS = True
MIN_PERIOD_DAYS = 30
CHUNK_PERCENTAGE = 1.0

OUTPUT_DIR = Path.cwd() / "backtest_results"


PRICE_PATH = Path("/home/amartinezg/shared/InputData/fgomez/backtest_tmp/prices")
PRICE_FILES = {"US_MIDLARGE_CAP": "us_ml.csv", "US_SMALL_CAP": "us_s.csv"}
EXCESS_TO_MARKET = True
WINDSOR_LOW, WINDSOR_HIGH = -1.0, 1.5
NON_TRADING_PCT = 0.75

BACKTEST_PER_UNIVERSE = False
SIGNAL_LAG_DAYS = 0

rebalancing_frequency = "D"
exposure_long, exposure_short = 0.5, 0.5
cap = 0.05
daily_standardization = False
plot_performance = True

# ---------------------------------------------------------------------------
# Chunk DataFrame helpers (same as benchmark.py)
# ---------------------------------------------------------------------------

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

def signal_construction(df: pd.DataFrame) -> pd.DataFrame:
    """Build daily entity-level signals from deduplicated chunk rows.

    Expects a DataFrame already exploded by entity (e.g. from ``_load_benchmark_raw``).
    Assigns a NYSE trading-date, deduplicates, and aggregates relevance * sentiment
    into a daily score per entity.
    """
    relevance_col = "chunk_relevance" if "chunk_relevance" in df.columns else "relevance"
    sentiment_col = "chunk_sentiment" if "chunk_sentiment" in df.columns else "sentiment"
    chunk_col = "chunk_index" if "chunk_index" in df.columns else "chunk_cnum"
    date_col = "timestamp" if "timestamp" in df.columns else "date"

    required_cols = {"entity_id", "doc_id", chunk_col, relevance_col, sentiment_col, date_col}
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for daily basic signals: {missing_cols}")

    basic_long = df[
        ["entity_id", "doc_id", chunk_col, date_col, relevance_col, sentiment_col]
    ].copy()
    basic_long = basic_long.dropna(subset=[relevance_col, sentiment_col, date_col])

    # Assign trading date so we aggregate before 15:30 NY (no look-ahead for close-to-close NYSE).
    # Trading date T = [15:30 NY on T-1, 15:30 NY on T). Add 8.5h so 15:30 becomes midnight.
    eastern = ZoneInfo("America/New_York")
    dt_utc = pd.to_datetime(basic_long[date_col], errors="coerce", utc=True)
    dt_ny = dt_utc.dt.tz_convert(eastern)
    basic_long["date"] = (dt_ny + pd.Timedelta(hours=8.5)).dt.date
    basic_long = basic_long.dropna(subset=["date"])

    basic_long = basic_long.drop_duplicates(subset=["entity_id", "date", "doc_id", chunk_col])

    if basic_long.empty:
        return pd.DataFrame(columns=["rp_entity_id", "date", "score", "volume", "entity_name"])

    basic_long["score_component"] = basic_long[relevance_col] * basic_long[sentiment_col]

    entity_daily_agg = (
        basic_long.groupby(["entity_id", "date"], as_index=False)
        .agg(
            score=("score_component", "sum"),
            volume=("entity_id", "size"),
        )
        .rename(columns={"entity_id": "rp_entity_id"})
    )

    df_universe = pd.read_csv(UNIVERSE_CSV)
    df_universe = df_universe.rename(columns={"id": "rp_entity_id", "name": "entity_name"})

    df_basic_signals = entity_daily_agg.merge(
        df_universe[["rp_entity_id", "entity_name"]],
        on="rp_entity_id",
        how="left",
    )

    df_basic_signals = df_basic_signals[["rp_entity_id", "date", "score", "volume", "entity_name"]]
    df_basic_signals = df_basic_signals.rename(columns={'date': 'DATE', 'rp_entity_id': 'RP_ENTITY_ID'})
    
    df_basic_signals = df_basic_signals.sort_values(["RP_ENTITY_ID", "DATE"]).reset_index(drop=True)
    df_basic_signals['DATE'] = pd.to_datetime(df_basic_signals['DATE'])

    basic_signals_path = OUTPUT_DIR / "df_basic_signals_daily.csv"
    #df_basic_signals.to_csv(basic_signals_path, index=False)

    return df_basic_signals

def _timestamp_to_date(ts: Any) -> str | None:
    """Convert various timestamp formats to YYYY-MM-DD."""
    if ts is None:
        return None
    if isinstance(ts, str):
        stripped = ts.strip()
        if len(stripped) >= 10 and stripped[4] == "-" and stripped[7] == "-":
            return stripped[:10]
        try:
            dt = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None
    if hasattr(ts, "strftime"):
        return ts.strftime("%Y-%m-%d")
    return None

def _load_benchmark_raw(parquet_path: Path | str, deduplicate: bool = False) -> pd.DataFrame:
    """Load benchmark parquet and explode entity_ids without aggregation.

    Returns one row per (method, entity_id, date, doc_id, chunk_index) with
    the raw relevance score.
    """
    df = pd.read_parquet(parquet_path)
    if deduplicate:
        # Deduplicate by chunk_text
        df = df.drop_duplicates(subset=["chunk_text"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    exploded = df.explode("entity_ids").dropna(subset=["entity_ids"])
    exploded = exploded.rename(columns={"entity_ids": "entity_id"})

    deduped = exploded.drop_duplicates(
        subset=["method", "entity_id", "date", "doc_id", "chunk_index"]
    )
    return deduped[["method", "timestamp","entity_id", "date", "doc_id", "chunk_index", "relevance", "sentiment"]]

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


def results_to_chunks_df(
    results_raw: list[dict[str, Any]],
    *,
    universe_entity_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Convert raw search results into a chunk-level DataFrame."""
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
                entity_ids = list(set([eid for eid in entity_ids if eid in universe_entity_ids]))

            rows.append(
                {
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
                }
            )

    if not rows:
        return pd.DataFrame(columns=_CHUNKS_DF_COLUMNS)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Retrieval runners
# ---------------------------------------------------------------------------


def run_full_grid(
    companies: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full grid: one request per entity batch, full date range, no time splitting."""
    num_batches = math.ceil(len(companies) / BATCH_SIZE)
    total_chunk_budget = num_batches * MAX_CHUNKS_PER_REQUEST

    print(f"\n{'='*70}")
    print("FULL GRID (no splitting)")
    print(f"{'='*70}")
    print(f"  Companies : {len(companies)}")
    print(f"  Batches   : {num_batches}")
    print(f"  Max chunks: {MAX_CHUNKS_PER_REQUEST}/request")
    print(f"  Budget    : {total_chunk_budget:,}")

    t0 = time.perf_counter()
    results_raw = execute_full_grid_search(
        text=TEXT,
        universe_csv_path=UNIVERSE_CSV,
        start_date=START_DATE,
        end_date=END_DATE,
        batch_size=BATCH_SIZE,
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        max_chunks_per_request=MAX_CHUNKS_PER_REQUEST,
    )
    elapsed = time.perf_counter() - t0

    chunks_df = results_to_chunks_df(results_raw, universe_entity_ids=set(companies))
    chunks_df = chunks_df.assign(method="full_grid")

    meta = {
        "method": "full_grid",
        "num_companies": len(companies),
        "num_batches": num_batches,
        "num_windows": 1,
        "max_chunks_per_request": MAX_CHUNKS_PER_REQUEST,
        "total_chunk_budget": total_chunk_budget,
        "execute_time_s": round(elapsed, 2),
        "num_documents": len(results_raw),
        "num_chunks_retrieved": len(chunks_df),
    }

    print(f"  Documents : {len(results_raw)}")
    print(f"  Chunks    : {len(chunks_df):,}")
    print(f"  Time      : {elapsed:.2f}s")
    return chunks_df, meta


def run_full_grid_split(
    companies: list[str],
    *,
    total_chunk_budget: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full grid with time-window splitting (window_days=WINDOW_DAYS)."""
    num_batches = math.ceil(len(companies) / BATCH_SIZE)

    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days + 1
    num_windows = math.ceil(total_days / WINDOW_DAYS)

    if total_chunk_budget is None:
        total_chunk_budget = num_batches * MAX_CHUNKS_PER_REQUEST

    adjusted_max_chunks = max(1, total_chunk_budget // (num_batches * num_windows))

    print(f"\n{'='*70}")
    print(f"FULL GRID + TIME SPLIT ({WINDOW_DAYS}-day windows)")
    print(f"{'='*70}")
    print(f"  Companies : {len(companies)}")
    print(f"  Batches   : {num_batches}")
    print(f"  Windows   : {num_windows}")
    print(f"  Max chunks: {adjusted_max_chunks}/request (budget={total_chunk_budget:,})")

    t0 = time.perf_counter()
    results_raw = execute_full_grid_search(
        text=TEXT,
        universe_csv_path=UNIVERSE_CSV,
        start_date=START_DATE,
        end_date=END_DATE,
        batch_size=BATCH_SIZE,
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        max_chunks_per_request=adjusted_max_chunks,
        window_days=WINDOW_DAYS,
    )
    elapsed = time.perf_counter() - t0

    chunks_df = results_to_chunks_df(results_raw, universe_entity_ids=set(companies))
    chunks_df = chunks_df.assign(method="full_grid_split")

    meta = {
        "method": "full_grid_split",
        "num_companies": len(companies),
        "num_batches": num_batches,
        "num_windows": num_windows,
        "window_days": WINDOW_DAYS,
        "max_chunks_per_request": adjusted_max_chunks,
        "total_chunk_budget": total_chunk_budget,
        "execute_time_s": round(elapsed, 2),
        "num_documents": len(results_raw),
        "num_chunks_retrieved": len(chunks_df),
    }

    print(f"  Documents : {len(results_raw)}")
    print(f"  Chunks    : {len(chunks_df):,}")
    print(f"  Time      : {elapsed:.2f}s")
    return chunks_df, meta


def run_smart_batching(
    companies: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Smart Batching: plan (volume-aware) + execute."""
    print(f"\n{'='*70}")
    print("SMART BATCHING (plan + execute)")
    print(f"{'='*70}")
    print(f"  Companies         : {len(companies)}")
    print(f"  Volume query mode : {VOLUME_QUERY_MODE}")

    t_plan = time.perf_counter()
    plan = plan_search(
        text=TEXT,
        universe_csv_path=UNIVERSE_CSV,
        start_date=START_DATE,
        end_date=END_DATE,
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        volume_query_mode=VOLUME_QUERY_MODE,
        max_iterations_per_batch=MAX_ITERATIONS_PER_BATCH,
        apply_volume_splits=APPLY_VOLUME_SPLITS,
        min_period_days=MIN_PERIOD_DAYS,
        min_entities_per_basket=BATCH_SIZE,
    )
    plan_time = time.perf_counter() - t_plan

    baskets = plan.get("baskets", [])
    total_expected = plan.get("total_expected_chunks", 0)
    chunk_pct = CHUNK_PERCENTAGE

    print(f"  Baskets           : {len(baskets)}")
    print(f"  Expected chunks   : {total_expected:,}")
    print(f"  Chunk percentage  : {chunk_pct*100:.2f}%")
    print(f"  Plan time         : {plan_time:.2f}s")

    t_exec = time.perf_counter()
    results_raw = execute_search(
        search_plan=plan,
        chunk_percentage=chunk_pct,
        requests_per_minute=REQUESTS_PER_MINUTE,
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        second_pass=False,
    )
    exec_time = time.perf_counter() - t_exec

    chunks_df = results_to_chunks_df(results_raw, universe_entity_ids=set(companies))
    chunks_df = chunks_df.assign(method="smart_batching")

    meta = {
        "method": "smart_batching",
        "num_companies": len(companies),
        "num_baskets": len(baskets),
        "total_expected_chunks": total_expected,
        "chunk_percentage": chunk_pct,
        "plan_time_s": round(plan_time, 2),
        "execute_time_s": round(exec_time, 2),
        "total_time_s": round(plan_time + exec_time, 2),
        "num_documents": len(results_raw),
        "num_chunks_retrieved": len(chunks_df),
    }

    print(f"  Documents         : {len(results_raw)}")
    print(f"  Chunks            : {len(chunks_df):,}")
    print(f"  Exec time         : {exec_time:.2f}s")
    print(f"  Total time        : {plan_time + exec_time:.2f}s")
    return chunks_df, meta


# ---------------------------------------------------------------------------
# Prices retrieve
# ---------------------------------------------------------------------------

def load_and_prepare_prices(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["DATE"] = pd.to_datetime(df["DATE"])
    ret = df["USD_TOT_RET_CC"].astype(float) / 100.0
    ret = ret.clip(lower=WINDSOR_LOW, upper=WINDSOR_HIGH)
    df["RETURNS_T0"] = ret
    n_zero_vol = df.groupby("DATE").apply(lambda g: (g["VOLUME"].fillna(0) == 0).sum(), include_groups=False)
    n_total = df.groupby("DATE").size()
    pct_zero = n_zero_vol / n_total
    bad_dates = pct_zero[pct_zero > NON_TRADING_PCT].index
    df = df[~df["DATE"].isin(bad_dates)].copy()

    # Build full entity x trading-date grid first.
    df = df[["DATE", "RP_ENTITY_ID", "RETURNS_T0"]].drop_duplicates(subset=["DATE", "RP_ENTITY_ID"])
    all_dates = np.sort(df["DATE"].unique())
    all_entities = np.sort(df["RP_ENTITY_ID"].dropna().unique())
    full_idx = pd.MultiIndex.from_product(
        [all_entities, all_dates], names=["RP_ENTITY_ID", "DATE"]
    )
    df = (
        df.set_index(["RP_ENTITY_ID", "DATE"])
        .reindex(full_idx)
        .reset_index()
        .sort_values(["RP_ENTITY_ID", "DATE"])
        .reset_index(drop=True)
    )   

    return df[["DATE", "RP_ENTITY_ID", "RETURNS_T0"]]


# ---------------------------------------------------------------------------
# Backtest functions
# ---------------------------------------------------------------------------

def _run_one_backtest(df_merge: pd.DataFrame, col_signal: str, univ_label: str, plot_name: str | None = None) -> None:
    df_bt = df_merge.copy()
    df_bt["SECURITY_ID"] = df_bt["RP_ENTITY_ID"]
    cols_bt = ["DATE", "SECURITY_ID", "RP_ENTITY_ID", col_signal, "RETURNS_T0", "RETURNS_T1"]
    df_bt = df_bt[cols_bt].dropna(subset=[col_signal])
    if df_bt.empty:
        print(f"Skip {univ_label} / {col_signal}: no rows after dropna")
        return
    returns, _, statistics = portfolio_backtesting_pipeline(
        df_signal_and_returns=df_bt,
        start_date=start_date,
        end_date=end_date,
        rebalancing_frequency=rebalancing_frequency,
        exposure_long=exposure_long,
        exposure_short=exposure_short,
        cap=cap,
        column_signal=col_signal,
        daily_standardization=daily_standardization,
        plot_performance=plot_name,
    )
    statistics["universe"] = univ_label
    statistics["signal_column"] = col_signal

    return returns, statistics
    #results_list.append(statistics)
    #returns_by_signal[univ_label + "|" + col_signal] = returns
    #print(f"Done: {univ_label} / {col_signal}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def retrieve_chunks() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"API Base URL : {API_BASE_URL}")
    print(f"Universe CSV : {UNIVERSE_CSV}")
    print(f"Text         : {TEXT}")
    print(f"Date range   : {START_DATE} → {END_DATE}")

    companies = load_universe_from_csv(UNIVERSE_CSV)

    all_meta: list[dict[str, Any]] = []

    if True:
        # 1. Full grid + time-window splitting
        fgs_df, fgs_meta = run_full_grid_split(companies)
        all_meta.append(fgs_meta)

        fgs_path = OUTPUT_DIR / "backtest_full_grid_split.parquet"
        fgs_df.to_parquet(fgs_path, index=False)
        print(f"  Saved → {fgs_path}  ({len(fgs_df):,} rows)")

    # 2. Smart Batching (chunk_percentage controlled via CHUNK_PERCENTAGE constant)
    sb_df, sb_meta = run_smart_batching(companies)
    all_meta.append(sb_meta)

    sb_path = OUTPUT_DIR / "backtest_smart_batching.parquet"
    sb_df.to_parquet(sb_path, index=False)
    print(f"  Saved → {sb_path}  ({len(sb_df):,} rows)")

    # ---------------------------------------------------------------------------
    # Persist summary
    # ---------------------------------------------------------------------------
    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "text": TEXT,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "universe_csv": UNIVERSE_CSV,
            "num_companies": len(companies),
        },
        "methods": all_meta,
    }

    json_path = OUTPUT_DIR / "backtest_summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {json_path}")

    # Quick comparison table
    print(f"\n{'='*70}")
    print("BACKTEST RESULTS SUMMARY")
    print(f"{'='*70}")
    header = f"{'Method':<22}  {'Docs':>8}  {'Chunks':>10}  {'Time(s)':>8}  {'Budget':>10}"
    print(header)
    print("-" * len(header))
    for m in all_meta:
        t = m.get("total_time_s", m.get("execute_time_s", 0))
        budget = m.get("total_chunk_budget", m.get("total_expected_chunks", "—"))
        budget_str = f"{budget:,}" if isinstance(budget, int) else str(budget)
        print(
            f"{m['method']:<22}  {m['num_documents']:>8}  "
            f"{m['num_chunks_retrieved']:>10,}  {t:>8.2f}  {budget_str:>10}"
        )

def read_chunks() -> None:
    fgs_path = OUTPUT_DIR / "backtest_full_grid_split.parquet"
    fgs_df = _load_benchmark_raw(fgs_path,deduplicate=True)
    fgs_singal = signal_construction(fgs_df)

    sb_path = OUTPUT_DIR / "backtest_smart_batching.parquet"
    sb_df = _load_benchmark_raw(sb_path)
    sb_singal = signal_construction(sb_df)

    return fgs_singal, sb_singal

def load_prices() -> None:
    all_returns = []
    for univ_name, filename in PRICE_FILES.items():
        path = PRICE_PATH / filename
        if not path.exists():
            raise FileNotFoundError(f"Price file not found: {path}")
        r = load_and_prepare_prices(path)
        r["UNIV"] = univ_name
        all_returns.append(r)

    df_returns = pd.concat(all_returns, ignore_index=True)

    if EXCESS_TO_MARKET:
        if BACKTEST_PER_UNIVERSE:
            daily_avg = df_returns.groupby(["UNIV", "DATE"])["RETURNS_T0"].transform("mean")
        else:
            daily_avg = df_returns.groupby(["DATE"])["RETURNS_T0"].transform("mean")
        df_returns["RETURNS_T0"] = df_returns["RETURNS_T0"] - daily_avg

    df_returns = df_returns.sort_values(["UNIV", "RP_ENTITY_ID", "DATE"]).reset_index(drop=True)
    df_returns["RETURNS_T1"] = df_returns.groupby(["UNIV", "RP_ENTITY_ID"])["RETURNS_T0"].shift(-1)             
    df_returns['DATE'] = pd.to_datetime(df_returns['DATE'])
    return df_returns


if __name__ == "__main__":
    #retrieve_chunks()
    
    if True:
        fgs, sb = read_chunks()

        prices = load_prices()        
        fgs_merged = prices.merge(
        fgs,
        on=["DATE", "RP_ENTITY_ID"],
        how="inner"
        )
        sb_merged = prices.merge(
        sb,
        on=["DATE", "RP_ENTITY_ID"],
        how="inner"
        )
        
        start_date = fgs_merged["DATE"].min()
        end_date = fgs_merged["DATE"].max()

    
        a,b = _run_one_backtest(fgs_merged, 'score', "combined", plot_name="fgs_plot_deduplicated.png")
        a,b = _run_one_backtest(sb_merged, 'score', "combined", plot_name="sb_plot.png")

