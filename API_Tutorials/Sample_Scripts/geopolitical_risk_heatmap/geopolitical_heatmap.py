# Requirements: pip install requests matplotlib seaborn pandas numpy
"""
G7 Geopolitical Risk Heatmap
-----------------------------
1. Calls the Volume API for each G7 country with a descriptive query over the
   last 90 days (parallel requests).
2. Aggregates daily counts into Monday-anchored weekly buckets.
3. Builds a country × week matrix, normalized row-wise (each country peaks at 1).
4. Renders a seaborn heatmap → geopolitical_heatmap.png
"""

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

API_KEY = os.environ["BIGDATA_API_KEY"]
BASE_URL = "https://api.bigdata.com"
HEADERS = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=90)
START_TS = f"{START_DATE.isoformat()}T00:00:00Z"
END_TS = f"{END_DATE.isoformat()}T23:59:59Z"

# Descriptive phrases per country — richer queries produce stronger
# semantic matches because embeddings are computed at chunk level.
G7 = {
    "United States": (
        "geopolitical risk United States political tensions military conflict "
        "trade sanctions foreign policy US diplomacy"
    ),
    "United Kingdom": (
        "geopolitical risk United Kingdom political instability foreign policy "
        "UK diplomacy Brexit sanctions military"
    ),
    "Germany": (
        "geopolitical risk Germany political tensions European security "
        "German foreign policy NATO energy sanctions"
    ),
    "France": (
        "geopolitical risk France political instability foreign policy "
        "French military diplomacy NATO European defense"
    ),
    "Italy": (
        "geopolitical risk Italy political crisis European stability "
        "Italian government foreign policy Mediterranean"
    ),
    "Japan": (
        "geopolitical risk Japan regional tensions military security "
        "Japanese foreign policy China Taiwan Strait"
    ),
    "Canada": (
        "geopolitical risk Canada foreign policy North America political tensions "
        "Canadian diplomacy trade sanctions Arctic"
    ),
}


# ── 1. Fetch daily volume for one country query ───────────────────────────────

def fetch_volume(country: str, query_text: str) -> tuple[str, list[dict]]:
    body = {
        "query": {
            "text": query_text,
            "auto_enrich_filters": False,
            "filters": {
                "timestamp": {"start": START_TS, "end": END_TS},
            },
        }
    }
    resp = requests.post(f"{BASE_URL}/v1/search/volume", headers=HEADERS, json=body)
    resp.raise_for_status()
    volume = resp.json().get("results", {}).get("volume", [])
    # Normalise field name: API may return "date" or "day"
    for entry in volume:
        if "day" in entry and "date" not in entry:
            entry["date"] = entry["day"]
    return country, volume


# ── 2. Aggregate daily → weekly (Monday-anchored) ────────────────────────────

def to_weekly(volume: list[dict]) -> dict[date, int]:
    weekly: dict[date, int] = defaultdict(int)
    for entry in volume:
        d = datetime.strptime(entry["date"], "%Y-%m-%d").date()
        week_start = d - timedelta(days=d.weekday())
        weekly[week_start] += entry.get("chunks", 0)
    return dict(sorted(weekly.items()))


# ── 3. Build country × week DataFrame (row-normalized) ───────────────────────

def build_matrix(country_weekly: dict[str, dict[date, int]]) -> pd.DataFrame:
    # Union of all week keys
    all_weeks = sorted(
        {week for weekly in country_weekly.values() for week in weekly}
    )

    rows = {}
    for country, weekly in country_weekly.items():
        row = [weekly.get(w, 0) for w in all_weeks]
        max_val = max(row) if row else 0
        rows[country] = [v / max_val if max_val > 0 else 0.0 for v in row]

    col_labels = [w.strftime("%b %d") for w in all_weeks]
    return pd.DataFrame(rows, index=col_labels).T


# ── 4. Render heatmap ─────────────────────────────────────────────────────────

def plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 5))

    sns.heatmap(
        df,
        ax=ax,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.4,
        linecolor="#e0e0e0",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"label": "Normalized chunk volume (0 = min, 1 = peak week)", "shrink": 0.7},
    )

    ax.set_title(
        f"G7 Geopolitical Risk — Weekly News Volume Heatmap\n"
        f"{START_DATE} → {END_DATE}  (row-normalized: 1.0 = peak week per country)",
        fontsize=13,
        pad=14,
    )
    ax.set_xlabel("Week starting (Monday)", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=10, rotation=0)

    plt.tight_layout()
    plt.savefig("geopolitical_heatmap.png", dpi=150)
    print("\nHeatmap saved → geopolitical_heatmap.png")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Window: {START_DATE} → {END_DATE}  ({len(G7)} countries)\n")

    # Fetch all countries in parallel
    country_weekly: dict[str, dict[date, int]] = {}
    with ThreadPoolExecutor(max_workers=len(G7)) as pool:
        futures = {
            pool.submit(fetch_volume, country, query): country
            for country, query in G7.items()
        }
        for future in as_completed(futures):
            country, volume = future.result()
            weekly = to_weekly(volume)
            total = sum(weekly.values())
            peak = max(weekly.values(), default=0)
            print(
                f"  {country:<16}  {len(weekly)} weeks  |  "
                f"total chunks: {total:,}  |  peak week: {peak:,}"
            )
            country_weekly[country] = weekly

    # Preserve G7 dict order in the matrix rows
    country_weekly_ordered = {c: country_weekly[c] for c in G7 if c in country_weekly}

    print("\nBuilding matrix and rendering heatmap...")
    df = build_matrix(country_weekly_ordered)
    print(df.to_string())

    plot(df)

    # Summary: hottest week per country
    print(f"\n{'Country':<18} {'Peak week':<14} {'Peak chunks':>12}")
    print("─" * 46)
    for country, weekly in country_weekly_ordered.items():
        if not weekly:
            continue
        peak_week = max(weekly, key=weekly.get)
        print(f"  {country:<16} {str(peak_week):<14} {weekly[peak_week]:>12,}")


if __name__ == "__main__":
    main()
