# Requirements: pip install requests matplotlib
"""
Macro Theme Radar
-----------------
1. Calls the Volume API for each theme over the last 90 days.
2. Aggregates daily counts into weekly buckets.
3. Normalizes each series to 0–1.
4. Plots all themes on a single line chart, saved to macro_theme_radar.png.
"""

import os
from collections import defaultdict
from datetime import date, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from datetime import datetime

API_KEY = os.environ["BIGDATA_API_KEY"]
BASE_URL = "https://api.bigdata.com"
HEADERS = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

# Descriptive phrases work much better than single words — embeddings are
# computed at chunk level, so a richer query produces stronger semantic matches.
THEMES = {
    "inflation":    "inflation rising prices CPI consumer price index purchasing power",
    "recession":    "recession economic contraction GDP decline slowdown downturn",
    "rate hike":    "interest rate hike Federal Reserve central bank monetary tightening",
    "China trade":  "China trade tariffs exports imports trade war economic relations",
}

END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=90)
START_TS = f"{START_DATE.isoformat()}T00:00:00Z"
END_TS = f"{END_DATE.isoformat()}T23:59:59Z"


# ── 1. Fetch daily volume for one theme ──────────────────────────────────────

def fetch_volume(query_text: str) -> list[dict]:
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
    return volume


# ── 2. Aggregate daily → weekly (Monday-anchored) ────────────────────────────

def to_weekly(volume: list[dict]) -> dict[date, int]:
    weekly: dict[date, int] = defaultdict(int)
    for entry in volume:
        d = datetime.strptime(entry["date"], "%Y-%m-%d").date()
        # Anchor each day to its Monday
        week_start = d - timedelta(days=d.weekday())
        weekly[week_start] += entry.get("chunks", 0)
    return dict(sorted(weekly.items()))


# ── 3. Normalize series to 0–1 ───────────────────────────────────────────────

def normalize(series: dict[date, int]) -> dict[date, float]:
    if not series:
        return {}
    max_val = max(series.values())
    if max_val == 0:
        return {k: 0.0 for k in series}
    return {k: v / max_val for k, v in series.items()}


# ── 4. Plot all themes ────────────────────────────────────────────────────────

def plot(theme_series: dict[str, dict[date, float]]) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    markers = ["o", "s", "^", "D"]

    for (theme, series), color, marker in zip(theme_series.items(), colors, markers):
        if not series:
            print(f"  WARNING: no data for '{theme}', skipping.")
            continue
        dates = list(series.keys())
        values = list(series.values())
        ax.plot(
            dates, values,
            label=f'"{theme}"',
            color=color,
            marker=marker,
            markersize=5,
            linewidth=2,
        )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    fig.autofmt_xdate()

    ax.set_title(
        f"Macro Theme Radar — Normalized Weekly Volume\n"
        f"{START_DATE} → {END_DATE}  (1.0 = peak week for each theme)",
        fontsize=13,
    )
    ax.set_ylabel("Normalized chunk volume (0–1)")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", alpha=0.15)

    plt.tight_layout()
    plt.savefig("macro_theme_radar.png", dpi=150)
    print("\nChart saved → macro_theme_radar.png")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Window: {START_DATE} → {END_DATE}  ({len(THEMES)} themes)\n")

    theme_series: dict[str, dict[date, float]] = {}
    weekly_raw: dict[str, dict[date, int]] = {}

    for label, query_text in THEMES.items():
        print(f"  Fetching volume: \"{label}\"...")
        daily = fetch_volume(query_text)
        weekly = to_weekly(daily)
        total_chunks = sum(weekly.values())
        peak = max(weekly.values(), default=0)
        print(f"    {len(weekly)} weeks  |  total chunks: {total_chunks}  |  peak week: {peak}")
        weekly_raw[label] = weekly
        theme_series[label] = normalize(weekly)

    plot(theme_series)

    # Print summary table
    print(f"\n{'Theme':<18} {'Peak week':<14} {'Peak chunks':>12}")
    print("─" * 46)
    for theme in weekly_raw:
        weekly = weekly_raw[theme]
        if not weekly:
            continue
        peak_week = max(weekly, key=weekly.get)
        peak_val = weekly[peak_week]
        print(f"  {theme:<16} {str(peak_week):<14} {peak_val:>12,}")


if __name__ == "__main__":
    main()
