# Requirements: pip install requests matplotlib
"""
Repsol + Oil Crisis — Volume Spike + Top Chunks
------------------------------------------------
1. Resolves Repsol's entity ID via Knowledge Graph.
2. Fetches daily document volume for "Repsol oil crisis" over the last 30 days.
3. Plots the time series (saves to repsol_volume.png).
4. Finds the day with the highest chunk volume.
5. Fetches and prints the top 10 most relevant chunks from that day.
"""

import os
from datetime import date, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from datetime import datetime

API_KEY = os.environ["BIGDATA_API_KEY"]
BASE_URL = "https://api.bigdata.com"
HEADERS = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=30)
START_TS = f"{START_DATE.isoformat()}T00:00:00Z"
END_TS = f"{END_DATE.isoformat()}T23:59:59Z"
QUERY_TEXT = "Repsol oil crisis"


# ── 1. Resolve Repsol entity ID ───────────────────────────────────────────────

def resolve_company_id(name: str) -> tuple[str, str]:
    resp = requests.post(
        f"{BASE_URL}/v1/knowledge-graph/companies",
        headers=HEADERS,
        json={"query": name, "types": ["PUBLIC"]},
    )
    resp.raise_for_status()
    companies = resp.json().get("results", [])
    entity_id = companies[0]["id"]
    entity_name = companies[0]["name"]
    print(f"Resolved '{name}' → {entity_name} (id={entity_id})")
    return entity_id, entity_name


# ── 2. Fetch daily volume ─────────────────────────────────────────────────────

def fetch_volume(entity_id: str) -> list[dict]:
    body = {
        "query": {
            "text": QUERY_TEXT,
            "auto_enrich_filters": False,
            "filters": {
                "timestamp": {"start": START_TS, "end": END_TS},
                "entity": {"any_of": [entity_id], "all_of": [], "none_of": []},
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


# ── 3. Plot volume time series ────────────────────────────────────────────────

def plot_volume(volume: list[dict], peak_date: str, entity_name: str) -> None:
    dates = [datetime.strptime(e["date"], "%Y-%m-%d") for e in volume]
    chunks = [e.get("chunks", 0) for e in volume]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(dates, chunks, color="#4C72B0", alpha=0.7, label="Chunks / day")

    # Highlight peak day
    peak_dt = datetime.strptime(peak_date, "%Y-%m-%d")
    peak_val = next(e.get("chunks", 0) for e in volume if e["date"] == peak_date)
    ax.bar([peak_dt], [peak_val], color="#DD4444", label=f"Peak: {peak_date}")
    ax.annotate(
        f"Peak\n{peak_val} chunks",
        xy=(peak_dt, peak_val),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        color="#DD4444",
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    fig.autofmt_xdate()
    ax.set_title(f'Daily chunk volume — "{QUERY_TEXT}" ({START_DATE} → {END_DATE})')
    ax.set_ylabel("Chunks")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("repsol_volume.png", dpi=150)
    print("\nChart saved → repsol_volume.png")
    plt.show()


# ── 4. Fetch top 10 chunks for peak day ──────────────────────────────────────

def fetch_top_chunks(entity_id: str, peak_date: str) -> list[dict]:
    day_start = f"{peak_date}T00:00:00Z"
    day_end = f"{peak_date}T23:59:59Z"
    body = {
        "query": {
            "text": QUERY_TEXT,
            "auto_enrich_filters": False,
            "filters": {
                "timestamp": {"start": day_start, "end": day_end},
                "entity": {"any_of": [entity_id], "all_of": [], "none_of": []},
            },
            "ranking_params": {"freshness_boost": 0, "source_boost": 0},
            "max_chunks": 10,
        }
    }
    resp = requests.post(f"{BASE_URL}/v1/search", headers=HEADERS, json=body)
    resp.raise_for_status()

    chunks: list[dict] = []
    for doc in resp.json().get("results", []):
        for ch in doc.get("chunks", []):
            chunks.append(
                {
                    "headline": doc.get("headline", ""),
                    "source": doc.get("source", {}).get("name", ""),
                    "url": doc.get("url", ""),
                    "text": ch.get("text", ""),
                    "relevance": ch.get("relevance", 0),
                }
            )
            if len(chunks) >= 10:
                return chunks
    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Window: {START_DATE} → {END_DATE}\n{'=' * 70}")

    entity_id, entity_name = resolve_company_id("Repsol")

    print(f"\nFetching daily volume for '{QUERY_TEXT}'...")
    volume = fetch_volume(entity_id)

    if not volume:
        print("No volume data returned for this period.")
        return

    # Find peak day by chunk count
    peak_entry = max(volume, key=lambda e: e.get("chunks", 0))
    peak_date = peak_entry["date"]
    peak_chunks = peak_entry.get("chunks", 0)
    peak_docs = peak_entry.get("documents", 0)

    print(f"Peak day: {peak_date}  |  chunks={peak_chunks}  docs={peak_docs}")

    plot_volume(volume, peak_date, entity_name)

    print(f"\nTop 10 chunks from {peak_date}:\n{'─' * 70}")
    top_chunks = fetch_top_chunks(entity_id, peak_date)

    if not top_chunks:
        print("No chunks found for the peak day.")
        return

    for i, ch in enumerate(top_chunks, start=1):
        print(f"\n[{i:02d}] relevance={ch['relevance']:.3f} | {ch['source']}")
        print(f"     {ch['headline']}")
        print(f"     {ch['text'][:300].strip()}")
        if ch["url"]:
            print(f"     {ch['url']}")


if __name__ == "__main__":
    main()
