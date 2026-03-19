# Requirements: pip install requests
"""
Iran Volume Spike + Co-mention Analysis
-----------------------------------------
1. Resolves "Iran" to a place entity ID via Co-mentions auto-enrichment
   (places[0] pattern — same as people lookup, no dedicated places endpoint).
2. Fetches daily chunk volume for Iran over the last 90 days.
3. Detects spike days: any day where chunks > mean + SPIKE_STD * std.
4. Defines the spike window as first_spike_date → last_spike_date
   (±SPIKE_PADDING days buffer on each side).
5. Runs Co-mentions for Iran over the spike window with limit=1000,
   then ranks all returned entity categories by headline count.
6. Prints a spike timeline and a ranked co-mention table.
"""

import os
import statistics
from datetime import date, datetime, timedelta

import requests

API_KEY = os.environ["BIGDATA_API_KEY"]
BASE_URL = "https://api.bigdata.com"
HEADERS = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

SPIKE_STD     = 1.5   # days above mean + N*std are spikes
SPIKE_PADDING = 3     # extra days added around the spike window for co-mentions
TOP_N         = 20    # co-mention rows to display per category
FETCH_N       = 1000  # co-mention limit (shared across all entity types)

END_DATE   = date.today()
START_DATE = END_DATE - timedelta(days=90)
START_TS   = f"{START_DATE.isoformat()}T00:00:00Z"
END_TS     = f"{END_DATE.isoformat()}T23:59:59Z"


# ── 1. Resolve "Iran" → place entity ID ──────────────────────────────────────

def resolve_place(name: str) -> tuple[str, str]:
    """
    Uses Co-mentions auto_enrich_filters to resolve a place name.
    places[0] = the focal place entity (same pattern as people lookup).
    Returns (entity_id, resolved_name).
    """
    body = {
        "query": {
            "text": name,
            "auto_enrich_filters": True,
            "filters": {"timestamp": {"start": START_TS, "end": END_TS}},
        },
        "limit": 5,
    }
    resp = requests.post(
        f"{BASE_URL}/v1/search/co-mentions/entities", headers=HEADERS, json=body
    )
    resp.raise_for_status()
    places = resp.json().get("results", {}).get("places", [])
    if not places:
        raise ValueError(f"Could not resolve place '{name}' via co-mentions")
    entity_id = places[0]["id"]

    # Confirm name via entities/id
    info_resp = requests.post(
        f"{BASE_URL}/v1/knowledge-graph/entities/id",
        headers=HEADERS,
        json={"values": [entity_id]},
    )
    info_resp.raise_for_status()
    resolved_name = (
        info_resp.json().get("results", {}).get(entity_id, {}).get("name", name)
    )
    return entity_id, resolved_name


# ── 2. Fetch daily volume ─────────────────────────────────────────────────────

def fetch_volume(entity_id: str) -> list[dict]:
    body = {
        "query": {
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
    for entry in volume:
        if "day" in entry and "date" not in entry:
            entry["date"] = entry["day"]
    return sorted(volume, key=lambda e: e["date"])


# ── 3. Detect spikes ──────────────────────────────────────────────────────────

def detect_spikes(volume: list[dict]) -> tuple[list[dict], float, float, float]:
    """
    Returns (spike_days, mean, stdev, threshold).
    spike_days: entries where chunks > mean + SPIKE_STD * stdev.
    """
    counts = [e.get("chunks", 0) for e in volume]
    mean   = statistics.mean(counts) if counts else 0
    stdev  = statistics.stdev(counts) if len(counts) > 1 else 0
    threshold = mean + SPIKE_STD * stdev
    spike_days = [e for e in volume if e.get("chunks", 0) >= threshold]
    return spike_days, mean, stdev, threshold


# ── 4. Fetch co-mentions for the spike window ─────────────────────────────────

def fetch_comentions(entity_id: str, window_start: date, window_end: date) -> dict:
    ws = f"{window_start.isoformat()}T00:00:00Z"
    we = f"{window_end.isoformat()}T23:59:59Z"
    body = {
        "query": {
            "auto_enrich_filters": False,
            "filters": {
                "timestamp": {"start": ws, "end": we},
                "entity": {"any_of": [entity_id], "all_of": [], "none_of": []},
            },
        },
        "limit": FETCH_N,
    }
    resp = requests.post(
        f"{BASE_URL}/v1/search/co-mentions/entities", headers=HEADERS, json=body
    )
    resp.raise_for_status()
    return resp.json().get("results", {})


# ── 5. Resolve entity IDs → names ────────────────────────────────────────────

def resolve_entities(entity_ids: list[str]) -> dict[str, dict]:
    resolved: dict[str, dict] = {}
    for i in range(0, len(entity_ids), 100):
        batch = entity_ids[i : i + 100]
        resp = requests.post(
            f"{BASE_URL}/v1/knowledge-graph/entities/id",
            headers=HEADERS,
            json={"values": batch},
        )
        resp.raise_for_status()
        for eid, info in resp.json().get("results", {}).items():
            resolved[eid] = {
                "name":    info.get("name", eid),
                "country": info.get("country") or "—",
                "type":    info.get("type") or info.get("category") or "—",
            }
    return resolved


# ── 6. Print spike timeline ───────────────────────────────────────────────────

def print_timeline(volume: list[dict], spike_days: list[dict], threshold: float,
                   mean: float, stdev: float, entity_name: str) -> None:
    spike_dates = {e["date"] for e in spike_days}
    peak = max(volume, key=lambda e: e.get("chunks", 0))

    print(f"\n{'═' * 70}")
    print(f"  Volume timeline — {entity_name}  |  {START_DATE} → {END_DATE}")
    print(f"  Mean: {mean:,.0f}  |  Std: {stdev:,.0f}  |  "
          f"Threshold (mean + {SPIKE_STD}σ): {threshold:,.0f}")
    print(f"{'═' * 70}")

    bar_max = max(e.get("chunks", 0) for e in volume) or 1
    bar_width = 40

    for entry in volume:
        d      = entry["date"]
        chunks = entry.get("chunks", 0)
        bar_n  = int(chunks / bar_max * bar_width)
        bar    = "█" * bar_n
        spike_marker = " ◄ SPIKE" if d in spike_dates else ""
        peak_marker  = " ◄ PEAK"  if d == peak["date"] else ""
        marker = peak_marker or spike_marker
        print(f"  {d}  {bar:<{bar_width}}  {chunks:>6,}{marker}")

    print(f"\n  Spike days ({len(spike_days)} total):")
    if spike_days:
        for s in spike_days:
            print(f"    {s['date']}   chunks: {s.get('chunks', 0):>6,}   "
                  f"docs: {s.get('documents', 0):>5,}   "
                  f"sentiment: {(s.get('sentiment') or 0.0):>+.4f}")
    else:
        print("    (none — try lowering SPIKE_STD)")


# ── 7. Print co-mention table ─────────────────────────────────────────────────

def print_comentions(raw: dict, entity_info: dict[str, dict],
                     window_start: date, window_end: date, entity_name: str) -> None:
    print(f"\n  Co-mentions with {entity_name}  |  {window_start} → {window_end}")
    print(f"  (ranked by headline count, top {TOP_N} per category)\n")

    CATEGORY_ORDER = ["companies", "people", "places", "organizations", "products", "concepts"]

    for cat in CATEGORY_ORDER:
        entities = raw.get(cat, [])
        if not entities:
            continue

        # Sort by headlines desc, chunks as tiebreaker
        entities.sort(
            key=lambda e: (e.get("total_headlines_count", 0), e.get("total_chunks_count", 0)),
            reverse=True,
        )
        entities = entities[:TOP_N]

        col_rank = 4
        col_name = 36
        col_ctry = 8
        col_hl   = 10
        col_ch   = 10

        header = (
            f"  {'#':>{col_rank}}  {'Name':<{col_name}}  "
            f"{'Ctry':<{col_ctry}}  {'Headlines':>{col_hl}}  {'Chunks':>{col_ch}}"
        )
        divider = "  " + "─" * (len(header) - 2)

        print(f"  ▌ {cat.upper()}")
        print(header)
        print(divider)

        for rank, entity in enumerate(entities, start=1):
            eid       = entity["id"]
            info      = entity_info.get(eid, {})
            name      = info.get("name", eid)[:col_name]
            country   = info.get("country", "—")
            headlines = entity.get("total_headlines_count", 0)
            chunks    = entity.get("total_chunks_count")
            hl_disp   = f"{headlines:,}" if headlines else "—"
            ch_disp   = f"{chunks:,}"    if chunks    else "—"

            print(
                f"  {rank:>{col_rank}}  {name:<{col_name}}  "
                f"{country:<{col_ctry}}  {hl_disp:>{col_hl}}  {ch_disp:>{col_ch}}"
            )
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Resolving 'Iran' entity ID...")
    entity_id, entity_name = resolve_place("Iran")
    print(f"  → {entity_name}  [{entity_id}]")

    print(f"\nFetching daily volume  ({START_DATE} → {END_DATE})...")
    volume = fetch_volume(entity_id)
    print(f"  → {len(volume)} days")

    spike_days, mean, stdev, threshold = detect_spikes(volume)
    print(f"  → {len(spike_days)} spike day(s) above threshold ({threshold:,.0f} chunks)")

    # Define co-mention window: first spike → last spike, ±SPIKE_PADDING days
    if spike_days:
        spike_dates = [datetime.strptime(e["date"], "%Y-%m-%d").date() for e in spike_days]
        window_start = min(spike_dates) - timedelta(days=SPIKE_PADDING)
        window_end   = max(spike_dates) + timedelta(days=SPIKE_PADDING)
    else:
        peak_date    = datetime.strptime(
            max(volume, key=lambda e: e.get("chunks", 0))["date"], "%Y-%m-%d"
        ).date()
        window_start = peak_date - timedelta(days=7)
        window_end   = peak_date + timedelta(days=7)
        print(f"  (no spikes detected — using ±7 days around peak: {peak_date})")

    # Clamp to overall window
    window_start = max(window_start, START_DATE)
    window_end   = min(window_end, END_DATE)

    print_timeline(volume, spike_days, threshold, mean, stdev, entity_name)

    print(f"\nFetching co-mentions  ({window_start} → {window_end})...")
    raw = fetch_comentions(entity_id, window_start, window_end)
    total_entities = sum(len(v) for v in raw.values())
    print(f"  → {total_entities} entities across {len(raw)} categories")

    # Resolve all entity IDs
    all_ids = [e["id"] for entities in raw.values() for e in entities]
    print(f"  Resolving {len(all_ids)} entity IDs...")
    entity_info = resolve_entities(all_ids)

    print_comentions(raw, entity_info, window_start, window_end, entity_name)


if __name__ == "__main__":
    main()
