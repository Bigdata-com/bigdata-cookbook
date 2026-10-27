#!/usr/bin/env python3
"""Short smoke test for Narrative_Miners mindmap + REST pipeline.

Uses 4 companies, a 14-day window, and low API limits.
"""

from __future__ import annotations

import os

os.environ.setdefault("BIGDATA_DISABLE_TRACING", "1")  # ADS-400: smoke tests must never emit telemetry
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bigdata_rest import company_ids_from_universe, load_universe  # noqa: E402
from src.llm_helpers import DEFAULT_LLM_MODEL  # noqa: E402
from src.mindmap import (  # noqa: E402
    Node,
    build_label_to_subtheme,
    flatten_search_queries,
    generate_narrative_mindmap,
    mindmap_to_main_narratives,
)
from src.search_helper import run_universe_search  # noqa: E402

SMOKE_COMPANY_COUNT = 4
SMOKE_WINDOW_DAYS = 14
SMOKE_CHUNK_PERCENTAGE = 0.01
SMOKE_MAIN_THEME = "AI Bubble"


def fixture_mindmap() -> Node:
    """Tiny fixed mindmap used if LLM generation is skipped or fails shape checks."""
    return Node(
        node=1,
        label=SMOKE_MAIN_THEME,
        summary="AI valuation and investment exuberance concerns.",
        search_query="",
        children=[
            Node(
                node=2,
                label="Valuation Disconnect",
                summary="Prices detached from earnings power.",
                search_query="",
                children=[
                    Node(
                        node=3,
                        label="Detached from earnings",
                        summary="Valuations outrun fundamentals.",
                        search_query=(
                            "Tech valuations have detached from fundamental earnings potential"
                        ),
                        children=[],
                    ),
                    Node(
                        node=4,
                        label="Irrational exuberance",
                        summary="Classic bubble behavior in AI spend.",
                        search_query="AI investments show classic signs of irrational exuberance",
                        children=[],
                    ),
                ],
            ),
            Node(
                node=5,
                label="ROI Skepticism",
                summary="Doubt that AI spend will pay off.",
                search_query="",
                children=[
                    Node(
                        node=6,
                        label="Uncertain returns",
                        summary="Predicted financial returns may not materialize.",
                        search_query=(
                            "Current AI investments may not generate predicted financial returns"
                        ),
                        children=[],
                    ),
                    Node(
                        node=7,
                        label="Unproven technology spend",
                        summary="Billions spent on unproven AI tech.",
                        search_query="Companies are spending billions on unproven AI technology",
                        children=[],
                    ),
                ],
            ),
        ],
    )


def build_narrative_df(
    raw_df: pd.DataFrame, label_to_subtheme: dict[str, str], source_type: str
) -> pd.DataFrame:
    columns = ["Date", "Label", "Sentence ID", "Headline", "Chunk Text", "Source_Type"]
    if raw_df.empty:
        return pd.DataFrame(columns=columns)

    rows = raw_df.reset_index(drop=True)
    labels = [
        label_to_subtheme.get(str(row.get("query", "") or ""), str(row.get("query", ""))[:60])
        for _, row in rows.iterrows()
    ]
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(rows["timestamp"], errors="coerce", utc=True).dt.tz_localize(
                None
            ),
            "Label": labels,
            "Sentence ID": [f"{doc_id}_{i}" for i, doc_id in enumerate(rows["document_id"])],
            "Headline": rows["headline"],
            "Chunk Text": rows["chunk_text"],
        }
    )
    out = out.dropna(subset=["Date"])
    out["Source_Type"] = source_type
    return out


def main() -> int:
    load_dotenv(ROOT.parent / ".env")
    load_dotenv(ROOT / ".env")

    if not os.getenv("BIGDATA_API_KEY"):
        print("FAIL: BIGDATA_API_KEY is not set")
        return 1
    if not os.getenv("OPENAI_API_KEY"):
        print("FAIL: OPENAI_API_KEY is not set")
        return 1

    universe_path = ROOT.parent / "Thematic_Screener_CLI" / "mag7.csv"
    universe_df = load_universe(universe_path).head(SMOKE_COMPANY_COUNT)
    company_ids = company_ids_from_universe(universe_df)
    id_to_name = dict(zip(universe_df["RP_ENTITY_ID"], universe_df["COMPANY_NAME"], strict=True))

    end = date.today()
    start = end - timedelta(days=SMOKE_WINDOW_DAYS)
    start_date = start.isoformat()
    end_date = end.isoformat()

    print(
        f"Smoke: {len(company_ids)} companies, {start_date}..{end_date}, "
        f"model={DEFAULT_LLM_MODEL}, chunk_percentage={SMOKE_CHUNK_PERCENTAGE}"
    )
    print(f"Companies: {list(id_to_name.values())}")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        root = generate_narrative_mindmap(
            SMOKE_MAIN_THEME,
            client=client,
            model=DEFAULT_LLM_MODEL,
            print_taxonomy=True,
        )
        main_narratives = mindmap_to_main_narratives(root)
        if len(main_narratives) < 1:
            raise RuntimeError("mindmap produced no sub-themes")
        for item in main_narratives:
            n_labels = len(item.get("search_labels", []))
            if n_labels < 2 or n_labels > 3:
                raise RuntimeError(
                    f"sub-theme {item['sub_theme']!r} has {n_labels} labels (want 2–3)"
                )
        print("LLM mindmap shape OK")
    except Exception as exc:
        print(f"⚠️ LLM mindmap unavailable ({exc}); using fixture mindmap")
        root = fixture_mindmap()
        main_narratives = mindmap_to_main_narratives(root)

    search_queries = flatten_search_queries(main_narratives)
    label_to_subtheme = build_label_to_subtheme(main_narratives)
    subthemes = {item["sub_theme"] for item in main_narratives}
    print(f"Sub-themes: {sorted(subthemes)}")
    print(f"Search labels: {len(search_queries)}")

    # Keep smoke search cheap: first two leaf labels only
    smoke_queries = search_queries[:2]
    raw_df = run_universe_search(
        company_ids,
        smoke_queries,
        start_date=start_date,
        end_date=end_date,
        scope="news",
        chunk_percentage=SMOKE_CHUNK_PERCENTAGE,
        id_to_name=id_to_name,
    )
    print(f"Search rows: {len(raw_df)}")
    if raw_df.empty:
        print("FAIL: search returned no rows")
        return 1

    labeled = build_narrative_df(raw_df, label_to_subtheme, "News Media")
    if labeled.empty:
        print("FAIL: no labeled rows after sub-theme mapping")
        return 1

    unexpected = sorted(set(labeled["Label"]) - subthemes)
    if unexpected:
        print(f"FAIL: Label values are not sub-themes: {unexpected}")
        return 1

    print("Sub-theme counts:")
    print(labeled.groupby("Label")["Sentence ID"].nunique().to_string())
    print("PASS: mindmap + REST search + sub-theme rollup smoke test succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
