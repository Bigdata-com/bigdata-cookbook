#!/usr/bin/env python3
"""Short smoke test for Narrative_Miners REST + OpenAI pipeline.

Uses 4 companies, a 14-day window, and low API limits.
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bigdata_rest import company_ids_from_universe, load_universe  # noqa: E402
from src.llm_helpers import DEFAULT_LLM_MODEL, sampling_params_for_model  # noqa: E402
from src.search_helper import run_universe_search  # noqa: E402

SMOKE_COMPANY_COUNT = 4
SMOKE_WINDOW_DAYS = 14
SMOKE_QUERIES = [
    "Tech valuations have detached from fundamental earnings potential",
]
SMOKE_CHUNK_PERCENTAGE = 0.01
SMOKE_MAX_OPENAI_LABELS = 2


class NarrativeLabel(str, Enum):
    VALUATION_DISCONNECT = "Valuation Disconnect"
    ROI_SKEPTICISM = "ROI Skepticism"
    INVESTMENT_SURGE = "Investment Surge"
    REGULATORY_RISK = "Regulatory Risk"
    BUBBLE_COMPARISON = "Bubble Comparison"


class NarrativeClassification(BaseModel):
    label: NarrativeLabel


def label_with_openai(client: OpenAI, text: str, model: str = DEFAULT_LLM_MODEL) -> str:
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the AI-bubble narrative expressed in the following text "
                    "snippet into exactly one label."
                ),
            },
            {"role": "user", "content": text[:600]},
        ],
        response_format=NarrativeClassification,
        **sampling_params_for_model(model, temperature=0.0),
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned no parsed label")
    return parsed.label.value


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

    raw_df = run_universe_search(
        company_ids,
        SMOKE_QUERIES,
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

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    labels: list[str] = []
    for i, row in raw_df.head(SMOKE_MAX_OPENAI_LABELS).iterrows():
        label = label_with_openai(client, str(row["chunk_text"]))
        labels.append(label)
        print(f"  labeled row {i}: {label}")

    if not labels:
        print("FAIL: no OpenAI labels produced")
        return 1

    print("PASS: REST search + gpt-5.6-luna labeling smoke test succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
