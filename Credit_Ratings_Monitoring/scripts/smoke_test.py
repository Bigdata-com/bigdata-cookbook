#!/usr/bin/env python3
"""Short smoke test for Credit_Ratings_Monitoring REST + Luna migration."""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.constants import DEFAULT_LLM_MODEL
from src.feature_extractor import FeatureExtractor
from src.knowledge_graph_manager import get_entity_ids
from src.openai_sampling import sampling_params_for_model
from src.search_enhanced import search_credit_ratings

SMOKE_COMPANIES = ["Tesla", "Apple", "Microsoft", "Amazon", "Ford"]
SMOKE_DAYS = 14
SMOKE_QUERY = "credit rating outlook downgrade upgrade affirm"


def _assert_no_sdk() -> None:
    forbidden = ("bigdata_client",)
    for name in forbidden:
        if importlib.util.find_spec(name) is not None:
            raise RuntimeError(f"SDK module still importable: {name}")


def _assert_luna_safe() -> None:
    params = sampling_params_for_model(DEFAULT_LLM_MODEL, temperature=0)
    if params:
        raise RuntimeError(f"Luna model should omit sampling params, got {params}")


def main() -> int:
    print("=== Credit Ratings Monitoring smoke test ===")
    _assert_no_sdk()
    _assert_luna_safe()
    print(f"checklist: no SDK imports — PASS")
    print(f"checklist: luna-safe sampling ({DEFAULT_LLM_MODEL}) — PASS")

    company_names = SMOKE_COMPANIES[:4]
    end_date = dt.date.today().isoformat()
    start_date = (dt.date.today() - dt.timedelta(days=SMOKE_DAYS)).isoformat()
    print(f"window: {start_date} .. {end_date} ({SMOKE_DAYS}d)")
    print(f"companies: {company_names}")

    companies, full_names, _company_objects = get_entity_ids(company_names)
    if not companies:
        raise RuntimeError("REST entity lookup returned no IDs")
    print(f"REST entity lookup: {len(companies)} ids — PASS")

    id_to_name = dict(zip(companies, full_names, strict=True))
    df = search_credit_ratings(
        company_ids=companies,
        queries=[SMOKE_QUERY],
        start_date=start_date,
        end_date=end_date,
        id_to_name=id_to_name,
        basket_filtered_entities=False,
        chunk_percentage=0.05,
    )
    if df.empty:
        raise RuntimeError("Search returned zero rows")
    entity_names = set(df["entity_name"].astype(str).tolist())
    if len(entity_names) < 2:
        raise RuntimeError(
            f"basket_filtered_entities=False expected co-mentioned entities, got {entity_names}"
        )
    print(
        f"REST search: {len(df)} rows, {len(entity_names)} distinct entities "
        f"(basket_filtered_entities=False) — PASS"
    )

    sample = df.head(3)
    extractor = FeatureExtractor(llm_model=DEFAULT_LLM_MODEL)
    prompts = extractor.get_prompts_for_labeler(
        sample["chunk_text"].astype(str).tolist(),
        [
            {"entity_name": row["entity_name"], "headline": row["headline"]}
            for _, row in sample.iterrows()
        ],
    )
    results = extractor._run_labeling_prompts(  # noqa: SLF001
        prompts,
        extractor.labeling_prompt,
        max_workers=3,
    )
    ok = sum(1 for r in results if r.get("response") and "error" not in r["response"].lower())
    if ok == 0:
        raise RuntimeError(f"Luna labeling failed: {results}")
    print(f"OpenAI labeling ({DEFAULT_LLM_MODEL}): {ok}/{len(results)} ok — PASS")

    print("\n=== SMOKE TEST PASS ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n=== SMOKE TEST FAIL ===\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
