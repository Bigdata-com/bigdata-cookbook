#!/usr/bin/env python3
"""Short smoke test for Rising_Bond_Spread_Risks migration."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT.parent / ".env")
load_dotenv(ROOT / ".env")


def _check_no_sdk_imports() -> None:
    src_dir = ROOT / "src"
    forbidden_modules = {
        "bigdata",
        "bigdata_client",
        "bigdata_research_tools",
    }
    for path in src_dir.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in forbidden_modules:
                        raise AssertionError(
                            f"Forbidden import {alias.name!r} in {path.name}"
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if root in forbidden_modules:
                    raise AssertionError(
                        f"Forbidden import from {node.module!r} in {path.name}"
                    )


def _check_luna_defaults() -> None:
    from src.openai_utils import DEFAULT_LLM_MODEL, sampling_params_for_model

    assert DEFAULT_LLM_MODEL == "gpt-5.6-luna"
    assert sampling_params_for_model("gpt-5.6-luna", temperature=0.0) == {}
    assert sampling_params_for_model("gpt-4o-mini", temperature=0.0) == {"temperature": 0.0}

    from src.simple_labeler import SimpleLabeler
    from src.report_generator import SummaryGenerator

    assert SimpleLabeler().model == "gpt-5.6-luna"
    assert SummaryGenerator().model == "gpt-5.6-luna"


def _check_get_scored_df() -> None:
    from src.entity_risk_prompt_labeler import get_scored_df

    df = pd.DataFrame(
        {
            "Entity": ["NVIDIA Corp.", "NVIDIA Corp.", "Microsoft Corp."],
            "Country": ["United States"] * 3,
            "label": ["ai_debt_credit_spread_risk", "hyperscaler_leverage_concern", "unassigned"],
        }
    )
    scored = get_scored_df(df, index_columns=["Entity", "Country"], pivot_column="label")
    assert "Composite Score" in scored.columns
    assert scored.loc[0, "Entity"] == "NVIDIA Corp."


def _check_basket_filtered_entities() -> None:
    text = (ROOT / "src" / "search_helper.py").read_text()
    assert "basket_filtered_entities=True" in text


def _check_imports() -> None:
    modules = [
        "src.bigdata_rest",
        "src.search_helper",
        "src.search_entities",
        "src.entity_risk_prompt_labeler",
        "src.simple_labeler",
        "src.report_generator",
        "src.narrative_dashboard",
        "src.visualization_tool",
    ]
    for module in modules:
        __import__(module)


def _optional_live_search() -> None:
    if not os.getenv("BIGDATA_API_KEY"):
        print("SKIP live search: BIGDATA_API_KEY not set")
        return

    from src.bigdata_rest import company_ids_from_universe, load_universe
    from src.search_helper import run_universe_search

    universe = load_universe(ROOT.parent / "Thematic_Screener_CLI" / "40_companies.csv")
    universe = universe.loc[
        universe["COMPANY_NAME"].isin(["NVIDIA Corp.", "Microsoft Corp.", "Amazon.com Inc."])
    ]
    ids = company_ids_from_universe(universe)
    df = run_universe_search(
        company_ids=ids,
        queries=["corporate bond spread widening AI capex"],
        start_date="2026-01-01",
        end_date="2026-02-01",
        scope="news",
        chunk_percentage=0.01,
        id_to_name=dict(zip(universe["RP_ENTITY_ID"], universe["COMPANY_NAME"])),
    )
    print(f"live search rows: {len(df)}")


def main() -> None:
    _check_no_sdk_imports()
    _check_luna_defaults()
    _check_get_scored_df()
    _check_basket_filtered_entities()
    _check_imports()
    _optional_live_search()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
