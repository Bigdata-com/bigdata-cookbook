#!/usr/bin/env python3
"""Short smoke test for Report_Generator_Specialized_Report_Tariffs (REST + luna)."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
load_dotenv(PROJECT_DIR / ".env")

from src.bigdata_rest import company_ids_from_universe, load_universe
from src.label.label_process import LabelProcessor
from src.mindmap.generate_trees import generate_themes_tree_dict, get_most_granular_elements
from src.openai_compat import DEFAULT_LLM_MODEL
from src.report_generator import GenerateReport
from src.search.content_retrieval import DataRetriever


def main() -> int:
    if not os.getenv("BIGDATA_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        print("FAIL: missing BIGDATA_API_KEY or OPENAI_API_KEY")
        return 1

    main_theme = "US Import Tariffs Corporate Risk Impact Analysis"
    focus = (
        "Provide a detailed taxonomy of risks describing how new American import tariffs "
        "will impact worldwide companies, their operations and strategy."
    )
    llm_model = DEFAULT_LLM_MODEL

    universe_path = PROJECT_DIR.parent / "Thematic_Screener_CLI" / "40_companies.csv"
    universe_df = load_universe(universe_path).head(3).reset_index(drop=True)
    company_ids = company_ids_from_universe(universe_df)
    id_to_name = dict(zip(universe_df["RP_ENTITY_ID"], universe_df["COMPANY_NAME"]))
    list_entities = [SimpleNamespace(id=eid, name=name) for eid, name in id_to_name.items()]

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
    document_limit_news = 5
    document_limit_filings = 3

    print(f"1/5 taxonomy ({llm_model})...")
    themes_tree_dict = generate_themes_tree_dict(main_theme, focus)
    leaves = get_most_granular_elements(themes_tree_dict[main_theme], "Summary")
    print(f"   leaves={len(leaves)}")

    print("2/5 news search (smart-batching, basket_filtered_entities=True)...")
    retriever = DataRetriever(
        company_ids=company_ids,
        id_to_name=id_to_name,
        document_limit=document_limit_news,
        sortby="relevance",
        search_freq="M",
        start_date_query=start_date,
        end_date_query=end_date,
    )
    df_news = retriever.retrieve(
        themes_tree_dict=themes_tree_dict,
        list_specific_themes=[main_theme],
        document_type="news",
    )
    if df_news is None or df_news.empty:
        print("FAIL: no news chunks retrieved")
        return 1
    df_news = df_news.head(document_limit_news)
    print(f"   chunks={len(df_news)}")

    print("3/5 labeling...")
    label_processor = LabelProcessor(
        list_entities=list_entities,
        themes_tree_dict=themes_tree_dict,
        list_specific_themes=[main_theme],
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    df_labeled = label_processor.run_label_process(df_news)
    if df_labeled is None or df_labeled.empty:
        print("FAIL: labeling returned no rows")
        return 1
    print(f"   labeled={len(df_labeled)}")

    print("4/5 report generation...")
    report_generator = GenerateReport(
        universe_df=universe_df,
        main_theme=main_theme,
        focus=focus,
        llm_model=llm_model,
        api_key=os.getenv("OPENAI_API_KEY"),
        start_date=start_date,
        end_date=end_date,
        search_frequency="M",
        document_limit_news=document_limit_news,
        document_limit_filings=document_limit_filings,
        batch_size=1,
        themes_tree_dict=themes_tree_dict,
    )
    report = report_generator.generate_report(
        df_labeled=df_labeled,
        news_search_fallback=False,
    )
    print(f"   theme_rows={len(report.report_by_theme) if report.report_by_theme is not None else 0} "
          f"company_rows={len(report.report_by_company)}")

    print("5/5 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
