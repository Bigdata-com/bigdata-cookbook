"""Short end-to-end smoke test for Report_Generator_Regulatory_Issues_in_Tech."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

load_dotenv(PROJECT_DIR / ".env")

from src.bigdata_rest import load_universe
from src.report_generator import GenerateReport


def main() -> int:
    if not os.getenv("BIGDATA_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        print("FAIL: missing BIGDATA_API_KEY or OPENAI_API_KEY")
        return 1

    universe_path = PROJECT_DIR.parent / "Thematic_Screener_CLI" / "40_companies.csv"
    universe_df = load_universe(universe_path).head(5).reset_index(drop=True)

    generator = GenerateReport(
        universe_df=universe_df,
        general_theme="Regulatory Issues",
        list_specific_focus=["AI"],
        llm_model="gpt-5.6-luna",
        api_key=os.environ["OPENAI_API_KEY"],
        start_date="2026-07-15",
        end_date="2026-07-21",
        fiscal_year=2026,
        search_frequency="M",
        document_limit_news=1,
        document_limit_filings=0,
        document_limit_transcripts=0,
        batch_size=1,
        chunk_percentage=0.02,
    )

    report = generator.generate_report()
    df = report.report_by_company
    print(f"PASS: report rows={len(df)} companies={df['entity_name'].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
