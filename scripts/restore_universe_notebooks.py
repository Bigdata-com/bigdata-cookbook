#!/usr/bin/env python3
"""Restore sovereign and crypto entity universes in migrated notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def set_cell_source(cells: list[dict], index: int, source: str) -> None:
    lines = source.split("\n")
    cells[index]["source"] = [line + "\n" for line in lines[:-1]] + (
        [lines[-1]] if lines[-1] else []
    )
    if lines and lines[-1] == "":
        cells[index]["source"] = [line + "\n" for line in lines[:-1]]
    if cells[index]["cell_type"] == "code":
        cells[index]["outputs"] = []
        cells[index]["execution_count"] = None
    else:
        cells[index].pop("outputs", None)
        cells[index].pop("execution_count", None)


def patch_rising_bond(nb_path: Path) -> None:
    with nb_path.open(encoding="utf-8") as handle:
        nb = json.load(handle)

    cells = nb["cells"]

    set_cell_source(
        cells,
        18,
        """import datetime

import pandas as pd

from src.bigdata_rest import load_sovereign_universe
from src.search_entities import search_by_entities, post_process_dataframe
from src.entity_risk_prompt_labeler import label_search_results, get_scored_df
from src.sentiment_analysis import create_full_grid_indicators
from src.narrative_dashboard import get_narrative_windows, summarize_narratives, display_dashboard
from src.report_generator import SummaryGenerator
from src.visualization_tool import display_risk_figures_cookbooks

print("✅ Modules imported")
""",
    )

    set_cell_source(
        cells,
        19,
        """## Defining Your Risk Analysis Parameters

To perform a sovereign risk analysis, we need to define several key parameters:

- **Main Theme** (`main_theme`): The risk scenario to analyze (e.g. Spillovers from Rising Bond Spreads in Western Europe)
- **Focus** (`analyst_focus`): The analyst focus that provides an expert perspective on the scenario and helps break it down into risks
- **Entity Universe** (`dict_country_bank`, `entities_countries`, `entities_banks`): Western European countries plus their central banks. Countries are the primary entities scored in the analysis; central banks are searched separately to maximize coverage, then mapped back to their country.
- **Time Period** (`start_date` and `end_date`): The date range over which to run the search
- **Document Type / Scope** (`document_type`): Which document categories to search over (`news`, `filings`, `transcripts`, `all`) -- passed to `bigdata-smart-batching`'s `category` filter
- **Model Selection** (`llm_model`): The LLM model used to label the search result chunks
- Removed/not reimplemented after the SDK migration: `sources`, `fiscal_year`, `rerank_threshold`, and control-entity co-mention filtering are not exposed by the REST search helpers in `src/`.
""",
    )

    set_cell_source(
        cells,
        20,
        """# ===== Theme Definition =====
main_theme = "Spillovers from Rising Bond Spreads in Western Europe"
analyst_focus = (
    "Generate a mind map of current and future risks that western european countries "
    "can face as a result of rising bond spreads."
)

# ===== Entity Universe (Western Europe countries + central banks) =====
universe_path = "data/western_europe_countries_banks.csv"
countries_df, banks_df, dict_country_bank, id_to_country = load_sovereign_universe(universe_path)

entities_countries = countries_df["RP_ENTITY_ID"].tolist()
entities_banks = banks_df["RP_ENTITY_ID"].tolist()
id_to_name_countries = dict(zip(countries_df["RP_ENTITY_ID"], countries_df["ENTITY_NAME"]))
id_to_name_banks = dict(zip(banks_df["RP_ENTITY_ID"], banks_df["ENTITY_NAME"]))

control_entities = None  # control-entity co-mention filtering not exposed by the REST search helper

# ===== LLM Specification =====
llm_model = "gpt-5.6-luna"

# ===== Docs Configuration =====
document_type = "news"
fiscal_year = None

# ===== Specify Time Range =====
start_date = "2025-07-01"
end_date = "2025-10-03"

print(f"Countries: {list(id_to_name_countries.values())}")
print(f"Central banks: {list(id_to_name_banks.values())}")
print(f"Date range: {start_date} to {end_date}")
""",
    )

    set_cell_source(
        cells,
        21,
        """## Define a Risk Taxonomy

The original notebook used `bigdata-research-tools`' LLM-powered `RiskAnalyzer.generate_risk_tree` to mind-map the theme into a taxonomy tree. That functionality was removed with the SDK and is **not reimplemented** here -- `src/entity_risk_prompt_labeler.generate_risk_tree` intentionally still raises `NotImplementedError`. Instead we define a small, predefined taxonomy of sovereign bond-spillover sub-themes aligned with the Western European country universe.
""",
    )

    set_cell_source(
        cells,
        22,
        """# generate_risk_tree() (bigdata-research-tools) is removed and NOT reimplemented
# (src/entity_risk_prompt_labeler.generate_risk_tree raises NotImplementedError
# by design -- calling it would stop the notebook here). We use a small
# predefined taxonomy instead of an LLM-generated mind map.

taxonomy = {
    "sovereign_refinancing_risk": (
        "Rising sovereign bond spreads increase refinancing costs and rollover risk "
        "for Western European government debt."
    ),
    "fiscal_sustainability_pressure": (
        "Widening sovereign spreads signal mounting fiscal sustainability pressures "
        "across Western European countries."
    ),
    "financial_contagion_spillover": (
        "Cross-border financial contagion from rising bond spreads threatens "
        "macroeconomic stability in Western Europe."
    ),
}
for label, summary in taxonomy.items():
    print(f"- {label}: {summary}")
""",
    )

    set_cell_source(
        cells,
        26,
        """## Retrieve Content using Bigdata’s Search Capabilities

With the taxonomy and screening parameters defined, we use `bigdata-smart-batching` (via `src/search_helper.run_universe_search`) to plan and execute the search. Two parameters control cost:

- **Chunk Percentage** (`chunk_percentage`): The fraction of the estimated relevant chunks to actually retrieve (kept at 1-2% for cost control).
- **Requests Per Minute** (`requests_per_minute`): Rate limit for the underlying search execution.

The original notebook ran two searches: one over Western European countries and one over their central banks. Bank hits are mapped back to the parent country before scoring.
""",
    )

    set_cell_source(
        cells,
        28,
        """Retrieve news content for Western European countries, screened against the bond-spillover taxonomy defined above.
""",
    )

    set_cell_source(
        cells,
        29,
        """df_sentences_countries = search_by_entities(
    entities=entities_countries,
    sentences=node_summaries,
    start_date=start_date,
    end_date=end_date,
    id_to_name=id_to_name_countries,
    scope=document_type,
    chunk_percentage=chunk_percentage,
    requests_per_minute=requests_per_minute,
)
df_sentences_countries = df_sentences_countries.rename(columns={"sentiment": "bigdata_sentiment"})
df_sentences_countries = df_sentences_countries.loc[df_sentences_countries["entity_id"] != ""].copy()
df_sentences_countries["Country"] = df_sentences_countries["entity_id"].map(id_to_country)
print(f"Retrieved {len(df_sentences_countries)} country-attributed chunks")
df_sentences_countries.head()
""",
    )

    set_cell_source(
        cells,
        30,
        """Retrieve news content for the central banks paired with each country. Results are mapped back to the parent country before merging with the country search.
""",
    )

    set_cell_source(
        cells,
        31,
        """df_sentences_banks = search_by_entities(
    entities=entities_banks,
    sentences=node_summaries,
    start_date=start_date,
    end_date=end_date,
    id_to_name=id_to_name_banks,
    scope=document_type,
    chunk_percentage=chunk_percentage,
    requests_per_minute=requests_per_minute,
)
df_sentences_banks = df_sentences_banks.rename(columns={"sentiment": "bigdata_sentiment"})
df_sentences_banks = df_sentences_banks.loc[df_sentences_banks["entity_id"] != ""].copy()

reverse_dict = {bank_name: country_name for country_name, bank_name in dict_country_bank.items()}
df_sentences_banks_mapped = df_sentences_banks.copy()
df_sentences_banks_mapped["entity_name"] = df_sentences_banks_mapped["entity_name"].map(reverse_dict)
df_sentences_banks_mapped["Country"] = df_sentences_banks_mapped["entity_name"]
print(f"Retrieved {len(df_sentences_banks_mapped)} central-bank-attributed chunks")
df_sentences_banks_mapped.head()
""",
    )

    set_cell_source(
        cells,
        32,
        """Combine country and central-bank searches, deduplicate overlapping evidence, and cap the number of rows sent to the OpenAI labeler for cost control.
""",
    )

    set_cell_source(
        cells,
        33,
        """df_combined = pd.concat(
    [df_sentences_banks_mapped, df_sentences_countries],
    ignore_index=True,
)
df_combined = df_combined.drop_duplicates(
    subset=["entity_name", "document_id", "chunk_text"],
    keep="first",
)

MAX_LABELING_ROWS = 10_000
if len(df_combined) > MAX_LABELING_ROWS:
    df_combined = (
        df_combined.sort_values("timestamp", ascending=False)
        .head(MAX_LABELING_ROWS)
        .reset_index(drop=True)
    )

print(f"{len(df_combined)} rows queued for labeling (cost control cap: {MAX_LABELING_ROWS})")
""",
    )

    with nb_path.open("w", encoding="utf-8") as handle:
        json.dump(nb, handle, indent=1, ensure_ascii=False)
        handle.write("\n")

    print(f"Patched {nb_path}")


def patch_crypto(nb_path: Path) -> None:
    with nb_path.open(encoding="utf-8") as handle:
        nb = json.load(handle)

    cells = nb["cells"]

    set_cell_source(
        cells,
        17,
        """ ## Defining your Screening Parameters



 -  **Main Theme** (``main_theme``): The central concept to explore

 -  **Entity Universe** (``entity_ids`` / ``entity_names``): The set of cryptocurrencies to screen. The original notebook used the Bigdata.com **Top 15 Cryptos** watchlist; here we load the same 15 token entities from ``data/top_15_cryptos.csv`` (resolved once via the REST knowledge graph).

 -  **Time Period** (``start_date`` and ``end_date``): The date range over which to run the search

 -  **Document Limit** (``document_limit``): The maximum number of documents to return per entity/theme query to the Bigdata API.

 -  **Model Selection** (``llm_model``): The OpenAI model used to generate the theme taxonomy and label the search result chunks.
""",
    )

    set_cell_source(
        cells,
        18,
        """# ===== Theme Definition =====
main_theme = "Crypto Institutional Adoption"
focus = "Include know your customer (KYC) and anti-money laundering (AML) themes"

# ===== Entity Universe (Top 15 Cryptos) =====
universe_path = "data/top_15_cryptos.csv"
entity_ids, entity_names = load_crypto_universe(universe_path)

# ===== LLM Specification =====
llm_model = "gpt-5.6-luna"

# ===== Specify Time Range =====
start_date = "2025-01-01"
end_date = "2025-09-08"

print(f"Cryptos ({len(entity_ids)}): {list(entity_names.values())}")
""",
    )

    # Ensure import exists in setup cell
    setup_source = "".join(cells[10]["source"])
    if "load_crypto_universe" not in setup_source:
        setup_source = setup_source.replace(
            "from src.bigdata_rest import BigdataRestClient",
            "from src.bigdata_rest import BigdataRestClient, load_crypto_universe",
        )
        set_cell_source(cells, 10, setup_source.rstrip("\n"))

    with nb_path.open("w", encoding="utf-8") as handle:
        json.dump(nb, handle, indent=1, ensure_ascii=False)
        handle.write("\n")

    print(f"Patched {nb_path}")


def main() -> None:
    patch_rising_bond(ROOT / "Rising_Bond_Spread_Risks" / "Rising_Bond_Spread_Risks.ipynb")
    patch_crypto(ROOT / "Screener_for_Crypto" / "Screener_for_Crypto.ipynb")


if __name__ == "__main__":
    main()
