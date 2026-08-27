#!/usr/bin/env python3
"""Restore pre-migration numeric/list defaults in migrated cookbook notebooks.

Keeps REST + CSV universe patterns; removes cost-control smoke trims so HTML
regeneration matches original demo scale as closely as possible.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_nb(path: Path) -> dict:
    return json.loads(path.read_text())


def save_nb(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def replace_in_nb(nb: dict, old: str, new: str) -> int:
    count = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if old not in src:
            continue
        src2 = src.replace(old, new)
        if src2 != src:
            cell["source"] = [line + "\n" for line in src2.split("\n")]
            if cell["source"] and cell["source"][-1] == "\n":
                cell["source"][-1] = ""
            # Prefer keeping trailing newline style of notebooks:
            if src.endswith("\n") and not "".join(cell["source"]).endswith("\n"):
                cell["source"] = [line + "\n" for line in src2.split("\n")[:-1]] + (
                    [src2.split("\n")[-1] + "\n"] if src2.split("\n")[-1] else []
                )
            # Simpler: write as single-string lines matching nbformat list-of-lines
            lines = src2.splitlines(keepends=True)
            if not lines:
                cell["source"] = []
            else:
                cell["source"] = lines
            count += 1
    return count


def set_cell_containing(nb: dict, needle: str, new_src: str) -> bool:
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if needle in src:
            lines = new_src.splitlines(keepends=True)
            if new_src and not new_src.endswith("\n"):
                lines = (new_src + "\n").splitlines(keepends=True)
            cell["source"] = lines
            return True
    return False


def git_show(rev_path: str) -> dict:
    raw = subprocess.check_output(["git", "show", rev_path], cwd=ROOT)
    return json.loads(raw)


def cell_with(nb: dict, needle: str) -> str:
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if needle in src:
            return src
    raise KeyError(needle)


def restore_ai_cost() -> None:
    path = ROOT / "AI_Cost_Cutting_Market_Analysis/AI_Cost_Cutting_Market_Analysis.ipynb"
    nb = load_nb(path)
    replace_in_nb(
        nb,
        'universe_df = load_universe(universe_path).head(5)  # small smoke-test universe to control API cost',
        "universe_df = load_universe(universe_path)  # full shared demo universe",
    )
    replace_in_nb(nb, 'start_date = "2025-01-01"', 'start_date = "2024-01-01"')
    replace_in_nb(nb, 'end_date = "2025-01-14"', 'end_date = "2024-06-30"')
    replace_in_nb(
        nb,
        "sentences=provider_sentences_list[:3],  # cost control: fewer narrative queries",
        "sentences=provider_sentences_list,",
    )
    replace_in_nb(
        nb,
        "sentences=user_sentences_list[:3],  # cost control: fewer narrative queries",
        "sentences=user_sentences_list,",
    )
    replace_in_nb(nb, "chunk_percentage=0.02,", "chunk_percentage=0.05,")
    # Labeling: use full retrieved set
    old_label = """# Cost control: cap the OpenAI labeling call to ~10 rows total, balanced across
# provider- and user-narrative search hits so the small sample can plausibly
# contain both roles.
if not df_all.empty:
    df_all_for_labeling = pd.concat([
        df_all[df_all['role_hint'] == 'provider'].head(5),
        df_all[df_all['role_hint'] == 'user'].head(5),
    ]).drop_duplicates(subset=['sentence_id']).reset_index(drop=True)
else:
    df_all_for_labeling = df_all"""
    new_label = """# Label the full retrieved set (pre-migration scale).
df_all_for_labeling = df_all.copy() if not df_all.empty else df_all"""
    replace_in_nb(nb, old_label, new_label)
    # Also fix trimmed comment on provider list if present
    replace_in_nb(
        nb,
        "provider_sentences_list = [  # trimmed to 3 for cost control",
        "provider_sentences_list = [",
    )
    save_nb(path, nb)
    print("restored AI_Cost_Cutting")


def restore_ai_revenue() -> None:
    path = ROOT / "AI_Revenue_Generation_Market_Analysis/AI_Revenue_Generation_Market_Analysis.ipynb"
    nb = load_nb(path)
    replace_in_nb(nb, "universe_df = load_universe(universe_path).head(5)", "universe_df = load_universe(universe_path)")
    replace_in_nb(nb, 'start_date = "2025-01-01"', 'start_date = "2024-01-01"')
    replace_in_nb(nb, 'end_date = "2025-01-14"', 'end_date = "2024-06-30"')
    replace_in_nb(nb, "chunk_percentage = 0.02", "chunk_percentage = 0.05")
    replace_in_nb(nb, "provider_sentences_list = provider_sentences_list[:4]", "provider_sentences_list = provider_sentences_list")
    replace_in_nb(nb, "user_sentences_list = user_sentences_list[:4]", "user_sentences_list = user_sentences_list")
    # labeling head caps
    src_all = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    # Common pattern: head(5) then head(10) for labeling
    for old, new in [
        (
            "df_all_for_labeling = pd.concat([\n        df_all[df_all['role_hint'] == 'provider'].head(5),\n        df_all[df_all['role_hint'] == 'user'].head(5),\n    ]).drop_duplicates(subset=['sentence_id']).reset_index(drop=True)",
            "df_all_for_labeling = df_all.copy()",
        ),
        ("df_all.head(10)", "df_all"),
        ("df_all_for_labeling = df_all.head(10)", "df_all_for_labeling = df_all.copy()"),
    ]:
        replace_in_nb(nb, old, new)
    save_nb(path, nb)
    print("restored AI_Revenue")


def restore_board() -> None:
    path = ROOT / "Board_Management_Monitoring/Board_Management_Monitoring.ipynb"
    nb = load_nb(path)
    pre = git_show("ae3f24b^:Board_Management_Monitoring/Board_Management_Monitoring.ipynb")
    pre_src = cell_with(pre, "management_themes = [")
    # Extract theme/period blocks from pre
    m = re.search(
        r"(management_themes = \[.*?\n\])\n\n(board_themes = \[.*?\n\])\n\n# Time Ranges\n(date_periods = \[.*?\n\])",
        pre_src,
        re.S,
    )
    if not m:
        raise RuntimeError("Could not parse pre-migration board themes/periods")
    management_themes, board_themes, date_periods = m.group(1), m.group(2), m.group(3)
    new_params = f"""# Themes (restored to pre-migration full lists)
{management_themes}

{board_themes}

# Time Ranges (restored to pre-migration full quarterly windows)
{date_periods}

# Company universe: CSV with RP_ENTITY_ID + COMPANY_NAME (no watchlists).
# Full shared demo universe (pre-migration searched person+company; REST uses CSV).
universe_path = "../Thematic_Screener_CLI/40_companies.csv"
universe_df = load_universe(universe_path)
company_ids = company_ids_from_universe(universe_df)
id_to_name = dict(zip(universe_df["RP_ENTITY_ID"], universe_df["COMPANY_NAME"]))

print(f"Universe ({{len(universe_df)}} companies):")
print(universe_df.to_string(index=False))
"""
    set_cell_containing(nb, "management_themes = [", new_params)

    new_search = '''# REST + smart-batching: run full theme list across all restored date periods.
# (SDK strict/relaxed/relaxed_post modes have no REST equivalent.)

theme_type_by_query = {q: "management" for q in management_themes}
theme_type_by_query.update({q: "board" for q in board_themes})
queries = management_themes + board_themes

frames = []
with timer("REST / smart-batching search execution (all periods)"):
    for start_date, end_date in date_periods:
        df_period = run_universe_search(
            company_ids=company_ids,
            queries=queries,
            start_date=start_date,
            end_date=end_date,
            scope="all",
            chunk_percentage=0.05,
            id_to_name=id_to_name,
        )
        if not df_period.empty:
            df_period = df_period.copy()
            df_period["period_start"] = start_date
            df_period["period_end"] = end_date
            frames.append(df_period)
        print(f"  {start_date} -> {end_date}: {0 if df_period is None else len(df_period)} rows")

df_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
print(
    f"Retrieved {len(df_raw)} chunk-level rows for {len(company_ids)} companies "
    f"over {len(date_periods)} periods"
)

if len(df_raw) > 0:
    df_processed = df_raw.copy()
    df_processed["Date"] = pd.to_datetime(df_processed["timestamp"], errors="coerce", utc=True)
    df_processed["SourceID"] = df_processed["source_id"]
    df_processed["Headline"] = df_processed["headline"]
    df_processed["QueryMeta"] = df_processed.apply(
        lambda r: f"{r['entity_name']} | {r['query']} ({theme_type_by_query.get(r['query'], 'management')})",
        axis=1,
    )
else:
    df_processed = pd.DataFrame(
        columns=["Date", "SourceID", "Headline", "QueryMeta", "document_id", "url"]
    )

search_modes = ["strict", "relaxed", "relaxed_post"]
results_files = {}
source_type = "all"

for search_mode in search_modes:
    output_file = os.path.join(
        output_dir, f"board_monitoring_{search_mode}_{source_type}_sources.csv"
    )
    df_processed.to_csv(output_file, index=False, encoding="utf-8-sig")
    results_files[search_mode] = output_file
    print(f"{search_mode.upper()} mode: {df_processed.shape[0]} documents saved to {output_file}")
'''
    set_cell_containing(nb, "theme_type_by_query", new_search)
    save_nb(path, nb)
    print("restored Board_Management")


def restore_credit() -> None:
    path = ROOT / "Credit_Ratings_Monitoring/Credit_Ratings_Monitoring.ipynb"
    nb = load_nb(path)
    replace_in_nb(
        nb,
        "start_date_query = (dt.date.today() - dt.timedelta(days=90)).isoformat()",
        "start_date_query = '2021-09-01'",
    )
    replace_in_nb(
        nb,
        "end_date_query = dt.date.today().isoformat()",
        "end_date_query = '2025-01-01'",
    )
    replace_in_nb(nb, "chunk_percentage = 0.3", "chunk_percentage = 0.05")
    save_nb(path, nb)
    print("restored Credit_Ratings")


def restore_daily_cb() -> None:
    path = ROOT / "Daily_Digest_Central_Banks/Daily_Digest_Central_Banks.ipynb"
    nb = load_nb(path)
    # dates already match pre in some versions; force them
    replace_in_nb(nb, "seeds=[123, 456])", "seeds=[123, 123456, 123456789, 456789, 789])")
    # restore full keyword search if trimmed
    for old, new in [
        (
            "search_keywords = [main_theme]\nif keywords:\n    search_keywords.append(keywords[0])\nsearch_keywords = list(dict.fromkeys(search_keywords))",
            "search_keywords = keywords if keywords else [main_theme]",
        ),
    ]:
        replace_in_nb(nb, old, new)
    # ensure dates
    replace_in_nb(nb, "start_query = '2025-08-18'", "start_query = '2025-08-18'")
    replace_in_nb(nb, "end_query = '2025-08-26'", "end_query = '2025-08-26'")
    save_nb(path, nb)
    print("restored Daily_Digest_Central_Banks")


def restore_daily_crude() -> None:
    path = ROOT / "Daily_Digest_Crude_Oil/Daily_Digest_Crude_Oil.ipynb"
    nb = load_nb(path)
    replace_in_nb(nb, "start_query = '2026-08-13'", "start_query = '2025-12-11'")
    replace_in_nb(nb, "end_query = '2026-08-13'", "end_query = '2025-12-18'")
    replace_in_nb(nb, "seeds=[123, 456])", "seeds=[123, 123456, 123456789, 456789, 789])")
    replace_in_nb(nb, "document_limit = 5   # Maximum number of chunks to retrieve per query to Bigdata API (cost control)", "document_limit = 10")
    replace_in_nb(nb, "document_limit = 5", "document_limit = 10")
    old = """# Cost control: 1-2 broad crude-oil queries on a single-day smoke window.
search_keywords = [main_theme]
if keywords:
    search_keywords.append(keywords[0])
search_keywords = list(dict.fromkeys(search_keywords))
print(f"Searching Bigdata for: {search_keywords}")"""
    new = """# Full lexicon keywords (pre-migration scale).
search_keywords = keywords if keywords else [main_theme]
print(f"Searching Bigdata for: {search_keywords}")"""
    replace_in_nb(nb, old, new)
    save_nb(path, nb)
    print("restored Daily_Digest_Crude_Oil")


def restore_election() -> None:
    path = ROOT / "Election_Monitor/Trump_Reelection_Impact_Analysis.ipynb"
    nb = load_nb(path)
    pre = git_show("39d1856^:Election_Monitor/Trump_Reelection_Impact_Analysis.ipynb")
    pre_sentences = cell_with(pre, "trump_positive_sentences_list = [")
    # Keep document_type from current (transcripts) or restore? Pre used DocumentType - current uses transcripts.
    # Keep transcripts as migrated default for REST scope string.
    new_sentences = pre_sentences
    # Strip SDK DocumentType if present at end and append current document_type
    if "document_type" not in new_sentences:
        new_sentences = new_sentences.rstrip() + '\n\n# Document Configuration\ndocument_type = "transcripts"\n'
    else:
        new_sentences = re.sub(
            r"document_type\s*=\s*.*",
            'document_type = "transcripts"',
            new_sentences,
        )
    set_cell_containing(nb, "trump_positive_sentences_list = [", new_sentences)

    replace_in_nb(
        nb,
        """universe = load_universe("../Thematic_Screener_CLI/40_companies.csv").head(5)
company_ids = company_ids_from_universe(universe)
id_to_name = dict(zip(universe["RP_ENTITY_ID"], universe["COMPANY_NAME"]))""",
        """universe = load_universe("../Thematic_Screener_CLI/40_companies.csv")
company_ids = company_ids_from_universe(universe)
id_to_name = dict(zip(universe["RP_ENTITY_ID"], universe["COMPANY_NAME"]))""",
    )
    replace_in_nb(nb, 'start_date = "2024-07-01"', 'start_date = "2024-01-01"')
    replace_in_nb(nb, 'end_date = "2024-11-15"', 'end_date = "2024-06-30"')
    replace_in_nb(nb, "chunk_percentage = 0.02", "chunk_percentage = 0.05")
    # remove labeling cap
    old_cap = """# Cost control: cap the number of chunks sent to the OpenAI labeler
MAX_ROWS_TO_LABEL = 30
if len(search_results) > MAX_ROWS_TO_LABEL:
    search_results = search_results.head(MAX_ROWS_TO_LABEL).reset_index(drop=True)

print(f"✅ Retrieved {len(search_results)} chunk-level rows across {len(company_ids)} companies")"""
    new_cap = """print(f"✅ Retrieved {len(search_results)} chunk-level rows across {len(company_ids)} companies")"""
    replace_in_nb(nb, old_cap, new_cap)
    save_nb(path, nb)
    print("restored Election_Monitor")


def restore_liquid() -> None:
    path = ROOT / "Liquid_Cooling_Market_Watch/Liquid_Cooling_Market_Watch.ipynb"
    nb = load_nb(path)
    replace_in_nb(
        nb,
        'universe_df = load_universe("../Thematic_Screener_CLI/40_companies.csv").head(5)',
        'universe_df = load_universe("../Thematic_Screener_CLI/40_companies.csv")',
    )
    replace_in_nb(nb, 'start_date = "2025-06-01"', 'start_date = "2021-01-01"')
    replace_in_nb(nb, 'end_date = "2025-06-14"', 'end_date = "2024-06-30"')
    replace_in_nb(nb, "provider_sentences_list = provider_sentences_list[:3]", "provider_sentences_list = provider_sentences_list")
    replace_in_nb(nb, "adopter_sentences_list = adopter_sentences_list[:3]", "adopter_sentences_list = adopter_sentences_list")
    replace_in_nb(nb, "chunk_percentage = 0.02", "chunk_percentage = 0.05")
    old_label = """# Cost control: cap the OpenAI labeling call to ~10 rows total, balanced across
# provider- and adopter-narrative search hits so the small sample can plausibly
# contain both roles.
if not df_all.empty:
    provider_mask = df_all.get('role_hint', pd.Series(dtype=str)) == 'provider'
    if 'role_hint' not in df_all.columns:
        # fallback if role_hint missing
        df_all_for_labeling = df_all.head(10).reset_index(drop=True)
    else:
        df_all_for_labeling = pd.concat([
            df_all[provider_mask].head(5),
            df_all[~provider_mask].head(5),
        ]).drop_duplicates().reset_index(drop=True)
else:
    df_all_for_labeling = df_all"""
    # Try several variants of labeling cap
    for old in [
        old_label,
        """# Cost control: cap the OpenAI labeling call to ~10 rows total, balanced across
# provider- and adopter-narrative search hits so the small sample can plausibly
# contain both roles.
if not df_all.empty:
    provider_mask = df_all['role_hint'] == 'provider' if 'role_hint' in df_all.columns else pd.Series([False]*len(df_all))
    df_all_for_labeling = pd.concat([
        df_all[provider_mask].head(5),
        df_all[~provider_mask].head(5),
    ]).drop_duplicates().reset_index(drop=True)
else:
    df_all_for_labeling = df_all""",
    ]:
        replace_in_nb(nb, old, "df_all_for_labeling = df_all.copy() if not df_all.empty else df_all")
    # Broader regex-like cleanup via cell scan
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "df_all[provider_mask].head(5)" in src or ("Cost control: cap the OpenAI labeling" in src and "df_all" in src):
            src2 = re.sub(
                r"# Cost control: cap the OpenAI labeling[\s\S]*?df_all_for_labeling = df_all",
                "df_all_for_labeling = df_all.copy() if not df_all.empty else df_all\nelse:\n    df_all_for_labeling = df_all",
                src,
                count=1,
            )
            # If that mangled, simpler approach:
            if "head(5)" in src and "df_all_for_labeling" in src:
                src2 = re.sub(
                    r"if not df_all\.empty:[\s\S]*?df_all_for_labeling = pd\.concat\(\[[\s\S]*?\]\)\.drop_duplicates[\s\S]*?else:\s*df_all_for_labeling = df_all",
                    "df_all_for_labeling = df_all.copy() if not df_all.empty else df_all",
                    src,
                    count=1,
                )
                cell["source"] = src2.splitlines(keepends=True)
    save_nb(path, nb)
    print("restored Liquid_Cooling")


def restore_narrative() -> None:
    path = ROOT / "Narrative_Miners/NarrativeMiner.ipynb"
    nb = load_nb(path)
    pre = git_show("bd1b385^:Narrative_Miners/NarrativeMiner.ipynb")
    pre_src = cell_with(pre, "main_narratives = [")
    m = re.search(r"main_narratives = \[.*?\]", pre_src, re.S)
    if not m:
        raise RuntimeError("narrative list missing")
    narratives = m.group(0)
    new_cfg = f"""# AI Bubble Narratives (used as REST search queries) — full pre-migration list
{narratives}

# OpenAI model used for narrative labeling
llm_model = "gpt-5.6-luna"

# Pre-migration time window (~1 year)
start_date = "2024-03-01"
end_date = "2025-03-28"

# Moderate retrieval (5% of matching chunks)
chunk_percentage = 0.05

# Label all retrieved rows via OpenAI (no smoke cap)
max_openai_labels = 10_000
"""
    set_cell_containing(nb, "main_narratives = [", new_cfg)
    save_nb(path, nb)
    print("restored Narrative_Miners")


def restore_pricing() -> None:
    path = ROOT / "Pricing_Power_Analysis/Pricing Power.ipynb"
    nb = load_nb(path)
    new_universe = '''# ===== Customizable Parameters =====

# Company Universe: full shared demo CSV (pre-migration Top US 100 watchlist).
# 40_companies.csv is the largest shared public demo universe in-repo.
company_universe = load_universe("../Thematic_Screener_CLI/40_companies.csv")
# Sector metadata is not in the CSV; default to Unknown for chart grouping.
if "SECTOR" not in company_universe.columns:
    company_universe = company_universe.copy()
    company_universe["SECTOR"] = "Unknown"
company_ids = company_ids_from_universe(company_universe)
id_to_name = dict(zip(company_universe["RP_ENTITY_ID"], company_universe["COMPANY_NAME"]))
id_to_sector = dict(zip(company_universe["RP_ENTITY_ID"], company_universe["SECTOR"]))

# LLM Specification (gpt-5.6-luna: omit temperature/top_p — see sampling_params_for_model)
llm_model = "gpt-5.6-luna"

# Search Frequency (used for weekly aggregation downstream)
search_frequency = "W"

# Specify Time Range — restored pre-migration window
start_date = "2024-01-01"
end_date = "2024-06-01"

# Search scope + full retrieval
search_scope = "news"
chunk_percentage = 0.05
requests_per_minute = 200
max_labeling_rows = 10_000  # label full retrieved set
'''
    set_cell_containing(nb, "Company Universe: small, explicit list", new_universe)
    # Also try if needle differs
    if "max_labeling_rows = 10" in json.dumps(nb):
        replace_in_nb(nb, "max_labeling_rows = 10  # cap OpenAI labeling calls per theme", "max_labeling_rows = 10_000")
        replace_in_nb(nb, "chunk_percentage = 0.02", "chunk_percentage = 0.05")
        replace_in_nb(nb, 'start_date = "2026-07-15"', 'start_date = "2024-01-01"')
        replace_in_nb(nb, 'end_date = "2026-08-01"', 'end_date = "2024-06-01"')
    # Restore min_companies if lowered
    replace_in_nb(
        nb,
        "plot_top_companies_by_sector(df_negative_relevant, min_companies=1, title_suffix=\"(Lack of Pricing Power)\", top_sectors=4)",
        "plot_top_companies_by_sector(df_negative_relevant, min_companies=3, title_suffix=\"(Lack of Pricing Power)\", top_sectors=4)",
    )
    replace_in_nb(
        nb,
        "plot_top_companies_by_sector(df_positive_relevant, min_companies=1, title_suffix=\"(Pricing Power)\", top_sectors=4)",
        "plot_top_companies_by_sector(df_positive_relevant, min_companies=3, title_suffix=\"(Pricing Power)\", top_sectors=4)",
    )
    save_nb(path, nb)
    print("restored Pricing_Power")


def restore_ai_threats() -> None:
    path = ROOT / "Report_Generator_AI_Threats/Report Generator_ AI Disruption Risk.ipynb"
    nb = load_nb(path)
    replace_in_nb(
        nb,
        "universe_df = load_universe(universe_path).head(5).reset_index(drop=True)  # small slice for cost control",
        "universe_df = load_universe(universe_path).reset_index(drop=True)",
    )
    replace_in_nb(nb, 'start_date = "2026-07-01"', 'start_date = "2025-01-01"')
    replace_in_nb(nb, 'end_date = "2026-07-21"', 'end_date = "2025-04-20"')
    replace_in_nb(nb, "document_limit_news = 1", "document_limit_news = 10")
    replace_in_nb(nb, "document_limit_filings = 0", "document_limit_filings = 5")
    replace_in_nb(nb, "document_limit_transcripts = 0", "document_limit_transcripts = 5")
    replace_in_nb(nb, "chunk_percentage = 0.02", "chunk_percentage = 0.05")
    replace_in_nb(nb, "max_rows_to_label = 10   # cost control: cap rows sent to OpenAI for labeling/summarization", "max_rows_to_label = 10_000")
    replace_in_nb(nb, "fiscal_year = 2026", "fiscal_year = 2025")
    # remove head caps on retrieved dfs
    replace_in_nb(
        nb,
        "df_sentences_semantic_risk = df_sentences_semantic_risk.head(max_rows_to_label).reset_index(drop=True)\ndf_sentences_semantic_proactivity = df_sentences_semantic_proactivity.head(max_rows_to_label).reset_index(drop=True)",
        "df_sentences_semantic_risk = df_sentences_semantic_risk.head(max_rows_to_label).reset_index(drop=True) if len(df_sentences_semantic_risk) > max_rows_to_label else df_sentences_semantic_risk\ndf_sentences_semantic_proactivity = df_sentences_semantic_proactivity.head(max_rows_to_label).reset_index(drop=True) if len(df_sentences_semantic_proactivity) > max_rows_to_label else df_sentences_semantic_proactivity",
    )
    save_nb(path, nb)
    print("restored Report_Generator_AI_Threats")


def restore_regulatory() -> None:
    path = ROOT / "Report_Generator_Regulatory_Issues_in_Tech/Report Generator_ Regulatory Issues.ipynb"
    nb = load_nb(path)
    replace_in_nb(nb, "universe_df = load_universe(universe_path).head(5)", "universe_df = load_universe(universe_path)")
    replace_in_nb(
        nb,
        'list_specific_focus = [\'AI\', \'Data Privacy\']',
        "list_specific_focus = ['AI', 'Social Media', 'Hardware and Chips', 'E-commerce', 'Advertising']",
    )
    replace_in_nb(
        nb,
        'start_date = (datetime.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")',
        'start_date = "2025-01-01"',
    )
    replace_in_nb(
        nb,
        'end_date = datetime.now().strftime("%Y-%m-%d")',
        'end_date = "2025-04-20"',
    )
    replace_in_nb(nb, "CHUNK_PERCENTAGE = 0.01", "CHUNK_PERCENTAGE = 0.05")
    replace_in_nb(nb, "MAX_ROWS_TO_LABEL = 10", "MAX_ROWS_TO_LABEL = 10_000")
    replace_in_nb(nb, "fiscal_year = 2026", "fiscal_year = 2025")
    save_nb(path, nb)
    print("restored Report_Generator_Regulatory")


def restore_tariffs() -> None:
    path = ROOT / "Report_Generator_Specialized_Report_Tariffs/Report_Generator_Specialized_Report_Tariffs.ipynb"
    nb = load_nb(path)
    replace_in_nb(
        nb,
        'universe_path = "../Thematic_Screener_CLI/40_companies.csv"',
        'universe_path = "../Thematic_Screener_CLI/mag7.csv"',
    )
    replace_in_nb(
        nb,
        "universe_df = load_universe(universe_path).head(5).reset_index(drop=True)",
        "universe_df = load_universe(universe_path).reset_index(drop=True)",
    )
    replace_in_nb(
        nb,
        'start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")',
        'start_date = "2025-02-01"',
    )
    replace_in_nb(
        nb,
        'end_date = datetime.now().strftime("%Y-%m-%d")',
        'end_date = "2025-08-13"',
    )
    # remove news head cap comment if present — keep document_limit_news as cap semantics
    replace_in_nb(
        nb,
        'df_sentences_semantic = df_sentences_semantic.head(document_limit_news)',
        'df_sentences_semantic = df_sentences_semantic  # full retrieved set for labeling',
    )
    save_nb(path, nb)
    print("restored Report_Generator_Tariffs")


def restore_rising_bond() -> None:
    path = ROOT / "Rising_Bond_Spread_Risks/Rising_Bond_Spread_Risks.ipynb"
    nb = load_nb(path)
    # Expand company universe + restore dates + raise labeling + chunk
    replace_in_nb(
        nb,
        'target_companies = ["NVIDIA Corp.", "Microsoft Corp.", "Amazon.com Inc."]',
        'target_companies = None  # use full CSV universe',
    )
    # Fix filter when target_companies is None
    replace_in_nb(
        nb,
        'universe_df = universe_df.loc[universe_df["COMPANY_NAME"].isin(target_companies)].reset_index(drop=True)',
        'universe_df = universe_df if target_companies is None else universe_df.loc[universe_df["COMPANY_NAME"].isin(target_companies)].reset_index(drop=True)',
    )
    replace_in_nb(nb, "datetime.timedelta(days=30)", "datetime.timedelta(days=95)")
    # Also fix if written as end - 30 for start
    replace_in_nb(nb, "chunk_percentage = 0.02", "chunk_percentage = 0.05")
    replace_in_nb(nb, "MAX_LABELING_ROWS = 10", "MAX_LABELING_ROWS = 10_000")
    # Prefer fixed pre-migration dates if present as rolling
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "end_date_dt = datetime.date.today()" in src and "start_date_dt" in src:
            src2 = src.replace(
                "end_date_dt = datetime.date.today()",
                'end_date_dt = datetime.date.fromisoformat("2025-10-03")',
            )
            src2 = re.sub(
                r"start_date_dt = end_date_dt - datetime\.timedelta\(days=\d+\)",
                'start_date_dt = datetime.date.fromisoformat("2025-07-01")',
                src2,
            )
            cell["source"] = src2.splitlines(keepends=True)
    save_nb(path, nb)
    print("restored Rising_Bond_Spread_Risks")


def restore_risk_analyzer() -> None:
    path = ROOT / "Risk_Analyzer/Risk_Analyzer.ipynb"
    nb = load_nb(path)
    replace_in_nb(
        nb,
        '''universe = load_universe("../Thematic_Screener_CLI/40_companies.csv")
# Small slice to keep this demo's API/LLM spend low.
COMPANY_SLICE = ["E09E2B", "D8442A", "228D42", "DD3BB1", "0157B1"]  # NVIDIA, Apple, Microsoft, Tesla, Amazon
universe = universe[universe["RP_ENTITY_ID"].isin(COMPANY_SLICE)].reset_index(drop=True)
company_ids = company_ids_from_universe(universe)
id_to_name = dict(zip(universe["RP_ENTITY_ID"], universe["COMPANY_NAME"]))

# Static sector/industry metadata for this small demo slice (a production run
# would enrich this via a knowledge-graph lookup instead of hardcoding it).
industry_by_id = {
    "E09E2B": "Semiconductors",
    "D8442A": "Consumer Electronics",
    "228D42": "Software",
    "DD3BB1": "Automobiles",
    "0157B1": "Internet Retail",
}
sector_by_id = {
    "E09E2B": "Technology",
    "D8442A": "Technology",
    "228D42": "Technology",
    "DD3BB1": "Consumer Discretionary",
    "0157B1": "Consumer Discretionary",
}''',
        '''universe = load_universe("../Thematic_Screener_CLI/40_companies.csv")
company_ids = company_ids_from_universe(universe)
id_to_name = dict(zip(universe["RP_ENTITY_ID"], universe["COMPANY_NAME"]))

# Firmographic defaults (CSV has no sector/industry columns)
industry_by_id = {eid: "Unknown" for eid in company_ids}
sector_by_id = {eid: "Unknown" for eid in company_ids}''',
    )
    replace_in_nb(nb, 'start_date = "2025-04-02"', 'start_date = "2025-04-01"')
    replace_in_nb(nb, 'end_date = "2025-04-21"', 'end_date = "2025-06-30"')
    replace_in_nb(nb, "chunk_percentage = 0.02", "chunk_percentage = 0.05")
    replace_in_nb(nb, "MAX_CLASSIFY_ROWS = 5  # cap OpenAI classification calls for cost control", "MAX_CLASSIFY_ROWS = 10_000")
    save_nb(path, nb)
    print("restored Risk_Analyzer")


def restore_crypto() -> None:
    path = ROOT / "Screener_for_Crypto/Screener_for_Crypto.ipynb"
    nb = load_nb(path)
    # Expand to a larger crypto-exposed public company set + restore dates
    new_entities = '''# ===== Theme Definition =====
main_theme = "Crypto Institutional Adoption"
focus = "Include know your customer (KYC) and anti-money laundering (AML) themes"

# ===== Entity Universe =====
# Pre-migration used a Top-15 crypto watchlist. REST covers public companies, so
# we use a verified larger set of crypto-exposed public companies.
entity_ids = [
    "D69946",  # Coinbase Global Inc.
    "C72B8F",  # Strategy Inc. (MicroStrategy)
    "56AAC4",  # Riot Platforms Inc.
    "E09E2B",  # NVIDIA Corp. (crypto mining GPUs / infra)
    "228D42",  # Microsoft Corp.
    "0157B1",  # Amazon.com Inc.
    "D8442A",  # Apple Inc.
    "4A6F00",  # Meta Platforms Inc. (if present in KG; keep if valid)
]
entity_names = {
    "D69946": "Coinbase Global Inc.",
    "C72B8F": "Strategy Inc. (MicroStrategy)",
    "56AAC4": "Riot Platforms Inc.",
    "E09E2B": "NVIDIA Corp.",
    "228D42": "Microsoft Corp.",
    "0157B1": "Amazon.com Inc.",
    "D8442A": "Apple Inc.",
    "4A6F00": "Meta Platforms Inc.",
}

# ===== LLM Specification =====
llm_model = "gpt-5.6-luna"

# ===== Specify Time Range (restored pre-migration window) =====
start_date = "2025-01-01"
end_date = "2025-09-08"
'''
    # Check if Meta ID is in 40_companies
    import csv
    ids = {r["RP_ENTITY_ID"]: r["COMPANY_NAME"] for r in csv.DictReader(open(ROOT / "Thematic_Screener_CLI/40_companies.csv"))}
    # Filter entity_ids to ones we know + coinbase set
    known = {
        "D69946": "Coinbase Global Inc.",
        "C72B8F": "Strategy Inc. (MicroStrategy)",
        "56AAC4": "Riot Platforms Inc.",
    }
    for eid, name in ids.items():
        lname = name.lower()
        if any(k in lname for k in ["nvidia", "microsoft", "amazon", "apple", "meta", "tesla", "alphabet", "google", "block", "paypal", "robinhood"]):
            known[eid] = name
    entity_ids = list(known.keys())
    entity_names = known
    new_entities = f'''# ===== Theme Definition =====
main_theme = "Crypto Institutional Adoption"
focus = "Include know your customer (KYC) and anti-money laundering (AML) themes"

# ===== Entity Universe =====
# Pre-migration used Top-15 crypto watchlist. REST uses crypto-exposed public cos.
entity_ids = {entity_ids!r}
entity_names = {entity_names!r}

# ===== LLM Specification =====
llm_model = "gpt-5.6-luna"

# ===== Specify Time Range (restored pre-migration window) =====
start_date = "2025-01-01"
end_date = "2025-09-08"
'''
    set_cell_containing(nb, "entity_ids = [", new_entities)
    replace_in_nb(nb, "document_limit = 15", "document_limit = 10")
    replace_in_nb(
        nb,
        """# Cost control: cap OpenAI labeling calls to a handful of retrieved rows
MAX_ROWS_TO_LABEL = 25
df_sentences = df_sentences.head(MAX_ROWS_TO_LABEL).reset_index(drop=True)""",
        """# Label full retrieved set
MAX_ROWS_TO_LABEL = 10_000
if len(df_sentences) > MAX_ROWS_TO_LABEL:
    df_sentences = df_sentences.head(MAX_ROWS_TO_LABEL).reset_index(drop=True)""",
    )
    save_nb(path, nb)
    print("restored Screener_for_Crypto")


def restore_inflation() -> None:
    path = ROOT / "Tracking_Inflation_Drivers/Tracking_Inflation_Drivers.ipynb"
    nb = load_nb(path)
    replace_in_nb(nb, 'start_date = "2026-07-14"', 'start_date = "2025-01-01"')
    replace_in_nb(nb, 'end_date = "2026-08-13"', 'end_date = "2025-02-28"')
    replace_in_nb(nb, "chunk_percentage = 0.02", "chunk_percentage = 0.05")
    # remove query / labeling trims
    replace_in_nb(nb, "sentences_query[:8]", "sentences_query")
    replace_in_nb(nb, "df_sentences.head(10)", "df_sentences")
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "sentences_query" in src and "head(" in src:
            src2 = src.replace("sentences_query[:8]", "sentences_query")
            cell["source"] = src2.splitlines(keepends=True)
        if "df_sentences.head(10)" in src:
            src2 = src.replace("df_sentences.head(10)", "df_sentences")
            cell["source"] = src2.splitlines(keepends=True)
    save_nb(path, nb)
    print("restored Tracking_Inflation_Drivers")


def main() -> None:
    restore_ai_cost()
    restore_ai_revenue()
    restore_board()
    restore_credit()
    restore_daily_cb()
    restore_daily_crude()
    restore_election()
    restore_liquid()
    restore_narrative()
    restore_pricing()
    restore_ai_threats()
    restore_regulatory()
    restore_tariffs()
    restore_rising_bond()
    restore_risk_analyzer()
    restore_crypto()
    restore_inflation()
    print("ALL DONE")


if __name__ == "__main__":
    main()
