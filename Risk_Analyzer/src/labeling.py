"""Minimal OpenAI-based risk classification helpers.

Replaces ``bigdata_research_tools.workflows.risk_analyzer.RiskAnalyzer``'s
taxonomy generation + semantic labeling steps with direct OpenAI calls,
following the pattern used across this repo's migrated cookbooks
(see ``MIGRATION_NOTES.md`` and ``Thematic_Screener_CLI``).
"""

from __future__ import annotations

import json
import os

import pandas as pd
from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"
NOT_RELEVANT = "Not Relevant"


def _client(client: OpenAI | None = None) -> OpenAI:
    return client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_risk_scenarios(
    main_theme: str,
    focus: str = "",
    n: int = 4,
    model: str = DEFAULT_MODEL,
    client: OpenAI | None = None,
) -> list[str]:
    """Generate ``n`` specific risk sub-scenarios for a main risk theme."""
    prompt = f"""Generate {n} specific risk sub-scenarios for: "{main_theme}"

{"Focus: " + focus if focus else ""}

Return ONLY a JSON array of {n} short risk scenario descriptions (each one sentence):
["Risk scenario 1", "Risk scenario 2", ...]
"""
    response = _client(client).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    text = (response.choices[0].message.content or "").strip()
    if text.startswith("```json"):
        text = text.split("```json")[1].split("```")[0].strip()
    elif text.startswith("```"):
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


def classify_risk_chunks(
    df_chunks: pd.DataFrame,
    risk_scenarios: list[str],
    main_theme: str,
    model: str = DEFAULT_MODEL,
    max_rows: int = 10,
    client: OpenAI | None = None,
) -> pd.DataFrame:
    """Classify each search-hit chunk against a list of risk sub-scenarios.

    Takes rows produced by ``src.search_helper.run_universe_search`` (must
    include ``chunk_text``/``text`` and ``entity_name``) plus the risk
    scenario taxonomy, and returns a copy of the (row-capped) DataFrame with
    two new columns:

    - ``risk_label``: the single best-matching sub-scenario, or
      ``"Not Relevant"`` if the passage does not support any of them.
    - ``risk_score``: an integer 0-3 exposure severity score (0 when
      ``risk_label`` is ``"Not Relevant"``).
    """
    oai = _client(client)
    rows = df_chunks.head(max_rows).reset_index(drop=True).copy()

    labels: list[str] = []
    scores: list[int] = []
    for _, row in rows.iterrows():
        text = row.get("chunk_text") or row.get("text") or ""
        company_name = row.get("entity_name", "")
        prompt = f"""You are a risk analyst. Decide whether the passage below discusses this \
company's exposure to one of the listed risk sub-scenarios for the main risk theme.

Main theme: {main_theme}

Risk sub-scenarios (pick the single best match, or "{NOT_RELEVANT}" if none apply):
{json.dumps(risk_scenarios)}

Company: {company_name}
Passage: \"\"\"{text}\"\"\"

Return ONLY a JSON object like:
{{"label": "<one sub-scenario from the list above, or '{NOT_RELEVANT}'>", "score": <integer 0-3, 0 if "{NOT_RELEVANT}">}}
"""
        try:
            response = oai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            label = str(payload.get("label", NOT_RELEVANT))
            score = int(payload.get("score", 0))
            if label not in risk_scenarios:
                label, score = NOT_RELEVANT, 0
        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            label, score = NOT_RELEVANT, 0

        labels.append(label)
        scores.append(score)

    rows["risk_label"] = labels
    rows["risk_score"] = scores
    return rows


def build_company_risk_matrix(
    df_labeled: pd.DataFrame,
    risk_scenarios: list[str],
    industry_by_id: dict[str, str] | None = None,
    sector_by_id: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate labeled chunks into a company x sub-scenario exposure matrix.

    Produces a DataFrame shaped for ``src.tool.display_figures``:
    ``[RP_ENTITY_ID, Sector, Company, Industry, <risk sub-scenarios...>, Composite Score]``.
    """
    industry_by_id = industry_by_id or {}
    sector_by_id = sector_by_id or {}

    relevant = df_labeled[df_labeled["risk_label"] != NOT_RELEVANT]
    pivot = pd.pivot_table(
        relevant,
        index="entity_id",
        columns="risk_label",
        values="risk_score",
        aggfunc="sum",
        fill_value=0,
    )
    # Ensure every sub-scenario has a column, even with zero exposure.
    for scenario in risk_scenarios:
        if scenario not in pivot.columns:
            pivot[scenario] = 0
    pivot = pivot[risk_scenarios]

    name_by_id = (
        df_labeled.drop_duplicates("entity_id").set_index("entity_id")["entity_name"].to_dict()
    )

    df_company = pivot.reset_index().rename(columns={"entity_id": "RP_ENTITY_ID"})
    df_company["Company"] = df_company["RP_ENTITY_ID"].map(name_by_id)
    df_company["Industry"] = df_company["RP_ENTITY_ID"].map(industry_by_id).fillna("Diversified")
    df_company["Sector"] = df_company["RP_ENTITY_ID"].map(sector_by_id).fillna("Unclassified")
    df_company["Composite Score"] = df_company[risk_scenarios].sum(axis=1)

    df_company = df_company[
        ["RP_ENTITY_ID", "Sector", "Company", "Industry", *risk_scenarios, "Composite Score"]
    ]
    return df_company.sort_values("Composite Score", ascending=False).reset_index(drop=True)
