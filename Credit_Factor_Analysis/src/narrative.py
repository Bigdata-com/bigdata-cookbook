"""Step 3 of the workflow: turn catalyst rows + news evidence into a narrative.

Keeps the prompt-construction logic out of the notebook. The LLM call uses
the OpenAI SDK against the ``gpt-5.6-terra`` model, which only supports the
default sampling ``temperature`` (no custom value), so none is passed.
"""

from __future__ import annotations

import pandas as pd
from openai import OpenAI

MODEL_NAME = "gpt-5.6-terra"

SYSTEM_PROMPT = (
    "You are a credit research analyst. You explain, in plain language, why a "
    "company's credit-news sentiment score moved and what a credit analyst "
    "should watch next. Ground every claim in the catalyst data and news "
    "evidence provided — never invent facts. Be concise and specific: cite "
    "concrete figures (recall counts, dates, dollar amounts) from the evidence "
    "when they are available."
)


def _format_catalysts(catalyst_df: pd.DataFrame) -> str:
    cols = [
        "catalyst_rank",
        "catalyst_direction",
        "credit_event_group",
        "credit_event_type",
        "credit_factor_score",
        "timestamp_utc",
    ]
    rows = catalyst_df[cols].to_dict(orient="records")
    lines = [
        f"- [{r['catalyst_direction']}] rank {r['catalyst_rank']} | "
        f"{r['credit_event_group']} / {r['credit_event_type']} | "
        f"score {r['credit_factor_score']:+.3f} | {r['timestamp_utc']}"
        for r in rows
    ]
    return "\n".join(lines)


def _format_evidence(evidence_by_catalyst: dict[str, list[dict]]) -> str:
    blocks = []
    for catalyst_type, docs in evidence_by_catalyst.items():
        blocks.append(f"### Evidence for catalyst: {catalyst_type}")
        for doc in docs:
            snippet = doc["chunks"][0]["text"] if doc.get("chunks") else ""
            blocks.append(
                f"- \"{doc['headline']}\" — {doc['source']['name']} "
                f"({doc['timestamp']})\n  {snippet[:400]}"
            )
    return "\n".join(blocks)


def build_narrative(
    entity_name: str,
    horizon: str,
    catalyst_df: pd.DataFrame,
    evidence_by_catalyst: dict[str, list[dict]],
    client: OpenAI | None = None,
) -> str:
    """Call gpt-5.6-terra to synthesize a credit narrative from catalyst + news data."""
    client = client or OpenAI()

    user_prompt = f"""Company: {entity_name}
Credit-news factor horizon: {horizon}

Catalyst rows (from bigdata_get_credit_factor, ranked most extreme first):
{_format_catalysts(catalyst_df)}

News evidence retrieved via bigdata_search for the top negative catalysts:
{_format_evidence(evidence_by_catalyst)}

Write a short credit narrative with three sections:
1. **What moved** — one paragraph on why the credit-news sentiment score is negative this period.
2. **Why it matters for credit** — one paragraph connecting the catalysts to credit risk (cash flow, liability, reputational, regulatory exposure).
3. **What to watch next** — 3-4 bullet points of concrete, monitorable follow-ups (dates, filings, financial disclosures)."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=900,
    )
    return response.choices[0].message.content
