"""Short live smoke test for Screener_for_Crypto REST migration."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bigdata_rest import BigdataRestClient, load_crypto_universe  # noqa: E402
from src.openai_utils import sampling_params_for_model  # noqa: E402
from src.search_entities import post_process_dataframe, search_by_entities  # noqa: E402

LLM_MODEL = "gpt-5.6-luna"
MAIN_THEME = "Crypto Institutional Adoption"
FOCUS = "KYC and AML themes"
ENTITY_IDS, ENTITY_NAMES = load_crypto_universe(ROOT / "data" / "top_15_cryptos.csv")
ENTITY_IDS = [ENTITY_IDS[0]]
ENTITY_NAMES = {ENTITY_IDS[0]: ENTITY_NAMES[ENTITY_IDS[0]]}
START_DATE = "2025-01-01"
END_DATE = "2025-01-31"
DOCUMENT_LIMIT = 5
N_THEMES = 2
MAX_ROWS_TO_LABEL = 3

def generate_themes(client: OpenAI, main_theme: str, focus: str, n_themes: int) -> list[str]:
    prompt = f"""Generate {n_themes} specific sub-themes for analyzing: "{main_theme}"

{"Focus: " + focus if focus else ""}

Return ONLY a JSON array of strings."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        **sampling_params_for_model(LLM_MODEL, temperature=0.3),
    )
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    themes = json.loads(text)
    if not isinstance(themes, list) or not themes:
        raise RuntimeError(f"Theme generation returned invalid payload: {themes!r}")
    return [str(t) for t in themes]


def label_text(client: OpenAI, text: str, entity_name: str, themes: list[str]) -> dict[str, str]:
    prompt = f"""Analyze this text about {entity_name} and pick the closest theme.

Themes: {", ".join(themes)}

Text: {text}

Return JSON: {{"label": "closest theme or 'unclear'", "motivation": "brief explanation"}}"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        **sampling_params_for_model(LLM_MODEL, temperature=0.0),
    )
    return json.loads(response.choices[0].message.content)


def main() -> None:
    load_dotenv(ROOT / ".env")
    if not os.getenv("BIGDATA_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("FAIL: missing BIGDATA_API_KEY or OPENAI_API_KEY")

    rest_client = BigdataRestClient()
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    themes = generate_themes(openai_client, MAIN_THEME, FOCUS, N_THEMES)
    print(f"themes={len(themes)}")

    df = search_by_entities(
        entity_ids=ENTITY_IDS,
        entity_names=ENTITY_NAMES,
        sentences=themes[:1],
        start_date=START_DATE,
        end_date=END_DATE,
        rest_client=rest_client,
        document_limit=DOCUMENT_LIMIT,
    )
    print(f"search_rows={len(df)}")
    if df.empty:
        raise SystemExit("FAIL: search returned zero rows")

    sample = df.head(MAX_ROWS_TO_LABEL)
    labeled = 0
    for _, row in sample.iterrows():
        result = label_text(openai_client, row["text"], row["entity_name"], themes)
        if result.get("label") and result["label"] != "unclear":
            labeled += 1
    print(f"labeled_non_unclear={labeled}")
    if labeled == 0:
        raise SystemExit("FAIL: no non-unclear labels")

    sample = sample.copy()
    sample["label"] = sample.apply(
        lambda r: label_text(openai_client, r["text"], r["entity_name"], themes)["label"],
        axis=1,
    )
    sample["motivation"] = ""
    processed = post_process_dataframe(sample)
    print(f"processed_rows={len(processed)}")
    print("PASS")


if __name__ == "__main__":
    main()
