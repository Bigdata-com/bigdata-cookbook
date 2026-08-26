"""Election labeling using OpenAI (SDK removed)."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from openai import OpenAI


def sampling_params_for_model(model: str, *, temperature: float) -> dict[str, Any]:
    """Return OpenAI sampling kwargs accepted by ``model``.

    ``gpt-5.6-luna`` only supports default sampling, so temperature is omitted.
    """
    if "luna" in model.lower():
        return {}
    return {"temperature": temperature}


def replace_company_placeholders(row: pd.Series) -> str:
    """Replace TARGET_ENTITY and OTHER_ENTITY_N placeholders."""
    text = row["motivation"]
    if "entity_name" in row:
        text = text.replace("TARGET_ENTITY", row["entity_name"])
    if row.get("other_entities_map"):
        for entity_id, entity_name in row["other_entities_map"]:
            text = text.replace(f"OTHER_ENTITY_{entity_id}", entity_name)
    return text


class ElectionLabeler:
    """Screener labeler."""

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        label_prompt: Optional[str] = None,
        unknown_label: str = "unclear",
        temperature: float = 0,
        api_key: Optional[str] = None,
    ):
        """Initialize with OpenAI client."""
        self.llm_model = llm_model
        self.label_prompt = label_prompt
        self.unknown_label = unknown_label
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def get_labels(
        self,
        texts: list[str],
        max_workers: int = 50,
    ) -> pd.DataFrame:
        """Process labels for texts using OpenAI."""
        import concurrent.futures
        
        system_prompt = self.label_prompt or DEFAULT_TRUMP_REELECTION_PROMPT
        
        def label_single(idx_text: tuple[int, str]) -> dict[str, Any]:
            idx, text = idx_text
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"sentence_id: {idx}\ntext: {text}"}
                    ],
                    response_format={"type": "json_object"},
                    **sampling_params_for_model(self.llm_model, temperature=self.temperature),
                )
                content = response.choices[0].message.content
                parsed = json.loads(content)
                # Extract first key
                for sentence_id, data in parsed.items():
                    return {
                        "index": idx,
                        "motivation": data.get("motivation", ""),
                        "label": data.get("label", self.unknown_label),
                    }
            except Exception:
                return {"index": idx, "motivation": "", "label": self.unknown_label}
            return {"index": idx, "motivation": "", "label": self.unknown_label}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(label_single, enumerate(texts)))
        
        return pd.DataFrame(results).set_index("index")


DEFAULT_TRUMP_REELECTION_PROMPT = (
    f"""You are a financial analyst identifying companies expressing a view on the possible election of Donald Trump as President of the United States.
Your task is to determine if, based on the provided sentence, the Target Company is expressing a positive or negative view on Trump's election.

Instructions:
1. Label each sentence for the Target Company: 'P' if the company mention that Donald Trump's election will positively affect their business, 'N' if the company will be negatively impacted by Trump's re-election, 'U' if unrelated.
2. Evaluate each sentence individually, focusing on whether the company is mentioning Donald Trump.
3. Use only the information in the sentence; do not infer from outside knowledge.
4. Ensure the text expresses a clear view on the elections and the possible consequences. 
5. Mentioning that the elections are upcoming, or asking a question about the elections does not imply expressing a view.
6. A company that mentions how another company's business may be affected by the elections is not directly related to the elections unless its business is also affected.
7. You will be given a sentence ID, a company name, and the sentence text, for which you must assign the label. Your output should be a JSON object with a very brief motivation for the choice of the label and the label. The motivation must be one short sentence that starts with the company name and should explain why the label is 'Y', 'N', or 'U' for that company, without summarizing the text. Format the JSON like: {{"<ID>": {{"motivation": <motivation>, "label": <label>}}, ...}}.

Example sentences and evaluations:

Example 1:
Sentence: "Target Company executives mentioned that a Trump administration's tax policies would significantly boost our profit margins and allow for expanded operations."
Motivation: Target Company expects their business to benefit from Trump election.
Label: 'P'

Example 2:
Sentence: "Target Company's CEO warned that Trump's trade policies could disrupt our supply chain and increase costs substantially for our manufacturing operations."
Motivation: Target Company expects their business  to be negatively impacted by Trump's election.
Label: 'N'

Example 3:
Sentence: "Target Company analysts noted that the upcoming election between Trump and Biden could create market volatility."
Motivation: Target Company is providing general market commentary without mentioning specific business impact.
Label: 'U'

Example 4:
Sentence: "Target Company reported strong quarterly earnings despite concerns about the presidential election outcome."
Motivation: Target Company's earnings report does not express how Trump's election would affect their business.
Label: 'U'

Example 5:
Sentence: "On the Internet, we see news like the if Trump wins, the trade relations will change and the security situation will change. An of course, the relations with China and Taiwan comes into play. So the security is a big factor."
Motivation: Target_Company's discusses how Trump's election entails a security risk.
Label: 'N'

Example 6:
Sentence: "The former President Trump may win the upcoming presidential election. If that happens, what would be the strategy of Target Company or how will Target Company be prepared for the second Trump administration?"
Motivation: The sentence does not provide information about Target Company's view on the elections.
Label: 'U'
"""
)