"""Simple theme tree generator (replaces research-tools mindmap)."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from src.openai_utils import sampling_params_for_model


class ThemeTree:
    """Simple theme tree structure."""

    def __init__(self, tree_dict: dict[str, Any]) -> None:
        self.tree = tree_dict

    def get_terminal_label_summaries(self) -> dict[str, str]:
        """Extract leaf node labels -> summaries."""
        results: dict[str, str] = {}

        def traverse(node: dict[str, Any]) -> None:
            if not node.get("Children"):
                label = node.get("Label", "")
                summary = node.get("Summary", "")
                if label:
                    results[label] = summary
            else:
                for child in node["Children"]:
                    traverse(child)

        traverse(self.tree)
        return results


def generate_theme_tree(
    main_theme: str,
    focus: str,
    model: str = "gpt-5.6-luna",
) -> ThemeTree:
    """Generate a simple theme taxonomy using OpenAI (replaces research-tools).

    Args:
        main_theme: The general theme (e.g., "Regulatory Issues")
        focus: Specific focus area (e.g., "Data Privacy", "Antitrust")
        model: OpenAI model id for taxonomy generation

    Returns:
        ThemeTree with structured hierarchy
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = (
        f"Generate a hierarchical theme taxonomy for '{main_theme}' focused on '{focus}'.\n\n"
        f"Return ONLY a JSON tree (no markdown code fences, no commentary) with this structure:\n"
        f'{{"Label": "root label", "Summary": "description", "Children": [...]}}\n\n'
        f"Every node (including leaves) must have non-empty 'Label' and 'Summary' fields. "
        f"Each leaf 'Summary' should be a full sentence that explicitly connects the sub-theme "
        f"back to '{main_theme}' so it can be used verbatim as a document-voice search query.\n"
        f"Keep it 2-3 levels deep with 3-5 leaf nodes."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            **sampling_params_for_model(model, temperature=0.3),
        )
        raw = response.choices[0].message.content.strip()
        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
        if fenced:
            raw = fenced.group(1).strip()
        tree_dict = json.loads(raw)
        if not tree_dict.get("Label") or not tree_dict.get("Summary"):
            raise ValueError("Theme tree missing required Label/Summary fields")
        return ThemeTree(tree_dict)
    except Exception:
        return ThemeTree(
            {
                "Label": f"{main_theme} in {focus}",
                "Summary": f"{main_theme} related to {focus}",
                "Children": [
                    {"Label": f"{focus}_aspect_1", "Summary": f"First aspect of {focus}"},
                    {"Label": f"{focus}_aspect_2", "Summary": f"Second aspect of {focus}"},
                ],
            }
        )
