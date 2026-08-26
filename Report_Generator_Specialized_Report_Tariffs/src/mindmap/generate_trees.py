"""Simple theme tree generator (replaces research-tools mindmap)."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from src.openai_compat import DEFAULT_LLM_MODEL, sampling_params_for_model


class ThemeTree:
    """Simple theme tree structure."""

    def __init__(self, tree_dict: dict[str, Any]) -> None:
        self.tree = tree_dict

    def get_terminal_label_summaries(self) -> dict[str, str]:
        """Extract leaf node labels -> summaries."""
        results = {}

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


def generate_themes(
    main_theme: str,
    focus: str,
    model: str = DEFAULT_LLM_MODEL,
) -> dict[str, Any]:
    """Generate a simple theme taxonomy using OpenAI (replaces research-tools).

    Args:
        main_theme: The general theme (e.g., "Trade Policy")
        focus: Specific focus area (e.g., "Tariffs", "Export Controls")

    Returns:
        Dict with structured hierarchy
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = (
        f"Generate a hierarchical theme taxonomy for '{main_theme}' focused on '{focus}'.\n\n"
        f"Return a JSON tree with this structure:\n"
        f'{{"Label": "root label", "Summary": "description", "Children": [...]}}\n\n'
        f"Keep it small and cheap to search: 2 levels deep with at most 3 leaf nodes total."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            **sampling_params_for_model(model, temperature=0.3),
        )
        import json

        tree_dict = json.loads(response.choices[0].message.content.strip())
        return tree_dict
    except Exception:
        # Fallback to simple tree
        return {
            "Label": f"{main_theme} in {focus}",
            "Summary": f"{main_theme} related to {focus}",
            "Children": [
                {"Label": f"{focus}_aspect_1", "Summary": f"First aspect of {focus}"},
                {"Label": f"{focus}_aspect_2", "Summary": f"Second aspect of {focus}"},
            ],
        }


def get_most_granular_elements(tree_dict: dict[str, Any], key: str) -> list[Any]:
    """Return ``key`` (e.g. 'Summary' or 'Label') from every leaf node of a theme tree.

    Used by DataRetriever (to build the search queries from leaf 'Summary' text)
    and TopicSummarizerSector (to enumerate leaf 'Label' topics).
    """
    results: list[Any] = []

    def _walk(node: dict[str, Any]) -> None:
        children = node.get("Children") or []
        if not children:
            value = node.get(key)
            if value:
                results.append(value)
        else:
            for child in children:
                _walk(child)

    _walk(tree_dict)
    return results


def generate_themes_tree_dict(main_theme: str, focus: str = "") -> dict[str, Any]:
    """Generate a theme taxonomy and key it by ``main_theme``.

    Downstream consumers (LabelProcessor, DataRetriever, TopicSummarizerSector)
    look up ``themes_tree_dict[main_theme]``, so the dict key must match the
    exact ``main_theme`` string used to construct GenerateReport, regardless of
    the label the LLM assigns to the generated tree's root node.
    """
    tree = generate_themes(main_theme, focus)
    return {main_theme: tree}
