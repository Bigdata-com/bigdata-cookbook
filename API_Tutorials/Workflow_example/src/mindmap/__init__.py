"""Mindmap generation using OpenAI directly (SDK mindmap generators removed).

MIGRATION NOTE:
The original SDK-based tree generators (`generate_mindmap`, `generate_tree`,
`MindMapGenerator`) have been removed along with `bigdata-research-tools`.
`generate_theme_tree` / `generate_risk_tree` below are minimal OpenAI-based
replacements that return a small `MindMap` tree object with the same shape
the notebooks expect: `.get_terminal_label_summaries()` plus `.node`,
`.label`, `.summary`, `.children` for recursive traversal/plotting.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class MindMap:
    """A single node of a thematic decomposition tree."""

    node: str
    label: str
    summary: str
    children: list["MindMap"] = field(default_factory=list)

    def get_terminal_label_summaries(self) -> dict[str, str]:
        """Return {label: summary} for every leaf (terminal) node."""
        summaries: dict[str, str] = {}

        def _walk(node: "MindMap") -> None:
            if not node.children:
                summaries[node.label] = node.summary
            else:
                for child in node.children:
                    _walk(child)

        _walk(self)
        return summaries


def _parse_tree(data: dict, node_id: str = "0") -> MindMap:
    children_data = data.get("children") or []
    children = [
        _parse_tree(child, node_id=f"{node_id}.{i}") for i, child in enumerate(children_data)
    ]
    return MindMap(
        node=node_id,
        label=str(data.get("label", "")).strip(),
        summary=str(data.get("summary", "")).strip(),
        children=children,
    )


def _generate_tree(
    main_theme: str,
    analyst_focus: str,
    model: str,
    max_children: int,
    max_depth: int,
) -> MindMap:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for LLM-based tree generation")
    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are a financial research analyst. Decompose the given main theme into a "
        f"small hierarchical taxonomy: at most {max_depth} levels deep and at most "
        f"{max_children} children per node. {analyst_focus}\n"
        "Respond ONLY with a JSON object of the shape: "
        '{"label": <short label>, "summary": <one sentence>, '
        '"children": [ <same shape>, ... ]}. '
        "Leaf nodes (empty children list) must have a specific, news-searchable summary "
        "sentence describing a concrete, observable sub-theme."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Main theme: {main_theme}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    data = json.loads(response.choices[0].message.content)
    return _parse_tree(data)


def generate_theme_tree(
    main_theme: str,
    analyst_focus: str = "",
    model: str = DEFAULT_MODEL,
    max_children: int = 3,
    max_depth: int = 2,
) -> MindMap:
    """General thematic decomposition tree (OpenAI-based SDK mindmap replacement)."""
    return _generate_tree(main_theme, analyst_focus, model, max_children, max_depth)


def generate_risk_tree(
    main_theme: str,
    model: str = DEFAULT_MODEL,
    max_children: int = 3,
    max_depth: int = 2,
) -> MindMap:
    """Risk-oriented decomposition tree (OpenAI-based SDK mindmap replacement)."""
    focus = (
        "Frame every sub-theme as a specific, concrete risk or driver that could "
        "affect a company's performance."
    )
    return _generate_tree(main_theme, focus, model, max_children, max_depth)
