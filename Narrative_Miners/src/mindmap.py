"""LLM mindmap generation for Narrative Miners.

Expands a main theme into sub-themes with 2–3 search labels each, then flattens
leaves for REST search while keeping parent sub-theme mapping for rollups.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from openai import OpenAI
from pydantic import BaseModel, Field

from src.llm_helpers import DEFAULT_LLM_MODEL, sampling_params_for_model

BRANCH_COLORS: tuple[str, ...] = (
    "#12496b",
    "#4c9ac9",
    "#e8963c",
    "#6a994e",
    "#bc4749",
    "#7b6ba8",
)
TEXT_COLOR = "#243038"
GRID_COLOR = "#dce1e5"

SYSTEM_MESSAGE_NARRATIVE_MINDMAP = """
Forget all previous prompts.
You are assisting a professional analyst building a narrative-mining mind map.
The main market narrative theme is:
{main_theme}

Your objective is to decompose the theme into distinct narrative sub-themes that
analysts track across news, earnings transcripts, and filings.

Follow these rules strictly:

1. **Structure (exactly three levels)**
   - Root node: the main theme (`label` = "{main_theme}").
   - Mid-level nodes: 3–5 distinct narrative **sub-themes** (short titles, 2–5 words).
   - Leaf nodes: under each sub-theme, create **exactly 2 or 3** search labels.

2. **Sub-themes**
   - Each sub-theme is a coherent narrative bucket (e.g. valuation concerns, ROI
     skepticism, capital-expenditure intensity, regulatory overhang).
   - Sub-themes must be mutually distinct; do not overlap meanings.
   - Mid-level nodes use empty `search_query` ("").

3. **Search labels (leaves only)**
   - `label`: concise claim name for display (3–8 words).
   - `summary`: one short analyst phrase (max 25 words) describing the narrative.
   - `search_query`: a concrete sentence or claim phrase for semantic search against
     news, transcripts, and filings. Write as document/market language, not taxonomy
     meta-language. Example: "AI infrastructure costs are rising but revenue gains
     remain uncertain."
   - Do not copy `summary` verbatim into `search_query`.
   - Avoid vague labels; each leaf must be searchable and specific to {main_theme}.

4. **Format your response as valid JSON only**
   - Each node must include only these fields: `node`, `label`, `summary`,
     `search_query`, `children`.
   - `node`: integer identifier (unique across the tree).
   - `children`: array of child nodes; empty array for leaves.
   - Return only the JSON object. No markdown or commentary.

Example shape for theme "AI Bubble":
{{
  "node": 1,
  "label": "AI Bubble",
  "summary": "Market narrative that AI valuations and investment outpace fundamentals.",
  "search_query": "",
  "children": [
    {{
      "node": 2,
      "label": "Valuation Disconnect",
      "summary": "Prices and multiples detached from earnings power.",
      "search_query": "",
      "children": [
        {{
          "node": 3,
          "label": "Detached from earnings",
          "summary": "Tech valuations outrun fundamental earnings potential.",
          "search_query": "Tech valuations have detached from fundamental earnings potential",
          "children": []
        }},
        {{
          "node": 4,
          "label": "Aggressive projections",
          "summary": "Growth forecasts rest on unproven assumptions.",
          "search_query": "AI market projections rely on aggressive and unproven assumptions",
          "children": []
        }}
      ]
    }}
  ]
}}
"""

USER_MESSAGE_NARRATIVE_MINDMAP = "Your given Theme is: {main_theme}"


class Node(BaseModel):
    """Recursive node of the narrative mindmap returned by the LLM."""

    node: int
    label: str
    summary: str
    search_query: str = ""
    children: list[Node] = Field(default_factory=list)


Node.model_rebuild()


def generate_narrative_mindmap(
    main_theme: str,
    client: OpenAI | None = None,
    model: str = DEFAULT_LLM_MODEL,
    print_taxonomy: bool = True,
) -> Node:
    """Generate a theme → sub-theme → search-label mindmap via the LLM."""
    openai_client = client if client is not None else OpenAI()
    completion = openai_client.chat.completions.parse(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": SYSTEM_MESSAGE_NARRATIVE_MINDMAP.format(main_theme=main_theme),
            },
            {
                "role": "user",
                "content": USER_MESSAGE_NARRATIVE_MINDMAP.format(main_theme=main_theme),
            },
        ],
        **sampling_params_for_model(
            model,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=42,
        ),
    )
    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI returned empty mindmap content")
    root = Node.model_validate_json(content)
    if print_taxonomy:
        print_tree(root)
    return root


def print_tree(node: Node, indent: int = 0) -> None:
    """Print a simple indented view of the mindmap."""
    prefix = "  " * indent
    query_note = f"  [search: {node.search_query}]" if node.search_query else ""
    print(f"{prefix}- {node.label}{query_note}")
    for child in node.children:
        print_tree(child, indent + 1)


def save_mindmap(root: Node, path: str | Path) -> Path:
    """Write a mindmap so a later run can reuse the same sub-themes and search labels."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(root.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return destination


def load_mindmap(path: str | Path) -> Node:
    """Load a mindmap previously written by ``save_mindmap``."""
    return Node.model_validate_json(Path(path).read_text(encoding="utf-8"))


def mindmap_to_main_narratives(root: Node) -> list[dict[str, Any]]:
    """Convert mindmap root into ``main_narratives`` (sub-theme + search labels).

    Each top-level child of ``root`` is treated as a sub-theme. Leaf
    ``search_query`` values (falling back to ``label``) become search labels.
    """
    narratives: list[dict[str, Any]] = []
    for branch in root.children:
        labels = _leaf_search_texts(branch)
        if not labels and not branch.children:
            # Branch itself is a leaf under root
            text = (branch.search_query or branch.label).strip()
            if text:
                labels = [text]
        if not labels:
            continue
        narratives.append({"sub_theme": branch.label, "search_labels": labels})
    return narratives


def _leaf_search_texts(node: Node) -> list[str]:
    """Collect search strings from leaf descendants of ``node``."""
    if not node.children:
        text = (node.search_query or node.label).strip()
        return [text] if text else []
    out: list[str] = []
    for child in node.children:
        out.extend(_leaf_search_texts(child))
    return out


def flatten_search_queries(main_narratives: list[dict[str, Any]]) -> list[str]:
    """Flatten all search labels from ``main_narratives`` into a query list."""
    queries: list[str] = []
    for item in main_narratives:
        for label in item.get("search_labels", []):
            text = str(label).strip()
            if text:
                queries.append(text)
    return queries


def build_label_to_subtheme(main_narratives: list[dict[str, Any]]) -> dict[str, str]:
    """Map each search label to its parent sub-theme."""
    mapping: dict[str, str] = {}
    for item in main_narratives:
        sub_theme = str(item["sub_theme"])
        for label in item.get("search_labels", []):
            mapping[str(label)] = sub_theme
    return mapping


def _wrap(text: str, width: int = 28) -> str:
    words = str(text).split()
    if not words:
        return ""
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + len(word) + 1 <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    if len(lines) > 2:
        lines = [lines[0], lines[1].rstrip(".,") + "..."]
    return "\n".join(lines)


def _branch_groups(root: Node) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    for branch in root.children:
        leaves = _leaf_search_texts(branch)
        if not leaves and not branch.children:
            text = (branch.search_query or branch.label).strip()
            leaves = [text] if text else [branch.label]
        # Prefer short display labels for leaves when available
        display_leaves: list[str] = []
        if branch.children:
            for child in branch.children:
                if child.children:
                    display_leaves.extend(_leaf_search_texts(child))
                else:
                    display_leaves.append(child.label or child.search_query)
        else:
            display_leaves = leaves
        groups.append((branch.label, display_leaves or leaves))
    return groups


def plot_mindmap(root: Node) -> Figure:
    """Draw a left-to-right mindmap: theme → sub-themes → search labels."""
    grouped = _branch_groups(root)
    if not grouped:
        fig, ax = plt.subplots(figsize=(9, 2.4))
        ax.text(0.5, 0.5, "No mindmap branches to display.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    total_leaves = sum(len(leaves) for _, leaves in grouped)
    fig, ax = plt.subplots(figsize=(13, max(4.5, 0.62 * total_leaves + 1.4)))
    ax.axis("off")

    row = 0.0
    branch_anchors: list[tuple[float, str, str]] = []
    for branch_index, (branch, leaves) in enumerate(grouped):
        color = BRANCH_COLORS[branch_index % len(BRANCH_COLORS)]
        leaf_rows = [row + index for index in range(max(len(leaves), 1))]
        branch_center = sum(leaf_rows) / len(leaf_rows)
        branch_anchors.append((branch_center, branch, color))

        for leaf_row, leaf in zip(leaf_rows, leaves or [branch]):
            ax.plot([1.55, 2.0], [branch_center, leaf_row], color=color, linewidth=1.1, alpha=0.55)
            ax.text(
                2.05,
                leaf_row,
                leaf,
                va="center",
                ha="left",
                fontsize=10,
                color=TEXT_COLOR,
            )
        ax.text(
            1.5,
            branch_center,
            _wrap(branch, 28),
            va="center",
            ha="right",
            fontsize=11,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": "none"},
        )
        row += max(len(leaves), 1) + 0.6

    theme_center = sum(center for center, _, _ in branch_anchors) / len(branch_anchors)
    for branch_center, _, color in branch_anchors:
        ax.plot(
            [0.62, 0.95],
            [theme_center, branch_center],
            color=color,
            linewidth=1.4,
            alpha=0.7,
        )
    ax.text(
        0.6,
        theme_center,
        _wrap(root.label, 22),
        va="center",
        ha="right",
        fontsize=12,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": TEXT_COLOR, "edgecolor": "none"},
    )

    ax.set_xlim(-0.3, 4.6)
    ax.set_ylim(row, -1.2)
    fig.tight_layout()
    return fig
