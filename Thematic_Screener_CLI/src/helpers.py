from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from src.search_query import normalize_summary_to_search_query

if TYPE_CHECKING:
    from src.screener import Node


class _TaxonomyNode(Protocol):
    node: int
    label: str
    summary: str
    search_query: str
    children: list[object]


def get_leaf_labels(model: _TaxonomyNode) -> list[str]:
    """Return leaf ``label`` values in tree order."""
    if not model.children:
        return [model.label]
    labels: list[str] = []
    for child in model.children:
        labels.extend(get_leaf_labels(child))  # type: ignore[arg-type]
    return labels


def get_leaf_search_queries(
    model: _TaxonomyNode,
    *,
    fallback: bool = True,
    entity_type: str | None = None,
) -> list[str]:
    """Return leaf ``search_query`` values, optionally deriving missing queries."""
    if not model.children:
        query = str(model.search_query or "").strip()
        if query:
            return [query]
        if fallback:
            summary = str(model.summary or "").strip()
            if summary:
                return [normalize_summary_to_search_query(summary, entity_type=entity_type)]
            return [str(model.label or "").strip()]
        return [""]

    queries: list[str] = []
    for child in model.children:
        queries.extend(
            get_leaf_search_queries(child, fallback=fallback, entity_type=entity_type)
        )  # type: ignore[arg-type]
    return queries


def get_leaf_label_summary_options(model: _TaxonomyNode) -> list[str]:
    """Return ``Label: Summary`` strings for risk labeling prompts."""
    if not model.children:
        label = str(model.label or "").strip()
        summary = str(model.summary or "").strip()
        if summary:
            return [f"{label}: {summary}"]
        return [label]
    options: list[str] = []
    for child in model.children:
        options.extend(get_leaf_label_summary_options(child))  # type: ignore[arg-type]
    return options


def get_leaf_pairs(model: _TaxonomyNode, *, fallback: bool = True) -> list[tuple[str, str]]:
    """Return ``(label, search_query)`` pairs for each leaf."""
    if not model.children:
        label = str(model.label or "").strip()
        queries = get_leaf_search_queries(model, fallback=fallback)
        return [(label, queries[0] if queries else "")]
    pairs: list[tuple[str, str]] = []
    for child in model.children:
        pairs.extend(get_leaf_pairs(child, fallback=fallback))  # type: ignore[arg-type]
    return pairs


def get_leaf_summaries(model: _TaxonomyNode) -> list[str]:
    """Return leaf ``summary`` values (analyst taxonomy phrases)."""
    if not model.children:
        return [model.summary]
    summaries: list[str] = []
    for child in model.children:
        summaries.extend(get_leaf_summaries(child))  # type: ignore[arg-type]
    return summaries


def build_leaf_ancestry(model: _TaxonomyNode, ancestors: list[str] | None = None) -> dict[str, list[str]]:
    """Map each leaf ``label`` to ancestor labels (root first).

    Used to derive risk_factor/risk_channel from a leaf sub-scenario.
    """
    chain = ancestors or []
    if not model.children:
        return {model.label: list(chain)}

    mapping: dict[str, list[str]] = {}
    child_ancestors = [*chain, model.label]
    for child in model.children:
        mapping.update(build_leaf_ancestry(child, child_ancestors))  # type: ignore[arg-type]
    return mapping


def print_tree(node: Node | _TaxonomyNode | None, level: int = 0) -> None:
    """Print the taxonomy tree with label, summary, and search query."""
    if node is None:
        return
    suffix = ""
    if not node.children:
        query = str(getattr(node, "search_query", "") or "").strip()
        if query:
            suffix = f" | query: {query}"
    print("  " * level + f"Node {node.node}: {node.label} - {node.summary}{suffix}")
    for child in node.children:
        print_tree(child, level + 1)
