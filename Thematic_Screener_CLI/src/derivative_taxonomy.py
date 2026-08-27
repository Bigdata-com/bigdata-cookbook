"""Validators and compact previews for derivative-hop taxonomies."""

from __future__ import annotations

import re
from typing import Any

from src.helpers import build_leaf_ancestry, get_leaf_labels, get_leaf_pairs
from src.prompts import DERIVATIVE_BRANCH_LABELS
from src.search_query import has_exposure_meta_language

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_NEAR_DUPLICATE_JACCARD = 0.55


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(text) if len(token) > 2}


def _jaccard(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def derivative_branch_nodes(root: Any) -> list[Any]:
    """Return the three derivative branch nodes under ``root``."""
    return list(root.children)


def leaves_by_derivative_branch(root: Any) -> dict[str, list[str]]:
    """Map each derivative branch label to its leaf labels."""
    grouped: dict[str, list[str]] = {label: [] for label in DERIVATIVE_BRANCH_LABELS}
    ancestry = build_leaf_ancestry(root)
    for leaf_label, ancestors in ancestry.items():
        for branch in DERIVATIVE_BRANCH_LABELS:
            if branch in ancestors:
                grouped[branch].append(leaf_label)
                break
    return grouped


def derivative_preview(root: Any) -> dict[str, list[str]]:
    """Compact branch → leaf labels map for notebook/CLI display."""
    return leaves_by_derivative_branch(root)


def leaf_branch_map(root: Any) -> dict[str, str]:
    """Map each leaf label to the derivative branch it sits under."""
    return {
        leaf: branch
        for branch, leaves in leaves_by_derivative_branch(root).items()
        for leaf in leaves
    }


def validate_derivatives_taxonomy(root: Any) -> list[dict[str, str]]:
    """Return findings for a derivatives-style taxonomy tree.

    An empty list means the tree matches the required 1st/2nd/3rd hop shape.
    """
    findings: list[dict[str, str]] = []
    children = derivative_branch_nodes(root)
    child_labels = [_normalize_label(child.label) for child in children]
    expected = [_normalize_label(label) for label in DERIVATIVE_BRANCH_LABELS]

    if child_labels != expected:
        findings.append(
            {
                "severity": "high",
                "check": "derivative_branches",
                "message": (
                    "Root must have exactly three children labeled "
                    f"{list(DERIVATIVE_BRANCH_LABELS)}; found {[child.label for child in children]}"
                ),
            }
        )
        return findings

    first_queries: list[str] = []
    later_queries: list[tuple[str, str]] = []

    for child, expected_label in zip(children, DERIVATIVE_BRANCH_LABELS, strict=True):
        if str(child.search_query or "").strip():
            findings.append(
                {
                    "severity": "medium",
                    "check": "branch_search_query",
                    "message": f"Branch '{expected_label}' should use an empty search_query.",
                }
            )
        leaf_labels = get_leaf_labels(child)
        if not leaf_labels:
            findings.append(
                {
                    "severity": "high",
                    "check": "empty_derivative_branch",
                    "message": f"Branch '{expected_label}' has no leaf exposure pathways.",
                }
            )
            continue
        for label, query in get_leaf_pairs(child, fallback=False):
            if not query.strip():
                findings.append(
                    {
                        "severity": "high",
                        "check": "missing_search_query",
                        "message": f"Leaf '{label}' is missing document-voice search_query text.",
                    }
                )
            elif has_exposure_meta_language(query):
                findings.append(
                    {
                        "severity": "medium",
                        "check": "search_query_meta_language",
                        "message": (
                            f"Leaf '{label}' search_query still uses exposure-meta language."
                        ),
                    }
                )
            if expected_label == DERIVATIVE_BRANCH_LABELS[0]:
                first_queries.append(query)
            else:
                later_queries.append((expected_label, query))

    for branch_label, query in later_queries:
        for first_query in first_queries:
            if _jaccard(query, first_query) >= _NEAR_DUPLICATE_JACCARD:
                findings.append(
                    {
                        "severity": "medium",
                        "check": "derivative_query_overlap",
                        "message": (
                            f"{branch_label} search_query is too similar to a "
                            f"1st derivative query: {query!r}"
                        ),
                    }
                )
                break

    return findings
