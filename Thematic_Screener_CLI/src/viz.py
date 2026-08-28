"""Charts for derivative-hop thematic screening results.

Every function returns a Matplotlib :class:`~matplotlib.figure.Figure` so
notebooks can display or save it. Functions degrade to a placeholder figure
when the underlying frame is empty, so a thin run never breaks a report.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.derivative_taxonomy import leaf_branch_map, leaves_by_derivative_branch
from src.helpers import get_leaf_labels
from src.prompts import DERIVATIVE_BRANCH_LABELS

HOP_COLORS: dict[str, str] = {
    DERIVATIVE_BRANCH_LABELS[0]: "#12496b",
    DERIVATIVE_BRANCH_LABELS[1]: "#4c9ac9",
    DERIVATIVE_BRANCH_LABELS[2]: "#e8963c",
}
EXPOSURE_BRANCH_COLORS: tuple[str, ...] = (
    "#12496b",
    "#4c9ac9",
    "#e8963c",
    "#6a994e",
    "#bc4749",
    "#7b6ba8",
)
UNMAPPED_COLOR = "#b0b7bd"
GRID_COLOR = "#dce1e5"
TEXT_COLOR = "#243038"

HOP_COLUMN = "hop"

INDIRECT_HOPS: tuple[str, ...] = tuple(DERIVATIVE_BRANCH_LABELS[1:])
"""Hops beyond the direct one, where non-obvious exposure lives."""


def _hop_color(hop: str) -> str:
    return HOP_COLORS.get(hop, UNMAPPED_COLOR)


def _branch_color(index: int) -> str:
    return EXPOSURE_BRANCH_COLORS[index % len(EXPOSURE_BRANCH_COLORS)]


def _exposure_branch_groups(root: Any) -> list[tuple[str, list[str]]]:
    """Map each top-level branch to the leaf labels beneath it."""
    if not root.children:
        return []
    groups: list[tuple[str, list[str]]] = []
    for branch in root.children:
        leaves = get_leaf_labels(branch) if branch.children else [branch.label]
        groups.append((branch.label, leaves))
    return groups


def _has_derivative_branches(root: Any) -> bool:
    return any(leaves_by_derivative_branch(root).values())


def _style_axes(ax: Any, *, xgrid: bool = False, ygrid: bool = False) -> None:
    """Apply a clean, presentation-friendly axis style."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    if xgrid:
        ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)


def _empty_figure(message: str) -> Figure:
    """Return a placeholder figure used when there is nothing to plot."""
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color=TEXT_COLOR)
    ax.axis("off")
    fig.tight_layout()
    return fig


def attach_hop_column(evidence_df: pd.DataFrame, root: Any) -> pd.DataFrame:
    """Return a copy of ``evidence_df`` with a ``hop`` column derived from the taxonomy.

    The ``label`` assigned during labeling is a taxonomy leaf, so its branch
    ancestor identifies which derivative hop the evidence belongs to.
    """
    branch_map = leaf_branch_map(root)
    result = evidence_df.copy()
    if "label" not in result.columns:
        result[HOP_COLUMN] = pd.NA
        return result
    result[HOP_COLUMN] = result["label"].map(branch_map)
    return result


def _ordered_hops(evidence_df: pd.DataFrame) -> list[str]:
    present = set(evidence_df[HOP_COLUMN].dropna().unique())
    return [hop for hop in DERIVATIVE_BRANCH_LABELS if hop in present]


def _select_hops(evidence_df: pd.DataFrame, hops: Sequence[str] | None) -> pd.DataFrame:
    """Restrict evidence to ``hops``, which also narrows any ranking built from it.

    Ranking companies by total evidence buries later hops, because direct
    exposure produces far more disclosure than indirect exposure does.
    """
    if hops is None or HOP_COLUMN not in evidence_df.columns:
        return evidence_df
    return evidence_df[evidence_df[HOP_COLUMN].isin(list(hops))]


def _wrap(text: str, width: int = 34) -> str:
    """Soft-wrap a label onto at most two lines for axis readability."""
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


def plot_mindmap(root: Any) -> Figure:
    """Draw a mindmap for derivative or exposure taxonomies."""
    if _has_derivative_branches(root):
        return _plot_derivative_mindmap(root)
    return _plot_exposure_mindmap(root)


def _plot_exposure_mindmap(root: Any) -> Figure:
    """Draw an exposure-style mindmap: theme, branches, and leaf pathways."""
    grouped = _exposure_branch_groups(root)
    if not grouped:
        return _empty_figure("No taxonomy branches to display.")

    total_leaves = sum(len(leaves) for leaves in (_leaves for _, _leaves in grouped))
    fig, ax = plt.subplots(figsize=(13, max(4.5, 0.62 * total_leaves + 1.4)))
    ax.axis("off")

    row = 0.0
    branch_anchors: list[tuple[float, str, str]] = []
    for branch_index, (branch, leaves) in enumerate(grouped):
        color = _branch_color(branch_index)
        leaf_rows = [row + index for index in range(len(leaves))]
        branch_center = sum(leaf_rows) / len(leaf_rows)
        branch_anchors.append((branch_center, branch, color))

        for leaf_row, leaf in zip(leaf_rows, leaves, strict=True):
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
        row += len(leaves) + 0.6

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


def _plot_derivative_mindmap(root: Any) -> Figure:
    """Draw the derivative mindmap: theme, three hops, and their exposure pathways."""
    grouped = {
        branch: leaves
        for branch, leaves in leaves_by_derivative_branch(root).items()
        if leaves
    }
    if not grouped:
        return _empty_figure("No taxonomy branches to display.")

    total_leaves = sum(len(leaves) for leaves in grouped.values())
    fig, ax = plt.subplots(figsize=(13, max(4.5, 0.62 * total_leaves + 1.4)))
    ax.axis("off")

    row = 0.0
    hop_anchors: list[tuple[float, str]] = []
    for branch, leaves in grouped.items():
        color = _hop_color(branch)
        leaf_rows = [row + index for index in range(len(leaves))]
        hop_center = sum(leaf_rows) / len(leaf_rows)
        hop_anchors.append((hop_center, branch))

        for leaf_row, leaf in zip(leaf_rows, leaves, strict=True):
            ax.plot([1.55, 2.0], [hop_center, leaf_row], color=color, linewidth=1.1, alpha=0.55)
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
            hop_center,
            branch,
            va="center",
            ha="right",
            fontsize=11,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": "none"},
        )
        row += len(leaves) + 0.6

    theme_center = sum(center for center, _ in hop_anchors) / len(hop_anchors)
    for hop_center, branch in hop_anchors:
        ax.plot(
            [0.62, 0.95],
            [theme_center, hop_center],
            color=_hop_color(branch),
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


def plot_hop_coverage(evidence_df: pd.DataFrame) -> Figure:
    """Compare evidence volume and company breadth across the three hops."""
    if evidence_df.empty or HOP_COLUMN not in evidence_df.columns:
        return _empty_figure("No labeled evidence to summarize by hop.")

    hops = _ordered_hops(evidence_df)
    if not hops:
        return _empty_figure("No labeled evidence to summarize by hop.")

    quotes = [int((evidence_df[HOP_COLUMN] == hop).sum()) for hop in hops]
    companies = [
        int(evidence_df.loc[evidence_df[HOP_COLUMN] == hop, "company_name"].nunique())
        for hop in hops
    ]
    colors = [_hop_color(hop) for hop in hops]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, values, title in (
        (axes[0], quotes, "Evidence quotes"),
        (axes[1], companies, "Companies with exposure"),
    ):
        bars = ax.bar(hops, values, color=colors, width=0.6)
        ax.bar_label(bars, padding=3, fontsize=10, color=TEXT_COLOR)
        ax.set_title(title, fontsize=12, color=TEXT_COLOR, pad=12)
        ax.set_ylim(0, max(values) * 1.18 if max(values) else 1)
        _style_axes(ax, ygrid=True)

    fig.tight_layout()
    return fig


def plot_top_pathways(evidence_df: pd.DataFrame, top_n: int = 12) -> Figure:
    """Rank exposure pathways by how much evidence each one surfaced."""
    if evidence_df.empty or "label" not in evidence_df.columns:
        return _empty_figure("No labeled evidence to rank by pathway.")

    if HOP_COLUMN in evidence_df.columns:
        counts = (
            evidence_df.groupby(["label", HOP_COLUMN], dropna=False)
            .size()
            .reset_index(name="quotes")
            .sort_values("quotes", ascending=False)
            .head(top_n)
            .iloc[::-1]
        )
        bar_colors = [_hop_color(hop) for hop in counts[HOP_COLUMN]]
    else:
        counts = (
            evidence_df.groupby("label")
            .size()
            .reset_index(name="quotes")
            .sort_values("quotes", ascending=False)
            .head(top_n)
            .iloc[::-1]
        )
        bar_colors = [_branch_color(index) for index in range(len(counts))]

    if counts.empty:
        return _empty_figure("No labeled evidence to rank by pathway.")

    fig, ax = plt.subplots(figsize=(11, max(3.5, 0.46 * len(counts) + 1.4)))
    positions = range(len(counts))
    bars = ax.barh(
        list(positions),
        counts["quotes"],
        color=bar_colors,
        height=0.68,
    )
    ax.bar_label(bars, padding=4, fontsize=9, color=TEXT_COLOR)
    ax.set_yticks(list(positions))
    ax.set_yticklabels([_wrap(label, 40) for label in counts["label"]], fontsize=9)
    ax.set_xlabel("Evidence quotes", fontsize=10, color=TEXT_COLOR)
    ax.set_xlim(0, counts["quotes"].max() * 1.12)
    _style_axes(ax, xgrid=True)
    if HOP_COLUMN in counts.columns:
        _add_hop_legend(ax, counts[HOP_COLUMN].dropna().unique())
    fig.tight_layout()
    return fig


def _add_hop_legend(ax: Any, hops: Any) -> None:
    present = [hop for hop in DERIVATIVE_BRANCH_LABELS if hop in set(hops)]
    if not present:
        return
    handles = [plt.Rectangle((0, 0), 1, 1, color=_hop_color(hop)) for hop in present]
    ax.legend(
        handles,
        present,
        loc="lower right",
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_COLOR,
    )


def plot_top_companies(
    evidence_df: pd.DataFrame,
    top_n: int = 15,
    hops: Sequence[str] | None = None,
) -> Figure:
    """Rank companies by evidence volume, split by which hop the evidence came from.

    Pass ``hops=INDIRECT_HOPS`` to rank on indirect exposure alone, which keeps
    high-volume direct names from crowding out the later hops.
    """
    evidence_df = _select_hops(evidence_df, hops)
    if evidence_df.empty or "company_name" not in evidence_df.columns:
        return _empty_figure("No labeled evidence to rank by company.")

    if HOP_COLUMN not in evidence_df.columns:
        counts = (
            evidence_df.groupby("company_name")
            .size()
            .reset_index(name="quotes")
            .sort_values("quotes", ascending=True)
            .tail(top_n)
        )
        fig, ax = plt.subplots(figsize=(11, max(3.5, 0.44 * len(counts) + 1.5)))
        bars = ax.barh(
            counts["company_name"],
            counts["quotes"],
            color=_branch_color(0),
            height=0.7,
        )
        ax.bar_label(bars, padding=4, fontsize=9, color=TEXT_COLOR)
        ax.set_xlabel("Evidence quotes", fontsize=10, color=TEXT_COLOR)
        ax.set_xlim(0, counts["quotes"].max() * 1.12 if len(counts) else 1)
        ax.tick_params(axis="y", labelsize=9)
        _style_axes(ax, xgrid=True)
        fig.tight_layout()
        return fig

    matrix = (
        evidence_df.pivot_table(
            index="company_name",
            columns=HOP_COLUMN,
            values="sentence_id",
            aggfunc="count",
            fill_value=0,
        )
        if HOP_COLUMN in evidence_df.columns
        else pd.DataFrame()
    )
    if matrix.empty:
        return _empty_figure("No labeled evidence to rank by company.")

    hops = [hop for hop in DERIVATIVE_BRANCH_LABELS if hop in matrix.columns]
    matrix = matrix.loc[matrix[hops].sum(axis=1).sort_values(ascending=False).index]
    matrix = matrix.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(11, max(3.5, 0.44 * len(matrix) + 1.5)))
    left = pd.Series(0, index=matrix.index, dtype=float)
    for hop in hops:
        ax.barh(
            matrix.index,
            matrix[hop],
            left=left,
            color=_hop_color(hop),
            height=0.7,
            label=hop,
        )
        left = left + matrix[hop]

    for company, total in left.items():
        ax.text(
            total + max(left) * 0.01,
            company,
            int(total),
            va="center",
            fontsize=9,
            color=TEXT_COLOR,
        )

    ax.set_xlabel("Evidence quotes", fontsize=10, color=TEXT_COLOR)
    ax.set_xlim(0, max(left) * 1.12 if len(left) else 1)
    ax.tick_params(axis="y", labelsize=9)
    _style_axes(ax, xgrid=True)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=TEXT_COLOR)
    fig.tight_layout()
    return fig


def plot_company_hop_matrix(
    evidence_df: pd.DataFrame,
    top_n: int = 18,
    hops: Sequence[str] | None = None,
) -> Figure:
    """Show which hops each leading company is exposed through.

    Pass ``hops=INDIRECT_HOPS`` to rank and display indirect channels only.
    """
    evidence_df = _select_hops(evidence_df, hops)
    if evidence_df.empty or HOP_COLUMN not in evidence_df.columns:
        return _empty_figure("No labeled evidence for the exposure matrix.")

    matrix = evidence_df.pivot_table(
        index="company_name",
        columns=HOP_COLUMN,
        values="sentence_id",
        aggfunc="count",
        fill_value=0,
    )
    hops = [hop for hop in DERIVATIVE_BRANCH_LABELS if hop in matrix.columns]
    if matrix.empty or not hops:
        return _empty_figure("No labeled evidence for the exposure matrix.")

    matrix = matrix[hops]
    matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index].head(top_n)

    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.38 * len(matrix) + 1.6)))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(hops)))
    ax.set_xticklabels(hops, fontsize=10)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index, fontsize=9)

    peak = matrix.to_numpy().max()
    for row_index in range(len(matrix)):
        for col_index in range(len(hops)):
            value = int(matrix.iat[row_index, col_index])
            if value == 0:
                continue
            ax.text(
                col_index,
                row_index,
                value,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > peak * 0.55 else TEXT_COLOR,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, colors=TEXT_COLOR)
    fig.colorbar(image, ax=ax, shrink=0.7, label="Evidence quotes")
    fig.tight_layout()
    return fig


def plot_evidence_timeline(evidence_df: pd.DataFrame) -> Figure:
    """Track how evidence accumulates month by month."""
    if evidence_df.empty or "timestamp" not in evidence_df.columns:
        return _empty_figure("No timestamped evidence to plot.")

    dated = evidence_df.copy()
    dated["month"] = pd.to_datetime(
        dated["timestamp"], errors="coerce", format="mixed"
    ).dt.to_period("M")
    dated = dated.dropna(subset=["month"])
    if dated.empty:
        return _empty_figure("No timestamped evidence to plot.")

    if HOP_COLUMN in dated.columns:
        series = (
            dated.groupby(["month", HOP_COLUMN]).size().unstack(fill_value=0).sort_index()
        )
        hops = [hop for hop in DERIVATIVE_BRANCH_LABELS if hop in series.columns]
        months = [str(period) for period in series.index]
        positions = list(range(len(months)))

        fig, ax = plt.subplots(figsize=(12, 4.2))
        bottom = [0.0] * len(series)
        for hop in hops:
            values = series[hop].tolist()
            ax.bar(positions, values, bottom=bottom, color=_hop_color(hop), label=hop, width=0.68)
            bottom = [base + value for base, value in zip(bottom, values, strict=True)]
        ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT_COLOR)
    else:
        monthly = dated.groupby("month").size().sort_index()
        months = [str(period) for period in monthly.index]
        positions = list(range(len(months)))
        fig, ax = plt.subplots(figsize=(12, 4.2))
        ax.bar(positions, monthly.tolist(), color=_branch_color(0), width=0.68)

    ax.set_ylabel("Evidence quotes", fontsize=10, color=TEXT_COLOR)
    ax.set_xticks(positions)
    ax.set_xticklabels(months)
    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    _style_axes(ax, ygrid=True)
    fig.tight_layout()
    return fig
