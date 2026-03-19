"""
Load and visualize benchmark results from parquet + JSON files.

Provides load_chunks_df(), load_benchmark_summary(), and plotting helpers
that derive chunk-count, relevance, entity, and histogram charts from
the chunk-level DataFrame produced by benchmark.py.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS: dict[str, str] = {
    "full_grid": "Full grid",
    "full_grid_split": "Full grid (time split)",
    "smart_batching": "Smart Batching",
}

METHOD_ORDER: list[str] = ["full_grid", "full_grid_split", "smart_batching"]


def load_chunks_df(base_path: Path | str = Path("benchmark_results")) -> pd.DataFrame:
    """Load the chunk-level parquet produced by *benchmark.py*.

    The DataFrame contains one row per chunk with columns including
    ``method``, ``case_index``, ``date``, ``relevance``, ``entity_ids``, etc.
    """
    parquet_path = Path(base_path).with_suffix(".parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Chunk data not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_benchmark_summary(
    base_path: Path | str = Path("benchmark_results"),
) -> dict[str, Any]:
    """Load the lightweight JSON summary produced by *benchmark.py*."""
    json_path = Path(base_path).with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"Summary file not found: {json_path}")
    with json_path.open() as f:
        return json.load(f)


def _case_text_label(summary: dict[str, Any], case_idx: int, max_len: int = 40) -> str:
    """Build a short label for a case from the summary JSON."""
    for key in ("full_grid_results", "full_grid_results_split_by_window", "smart_batching_results"):
        results = summary.get(key, [])
        if case_idx < len(results):
            text = results[case_idx].get("text", f"Case {case_idx}")
            return text[:max_len] + "..." if len(text) > max_len else text
    return f"Case {case_idx}"


def _methods_in_df(df: pd.DataFrame) -> list[str]:
    """Return methods present in the DataFrame, in canonical order."""
    present = set(df["method"].unique())
    return [m for m in METHOD_ORDER if m in present]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_benchmark_timeseries(
    base_path: Path | str = Path("benchmark_results"),
    *,
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (12, 4),
) -> None:
    """Plot chunk count per date for each case, overlaying methods."""
    df = load_chunks_df(base_path)
    summary = load_benchmark_summary(base_path)
    metadata = summary.get("metadata", {})

    case_indices = sorted(df["case_index"].unique())
    methods = _methods_in_df(df)
    n_cases = len(case_indices)
    ncols = min(n_cases, 3)
    nrows = max(1, (n_cases + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows))
    if n_cases == 1:
        axes = [axes]
    else:
        axes = np.asarray(axes).flatten()

    for ax_idx, case_idx in enumerate(case_indices):
        ax = axes[ax_idx]
        case_df = df[df["case_index"] == case_idx]
        for method in methods:
            method_df = case_df[case_df["method"] == method]
            if method_df.empty:
                continue
            daily = method_df.groupby("date").size().reset_index(name="chunk_count")
            daily = daily.sort_values("date")
            ax.plot(
                daily["date"],
                daily["chunk_count"],
                label=METHOD_LABELS.get(method, method),
                marker=".",
                markersize=3,
            )
        ax.set_title(f"Case {case_idx + 1}: {_case_text_label(summary, case_idx)}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Chunk count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for j in range(n_cases, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Benchmark chunk count timeseries (universe: {metadata.get('universe_csv', 'N/A')})",
        fontsize=11,
    )
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def plot_relevance_timeseries(
    base_path: Path | str = Path("benchmark_results"),
    *,
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (12, 4),
) -> None:
    """Plot average relevance per date for each case, overlaying methods."""
    df = load_chunks_df(base_path)
    summary = load_benchmark_summary(base_path)
    metadata = summary.get("metadata", {})

    case_indices = sorted(df["case_index"].unique())
    methods = _methods_in_df(df)
    n_cases = len(case_indices)
    ncols = min(n_cases, 3)
    nrows = max(1, (n_cases + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows))
    if n_cases == 1:
        axes = [axes]
    else:
        axes = np.asarray(axes).flatten()

    for ax_idx, case_idx in enumerate(case_indices):
        ax = axes[ax_idx]
        case_df = df[df["case_index"] == case_idx]
        for method in methods:
            method_df = case_df[case_df["method"] == method].dropna(subset=["relevance"])
            if method_df.empty:
                continue
            daily = (
                method_df.groupby("date")["relevance"]
                .mean()
                .reset_index(name="avg_relevance")
                .sort_values("date")
            )
            ax.plot(
                daily["date"],
                daily["avg_relevance"],
                label=METHOD_LABELS.get(method, method),
                marker=".",
                markersize=3,
            )
        ax.set_title(f"Case {case_idx + 1}: {_case_text_label(summary, case_idx)}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Avg relevance")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for j in range(n_cases, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Benchmark average relevance by date (universe: {metadata.get('universe_csv', 'N/A')})",
        fontsize=11,
    )
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def plot_entities_timeseries(
    base_path: Path | str = Path("benchmark_results"),
    *,
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (12, 4),
) -> None:
    """Plot distinct entity detections per date for each case, overlaying methods."""
    df = load_chunks_df(base_path)
    summary = load_benchmark_summary(base_path)
    metadata = summary.get("metadata", {})

    case_indices = sorted(df["case_index"].unique())
    methods = _methods_in_df(df)
    n_cases = len(case_indices)
    ncols = min(n_cases, 3)
    nrows = max(1, (n_cases + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows))
    if n_cases == 1:
        axes = [axes]
    else:
        axes = np.asarray(axes).flatten()

    for ax_idx, case_idx in enumerate(case_indices):
        ax = axes[ax_idx]
        case_df = df[df["case_index"] == case_idx]
        for method in methods:
            method_df = case_df[case_df["method"] == method]
            if method_df.empty:
                continue
            exploded = method_df.explode("entity_ids").dropna(subset=["entity_ids"])
            if exploded.empty:
                continue
            daily = (
                exploded.groupby("date")["entity_ids"]
                .nunique()
                .reset_index(name="distinct_entities")
                .sort_values("date")
            )
            ax.plot(
                daily["date"],
                daily["distinct_entities"],
                label=METHOD_LABELS.get(method, method),
                marker=".",
                markersize=3,
            )
        ax.set_title(f"Case {case_idx + 1}: {_case_text_label(summary, case_idx)}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Distinct entities")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for j in range(n_cases, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Distinct entity detections over time (universe: {metadata.get('universe_csv', 'N/A')})",
        fontsize=11,
    )
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def plot_relevance_histogram(
    base_path: Path | str = Path("benchmark_results"),
    *,
    bins: int = 20,
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (12, 4),
) -> None:
    """Plot histogram of daily average relevance for each case and method."""
    df = load_chunks_df(base_path)
    summary = load_benchmark_summary(base_path)
    metadata = summary.get("metadata", {})

    case_indices = sorted(df["case_index"].unique())
    methods = _methods_in_df(df)
    n_cases = len(case_indices)
    ncols = min(n_cases, 3)
    nrows = max(1, (n_cases + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows))
    if n_cases == 1:
        axes = [axes]
    else:
        axes = np.asarray(axes).flatten()

    for ax_idx, case_idx in enumerate(case_indices):
        ax = axes[ax_idx]
        case_df = df[(df["case_index"] == case_idx) & df["relevance"].notna()]

        all_daily_avgs: list[float] = []
        for method in methods:
            method_df = case_df[case_df["method"] == method]
            if method_df.empty:
                continue
            daily_avg = method_df.groupby("date")["relevance"].mean()
            all_daily_avgs.extend(daily_avg.tolist())

        if not all_daily_avgs:
            continue
        bin_edges = np.linspace(min(all_daily_avgs), max(all_daily_avgs), bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        for method in methods:
            method_df = case_df[case_df["method"] == method]
            if method_df.empty:
                continue
            daily_avg = method_df.groupby("date")["relevance"].mean()
            counts, _ = np.histogram(daily_avg.values, bins=bin_edges)
            ax.plot(
                bin_centers,
                counts,
                label=METHOD_LABELS.get(method, method),
                marker=".",
                markersize=4,
                linewidth=1.5,
            )

        ax.set_title(f"Case {case_idx + 1}: {_case_text_label(summary, case_idx)}")
        ax.set_xlabel("Relevance")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    for j in range(n_cases, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Relevance distribution by case (universe: {metadata.get('universe_csv', 'N/A')})",
        fontsize=11,
    )
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    base = "benchmark_results"
    plot_benchmark_timeseries(base_path=base, output_path="benchmark_results.png")
    plot_relevance_timeseries(base_path=base, output_path="benchmark_relevance.png")
    plot_relevance_histogram(base_path=base, output_path="benchmark_relevance_histogram.png")
    plot_entities_timeseries(base_path=base, output_path="benchmark_entities.png")
