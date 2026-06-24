"""Isolated run-directory management for the screener CLI.

Each pipeline run writes all of its artifacts (themes, plans, search
results, and output CSVs) into a dedicated directory so that concurrent or
repeated runs never clobber each other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "config.json"
THEMES_FILENAME = "themes.txt"
SEARCH_QUERIES_FILENAME = "search_queries.txt"
TAXONOMY_TREE_FILENAME = "taxonomy_tree.json"
PLANS_DIRNAME = "plans"
RESULTS_DIRNAME = "results"
RESULTS_FILENAME = "results.json"
LABELED_SENTENCES_FILENAME = "labeled_sentences.csv"
COMPANY_SUMMARIES_FILENAME = "company_summaries.csv"
SCREENER_RESULTS_FILENAME = "screener_results.csv"


def default_run_name() -> str:
    """Return a UTC timestamp suitable as a unique run directory name."""
    return datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class RunContext:
    """Resolves and creates the paths for a single isolated run.

    Attributes
    ----------
    runs_root:
        Parent directory that holds all runs.
    run_name:
        Unique name for this run; the run directory is ``runs_root/run_name``.
    """

    runs_root: Path
    run_name: str

    @classmethod
    def create(cls, runs_root: str | Path, run_name: str | None) -> RunContext:
        """Build a context, generating a timestamped name when none is given."""
        resolved_name = run_name if run_name else default_run_name()
        return cls(runs_root=Path(runs_root), run_name=resolved_name)

    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.run_name

    @property
    def config_path(self) -> Path:
        return self.run_dir / CONFIG_FILENAME

    @property
    def themes_path(self) -> Path:
        return self.run_dir / THEMES_FILENAME

    @property
    def search_queries_path(self) -> Path:
        return self.run_dir / SEARCH_QUERIES_FILENAME

    @property
    def taxonomy_tree_path(self) -> Path:
        return self.run_dir / TAXONOMY_TREE_FILENAME

    @property
    def plans_dir(self) -> Path:
        return self.run_dir / PLANS_DIRNAME

    @property
    def results_dir(self) -> Path:
        return self.run_dir / RESULTS_DIRNAME

    @property
    def results_path(self) -> Path:
        return self.results_dir / RESULTS_FILENAME

    @property
    def labeled_sentences_path(self) -> Path:
        return self.run_dir / LABELED_SENTENCES_FILENAME

    @property
    def company_summaries_path(self) -> Path:
        return self.run_dir / COMPANY_SUMMARIES_FILENAME

    @property
    def screener_results_path(self) -> Path:
        return self.run_dir / SCREENER_RESULTS_FILENAME

    def ensure_run_dir(self) -> None:
        """Create the run directory (and runs root) if missing."""
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def ensure_plans_dir(self) -> None:
        self.plans_dir.mkdir(parents=True, exist_ok=True)

    def ensure_results_dir(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self) -> dict[str, Any]:
        """Return the persisted run config, or an empty dict when absent."""
        if not self.config_path.exists():
            return {}
        with self.config_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def save_config(self, config: dict[str, Any]) -> None:
        """Persist the run config, merging onto any existing values."""
        self.ensure_run_dir()
        merged = self.load_config()
        merged.update({key: value for key, value in config.items() if value is not None})
        with self.config_path.open("w", encoding="utf-8") as handle:
            json.dump(merged, handle, indent=2, sort_keys=True)

    def read_themes(self) -> list[str]:
        """Read one exposure label per non-empty line from ``themes.txt``."""
        if not self.themes_path.exists():
            raise FileNotFoundError(
                f"themes file not found at {self.themes_path}; run the 'generate-labels' step first"
            )
        lines = self.themes_path.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()]

    def write_themes(self, labels: list[str]) -> None:
        """Write exposure labels to ``themes.txt`` (one per line)."""
        self.ensure_run_dir()
        self.themes_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

    def read_search_queries(self) -> list[str]:
        """Read Bigdata search query text aligned with ``themes.txt``."""
        labels = self.read_themes()
        if self.search_queries_path.exists():
            lines = self.search_queries_path.read_text(encoding="utf-8").splitlines()
            queries = [line.strip() for line in lines if line.strip()]
            if len(queries) != len(labels):
                raise ValueError(
                    "search_queries.txt line count "
                    f"({len(queries)}) does not match themes.txt ({len(labels)})"
                )
            return queries

        if self.taxonomy_tree_path.exists():
            from src.helpers import get_leaf_search_queries
            from src.screener import Node

            root = Node.model_validate_json(self.taxonomy_tree_path.read_text(encoding="utf-8"))
            queries = get_leaf_search_queries(root)
            if len(queries) == len(labels):
                return queries

        from src.search_query import normalize_summary_to_search_query

        return [normalize_summary_to_search_query(label) for label in labels]

    def write_search_queries(self, search_queries: list[str]) -> None:
        """Write retrieval queries to ``search_queries.txt`` (one per line)."""
        self.ensure_run_dir()
        self.search_queries_path.write_text("\n".join(search_queries) + "\n", encoding="utf-8")
