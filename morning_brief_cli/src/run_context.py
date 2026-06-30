"""Isolated run-directory management for the morning brief CLI.

Each pipeline run writes all of its artifacts (plans, search results, and
rendered briefs) into a dedicated directory so that concurrent or repeated
runs never clobber each other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "config.json"
PORTFOLIO_FILENAME = "portfolio.csv"
PLANS_DIRNAME = "plans"
RESULTS_DIRNAME = "results"
BRIEFS_DIRNAME = "briefs"


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
    def portfolio_path(self) -> Path:
        return self.run_dir / PORTFOLIO_FILENAME

    @property
    def plans_dir(self) -> Path:
        return self.run_dir / PLANS_DIRNAME

    @property
    def results_dir(self) -> Path:
        return self.run_dir / RESULTS_DIRNAME

    def results_path_for(self, topic_id: str) -> Path:
        """Return the results JSON path for a single topic."""
        return self.results_dir / f"{topic_id}.json"

    @property
    def briefs_dir(self) -> Path:
        return self.run_dir / BRIEFS_DIRNAME

    def ensure_run_dir(self) -> None:
        """Create the run directory (and runs root) if missing."""
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def ensure_plans_dir(self) -> None:
        self.plans_dir.mkdir(parents=True, exist_ok=True)

    def ensure_results_dir(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def ensure_briefs_dir(self) -> None:
        self.briefs_dir.mkdir(parents=True, exist_ok=True)

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
