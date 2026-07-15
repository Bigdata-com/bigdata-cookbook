"""Load company universe CSVs for monitoring."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

UNIVERSE_ID_COLUMN = "RP_ENTITY_ID"
UNIVERSE_ID_ALIASES: tuple[str, ...] = ("RP_ENTITY_ID", "RP_COMPANY_ID")
UNIVERSE_NAME_COLUMN = "COMPANY_NAME"


def _find_universe_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first matching column name, case-insensitive."""
    normalized = {str(column).upper(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.upper() in normalized:
            return normalized[candidate.upper()]
    return None


def load_universe(universe_path: str | Path) -> pd.DataFrame:
    """Load the company universe CSV.

    Accepts ``RP_ENTITY_ID`` (preferred) or legacy ``RP_COMPANY_ID``, plus
    ``COMPANY_NAME`` (or ``NAME`` / ``COMPANY`` aliases).
    """
    raw_df = pd.read_csv(universe_path)
    id_column = _find_universe_column(raw_df.columns, UNIVERSE_ID_ALIASES)
    if id_column is None:
        msg = (
            f"universe file {universe_path} is missing required ID column "
            f"(expected one of: {', '.join(UNIVERSE_ID_ALIASES)})"
        )
        raise ValueError(msg)
    name_column = _find_universe_column(
        raw_df.columns, (UNIVERSE_NAME_COLUMN, "NAME", "COMPANY")
    )
    if name_column is None:
        msg = (
            f"universe file {universe_path} is missing required name column "
            f"(expected one of: {UNIVERSE_NAME_COLUMN}, NAME, COMPANY)"
        )
        raise ValueError(msg)
    universe_df = pd.DataFrame(
        {
            UNIVERSE_ID_COLUMN: raw_df[id_column].astype(str).str.strip(),
            UNIVERSE_NAME_COLUMN: raw_df[name_column].astype(str).str.strip(),
        }
    )
    return universe_df.reset_index(drop=True)
