"""Query builder helpers (SDK removed — use search_helper instead).

MIGRATION NOTE:
SDK query-building functions have been removed.
Use `src.search_helper.run_universe_search` or build plain dict payloads for /v1/search.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class EntitiesToSearch:
    """Deprecated — kept for reference only."""
    people: list[str] | None = None
    product: list[str] | None = None
    org: list[str] | None = None
    place: list[str] | None = None
    topic: list[str] | None = None
    concepts: list[str] | None = None
    companies: list[str] | None = None


def build_similarity_queries(*args, **kwargs):
    """Deprecated — removed with SDK."""
    raise NotImplementedError(
        "build_similarity_queries removed with SDK. Use `src.search_helper.run_universe_search`."
    )


def build_batched_query(*args, **kwargs):
    """Deprecated — removed with SDK."""
    raise NotImplementedError(
        "build_batched_query removed with SDK. Use `src.search_helper.run_universe_search`."
    )


def create_date_intervals(
    start_date: str, end_date: str, freq: str
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Generates date intervals based on a specified frequency within a given start and end date range.

    Args:
        start_date (str):
            The start date in 'YYYY-MM-DD' format.
        end_date (str):
            The end date in 'YYYY-MM-DD' format.
        freq (str):
            The frequency for intervals. Supported values:
                - 'Y': Yearly intervals.
                - 'M': Monthly intervals.
                - 'W': Weekly intervals.
                - 'D': Daily intervals.

    Returns:
        List[Tuple[pd.Timestamp, pd.Timestamp]]:
            A list of tuples, where each tuple contains the start and end timestamp
            of an interval. The intervals are inclusive of the start and exclusive of the next start.

    Raises:
        ValueError: If the provided frequency is invalid.

    Operation:
        1. Converts the `start_date` and `end_date` strings to `pd.Timestamp` objects.
        2. Adjusts the frequency for yearly ('Y') and monthly ('M') intervals to align with period starts:
           - 'Y' → 'AS' (Year Start).
           - 'M' → 'MS' (Month Start).
        3. Uses `pd.date_range` to generate a range of dates based on the frequency.
        4. Creates tuples representing start and end times for each interval:
           - The start time is set to midnight (00:00:00).
           - The end time is set to the last second of the interval (23:59:59).
        5. Ensures the final interval includes the specified `end_date`.

    Notes:
        - The intervals are inclusive of the start and exclusive of the next start time.
        - For invalid frequencies, a `ValueError` is raised to indicate the issue.
    """
    # Convert start and end dates to pandas Timestamps
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Adjust frequency for yearly and monthly to use appropriate start markers
    # 'AS' for year start, 'MS' for month start
    adjusted_freq = freq.replace("Y", "AS").replace("M", "MS")

    # Generate date range based on the adjusted frequency
    try:
        date_range = pd.date_range(start=start_date, end=end_date, freq=adjusted_freq)
    except ValueError:
        raise ValueError("Invalid frequency. Use 'Y', 'M', 'W', or 'D'.")

    # Create intervals
    intervals = []
    for i in range(len(date_range) - 1):
        intervals.append(
            (
                date_range[i].replace(hour=0, minute=0, second=0),
                (date_range[i + 1] - pd.Timedelta(seconds=1)).replace(
                    hour=23, minute=59, second=59
                ),
            )
        )

    # Handle the last range to include the full end_date
    intervals.append(
        (
            date_range[-1].replace(hour=0, minute=0, second=0),
            end_date.replace(hour=23, minute=59, second=59),
        )
    )

    return intervals


def create_date_ranges(
    start_date: str, end_date: str, freq: str
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Generates date range tuples (SDK AbsoluteDateRange removed)."""
    return create_date_intervals(start_date, end_date, freq=freq)