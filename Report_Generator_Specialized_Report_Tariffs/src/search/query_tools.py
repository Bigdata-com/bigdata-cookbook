"""
Module that includes simple, flexible functions to build 
hybrid query objects and generate date ranges at any desired frequency.

MIGRATION NOTE: SDK imports removed. Only date-range helpers remain.
If SDK query builders are needed, use bigdata-smart-batching or REST API directly.

Copyright (C) 2024, RavenPack | Bigdata.com. All rights reserved.
Author: Jelena Starovic (jstarovic@ravenpack.com)
"""

from typing import List, Tuple
import pandas as pd


def create_date_intervals(
    sd: str, ed: str, freq: str
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Create date intervals for a given frequency.

    :param sd:
        The start date in 'YYYY-MM-DD' format.
    :param ed:
        The end date in 'YYYY-MM-DD' format.
    :param freq:
        Frequency string ('Y' for yearly, 'M' for monthly, 'W' for weekly, 'D' for daily).
    :return:
        List[Tuple[pd.Timestamp, pd.Timestamp]]: List of start and end date tuples.
    """
    # Convert start and end dates to pandas Timestamps
    start_date = pd.Timestamp(sd)
    end_date = pd.Timestamp(ed)

    # Adjust frequency for yearly and monthly to use appropriate start markers
    # 'AS' for year start, 'MS' for month start
    adjusted_freq = {"Y": "AS", "M": "MS"}.get(freq, freq)

    # Generate date range based on the adjusted frequency
    try:
        date_range = pd.date_range(start=start_date, end=end_date, freq=adjusted_freq)
    except ValueError:
        raise ValueError("Invalid frequency. Use 'Y', 'M', 'W', or 'D'.")
        
    try:
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
    except: 
        raise ValueError(f"Frequency {adjusted_freq} is longer than date range ({sd}, {ed}). Please provide a longer date range, or a shorter frequency")
