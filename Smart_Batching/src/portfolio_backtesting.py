from typing import Union
from itertools import product
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import warnings

# These are the standard data column names expected in the input DataFrame
COLUMN_DATE = 'DATE'
COLUMN_RETURN_CURRENT = 'RETURNS_T0'
COLUMN_RETURN_NEXT_DAY = 'RETURNS_T1'
COLUMN_SECURITY_ID = 'SECURITY_ID'

# These are the standard column names used in portfolio backtesting
COLUMN_DRIFT_FACTOR = 'drift_factor'
COLUMN_DRIFTING_WEIGHTS = 'drifting_weights'
COLUMN_IS_REBALANCING_DATE = 'is_rebalancing_date'
COLUMN_IS_SELECTED_SIGNAL = 'is_selected_signal'
COLUMN_NEXT_REBALANCING_DATE = 'next_rebalancing_date'
COLUMN_REBALANCING_DATE = 'rebalancing_date'
COLUMN_REBALANCING_PERIOD = 'rebalancing_period'
COLUMN_RETURNS_PORTFOLIO = 'portfolio_returns'
COLUMN_SCALED_DRIFTING_WEIGHTS = 'scaled_drifting_weights'
COLUMN_SELECTED_SIGNAL = 'selected_signal'
COLUMN_WEIGHTS = 'weights'

# These are the standard parameters
DEFAULT_ALLOW_ALL_MISSING_SIGNALS = False
DEFAULT_ALLOW_SOME_MISSING_SIGNALS = True

def constrain_position_weights(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """
    Constrain the position weights of a DataFrame to a specified cap.

    This function caps the long and short position weights in the given DataFrame to a specified cap.
    It reallocates the excess weight proportionally to the other positions.
    The long and short positions are determined by positive and negative weights.

    Args:
        df (pd.DataFrame): The DataFrame containing position weights.
        cap (float): The cap value for the position weights.

    Returns:
        pd.DataFrame: The DataFrame with the constrained position weights.
		
    """
    df = df.copy()
    
    cap_not_breached = \
        df[COLUMN_WEIGHTS].isna().all() or (
            df[COLUMN_WEIGHTS].abs().max() <= cap) or (
			cap is None)
    
    if cap_not_breached:
        return df
        
    def calculate_exposure(data: pd.DataFrame) -> pd.Series:
        """
        Calculate the total daily exposure of position weights in a DataFrame.

        This function calculates the exposure of position weights in the given
        DataFrame by grouping the data based on the COLUMN_DATE column and then
        summing the COLUMN_WEIGHTS values for each group.

        Args:
        data (pd.DataFrame): The DataFrame containing position weights.

        Returns:
        pd.Series: The Series representing the exposure calculated for each row in the DataFrame.
        """
        return data.groupby(COLUMN_DATE)[COLUMN_WEIGHTS].transform('sum')

    position_types = ['long', 'short']
    for position_type in position_types:
        if position_type == 'long':
            positions_outside_cap = df[COLUMN_WEIGHTS] > cap
            sign = 1
        else:
            positions_outside_cap = df[COLUMN_WEIGHTS] < -cap
            sign = -1

        if any(positions_outside_cap):
            dates_outside_cap = df[positions_outside_cap][COLUMN_DATE].unique()

            positions_to_update = df[
                df[COLUMN_DATE].isin(dates_outside_cap) &
                ((sign * df[COLUMN_WEIGHTS]) > 0)][[COLUMN_DATE, COLUMN_WEIGHTS]]

            positions_to_update['original_exposure'] = \
                calculate_exposure(positions_to_update)

            max_iterations = positions_to_update.groupby(COLUMN_DATE).apply(
                lambda x: sum(x[COLUMN_WEIGHTS].abs() < cap)).max() + 1
            iteration_count = 0
            while any(positions_to_update[COLUMN_WEIGHTS].abs() > cap):
                positions_to_update.loc[
                    positions_to_update[COLUMN_WEIGHTS].abs() > cap,
                    COLUMN_WEIGHTS] = sign * cap

                positions_to_update['updated_exposure'] = \
                    calculate_exposure(positions_to_update)

                positions_below_cap = (
                    sign * positions_to_update[COLUMN_WEIGHTS]) < cap

                positions_to_update['exposure_within_cap'] = \
                    calculate_exposure(
                        positions_to_update[positions_below_cap])

                positions_to_update['adjustment_factor'] = 1 + (
                    positions_to_update['original_exposure'] -
                    positions_to_update['updated_exposure']
                ) / positions_to_update['exposure_within_cap']

                positions_to_update.loc[positions_below_cap, COLUMN_WEIGHTS] = (
                    positions_to_update.loc[positions_below_cap, COLUMN_WEIGHTS] *
                    positions_to_update.loc[positions_below_cap, 'adjustment_factor'])

                iteration_count += 1

                assert iteration_count <= max_iterations, \
                    f"{position_type.capitalize()} position capping got stuck in infinity loop!"
            df.loc[positions_to_update.index,
                   COLUMN_WEIGHTS] = positions_to_update[COLUMN_WEIGHTS]
	
    return df

def get_rebalancing_period_ranges(
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        rebalancing_freq: str,
        expand_ranges: bool,
        **date_offset_kwargs) -> list[pd.Timestamp]:
    """
    Generate a list of rebalancing period ranges based on the specified start and end dates, rebalancing frequency,
    and optional date offset arguments.

    Args:
        start_date (Union[str, pd.Timestamp]): Start date of the period as a string or a pandas Timestamp.
            It should be the first day of the first rebalancing month.
        end_date (Union[str, pd.Timestamp]): End date of the period as a string or a pandas Timestamp.
        rebalancing_freq (str): Frequency of rebalancing periods, specified as a string compatible with pandas date_range.
            Daily - 'D', Weekly - 'W', Month start - 'MS', Quarter start - 'QS', Start of half-years - '6MS', Year start - 'YS'.
            More details can be specified together with the frequency. For example, 'W-MON', 'W-TUE', ... can specify which weekday
            will be the rebalancing day for a weekly frequency. Similarly with quarterly and annual frequencies. Full list can be found at
            https://pandas.pydata.org/pandas-docs/version/1.5/user_guide/timeseries.html#anchored-offsets
            Full list of possible rebalancing frequencies in the pandas library can be found at
            https://pandas.pydata.org/pandas-docs/version/1.5/user_guide/timeseries.html#offset-aliases
        expand_ranges (bool): Flag indicating whether to expand the rebalancing period ranges beyond the start and end dates.
            If True, the function will include rebalancing periods before the start date and after the end date to ensure
            full coverage within the specified range.
        **date_offset_kwargs: Optional keyword arguments for pandas DateOffset to modify the rebalancing period ranges.
            The date_offset_kwargs parameter in the get_rebalancing_period_ranges function allows for additional keyword
            arguments to modify the rebalancing period ranges using the pd.DateOffset function.
            Here are some possible date_offset_kwargs parameters and their effects:
                months: An integer representing the number of months to shift the rebalancing periods.
                weeks: An integer representing the number of weeks to shift the rebalancing periods.
                days: An integer representing the number of days to shift the rebalancing periods.
                weekday: An integer representing the desired weekday for the rebalancing periods (0 for Monday, 1 for Tuesday, and so on).
                week: An integer representing the desired week number for the rebalancing periods.
                day: An integer representing the desired day of the month for the rebalancing periods.
            These parameters can be used individually or in combination to create custom date offsets for the rebalancing periods.
            For example, specifying months=1 will shift the rebalancing periods by one month, while specifying weeks=2 and days=-1
            will shift the rebalancing periods by two weeks and one day earlier.
            Please refer to the pandas documentation on DateOffset for more details on available parameters and their usage:
            https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.tseries.offsets.DateOffset.html

    Returns:
        list[pd.Timestamp]: List of rebalancing period ranges as pandas Timestamp objects.

    Notes:
        - The rebalancing period ranges are determined using pandas' date_range function.
        - If date_offset_kwargs are provided, the rebalancing periods will be modified accordingly.
        - If expand_ranges=False, this function will only return timestamps that are in the interval defined by start_date and end_date.
          If the start_date does not correspond to the frequency (before any offsets are applied),
          the returned timestamps will start at the next valid timestamp, same for end_date,
          the returned timestamps will stop at the previous valid timestamp.

    Example:
        >>> start_date = "2023-01-01"
        >>> end_date = "2023-04-30"
        >>> rebalancing_freq = "MS"
        >>> get_rebalancing_period_ranges(start_date, end_date, rebalancing_freq, day=5, expand_ranges=False)
        [Timestamp('2023-01-05 00:00:00'), Timestamp('2023-02-05 00:00:00'), Timestamp('2023-03-05 00:00:00'), Timestamp('2023-04-05 00:00:00')]
        >>> get_rebalancing_period_ranges(start_date, end_date, rebalancing_freq, day=5, expand_ranges=True)
        [Timestamp('2022-12-05 00:00:00'), Timestamp('2023-01-05 00:00:00'), Timestamp('2023-02-05 00:00:00'), Timestamp('2023-03-05 00:00:00'), Timestamp('2023-04-05 00:00:00'), Timestamp('2023-05-05 00:00:00')]

    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    rebalancing_period_ranges = pd.date_range(
        start=start_date,
        end=end_date,
        freq=rebalancing_freq)

    main_offset = rebalancing_period_ranges.freq
    additional_offset = pd.DateOffset(**date_offset_kwargs)

    adj_rebalancing_period_ranges = \
        rebalancing_period_ranges + additional_offset # type: ignore

    if expand_ranges:
        while adj_rebalancing_period_ranges[0] > start_date: # type: ignore
            rebalancing_period_ranges = rebalancing_period_ranges.union(
                [rebalancing_period_ranges[0] - main_offset]) # type: ignore
            adj_rebalancing_period_ranges = rebalancing_period_ranges + additional_offset
        while adj_rebalancing_period_ranges[-1] <= end_date: # type: ignore
            rebalancing_period_ranges = rebalancing_period_ranges.union(
                [rebalancing_period_ranges[-1] + main_offset]) # type: ignore
            adj_rebalancing_period_ranges = rebalancing_period_ranges + additional_offset

    return adj_rebalancing_period_ranges.tolist()

def assign_rebalancing_periods(
        df: pd.DataFrame,
        rebalancing_period_ranges: list[pd.Timestamp],
        rebalancing_freq: str) -> pd.DataFrame:
    """
    Assign rebalancing period details to the DataFrame based on the specified parameters.

    Args:
        df (pd.DataFrame): Input DataFrame.
        rebalancing_period_ranges (Union[list[str], list[pd.Timestamp]]): List of rebalancing period ranges as strings or pandas Timestamps.
        rebalancing_freq (str): Frequency of rebalancing periods, specified as a string compatible with pandas date_range.
                                Daily - 'D', Weekly - 'W', Month start - 'MS', Quarter start - 'QS', Start of half-years - '6MS', Year start - 'YS'.
                                Full list of possible rebalancing frequencies in the pandas library can be found at
                                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

    Returns:
        pd.DataFrame: DataFrame with rebalancing period details assigned.

    Notes:
        - The rebalancing period ranges are defined by the list `rebalancing_period_ranges`.
        - The rebalancing periods are assigned using pd.cut function based on the dates in `df[column_dates]`.
        - The assigned rebalancing periods are stored in the `df[column_rebalancing_period]`.
        - The rebalancing dates are stored in the `df[column_rebalancing_date]`.
        - The `df[column_is_rebalancing_date]` is set to True for the rebalancing days and False for the non-rebalancing days.

    Example:
    >>> df = pd.DataFrame(
    ...     {'DATE': [pd.Timestamp('2022-01-01'), pd.Timestamp('2022-01-05'),
    ...               pd.Timestamp('2022-01-10'), pd.Timestamp('2022-01-15'),
    ...               pd.Timestamp('2022-01-20')]})
    >>> rebalancing_period_ranges = [
    ...     pd.Timestamp('2022-01-01'), pd.Timestamp('2022-01-10'),
    ...     pd.Timestamp('2022-01-20'), pd.Timestamp('2022-01-30')]
    >>> df = assign_rebalancing_periods(df, rebalancing_period_ranges, rebalancing_freq='10D')
    >>> print(df)
            DATE rebalancing_period rebalancing_date  is_rebalancing_day next_rebalancing_date
    0 2022-01-01          10D1-2022       2022-01-01                True            2022-01-10
    1 2022-01-05          10D1-2022       2022-01-01               False            2022-01-10
    2 2022-01-10          10D2-2022       2022-01-10                True            2022-01-20
    3 2022-01-15          10D2-2022       2022-01-10               False            2022-01-20
    4 2022-01-20          10D3-2022       2022-01-20                True                   NaT
    """
    rebalance_df = df[[COLUMN_DATE]].drop_duplicates().sort_values(
        COLUMN_DATE).reset_index(drop=True)

    if rebalancing_freq == 'D':
        # Every day is a rebalancing day
        rebalance_df[COLUMN_REBALANCING_PERIOD] = range(
            1, len(rebalance_df) + 1)
        rebalance_df[COLUMN_REBALANCING_DATE] = rebalance_df[COLUMN_DATE]
        rebalance_df[COLUMN_IS_REBALANCING_DATE] = True
    else:
        # For other re-balancing frequencies assign dates to re-balancing periods
        rebalance_df[COLUMN_REBALANCING_PERIOD] = pd.cut(
            rebalance_df[COLUMN_DATE], bins=rebalancing_period_ranges, # type: ignore
            labels=range(1, len(rebalancing_period_ranges)),
            right=False)

        rebalance_df[COLUMN_REBALANCING_PERIOD] = \
            rebalance_df[COLUMN_REBALANCING_PERIOD].astype(int)

        # Determine the rebalancing days
        rebalance_df[COLUMN_REBALANCING_DATE] = rebalance_df.groupby(
            COLUMN_REBALANCING_PERIOD)[COLUMN_DATE].transform('min')

        rebalance_df[COLUMN_IS_REBALANCING_DATE] = (
            rebalance_df[COLUMN_DATE] == rebalance_df[COLUMN_REBALANCING_DATE])

        first_record_with_valid_rebalancing_period = \
            rebalance_df[COLUMN_REBALANCING_PERIOD].first_valid_index()
        rebalance_df = rebalance_df.loc[first_record_with_valid_rebalancing_period:]

    # Determine the next rebalancing date
    if rebalancing_freq == 'D':
        rebalance_df[COLUMN_NEXT_REBALANCING_DATE] = \
            rebalance_df[COLUMN_REBALANCING_DATE].shift(-1)
    else:
        rebalance_df[COLUMN_REBALANCING_PERIOD] = \
            rebalance_df[COLUMN_REBALANCING_PERIOD].fillna(method='ffill')

        rebalance_df.loc[
            rebalance_df[COLUMN_IS_REBALANCING_DATE],
            COLUMN_NEXT_REBALANCING_DATE] = rebalance_df[
                rebalance_df[COLUMN_IS_REBALANCING_DATE]][COLUMN_REBALANCING_DATE]

        rebalance_df[COLUMN_NEXT_REBALANCING_DATE] = \
            rebalance_df[COLUMN_NEXT_REBALANCING_DATE].fillna(
                method='bfill').shift(-1)

    # Re-label rebalancing periods
    rebalance_df['__year'] = rebalance_df[COLUMN_DATE].dt.year
    rebalance_df['__cycle_count'] = \
        rebalance_df.groupby('__year')[COLUMN_IS_REBALANCING_DATE].cumsum()

    max_len = len(str(rebalance_df['__cycle_count'].max()))

    rebalance_df.loc[:, COLUMN_REBALANCING_PERIOD] = \
        rebalance_df['__cycle_count'].apply(
            lambda x: rebalancing_freq + str(int(x)).zfill(max_len))

    rebalance_df.loc[:, COLUMN_REBALANCING_PERIOD] = \
        rebalance_df[COLUMN_REBALANCING_PERIOD] + \
        '-' + rebalance_df['__year'].astype(str)

    return df.merge(rebalance_df.drop(columns=['__year', '__cycle_count']), on=COLUMN_DATE)


def drift_portfolio_weights(df: pd.DataFrame,
                            exposure_long: float,
                            exposure_short: float,
                            ) -> pd.DataFrame:
    """
    Calculates the drift-adjusted portfolio weights for the specified DataFrame and exposures.

    Args:
        df (pd.DataFrame): Input DataFrame containing portfolio data.
        exposure_long (float): Exposure for long positions.
        exposure_short (float): Exposure for short positions.

    Returns:
        pd.DataFrame: DataFrame with calculated drift-adjusted portfolio weights.

    Raises:
        AssertionError: Raised if the dates in the DataFrame are not monotonic increasing.

    Notes:
        - If any weights are missing on rebalancing days, a warning is issued.
        - The function performs drift adjustment of weights, if possible (non-daily rebalancing).
            First, it sets the weights to NaN for non-rebalancing days,
            Second, fills NaN values in 'return_current' column with 0, and
            Third, sets the drift factor to 1 on rebalancing days.
        - The drift factor is calculated as the cumulative product of (1 + return_current) within each rebalancing period.
        - The weights are multiplied by the drift factor, and then adjusted based on the specified exposures and the total exposure of each day.
        - Long positions are multiplied by exposure_long divided by the sum of positive weights on each day.
        - Short positions are multiplied by exposure_short divided by the sum of negative weights on each day, and then multiplied by -1.
    """
    assert df[COLUMN_DATE].is_monotonic_increasing, \
        'Dates in the DataFrame are not monotonicaly increasing.'
    df = df.copy()

    if any(df[df[COLUMN_IS_REBALANCING_DATE]][COLUMN_WEIGHTS].isna()):
        logging.warning('Missing some weights on rebalancing day')

    is_possible_to_drift = any(~df[COLUMN_IS_REBALANCING_DATE])

    if is_possible_to_drift:
        df.loc[~df[COLUMN_IS_REBALANCING_DATE], COLUMN_WEIGHTS] = np.nan

        df[COLUMN_WEIGHTS] = \
            df.groupby([COLUMN_REBALANCING_PERIOD, COLUMN_SECURITY_ID
                        ])[COLUMN_WEIGHTS].transform('first')

        df[COLUMN_DRIFT_FACTOR] = df[COLUMN_RETURN_CURRENT].fillna(0) + 1

        df.loc[df[COLUMN_IS_REBALANCING_DATE], COLUMN_DRIFT_FACTOR] = 1

        df[COLUMN_DRIFT_FACTOR] = \
            df.groupby([COLUMN_REBALANCING_PERIOD, COLUMN_SECURITY_ID
                        ])[COLUMN_DRIFT_FACTOR].cumprod()

        df[COLUMN_WEIGHTS] = df[COLUMN_WEIGHTS] * df[COLUMN_DRIFT_FACTOR]

        def calculate_exposure(data: pd.DataFrame) -> pd.Series:
            return data.groupby(COLUMN_DATE)[COLUMN_WEIGHTS].transform('sum')

        long_positions = df[COLUMN_WEIGHTS] > 0
        df.loc[long_positions, COLUMN_WEIGHTS] = (
            df.loc[long_positions, COLUMN_WEIGHTS] * exposure_long /
            calculate_exposure(df.loc[long_positions]))

        short_positions = df[COLUMN_WEIGHTS] < 0
        df.loc[short_positions, COLUMN_WEIGHTS] = (
            df.loc[short_positions, COLUMN_WEIGHTS] * exposure_short /
            calculate_exposure(df.loc[short_positions])) * -1

        df[COLUMN_WEIGHTS] = df[COLUMN_WEIGHTS].fillna(0)
        df.drop(columns=[COLUMN_DRIFT_FACTOR], inplace=True)

    return df

def convert_signals_to_weights(data: pd.DataFrame,
                               exposure_long: float,
                               exposure_short: float) -> pd.Series:
    """
    Convert signals in the DataFrame to weights based on the specified exposures.

    Args:
        data (pd.DataFrame): Input DataFrame containing signals.
        exposure_long (float): Exposure for long positions.
        exposure_short (float): Exposure for short positions.

    Returns:
        pd.Series: Series containing the calculated weights.

    Example:
        >>>     data = pd.DataFrame({
        ...         'selected_signal': [0.5, -0.2, 0.8, -0.4, 0.6, -0.3],
        ...         'is_rebalancing_date': [True, True, True, True, True, True],
        ...         'is_selected_signal': [True, True, True, True, True, True],
        ...         'DATE': ['2010-01-01', '2010-01-01', '2010-01-01', '2010-01-Exampl', '2010-01-02', '2010-01-02']
        ...     })
        >>>     result = convert_signals_to_weights(data, exposure_long=0.5, exposure_short=0.5)
        >>>     print(result)
        0    0.192308
        1   -0.500000
        2    0.307692
        3   -0.285714
        4    0.500000
        5   -0.214286
        dtype: float64
    """
	
    weights = pd.Series(None, index=data.index, dtype=float)
    weights.loc[data[COLUMN_IS_REBALANCING_DATE]] = 0

    mask_long_positions = data[COLUMN_SELECTED_SIGNAL] > 0

    weights.loc[mask_long_positions] = (
        exposure_long *
        data.loc[mask_long_positions, COLUMN_SELECTED_SIGNAL] /
        data[mask_long_positions].groupby([COLUMN_DATE])[
            COLUMN_SELECTED_SIGNAL].transform('sum'))

    mask_short_positions = data[COLUMN_SELECTED_SIGNAL] < 0

    weights.loc[mask_short_positions] = (
        -exposure_short *
        data.loc[mask_short_positions, COLUMN_SELECTED_SIGNAL] /
        data[mask_short_positions].groupby([COLUMN_DATE])[
            COLUMN_SELECTED_SIGNAL].transform('sum'))

    return weights

def round_and_normalize_weights(
    df: pd.DataFrame, 
    exposure_long: float, 
    exposure_short: float,
    column_weights: str = COLUMN_WEIGHTS,
    column_date: str = COLUMN_DATE
) -> pd.DataFrame:
    """
    Sets small weights to zero and rescales all weights so that each day's 
    gross exposure (sum(abs(weights))) matches target gross exposure.
    """
    df = df.copy()
    gross_target = exposure_long + exposure_short

    # Set very small weights to zero
    df.loc[df[column_weights].abs() < 1e-5, column_weights] = 0

    def _rescale(weights: pd.Series) -> pd.Series:
        gross = weights.abs().sum()
        if gross == 0 or np.isclose(gross, gross_target):
            return weights
        weights = weights * (gross_target / gross)
        weights[weights.abs() < 1e-8] = 0
        return weights

    df[column_weights] = df.groupby(column_date)[column_weights].transform(_rescale)
    return df


def construct_longshort_portfolio(df: pd.DataFrame,
								  exposure_long: float,
								  exposure_short: float,
								  cap: float,
								 ) -> pd.DataFrame:
    """
    Constructs a portfolio based on the specified DataFrame, exposures, and cap.

    Args:
        df (pd.DataFrame): Input DataFrame containing portfolio data.
        exposure_long (float): Exposure for long positions.
        exposure_short (float): Exposure for short positions.
        cap (float): Maximum absolute weight allowed for each position.
        If 'cap' is None, it means no cap will be applied

    Returns:
        pd.DataFrame: DataFrame representing the constructed portfolio.

    Notes:
        - The function assumes the DataFrame has columns required by the 'convert_signals_to_weights' and 'drift_portfolio_weights' functions.
        - The 'convert_signals_to_weights' function is used to calculate initial portfolio weights based on the signals in the DataFrame.
        - If the 'cap' parameter is provided, the 'constrain_position_weights' function is called to cap the weights at the specified value.
        - If there are non-rebalancing days in the DataFrame, the 'drift_portfolio_weights' function is called to perform drift adjustment of weights.
        - The DataFrame is returned with the constructed portfolio.

    Constructs a portfolio based on the specified parameters.

    Note:
        - The DataFrame is assumed to be a full grid. If it is not a full grid,
        the weight drift will re-allocate the weight for entities with missing records.
    """
    df = df.copy()

    df[COLUMN_WEIGHTS] = convert_signals_to_weights(
        data=df, exposure_long=exposure_long, exposure_short=exposure_short)

    if cap:
        df = constrain_position_weights(df=df, cap=cap)
	
    # If there are no non-rebalancing days, there is no point to drift.
    is_possible_to_drift = any(~df[COLUMN_IS_REBALANCING_DATE])

    if is_possible_to_drift:
        df = drift_portfolio_weights(
            df=df,
            exposure_long=exposure_long,
            exposure_short=exposure_short)

	# Trim very smal weights and rescale to reach full net portfolio exposure
    df = round_and_normalize_weights(
			df=df, 
			exposure_long=exposure_long, 
			exposure_short=exposure_short,
			column_weights=COLUMN_WEIGHTS,
			column_date=COLUMN_DATE
	)
	
    return df

def evaluate_portfolio_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates the returns and performance metrics of a portfolio based on the provided DataFrame.
    Computes and returns portfolio, long leg, and short leg cumulative log returns and returns.

    Args:
        df (pd.DataFrame): Input DataFrame containing portfolio data. The date column is assumed to be of datetime type.

    Returns:
        pd.DataFrame: DataFrame with evaluated portfolio returns and performance metrics.

    Raises:
        AssertionError: Raised if weekend records are present in the DataFrame, which can bias performance measures.

    Notes:
        - The function does not monitor the presence of non-trading days, which may bias the performance measures.
          For this reason the input DataFrame should not contain any non-trading days.
        - The function assumes the DataFrame has columns required for evaluating portfolio returns and performance metrics.
        - The function calculates the position returns for the next day based on the provided DataFrame.
        - The portfolio return is calculated by summing the position returns for each date.
        - Additional performance metrics such as log returns, cumulative log returns, and cumulative product are calculated.
        - The evaluated portfolio returns and performance metrics are returned as a DataFrame.
        - Depending on the input DataFrame, T1_RETURN may stand for actual return or market-adjusted return.
        - For a given date, T1_RETURN indicates the return on the subsequent day. The return is not actually earned until the next day.
        - To obtain the T0_RETURN as simple shift is sufficient (assuming there are no weekend and holiday dates).
        - T1_RETURN is chosen for practical purposes. This way we do not have a 0 return for the first trading day.
          Similarly, we do not lose the performance information about the last trading day.

    Example:
        >>> df = pd.DataFrame({
        ...     'DATE': ['2023-06-05', '2023-06-06', '2023-06-07', '2023-06-08'],
        ...     'T1_RETURN': [0.02, -0.03, 0.01, 0.005],
        ...     'weights': [0.7, 0.3, -0.5, 0.5]
        ... })
        >>> df['DATE'] = pd.to_datetime(df['DATE'])
        >>> evaluate_portfolio_returns(df)
        # Output:
        #         DATE  T1_RETURN  T1_LOG_RETURN  T1_CUM_LOG_RETURN  T1_CUM_PROD
        # 0 2023-06-05     0.0140       0.013903           0.013903     0.014000
        # 1 2023-06-06    -0.0090      -0.009041           0.004862     0.004874
        # 2 2023-06-07    -0.0050      -0.005013          -0.000150    -0.000150
        # 3 2023-06-08     0.0025       0.002497           0.002346     0.002349
    """
    # The following doesn't account for holidays.
    if not (df[COLUMN_DATE].dt.weekday + 1).max() < 6:
        logging.warning('Weekend days present in Input DataFrame, performance measure will be biased!')
        
    df = df.copy()
    
    df['position_returns_next_day'] = df[COLUMN_RETURN_NEXT_DAY].fillna(0) * df[COLUMN_WEIGHTS]
    
    df['position_returns_long_leg'] = np.where(
        df[COLUMN_WEIGHTS] > 0,
        df[COLUMN_RETURN_NEXT_DAY].fillna(0) * df[COLUMN_WEIGHTS],
        0.0,
    )
    
    df['position_returns_short_leg'] = np.where(
        df[COLUMN_WEIGHTS] < 0,
        df[COLUMN_RETURN_NEXT_DAY].fillna(0) * df[COLUMN_WEIGHTS],
        0.0,
    )
    
    portfolio_return = df.groupby(COLUMN_DATE).agg({
        'position_returns_next_day': 'sum',
        'position_returns_long_leg': 'sum',
        'position_returns_short_leg': 'sum',
    }).reset_index()
    
    portfolio_return.rename(columns={
        'position_returns_next_day': 'T1_RETURN',
        'position_returns_long_leg': 'T1_RETURN_LONG',
        'position_returns_short_leg': 'T1_RETURN_SHORT'
    }, inplace=True)
    
    for col in ['T1_RETURN', 'T1_RETURN_LONG', 'T1_RETURN_SHORT']:
        with np.errstate(divide='ignore', invalid='ignore'):
            log_ret = np.log1p(portfolio_return[col])
        portfolio_return[col.replace('RETURN', 'LOG_RETURN')] = log_ret
        portfolio_return[col.replace('RETURN', 'CUM_LOG_RETURN')] = np.nancumsum(log_ret)
        portfolio_return[col.replace('RETURN', 'CUM_PROD')] = np.cumprod(1 + portfolio_return[col].fillna(0)) - 1
        
    return portfolio_return

def plot_cumulative_logreturns(df: pd.DataFrame, name: str | None = None) -> None:
    """
    Plot cumulative logreturns for the full strategy, long leg, and short leg curves.

    Args:
        df: DataFrame with cumulative log return columns.
        name: If provided, saves the plot to a file with this name (e.g. "my_plot.png").
    """
    req_cols = ['T1_CUM_LOG_RETURN', 'T1_CUM_LOG_RETURN_LONG', 'T1_CUM_LOG_RETURN_SHORT']
    for c in req_cols:
        if c not in df:
            raise ValueError(f"DataFrame missing required column for plotting: {c}")
    plt.figure(figsize=(5, 3))
    colors = ['#000000', '#23a696', '#d44b4b']
    styles = ['-', '-', '-']
    labels = ['Long-Short', 'Long Leg', 'Short Leg']
    for col, color, style, label in zip(req_cols, colors, styles, labels):
        plt.plot(df['DATE'], df[col], linestyle=style, alpha=0.96, color=color, label=label)
    plt.ylabel("Cumulative logreturns")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if name:
        plt.savefig(name, dpi=150, bbox_inches='tight')

def filter_dates(df: pd.DataFrame,
                 start_date: Union[str, pd.Timestamp, None] = None,
                 end_date: Union[str, pd.Timestamp, None] = None
                 ) -> pd.DataFrame:
    """
    Filters a DataFrame based on a specified date range. Both range edges are inclusive.

    Args:
        df (pd.DataFrame): Input DataFrame to be filtered. The date column is assumed to be of datetime type.
        start_date (Union[str, pd.Timestamp, None], optional): Start date of the range.
            If not provided, the minimum possible date (pd.Timestamp.min) will be used.
        end_date (Union[str, pd.Timestamp, None], optional): End date of the range.
            If not provided, the maximum possible date (pd.Timestamp.max) will be used.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only the rows within the specified date range.

    Example:
        >>> df = pd.DataFrame({
        ...     'dates': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04'],
        ...     'values': [1, 2, 3, 4]
        ... })
        >>> filtered_df = filter_dates(df, 'dates', start_date='2022-01-02', end_date='2022-01-03')
        >>> print(filtered_df)

        >>> df = pd.DataFrame({
        ...     'DATE': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04'],
        ...     'values': [1, 2, 3, 4]
        ... })
        >>> filtered_df = filter_dates(df, start_date='2022-01-02', end_date='2022-01-03')
        >>> print(filtered_df)
        # Output:
        #            dates  values
        # 1  2022-01-02       2
        # 2  2022-01-03       3
    """
    df = df.copy()
    if start_date or end_date:
        if not start_date:
            start_date = pd.Timestamp.min
        if not end_date:
            end_date = pd.Timestamp.max
        df = df[df[COLUMN_DATE].between(start_date, end_date)]
    return df

def validate_rebalancing_periods(df: pd.DataFrame,
                                 column_signal: str,
                                 allow_all_missing_signals: bool,
                                 allow_some_missing_signals: bool
                                 ) -> None:
    """
    Validates the rebalancing periods in the DataFrame based on the specified parameters.
    It checks for missing signals on rebalancing days within each rebalancing period.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_signal (str): Name of the column containing the signals.
        allow_all_missing_signals (bool): Whether to allow rebalancing periods with all missing signals.
        allow_some_missing_signals (bool): Whether to allow rebalancing periods with some missing signals.

    Returns:
        None

    Raises:
        AssertionError: Raised if the validation conditions are not met.

    Notes:
        - The allow_all_missing_signals parameter determines whether rebalancing periods with all missing signals are allowed.
        - The allow_some_missing_signals parameter determines whether rebalancing periods with some missing signals are allowed.
        - If allow_all_missing_signals is False, the function raises an AssertionError if any rebalancing period has no signals on its rebalancing day.
        - If allow_some_missing_signals is False, the function raises an AssertionError if any rebalancing period has missing signals on its rebalancing day.
        - The rebalancing periods, dates, and signals are identified based on the specified column names in the DataFrame.

    Example:
        >>> df = pd.DataFrame({
        ...     'rebalancing_period': [1, 1, 2, 2, 3, 3],
        ...     'is_rebalancing_day': [True, True, True, True, True, True],
        ...     'DATE': ['2022-01-01', '2022-01-01', '2022-01-10', '2022-01-10', '2022-01-20', '2022-01-20'],
        ...     'signal': [1.2, 1.5, 0.8, np.nan, 1.1, 1.3]
        ... })

        >>> validate_rebalancing_periods(df, column_signal='signal', allow_all_missing_signals=False, allow_some_missing_signals=False)
        AssertionError: Some rebalancing periods have missing signals on rebalancing day
    """
    if not allow_all_missing_signals:
        assert df[
            df[column_signal].notna() &
            df[COLUMN_IS_REBALANCING_DATE]
        ][COLUMN_DATE].nunique() == df[
            df[COLUMN_IS_REBALANCING_DATE]
        ][COLUMN_DATE].nunique(), \
            'Some rebalancing periods have no signals on rebalancing day'

    if not allow_some_missing_signals:
        missing_some_signals = df[
            df[COLUMN_IS_REBALANCING_DATE]][column_signal].isna().any()
        assert missing_some_signals is False, \
            'Some rebalancing periods have missing signals on rebalancing day'

def calculate_portfolio_turnover(df: pd.DataFrame) -> dict[str, float]:
    """
    Calculate portfolio turnover and effective holding period.

    Args:
        df (pd.DataFrame): DataFrame containing the investment data.

    Returns:
        dict: A dictionary containing the portfolio turnover and effective
        holding period, average numbers for long and short positions, start and dates.

    Notes:
        - To have a more precise estimate of the turnover, we need to know where the weight drifted
        just before the rebalancing. For this reason we need non-adjusted T1_RETURNS.
        - We determine the T1_RETURN from T0_RETURN as T1_RETURN in the input DataFrame may be market
        adjusted returns, while T0_RETURN should be non-adjusted.
        - We obtain T1_RETURN using a full grid. This is an expensive operation, and could be minimized.
        One way is to have the non-adjusted T1_RETURN in the input data.
        - The division by 2 in the turnover calculation is used to have a 100% turnover accounting
        for the fact that each rebalancing involves both a buy and a sell. For example, if the long
        exposure is 100% and on day 1 we have the following portfolio {A: 100%, B: 0%} and the next
        day A is sold, while B is bought we will have {A: 0%, B: 100%}. This is because the portfolio
        is designed to have a 100% exposure. So the sum of absolute differences will be 200%.
    """
    data = df[
        [COLUMN_DATE, COLUMN_SECURITY_ID, COLUMN_WEIGHTS, COLUMN_RETURN_CURRENT
         ]].copy()

    # We require a full grid to obtain non-adjusted T1_RETURN
    is_not_full_grid = len(data) != \
        (data[COLUMN_SECURITY_ID].nunique() * data[COLUMN_DATE].nunique())

    if is_not_full_grid:
        unique_dates = sorted(data[COLUMN_DATE].unique().tolist())
        unique_securities = data[COLUMN_SECURITY_ID].unique().tolist()

        full_grid = pd.DataFrame(product(*[unique_dates, unique_securities]),
                                 columns=[COLUMN_DATE, COLUMN_SECURITY_ID])
        full_grid[COLUMN_DATE] = pd.to_datetime(full_grid[COLUMN_DATE])

        data = full_grid.merge(
            data, on=[COLUMN_DATE, COLUMN_SECURITY_ID], how='left')

    # Sort so dates are monotonic (required for shift logic and assertion)
    data = data.sort_values([COLUMN_DATE, COLUMN_SECURITY_ID]).reset_index(drop=True)
    assert data[COLUMN_DATE].is_monotonic_increasing, \
        'Dates must be monotonicaly increasing for turnover calculations.'

    data['T1_RETURN'] = data.groupby(COLUMN_SECURITY_ID)[
        COLUMN_RETURN_CURRENT].shift(-1)
	
    # Calculate next day weights due to drift, assuming no rebalancing
    data['T1_WEIGHT'] = \
        data[COLUMN_WEIGHTS] * (1 + data['T1_RETURN'].fillna(0))

    def calculate_total_by_dates(
            data: pd.DataFrame, column_name: str) -> pd.Series:
        return data.groupby(COLUMN_DATE)[column_name].transform('sum')

    # Drift one day
    long_positions = data[COLUMN_WEIGHTS] > 0
    short_positions = data[COLUMN_WEIGHTS] < 0

    for position_types in [long_positions, short_positions]:
        daily_position_exposures = \
            calculate_total_by_dates(data.loc[position_types], COLUMN_WEIGHTS)
        scaled_next_day_weights = (
            data.loc[position_types, 'T1_WEIGHT'] /
            calculate_total_by_dates(data.loc[position_types], 'T1_WEIGHT'))

        data.loc[position_types, 'T1_WEIGHT'] = \
            scaled_next_day_weights * daily_position_exposures

    data['T0_WEIGHT_BEFORE_REBALANCING']  = data.groupby(
        COLUMN_SECURITY_ID)['T1_WEIGHT'].shift(1).fillna(0)

    data['DIFF_WEIGHT'] = np.abs(data[COLUMN_WEIGHTS] - data['T0_WEIGHT_BEFORE_REBALANCING'] )
    daily_annualized_portfolio_turnover = \
        1 / data[COLUMN_DATE].nunique() * data['DIFF_WEIGHT'].sum() / 2

    effective_holding_period = 1 / daily_annualized_portfolio_turnover

    # Count number of positions when full gross portfolio exposure is satisfied.
    n_short_positions = data[data.weights!=0].assign(
        N_SHORT_POSITIONS=data[data.weights!=0].weights < 0
        ).groupby('DATE').N_SHORT_POSITIONS.sum().mean()
    n_long_positions = data[data.weights!=0].assign(
        N_LONG_POSITIONS=data[data.weights!=0].weights > 0
        ).groupby('DATE').N_LONG_POSITIONS.sum().mean()
    n_positions = data[data.weights!=0].assign(
        N_POSITIONS=data[data.weights!=0].weights != 0
        ).groupby('DATE').N_POSITIONS.sum().mean()

    return {
	    'TURNOVER': round(daily_annualized_portfolio_turnover, 2),
	    'EHP': round(effective_holding_period, 2),
	    'N_SHORT': int(round(n_short_positions)),
	    'N_LONG': int(round(n_long_positions)),
	    'SIZE': int(round(n_positions))
		}

def cross_sectional_standardization(df: pd.DataFrame,
                                    column_signal: str
                                    ) -> pd.DataFrame:
    """
    Standardize the daily cross-section of signals.

    Args:
        df (pd.DataFrame): Input DataFrame containing signal data. The date column is assumed to be of datetime type.

    Returns:
	    df (pd.DataFrame): DataFrame containing the transformed signal data.
    """
    df[column_signal] = (df[column_signal] - df[column_signal].mean()) / \
        df[column_signal].std()
	
    return df


def evaluate_performance(returns: pd.Series) -> dict[str, float]:
    """
    Evaluates the performance of a portfolio based on the provided returns.

    Args:
        returns (pd.Series): Series of returns, with dates in the index.

    Returns:
        dict[str, float]: Dictionary containing performance metrics.

    Notes:
        - Alternative measures could be used here.

    Example:
        >>> returns = pd.Series([0.05, 0.03, -0.02, 0.04], name='returns',
        ...                     index=pd.date_range('2001-01-01', freq='D', periods=4))
        >>> evaluate_performance(returns)

        # Output:
        # {'AR': 6.134119251544516, 'STD': 0.4876114479628968, 'IR': 12.5799}
    """

    annualized_cumulative_log_return = (1 + returns).map(np.log).mean() * 252
    annualized_return_std = (1 + returns).map(np.log).std() * np.sqrt(252)

    information_ratio = calculate_information_ratio(
        annualized_cumulative_log_return, annualized_return_std)

    max_drawdown = calculate_max_drawdown(returns)
	
    performance = {'AR': annualized_cumulative_log_return,
                   'STD': annualized_return_std,
                   'IR': information_ratio,
				   'MDD': max_drawdown}

    return performance

def calculate_information_ratio(annualized_return_mean: float,
                                annualized_return_std: float) -> float:
    """
    Calculates the information ratio based on the annualized return mean and annualized return standard deviation.

    Args:
        annualized_return_mean (float): Annualized return mean.
        annualized_return_std (float): Annualized return standard deviation.

    Returns:
        float: Information ratio.

    Example:
        >>> calculate_information_ratio(0.08, 0.05)
        1.6
    """
    if annualized_return_std is None or np.isclose(annualized_return_std, 0.0):
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        information_ratio = annualized_return_mean / annualized_return_std
    return round(float(information_ratio) if np.isfinite(information_ratio) else 0.0, 4)
	
def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculates the max drawdown as the maximum historical loss.

    Args:
        returns (pd.Series): Arithmetic returns.

    Returns:
        float: Max Drawdown.
    """
    drawdown = np.log(1 + returns).cumsum().cummax() - np.log(1 + returns).cumsum()
    max_drawdown = drawdown.max()
    return round(max_drawdown, 4)

def portfolio_backtesting_pipeline(
        df_signal_and_returns: pd.DataFrame,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        rebalancing_frequency: str,
        exposure_long: float,
        exposure_short: float,
        cap: float,
        column_signal: str,
        daily_standardization: bool,
        plot_performance: str | None = None,
        allow_all_missing_signals: bool = DEFAULT_ALLOW_ALL_MISSING_SIGNALS,
        allow_some_missing_signals: bool = DEFAULT_ALLOW_SOME_MISSING_SIGNALS,
        date_offset_params: dict = dict(),
        validate_rebalancing_period_signals: bool = False,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a long-short portfolio backtesting using a pipeline approach.

    Args:
        df_signal_and_returns (pd.DataFrame): DataFrame containing the signal and return data.
            The DataFrame contains a set of expected columns. Column names are defined as constants in the module.
        start_date (Union[str, pd.Timestamp]): Start date of the backtesting period.
        end_date (Union[str, pd.Timestamp]): End date of the backtesting period.
        rebalancing_frequency (str): Frequency of rebalancing (e.g., 'MS' for monthly, 'QS' for quarterly).
            For more aliases check https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        exposure_long (float): Long exposure value for the strategy.
        exposure_short (float): Short exposure value for the strategy.
        cap (float): Cap for the maximum allocation to one position. If cap equals 0, no cap is applied.
        column_signal (str): Column name for the signal in the input DataFrame.
		daily_standardization (bool): Boolean to perform daily cross-sectional standardization on signal.
        allow_all_missing_signals (bool, optional): Flag to allow that rebalancing days could have all missing signals.
            Defaults to value in DEFAULT_ALLOW_ALL_MISSING_SIGNALS.
        allow_some_missing_signals (bool, optional): Flag to allow the rebalancing days to have some missing signals.
            Defaults to value in DEFAULT_ALLOW_SOME_MISSING_SIGNALS.
        date_offset_params (dict, optional): Additional parameters for calculating rebalancing periods. Parameters in this
            dictionary are used with pd.DateOffset. Defaults to an empty dictionary.
        validate_rebalancing_period_signals (bool, optional): Flag to either use the validate_rebalancing_periods function or not.
            Defaults to False.

    Returns:
        tuple[pd.Series, pd.DataFrame, pd.DataFrame]: A tuple containing the portfolio returns,
        the modified DataFrame with portfolio weights, and performance metrics with turnover information.

    Notes:
        - The input DataFrame should contain only the signals that the user wants to trade on. That is the pipeline does
        not implement any signal selection.
        - Note the input DataFrame should be prepared in a way not contain weekends or holidays.
        - The procedure ensures input DataFrame contains signals for a grid of entities at each date.
    """

    # Check columns in the input DataFrame for consistency
    columns_expected_in_df = [
        column_signal,
        COLUMN_DATE,
        COLUMN_SECURITY_ID,
        COLUMN_RETURN_CURRENT,
        COLUMN_RETURN_NEXT_DAY
    ]
    for column in columns_expected_in_df:
        assert column in df_signal_and_returns, \
            f'Column "{column}" not in DataFrame'
    columns_internal_to_backtesting = [
        COLUMN_DRIFT_FACTOR,
        COLUMN_DRIFTING_WEIGHTS,
        COLUMN_IS_REBALANCING_DATE,
        COLUMN_IS_SELECTED_SIGNAL,
        COLUMN_NEXT_REBALANCING_DATE,
        COLUMN_REBALANCING_DATE,
        COLUMN_REBALANCING_PERIOD,
        COLUMN_RETURNS_PORTFOLIO,
        COLUMN_SCALED_DRIFTING_WEIGHTS,
        COLUMN_SELECTED_SIGNAL,
        COLUMN_WEIGHTS
    ]
    for column in columns_internal_to_backtesting:
        assert column not in df_signal_and_returns, \
            f'Column "{column}" already exists in DataFrame'
        
    assert not df_signal_and_returns.empty, \
        'Input DataFrame is empty'
    
    df_signal_and_returns = df_signal_and_returns.copy()
    
    df_signal_and_returns[COLUMN_DATE] = \
        pd.to_datetime(df_signal_and_returns[COLUMN_DATE])

	# Ensure the input DataFrame is a full grid of securities for each date   
    is_not_full_grid = len(df_signal_and_returns) != \
        (df_signal_and_returns[COLUMN_SECURITY_ID].nunique() * \
         df_signal_and_returns[COLUMN_DATE].nunique())
    
    if is_not_full_grid:
        unique_dates = df_signal_and_returns[COLUMN_DATE].unique().tolist()
        unique_securities = df_signal_and_returns[COLUMN_SECURITY_ID].unique().tolist()
		
        full_grid = pd.DataFrame(product(*[unique_dates, unique_securities]),
                                 columns=[COLUMN_DATE, COLUMN_SECURITY_ID])
		
        full_grid[COLUMN_DATE] = pd.to_datetime(full_grid[COLUMN_DATE])
		
        df_signal_and_returns = full_grid.merge(
            df_signal_and_returns, 
			on=[COLUMN_DATE, COLUMN_SECURITY_ID], 
			how='left')
    
    # Apply, if asked, daily cross-sectional standardization to signals
    if daily_standardization:	
        df_signal_and_returns = \
            df_signal_and_returns.groupby(COLUMN_DATE, group_keys=False). \
            apply(cross_sectional_standardization, column_signal=column_signal)
        
    if rebalancing_frequency == 'D':
        df_signal_and_returns[COLUMN_IS_REBALANCING_DATE] = True
        effective_start_date = start_date
        effective_end_date = end_date
        df_signal_and_returns = filter_dates(
            df_signal_and_returns,
            start_date=effective_start_date,
            end_date=effective_end_date).reset_index(drop=True)
    else:
        # Create the rebalancing period ranges between first and last days.
        rebalancing_period_start_date = start_date
        rebalancing_period_end_date = end_date

        # The parameter expand_ranges makes sure that the start_date and end_date
        # are within rebalancing periods. That is the rebalancing_period_ranges
        # may contain dates that are outside [start_date, end_date] range.
        rebalancing_period_ranges = get_rebalancing_period_ranges(
            start_date=rebalancing_period_start_date,
            end_date=rebalancing_period_end_date,
            rebalancing_freq=rebalancing_frequency,
            expand_ranges=True,
            **date_offset_params)
        
        effective_start_date = rebalancing_period_ranges[0]
        effective_end_date = rebalancing_period_ranges[-1]

        # Keep only data that is within selected rebalancing periods        
        df_signal_and_returns = filter_dates(
            df_signal_and_returns,
            start_date=effective_start_date,
            end_date=effective_end_date).reset_index(drop=True)

        # Assign records to rebalancing periods and indicate rebalancing days.
        df_signal_and_returns = assign_rebalancing_periods(
            df=df_signal_and_returns,
            rebalancing_period_ranges=rebalancing_period_ranges,
            rebalancing_freq=rebalancing_frequency)

    # Check whether there are rebalancing days that miss all or some signals
    if validate_rebalancing_period_signals:
        validate_rebalancing_periods(
            df=df_signal_and_returns,
            column_signal=column_signal,
            allow_all_missing_signals=allow_all_missing_signals,
            allow_some_missing_signals=allow_some_missing_signals)
        
    if rebalancing_frequency == 'D':
        # Daily rebalancing selects all signals to construct portfolios
        df_signal_and_returns[COLUMN_SELECTED_SIGNAL] = \
            df_signal_and_returns[column_signal]
    else:
        # Other rebalancing frequencies use only signals on rebalancing days to construct portfolios
        df_signal_and_returns[COLUMN_SELECTED_SIGNAL] = \
            df_signal_and_returns.loc[
                (df_signal_and_returns[COLUMN_IS_REBALANCING_DATE]) &
                (df_signal_and_returns[column_signal].notna()),
                column_signal]

    # Construct the portfolio using selected signals
    df_portfolio = construct_longshort_portfolio(
        df=df_signal_and_returns,
        exposure_long=exposure_long,
        exposure_short=exposure_short,
        cap=cap)

    # Trim the constructed portfolio data to desired start and end dates.
    # This is needed, as the effective start and end dates may be different.
    # For example, with start_date, end_date = '2010-01-01', '2010-12-31' and
    # quarterly rebalancing starting from February 1st, we need to have the
    # effective start date on 1st of November 2019, and effective end date on
    # 1st of February 2011. This way we construct portfolios for all days between
    # '2010-01-01' and '2010-12-31'.
    df_portfolio = df_portfolio[
        df_portfolio[COLUMN_DATE].between(
            start_date, end_date)].reset_index(drop=True)
    
    portfolio_returns = evaluate_portfolio_returns(df=df_portfolio)

    # Ensure portfolio constraints are satisfied.
    # Max allocation constraint must be satisfied at rebalancing dates only due to weights drift: 
    # this condition is always satisfied leading to underallocation when
    # condition is unfeasible.
    # Gross exposure constraint must be satisfied at every date.
    # WARNING: The breach flags is a working-progress and is not important.
    
    gross_exposure_breached = \
        (df_portfolio.groupby([COLUMN_DATE])[COLUMN_WEIGHTS]. \
                transform('sum').round(5) != exposure_long - exposure_short).any()
    
    if gross_exposure_breached:
        logging.warning('Gross portfolio exposure is not satisfied at every day!')
        
    mask_rebalancing_dates = df_portfolio[COLUMN_IS_REBALANCING_DATE] 
    
    mask_long_positions = df_portfolio[COLUMN_WEIGHTS] >= 0
    exposure_long_breached = \
        (df_portfolio[mask_long_positions & mask_rebalancing_dates]. \
                 groupby([COLUMN_DATE])[COLUMN_WEIGHTS]. \
                     transform('sum').round(5) != exposure_long).any()
    
    mask_short_positions = df_portfolio[COLUMN_WEIGHTS] <= 0
    exposure_short_breached = \
        (df_portfolio[mask_short_positions & mask_rebalancing_dates]. \
                 groupby([COLUMN_DATE])[COLUMN_WEIGHTS]. \
                     transform('sum').round(5) != - exposure_short).any()
    if exposure_long_breached \
        or exposure_short_breached:
             logging.warning('Long and/or short portfolio exposures are not satisfied at every rebalancing date!')

    # Evaluate portfolio performance
    performance = evaluate_performance(returns=portfolio_returns['T1_RETURN'])
    turnover = calculate_portfolio_turnover(df=df_portfolio)
    performance.update(turnover)
    portfolio_stats = pd.DataFrame.from_dict(performance, orient='index').T
    
    if plot_performance:
        plot_cumulative_logreturns(portfolio_returns, name=plot_performance)
        
    return portfolio_returns, df_portfolio, portfolio_stats

	