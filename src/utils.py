import pandas as pd
import numpy as np
import itertools
import random

from indicators import calculate_atr, calculate_bollinger_bands, calculate_macd, calculate_rsi, calculate_stochastic_oscillator

# Define the dtype for the structured array
param_dtype = np.dtype([
    ('starting_balance', np.float64),
    ('rsi_threshold_low', np.float64),
    ('rsi_threshold_high', np.float64),
    ('risk_per_trade', np.float64),
    ('stop_loss', np.float64),
    ('take_profit', np.float64),
    ('trailing_stop_loss', np.float64),
    ('trailing_take_profit', np.float64),
    ('threshold', np.float64),
    ('broker_fee', np.float64),
    ('slippage', np.float64),
    ('hold_time_limit', np.int64),  # Use -1 for None
    ('stoch_k_period', np.int64),
    ('stoch_d_period', np.int64),
    ('stoch_threshold_low', np.float64),
    ('stoch_threshold_high', np.float64),
    ('atr_period', np.int64),  # Add atr_period here
    ('support_resistance_timeframe', 'U10'),  # Added support_resistance_timeframe
    ('support_resistance_window', np.int64)  # Added support_resistance_window
])

def resample_data(data, timeframe='1min'):
    resampled = data.resample(timeframe).agg({
        'open': 'first',     # First open price
        'high': 'max',       # Highest price
        'low': 'min',        # Lowest low price
        'close': 'last',     # Last close price
        'volume': 'sum',     # Sum of volumes
        'trades': 'sum'      # Sum of trades
    })
    # Drop rows with any missing values caused by resampling
    resampled.dropna(inplace=True)
    return resampled

def get_random_slices(data, num_slices=5, slice_length=3000):
    if len(data) < slice_length:
        slice_length = len(data)  # Adjust slice length to fit the data
    random_slices = []
    max_start = len(data) - slice_length
    for _ in range(num_slices):
        start = random.randint(0, max_start)
        random_slices.append(data.iloc[start:start + slice_length])
    return random_slices

def generate_param_combinations(param_grid):
    keys, values = zip(*param_grid.items())
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))

def create_numba_params(params):
    hold_time_limit = params.get('hold_time_limit')
    if hold_time_limit is None:
        hold_time_limit = -1
    atr_period = params.get('atr_period', 14)
    support_resistance_timeframe = params.get('support_resistance_timeframe', '1H')
    support_resistance_window = params.get('support_resistance_window', 30)
    return np.array([(
        params.get('starting_balance', 250),
        params['rsi_threshold_low'],
        params['rsi_threshold_high'],
        params['risk_per_trade'],
        params['stop_loss'],
        params['take_profit'],
        params['trailing_stop_loss'],
        params['trailing_take_profit'],
        params['threshold'],
        params.get('broker_fee', 0.0005),
        params.get('slippage', 0.0002),
        hold_time_limit,
        params['stoch_k_period'],
        params['stoch_d_period'],
        params['stoch_threshold_low'],
        params['stoch_threshold_high'],
        atr_period,
        support_resistance_timeframe.encode('utf-8')[:10].decode('utf-8'),  # Ensure max length of 10
        support_resistance_window
    )], dtype=param_dtype)[0]

def precompute_data(
    data, 
    sma_periods_short, sma_periods_long,
    ema_periods,
    rsi_periods,
    bb_periods, bb_num_std_values,
    macd_fast_periods, macd_slow_periods, macd_signal_periods,
    stoch_k_periods, stoch_d_periods,
    atr_periods,
    support_resistance_timeframes, support_resistance_windows
):
    """
    Precompute all necessary indicators based on the provided parameter ranges.
    """
    data_dict = {}
    # Precompute indicators for multiple timeframes if necessary
    indicator_periods = {
        'sma_short': sma_periods_short,
        'sma_long': sma_periods_long,
        'ema_period': ema_periods,
        'rsi_period': rsi_periods,
        'bb_period': bb_periods,
        'bb_num_std': bb_num_std_values,
        'macd_fast': macd_fast_periods,
        'macd_slow': macd_slow_periods,
        'macd_signal': macd_signal_periods,
        'stoch_k_period': stoch_k_periods,
        'stoch_d_period': stoch_d_periods,
        'atr_period': atr_periods
    }

    resampled_data = resample_data(data, timeframe='1h')  # Using a fixed timeframe for support/resistance

    # Compute Stochastic Oscillator for all combinations
    for k_period in stoch_k_periods:
        for d_period in stoch_d_periods:
            k_values, d_values = calculate_stochastic_oscillator(
                resampled_data['high'], 
                resampled_data['low'], 
                resampled_data['close'], 
                k_period, 
                d_period
            )
            resampled_data[f'stoch_k_{k_period}_{d_period}'] = k_values
            resampled_data[f'stoch_d_{k_period}_{d_period}'] = d_values

    # Compute SMAs
    for sma_short in sma_periods_short:
        for sma_long in sma_periods_long:
            if sma_short >= sma_long:
                continue  # Avoid invalid SMA combinations
            resampled_data[f"sma_{sma_short}"] = resampled_data['close'].rolling(window=sma_short).mean()
            resampled_data[f"sma_{sma_long}"] = resampled_data['close'].rolling(window=sma_long).mean()

    # Compute EMAs
    for ema_period in ema_periods:
        resampled_data[f'ema_{ema_period}'] = resampled_data['close'].ewm(span=ema_period, adjust=False).mean()

    # Compute RSI
    for rsi_period in rsi_periods:
        resampled_data[f'rsi_{rsi_period}'] = calculate_rsi(resampled_data['close'], period=rsi_period)

    # Compute Bollinger Bands
    for bb_period in bb_periods:
        for bb_num_std in bb_num_std_values:
            upper_bb, lower_bb = calculate_bollinger_bands(
                resampled_data['close'], period=bb_period, num_std_dev=bb_num_std
            )
            resampled_data[f'upper_bb_{bb_period}_{bb_num_std:.1f}'] = upper_bb
            resampled_data[f'lower_bb_{bb_period}_{bb_num_std:.1f}'] = lower_bb

    # Compute MACD
    for macd_fast in macd_fast_periods:
        for macd_slow in macd_slow_periods:
            if macd_fast >= macd_slow:
                continue  # Avoid invalid MACD combinations
            for macd_signal in macd_signal_periods:
                macd_line, signal_line, macd_hist = calculate_macd(
                    resampled_data['close'], 
                    fast_period=macd_fast, 
                    slow_period=macd_slow, 
                    signal_period=macd_signal
                )
                resampled_data[f'macd_line_{macd_fast}_{macd_slow}_{macd_signal}'] = macd_line
                resampled_data[f'signal_line_{macd_fast}_{macd_slow}_{macd_signal}'] = signal_line
                resampled_data[f'macd_hist_{macd_fast}_{macd_slow}_{macd_signal}'] = macd_hist

    # Compute ATR
    for atr_period in atr_periods:
        resampled_data[f'atr_{atr_period}'] = calculate_atr(resampled_data, period=atr_period)

    # Drop NaNs after all computations
    resampled_data.dropna(inplace=True)
    data_dict['test'] = resampled_data

    # Compute Support and Resistance for all combinations
    for timeframe in support_resistance_timeframes:
        for window in support_resistance_windows:
            support_levels = resampled_data['low'].rolling(window=window).min().reindex(data.index, method='ffill').values
            resistance_levels = resampled_data['high'].rolling(window=window).max().reindex(data.index, method='ffill').values

            # Handle NaN values by setting to the overall min and max
            min_low = data['low'].min()
            max_high = data['high'].max()
            support_levels = np.where(np.isnan(support_levels), min_low, support_levels)
            resistance_levels = np.where(np.isnan(resistance_levels), max_high, resistance_levels)

            # Encode timeframe to bytes and truncate to 10 characters if necessary
            encoded_timeframe = timeframe.encode('utf-8')[:10].decode('utf-8')

            data_dict[f'support_levels_{encoded_timeframe}_{window}'] = support_levels
            data_dict[f'resistance_levels_{encoded_timeframe}_{window}'] = resistance_levels

    return data_dict

def collect_all_indicator_params():
    """
    Collect all possible indicator parameters based on the Optuna search space.
    This ensures that all necessary indicators are precomputed.
    """
    indicator_params = {
        'sma_periods_short': list(range(5, 55, 5)),      # 5 to 50 inclusive, step 5
        'sma_periods_long': list(range(50, 210, 10)),    # 50 to 200 inclusive, step 10
        'ema_periods': list(range(10, 55, 5)),           # 10 to 50 inclusive, step 5
        'rsi_periods': list(range(7, 36, 7)),            # 7 to 35 inclusive, step 7
        'bb_periods': list(range(20, 51, 10)),           # 20 to 50 inclusive, step 10
        'bb_num_std_values': [1.5, 2.0, 2.5],
        'macd_fast_periods': list(range(5, 13)),        # 5 to 12 inclusive
        'macd_slow_periods': list(range(20, 31)),        # 20 to 30 inclusive
        'macd_signal_periods': list(range(5, 16, 5)),    # 5 to 15 inclusive, step 5
        'stoch_k_periods': list(range(7, 22, 7)),        # 7 to 21 inclusive, step 7
        'stoch_d_periods': list(range(3, 10, 3)),        # 3 to 9 inclusive, step 3
        'atr_periods': list(range(7, 22)),               # 7 to 21 inclusive
        'support_resistance_timeframes': ['30m', '1h', '4h', '1d'],
        'support_resistance_windows': list(range(10, 201, 10))  # 10 to 200 inclusive, step 10
    }
    return indicator_params
