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
    ('threshold', np.int64),
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



def precompute_data(data, timeframes, indicator_params, support_resistance_timeframe, support_resistance_window):
    data_dict = {}
    # Precompute indicators for main timeframes
    for timeframe in timeframes:
        resampled_data = resample_data(data, timeframe=timeframe)
        resampled_data.dropna(inplace=True)  # Ensure no NaNs in the final dataset

        # Compute Stochastic Oscillator
        for k_period in indicator_params['stoch_k_periods']:
            for d_period in indicator_params['stoch_d_periods']:
                k_values, d_values = calculate_stochastic_oscillator(
                    resampled_data['high'], resampled_data['low'], resampled_data['close'], k_period, d_period
                )
                resampled_data[f'stoch_k_{k_period}_{d_period}'] = k_values
                resampled_data[f'stoch_d_{k_period}_{d_period}'] = d_values

        # Compute SMAs
        resampled_data[f"sma_{indicator_params['sma_short']}"] = resampled_data['close'].rolling(window=indicator_params['sma_short']).mean()
        resampled_data[f"sma_{indicator_params['sma_long']}"] = resampled_data['close'].rolling(window=indicator_params['sma_long']).mean()

        # Compute EMAs
        for period in indicator_params['ema_periods']:
            resampled_data[f'ema_{period}'] = resampled_data['close'].ewm(span=period, adjust=False).mean()

        # Compute RSI
        for period in indicator_params['rsi_periods']:
            resampled_data[f'rsi_{period}'] = calculate_rsi(resampled_data['close'], period=period)

        # Compute Bollinger Bands
        for period in indicator_params['bb_periods']:
            for num_std in indicator_params['bb_num_std']:
                num_std_str = f"{num_std:.1f}"
                upper_bb, lower_bb = calculate_bollinger_bands(
                    resampled_data['close'], period=period, num_std_dev=num_std
                )
                resampled_data[f'upper_bb_{period}_{num_std_str}'] = upper_bb
                resampled_data[f'lower_bb_{period}_{num_std_str}'] = lower_bb

        # Compute MACD
        for fast in indicator_params['macd_fast']:
            for slow in indicator_params['macd_slow']:
                for signal in indicator_params['macd_signal']:
                    macd_line, signal_line, macd_hist = calculate_macd(
                        resampled_data['close'], fast, slow, signal)
                    resampled_data[f'macd_line_{fast}_{slow}_{signal}'] = macd_line
                    resampled_data[f'signal_line_{fast}_{slow}_{signal}'] = signal_line
                    resampled_data[f'macd_hist_{fast}_{slow}_{signal}'] = macd_hist

        # Compute ATR
        for period in indicator_params['atr_periods']:
            resampled_data[f'atr_{period}'] = calculate_atr(resampled_data, period=period)

        # Drop NaNs
        resampled_data.dropna(inplace=True)
        data_dict[timeframe] = resampled_data

    # Precompute Support and Resistance based on different timeframe and window
    resampled_support = resample_data(data, timeframe=support_resistance_timeframe)
    resampled_support.dropna(inplace=True)

    # Calculate Support and Resistance using a rolling window
    support_levels = resampled_support['low'].rolling(window=support_resistance_window).min().reindex(data.index, method='ffill').values
    resistance_levels = resampled_support['high'].rolling(window=support_resistance_window).max().reindex(data.index, method='ffill').values

    # Handle NaN values by setting to the overall min and max
    min_low = data['low'].min()
    max_high = data['high'].max()
    support_levels = np.where(np.isnan(support_levels), min_low, support_levels)
    resistance_levels = np.where(np.isnan(resistance_levels), max_high, resistance_levels)

    data_dict['support_levels'] = support_levels
    data_dict['resistance_levels'] = resistance_levels

    return data_dict



def collect_indicator_params(param_grid):
    # Collect unique SMA periods
    sma_periods_set = set()
    for periods in param_grid.get('sma_periods', []):
        if periods is not None:  # Filter out None
            sma_periods_set.update(periods)
    sma_periods_unique = list(sma_periods_set)

    # Collect other unique indicator parameters
    rsi_periods_unique = list(set(param_grid.get('rsi_period', [])))
    bb_periods_unique = list(set(param_grid.get('bb_period', [])))
    bb_num_std_unique = list(set(param_grid.get('bb_num_std', [])))
    macd_fast_unique = list(set(param_grid.get('macd_fast', [])))
    macd_slow_unique = list(set(param_grid.get('macd_slow', [])))
    macd_signal_unique = list(set(param_grid.get('macd_signal', [])))
    ema_periods_unique = list(set(param_grid.get('ema_period', [])))
    atr_periods_unique = list(set(param_grid.get('atr_period', [14])))  # Assuming ATR period is fixed

    return {
        'sma_periods': sma_periods_unique,
        'rsi_periods': rsi_periods_unique,
        'bb_periods': bb_periods_unique,
        'bb_num_std': bb_num_std_unique,
        'macd_fast': macd_fast_unique,
        'macd_slow': macd_slow_unique,
        'macd_signal': macd_signal_unique,
        'ema_periods': ema_periods_unique,
        'atr_periods': atr_periods_unique
    }

def collect_indicator_params_from_params(params):
    # Handle Stochastic Oscillator parameters
    stoch_k_periods_unique = [params.get('stoch_k_period', 14)]  # Default to 14
    stoch_d_periods_unique = [params.get('stoch_d_period', 3)]   # Default to 3
    # Collect SMA and EMA periods
    sma_short = params.get('sma_short')
    sma_long = params.get('sma_long')
    ema_periods_unique = [params['ema_period']]

    # Extract other indicator parameters
    rsi_periods_unique = [params['rsi_period']]
    bb_periods_unique = [params['bb_period']]
    bb_num_std_unique = [params['bb_num_std']]
    macd_fast_unique = [params['macd_fast']]
    macd_slow_unique = [params['macd_slow']]
    macd_signal_unique = [params['macd_signal']]
    atr_periods_unique = [params.get('atr_period', 14)]  # Include ATR period

    return {
        'stoch_k_periods': stoch_k_periods_unique,
        'stoch_d_periods': stoch_d_periods_unique,
        'sma_short': sma_short,
        'sma_long': sma_long,
        'rsi_periods': rsi_periods_unique,
        'bb_periods': bb_periods_unique,
        'bb_num_std': bb_num_std_unique,
        'macd_fast': macd_fast_unique,
        'macd_slow': macd_slow_unique,
        'macd_signal': macd_signal_unique,
        'ema_periods': ema_periods_unique,
        'atr_periods': atr_periods_unique
    }



def adjust_indicator_periods(params, timeframe):
    timeframe_multipliers = {
        '1min': 1,
        '5min': 5,
        '15min': 15,
        '30min': 30,
        '1h': 60,
        '2h': 120,
    }
    multiplier = timeframe_multipliers[timeframe]
    adjusted_params = params.copy()
    adjusted_params['sma_periods'] = (
        int(params['sma_periods'][0] / multiplier),
        int(params['sma_periods'][1] / multiplier)
    )
    # Adjust other periods similarly
    return adjusted_params

def get_time_series_folds(data, k):
    fold_size = len(data) // k
    remainder = len(data) % k
    folds = []
    start = 0
    for i in range(k):
        extra = 1 if i < remainder else 0
        end = start + fold_size + extra
        folds.append(data.iloc[start:end])
        start = end
    return folds
