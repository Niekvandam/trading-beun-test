import numpy as np
import pandas as pd

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(series, period=20, num_std_dev=2):
    period = int(period)
    rolling_mean = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band


def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def calculate_stochastic_oscillator(high, low, close, k_period=14, d_period=3):
    if not isinstance(k_period, int):
        k_period = int(k_period)
    if not isinstance(d_period, int):
        d_period = int(d_period)
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_values = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_values = k_values.rolling(window=d_period).mean()
    return k_values, d_values


def calculate_atr(data, period=14):
    high = data['high']
    low = data['low']
    close = data['close']

    # True Range (TR) calculation
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR calculation
    atr = tr.rolling(window=period).mean()
    return atr

def collect_indicator_params_from_params(params):
    # Extract unique indicator parameters from the current params
    sma_periods_set = set(params['sma_periods'])
    rsi_periods_unique = [params['rsi_period']]
    bb_periods_unique = [params['bb_period']]
    bb_num_std_unique = [params['bb_num_std']]
    macd_fast_unique = [params['macd_fast']]
    macd_slow_unique = [params['macd_slow']]
    macd_signal_unique = [params['macd_signal']]
    ema_periods_unique = [params['ema_period']]
    atr_periods_unique = [14]  # Assuming ATR period is fixed
    
    return {
        'sma_periods': list(sma_periods_set),
        'rsi_periods': rsi_periods_unique,
        'bb_periods': bb_periods_unique,
        'bb_num_std': bb_num_std_unique,
        'macd_fast': macd_fast_unique,
        'macd_slow': macd_slow_unique,
        'macd_signal': macd_signal_unique,
        'ema_periods': ema_periods_unique,
        'atr_periods': atr_periods_unique
    }
