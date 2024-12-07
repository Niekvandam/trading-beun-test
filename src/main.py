import pandas as pd
from indicators import (
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_atr
)
from utils import (
    resample_data,
    precompute_data,
    collect_indicator_params
)
from backtest import backtest_strategy

# Load and preprocess data
data = pd.read_csv('E:\\programming\\test-trader\\src\\data\\btcusd.csv')
# Convert 'timestamp' to datetime and set as index
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
data.set_index('timestamp', inplace=True)

# Sort data by index to ensure chronological order
data.sort_index(inplace=True)

# Define parameter grid
param_grid = {
    'starting_balance': [250],  # Fixed to maintain consistency
    'sma_periods': [(50, 200), (20, 100)],  # Commonly effective SMA pairs
    'rsi_period': [14, 21],  # Use only the standard 14-period RSI for stability
    'rsi_threshold': [(30, 70), (20, 80)],  # Retain thresholds with most impact
    'bb_period': [20],  # Standard Bollinger Band period
    'bb_num_std': [2],  # Use 2 standard deviations for classic BB strategy
    'macd_fast': [12],  # Standard MACD fast period
    'macd_slow': [26],  # Standard MACD slow period
    'macd_signal': [9],  # Standard MACD signal period
    'ema_period': [21, 50],  # Common EMA periods
    'risk_per_trade': [0.02, 0.03],  # Moderate risk levels
    'stop_loss': [0.01, 0.02],  # Conservative stop-loss levels
    'take_profit': [0.03, 0.05],  # Realistic take-profit targets
    'trailing_stop_loss': [0.02, 0.03, 0.05],  # Conservative trailing stop-loss
    'trailing_take_profit': [0.02, 0.03, 0.05],  # Conservative trailing take-profit
    'weights': [
        {'sma': 2, 'rsi': 1, 'bb': 1, 'macd': 1, 'ema': 1},
        {'sma': 2, 'rsi': 1, 'bb': 1, 'macd': 1, 'ema': 1},
        {'sma': 2, 'rsi': 1, 'bb': 1, 'macd': 1, 'ema': 2},

    ],  # Two weighting strategies
    'threshold': [2, 3, 4, 5],  # Effective entry thresholds
    'broker_fee': [0.0002],  # Fixed broker fee
    'slippage': [0.00005],  # Fixed slippage
    'hold_time_limit': [None],  # No time limit for simplicity
}

# Collect unique indicator parameters
indicator_params = collect_indicator_params(param_grid)

# Define timeframes
timeframes = ['1T', '5T', '15T', '30T', '1H', '2H', '4H']

# Precompute data
data_dict = precompute_data(data, timeframes, indicator_params)

# Run backtest
backtest_results = backtest_strategy(data_dict, param_grid, timeframes)

# Analyze and save results
backtest_results.to_csv('results/backtest_results.csv', index=False)
