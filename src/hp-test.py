import pandas as pd
import numpy as np
import optuna
import multiprocessing
from datetime import datetime
import json

from tqdm import tqdm
from indicators import (
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_atr
)
from utils import (
    adjust_indicator_periods,
    get_time_series_folds,
    resample_data,
    precompute_data,
    collect_indicator_params_from_params
)
from backtest import backtest_strategy_hp
import optuna

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
# Load and preprocess data
data = pd.read_csv('E:\\programming\\test-trader\\src\\data\\btcusd.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
data.set_index('timestamp', inplace=True)
data.sort_index(inplace=True)

# Define timeframes
timeframes = ['1h', '2h', '4h', '1d'] # '1min', '5min', '15min', '30min', 
total_iterations = 15000
progress_bar = tqdm(total=total_iterations, desc="Optimization Progress")
def run_backtest_with_params(params):
    # Collect indicator parameters
    indicator_params = collect_indicator_params_from_params(params)
    
    # Precompute data for the specific timeframe
    data_dict = precompute_data(data, [params['timeframe']], indicator_params)
    
    # Run backtest on the specific timeframe
    backtest_results = backtest_strategy_hp(
        data_dict, [params], [params['timeframe']], num_slices=3, slice_length=3000
    )
    
    # Extract avg_roi and avg_trades
    avg_roi = backtest_results['avg_roi'].mean()
    avg_trades = backtest_results['avg_trades'].mean()
    
    return avg_roi, avg_trades

def run_k_fold_cross_validation(data, params, k, debug=False):
    indicator_params = collect_indicator_params_from_params(params)
    if debug:
        print(f"Indicator Parameters: {indicator_params}")
    # Create time-series folds
    data_folds = get_time_series_folds(data, k)
    results = []
    for i, data_fold in enumerate(data_folds):
        # Precompute data for the specific timeframe
        data_dict = precompute_data(data_fold, [params['timeframe']], indicator_params)
        # Run backtest on the specific timeframe
        backtest_results = backtest_strategy_hp(
            data_dict, [params], [params['timeframe']], num_slices=1, slice_length=len(data_fold)
        )
        # Extract avg_roi and avg_trades
        avg_roi = backtest_results['avg_roi'].mean()
        avg_trades = backtest_results['avg_trades'].mean()
        results.append({
            'fold': i + 1,
            'avg_roi': avg_roi,
            'avg_trades': avg_trades
        })
    return pd.DataFrame(results)

def objective(trial):
    global progress_bar
    # Suggest hyperparameters using Optuna's trial object
    sma_short = trial.suggest_int('sma_short', 5, 100, step=5)
    sma_long = trial.suggest_int('sma_long', 50, 400, step=10)
    if sma_short >= sma_long:
        # Enforce constraint by pruning the trial
        raise optuna.exceptions.TrialPruned()

    rsi_period = trial.suggest_int('rsi_period', 7, 35, step=7)
    rsi_threshold_low = trial.suggest_float('rsi_threshold_low', 10, 40)
    rsi_threshold_high = trial.suggest_float('rsi_threshold_high', 60, 90)
    bb_period = trial.suggest_int('bb_period', 20, 100, step=10)
    bb_num_std = trial.suggest_categorical('bb_num_std', [1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    macd_fast = trial.suggest_int('macd_fast', 5, 20, step=1)
    macd_slow = trial.suggest_int('macd_slow', 20, 40, step=2)
    if macd_fast >= macd_slow:
        raise optuna.exceptions.TrialPruned()
    macd_signal = trial.suggest_int('macd_signal', 5, 15, step=1)
    ema_period = trial.suggest_int('ema_period', 10, 100, step=5)
    risk_per_trade = trial.suggest_float('risk_per_trade', 0.005, 0.1)
    stop_loss = trial.suggest_float('stop_loss', 0.005, 0.05, log=True)
    take_profit = trial.suggest_float('take_profit', 0.01, 0.15, log=True)
    trailing_stop_loss = trial.suggest_float('trailing_stop_loss', 0.001, 0.02, log=True)
    trailing_take_profit = trial.suggest_float('trailing_take_profit', 0.001, 0.02, log=True)
    threshold = trial.suggest_float('threshold', 1, 5.5, step=0.5)
    stoch_k_period = trial.suggest_int('stoch_k_period', 7, 21, step=7)
    stoch_d_period = trial.suggest_int('stoch_d_period', 3, 9, step=3)
    stoch_threshold_low = trial.suggest_float('stoch_threshold_low', 10, 30)
    stoch_threshold_high = trial.suggest_float('stoch_threshold_high', 70, 90)
    timeframe = trial.suggest_categorical('timeframe', timeframes)

    # Prepare parameters dictionary
    params = {
        'stoch_k_period': stoch_k_period,
        'stoch_d_period': stoch_d_period,
        'stoch_threshold_low': stoch_threshold_low,
        'stoch_threshold_high': stoch_threshold_high,
        'starting_balance': 250,
        'sma_short': sma_short,
        'sma_long': sma_long,
        'rsi_period': rsi_period,
        'rsi_threshold_high': rsi_threshold_high,
        'rsi_threshold_low': rsi_threshold_low,
        'bb_period': bb_period,
        'bb_num_std': bb_num_std,
        'macd_fast': macd_fast,
        'macd_slow': macd_slow,
        'macd_signal': macd_signal,
        'ema_period': ema_period,
        'risk_per_trade': risk_per_trade,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'trailing_stop_loss': trailing_stop_loss,
        'trailing_take_profit': trailing_take_profit,
        'threshold': threshold,
        'broker_fee': 0.0002,
        'slippage': 0.00005,
        'hold_time_limit': None,
        'timeframe': timeframe,
        'atr_period': 14,
    }

    # Run backtest
    avg_roi, avg_trades = run_backtest_with_params(params)
    if avg_trades == 0:
        sharpe_ratio = float('-inf')  # Penalize strategies with no trades
    else:
        sharpe_ratio = avg_roi / (avg_trades ** 0.5)
    progress_bar.update(1)

    loss = -sharpe_ratio
    return loss

if __name__ == '__main__':
    # Create Optuna study
    study = optuna.create_study(direction='minimize')

    # Determine the number of parallel trials
    num_parallel_trials = multiprocessing.cpu_count()

    # Optimize the objective function
    study.optimize(objective, n_trials=total_iterations, n_jobs=num_parallel_trials)

    # Extract and print the best parameters
    best_params = study.best_params
    print("Best parameters:")
    print(best_params)

    # Define the number of folds for cross-validation
    k = 5  # Adjust as necessary based on your data size

    # Run k-fold cross-validation
    cv_results = run_k_fold_cross_validation(data, best_params, k)
    print("K-Fold Cross-Validation Results:")
    print(cv_results)

    # Optionally, analyze the results
    average_roi = cv_results['avg_roi'].mean()
    print(f"Average ROI across folds: {average_roi:.2f}%")

    # Get the current date
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M')

    # Create results dictionary
    results = {
        'best_params': best_params,
        'cv_results': cv_results.to_dict(orient='records'),
        'average_roi': average_roi
    }

    # Define the file path
    file_path = f'results/{current_date}.json'

    # Write results to JSON file
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {file_path}")
