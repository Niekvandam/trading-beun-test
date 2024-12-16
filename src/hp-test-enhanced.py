import pandas as pd
import optuna
from indicators import calculate_rsi, calculate_bollinger_bands, calculate_macd, calculate_atr
from utils import precompute_data, collect_indicator_params_from_params
from strategy_enhanced import trading_strategy_enhanced
from walk_forward import walk_forward_splits
import numpy as np
import multiprocessing
import json
from datetime import datetime

# Load data
data = pd.read_csv('src/data/btcusd.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
data.set_index('timestamp', inplace=True)
data.sort_index(inplace=True)

def run_walk_forward_backtest(data, params, precomputed_data, n_splits=4):
    folds = walk_forward_splits(data, n_splits=n_splits, train_size=0.7)
    fold_results = []
    
    for i, (train_data, test_data) in enumerate(folds):
        # Use precomputed data instead of recalculating
        data_dict = precomputed_data[i]
        test_res = data_dict['test']

        close = test_res['close'].values
        high = test_res['high'].values
        low = test_res['low'].values
        
        # Trend filter EMA
        higher_tf_ema = test_res[f'ema_{params["ema_period"]}'].values
        
        sma_short = test_res[f'sma_{params["sma_short"]}'].values
        sma_long = test_res[f'sma_{params["sma_long"]}'].values
        rsi = test_res[f'rsi_{params["rsi_period"]}'].values
        bb_str = f"{params['bb_num_std']:.1f}"
        upper_bb = test_res[f'upper_bb_{params["bb_period"]}_{bb_str}'].values
        lower_bb = test_res[f'lower_bb_{params["bb_period"]}_{bb_str}'].values
        macd_line = test_res[f'macd_line_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'].values
        signal_line = test_res[f'signal_line_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'].values
        ema = test_res[f'ema_{params["ema_period"]}'].values
        atr = test_res[f'atr_{params["atr_period"]}'].values
        stoch_k = test_res[f'stoch_k_{params["stoch_k_period"]}_{params["stoch_d_period"]}'].values
        stoch_d = test_res[f'stoch_d_{params["stoch_k_period"]}_{params["stoch_d_period"]}'].values

        # Retrieve support and resistance levels
        support_levels = data_dict['support_levels']
        resistance_levels = data_dict['resistance_levels']

        initial_condition_flags = (0.0, 0.0)

        final_balance, trades = trading_strategy_enhanced(
            close=close, high=high, low=low,
            sma_short=sma_short, sma_long=sma_long, ema=ema, rsi=rsi, stoch_k=stoch_k, stoch_d=stoch_d,
            upper_bb=upper_bb, lower_bb=lower_bb, macd_line=macd_line, signal_line=signal_line, atr=atr,
            starting_balance=params.get('starting_balance', 250),
            rsi_threshold_low=params['rsi_threshold_low'],
            rsi_threshold_high=params['rsi_threshold_high'],
            stoch_threshold_low=params['stoch_threshold_low'],
            stoch_threshold_high=params['stoch_threshold_high'],
            risk_per_trade=params['risk_per_trade'],
            stop_loss_pct=params['stop_loss'],
            take_profit_pct=params['take_profit'],
            trailing_stop_loss_pct=params['trailing_stop_loss'],
            trailing_take_profit_pct=params['trailing_take_profit'],
            threshold=params['threshold'],
            broker_fee=params.get('broker_fee', 0.0005),
            slippage=params.get('slippage', 0.0002),
            hold_time_limit=params.get('hold_time_limit', -1),
            atr_period=params['atr_period'],
            allow_short=params.get('allow_short', True),
            trend_filter=params.get('trend_filter', True),
            dynamic_sizing_factor=params.get('dynamic_sizing_factor', 0.1),
            higher_tf_ema=higher_tf_ema,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            initial_condition_flags=initial_condition_flags
        )

        initial_balance = params.get('starting_balance', 250)
        roi = (final_balance - initial_balance) / initial_balance * 100.0
        fold_results.append({'fold': i, 'roi': roi, 'trades': trades})
    
    fold_df = pd.DataFrame(fold_results)
    return fold_df

def calculate_metrics(fold_results):
    """
    Given a DataFrame of fold results with columns 'roi' and 'trades',
    compute additional metrics: Sharpe ratio, max drawdown, profit factor, etc.
    """
    # Extract ROI series
    rois = fold_results['roi'].values
    
    # Sharpe Ratio:
    # We treat each fold's ROI as one period's return. For simplicity, assume zero risk-free rate.
    if len(rois) > 1:
        mean_roi = np.mean(rois)
        std_roi = np.std(rois, ddof=1)
        sharpe = mean_roi / (std_roi + 1e-9)
    else:
        # With only one fold, Sharpe ratio doesn't make much sense
        sharpe = np.nan

    # Max Drawdown:
    # Convert ROI folds into a cumulative growth factor sequence (1 + roi/100)
    # Then compute drawdown.
    growth_factors = 1 + (rois / 100.0)
    cumulative = np.cumprod(growth_factors)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min() * 100.0  # in percentage

    # Profit Factor:
    # Approximate profits and losses from roi of each fold
    gains = rois[rois > 0].sum()
    losses = -rois[rois < 0].sum()
    if losses > 0:
        profit_factor = gains / losses
    else:
        profit_factor = np.inf if gains > 0 else 1.0

    # Add trades metrics:
    avg_trades = fold_results['trades'].mean()
    total_trades = fold_results['trades'].sum()
    
    metrics = {
        'mean_roi': np.mean(rois),
        'std_roi': np.std(rois, ddof=1),
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'average_trades': avg_trades,
        'total_trades': total_trades
    }

    return metrics

def objective(trial, precomputed_data):
    # Suggest parameters as before
    sma_short = trial.suggest_int('sma_short', 5, 50, step=5)
    sma_long = trial.suggest_int('sma_long', 50, 200, step=10)
    if sma_short >= sma_long:
        raise optuna.exceptions.TrialPruned()
    
    rsi_period = trial.suggest_int('rsi_period', 7, 35, step=7)
    rsi_threshold_low = trial.suggest_float('rsi_threshold_low', 20, 40)
    rsi_threshold_high = trial.suggest_float('rsi_threshold_high', 60, 80)
    bb_period = trial.suggest_int('bb_period', 20, 50, step=10)
    bb_num_std = trial.suggest_categorical('bb_num_std', [1.5, 2.0, 2.5])
    macd_fast = trial.suggest_int('macd_fast', 5, 12)
    macd_slow = trial.suggest_int('macd_slow', 20, 30)
    if macd_fast >= macd_slow:
        raise optuna.exceptions.TrialPruned()
    macd_signal = trial.suggest_int('macd_signal', 5, 15, step=5)
    ema_period = trial.suggest_int('ema_period', 10, 50, step=5)
    risk_per_trade = trial.suggest_float('risk_per_trade', 0.01, 0.05)
    stop_loss = trial.suggest_float('stop_loss', 0.01, 0.05, log=True)
    take_profit = trial.suggest_float('take_profit', 0.01, 0.1, log=True)
    trailing_stop_loss = trial.suggest_float('trailing_stop_loss', 0.005, 0.02, log=True)
    trailing_take_profit = trial.suggest_float('trailing_take_profit', 0.005, 0.02, log=True)
    threshold = trial.suggest_float('threshold', 2, 5, step=1)
    stoch_k_period = trial.suggest_int('stoch_k_period', 7, 21, step=7)
    stoch_d_period = trial.suggest_int('stoch_d_period', 3, 9, step=3)
    stoch_threshold_low = trial.suggest_float('stoch_threshold_low', 10, 30)
    stoch_threshold_high = trial.suggest_float('stoch_threshold_high', 70, 90)
    atr_period = trial.suggest_int('atr_period', 7, 21)
    
    # New parameters for Support and Resistance
    support_resistance_timeframe = trial.suggest_categorical('support_resistance_timeframe', ['30m', '1h', '4h', '1d'])
    support_resistance_window = trial.suggest_int('support_resistance_window', 10, 200, step=10)

    # Add multiple timeframes if desired
    timeframes = ['1min', '3min', '5min', '15min', '30min', '1h', '2h', '4h']
    timeframe = trial.suggest_categorical('timeframe', timeframes)


    params = {
        'starting_balance': 250,
        'sma_short': sma_short,
        'sma_long': sma_long,
        'rsi_period': rsi_period,
        'rsi_threshold_low': rsi_threshold_low,
        'rsi_threshold_high': rsi_threshold_high,
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
        'hold_time_limit': -1,
        'stoch_k_period': stoch_k_period,
        'stoch_d_period': stoch_d_period,
        'stoch_threshold_low': stoch_threshold_low,
        'stoch_threshold_high': stoch_threshold_high,
        'timeframe': timeframe,
        'atr_period': atr_period,
        'allow_short': True,
        'trend_filter': True,
        'dynamic_sizing_factor': 0.1,
        'support_resistance_timeframe': support_resistance_timeframe,
        'support_resistance_window': support_resistance_window
    }

    fold_results = run_walk_forward_backtest(data, params, precomputed_data)
    
    # Calculate metrics
    metrics = calculate_metrics(fold_results)
    
    # For optimization, let's use the Sharpe ratio as the primary metric
    # The optimizer tries to minimize loss, so we invert Sharpe ratio
    # by returning negative Sharpe ratio as loss. Higher Sharpe ratio = lower loss.
    if np.isnan(metrics['sharpe_ratio']):
        return float('-inf')  # no variance in returns means we can't really evaluate
    
    loss = metrics['sharpe_ratio']
    return loss

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    n_jobs = multiprocessing.cpu_count() - 1

    # Precompute indicators for all walk-forward splits
    n_splits = 4
    folds = walk_forward_splits(data, n_splits=n_splits, train_size=0.7)
    precomputed_data = []
    for train_data, test_data in folds:
        indicator_params = collect_indicator_params_from_params({
            # Define a default or representative parameter set for precomputing
            'sma_short': 20,
            'sma_long': 50,
            'rsi_period': 14,
            'bb_period': 20,
            'bb_num_std': 2.0,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'ema_period': 20,
            'stoch_k_period': 14,
            'stoch_d_period': 3,
            'atr_period': 14,
            'support_resistance_timeframe': '1h',
            'support_resistance_window': 30
        })
        data_dict = precompute_data(
            test_data, 
            [indicator_params['sma_periods'][0]],  # Assuming single timeframe for simplicity
            indicator_params, 
            '1h', 
            30
        )
        precomputed_data.append(data_dict)

    try:
        study.optimize(lambda trial: objective(trial, precomputed_data), n_trials=5000, n_jobs=n_jobs)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected! Saving the best parameters so far...")
    finally:
        # Save best parameters and results regardless of completion
        print("Best parameters so far:")
        print(study.best_params)
        
        best_params = study.best_params
        fold_results = run_walk_forward_backtest(data, best_params, precomputed_data, n_splits=n_splits)
        final_metrics = calculate_metrics(fold_results)
        print("Final Metrics:")
        print(final_metrics)
        
        # Get the top 100 best trials
        sorted_trials = sorted(
            study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True
        )
        top_100 = sorted_trials[:100]

        # Save the top 100 trials to a JSON file
        now = datetime.now()
        filename = f"src/results/backtest_{now.strftime('%Y%m%d_%H%M')}.json"
        top_100_results = [{'rank': i + 1, 'value': trial.value, 'params': trial.params} for i, trial in enumerate(top_100)]

        with open(filename, 'w') as f:
            json.dump(top_100_results, f, indent=4)

        print(f"\nTop 100 Trials saved to {filename}")
