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

def run_walk_forward_backtest(data, params, n_splits=4):
    folds = walk_forward_splits(data, n_splits=n_splits, train_size=0.7)
    fold_results = []
    
    for i, (train_data, test_data) in enumerate(folds):
        # Precompute indicators
        indicator_params = collect_indicator_params_from_params(params)
        timeframe = params.get('timeframe', '1H')
        data_dict = precompute_data(test_data, [timeframe], indicator_params)
        test_res = data_dict[timeframe]

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

def objective(trial):
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
        'starting_balance': 250,
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
    }

    fold_results = run_walk_forward_backtest(data, params, n_splits=4)
    
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
    study.optimize(objective, n_trials=50000, n_jobs=n_jobs)
    print("Best parameters:")
    print(study.best_params)

    best_params = study.best_params
    fold_results = run_walk_forward_backtest(data, best_params, n_splits=4)
    final_metrics = calculate_metrics(fold_results)
    print("Final Metrics:")
    print(final_metrics)

    # Get the top 100 best trials
    # Sort all trials by their objective value (sharpe ratio here) in descending order
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)
    top_100 = sorted_trials[:1000]

    # Save the top 100 trials to a JSON file

    now = datetime.now()
    filename = f"src/results/backtest_{now.strftime('%Y%m%d_%H%M')}.json"

    top_100_results = [{'rank': i + 1, 'value': trial.value, 'params': trial.params} for i, trial in enumerate(top_100)]
    
    with open(filename, 'w') as f:
        json.dump(top_100_results, f, indent=4)

    print(f"\nTop 100 Trials saved to {filename}")