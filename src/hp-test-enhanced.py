import pandas as pd
import optuna
from indicators import calculate_rsi, calculate_bollinger_bands, calculate_macd, calculate_atr
from utils import precompute_data, collect_all_indicator_params, create_numba_params, get_random_slices
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
        
        # Retrieve support and resistance levels based on the parameter's timeframe and window
        support_resistance_timeframe = params['support_resistance_timeframe']
        support_resistance_window = params['support_resistance_window']
        support_key = f'support_levels_{support_resistance_timeframe}_{support_resistance_window}'
        resistance_key = f'resistance_levels_{support_resistance_timeframe}_{support_resistance_window}'
        
        support_levels = data_dict['test'][support_key].values  # Updated to pass NumPy array
        resistance_levels = data_dict['test'][resistance_key].values  # Updated to pass NumPy array

        # Retrieve higher timeframe EMA
        higher_tf_ema_key = f'ema_{params["ema_period"]}'
        if higher_tf_ema_key not in test_res.columns:
            raise KeyError(higher_tf_ema_key)
        higher_tf_ema = test_res[higher_tf_ema_key].values

        # Retrieve indicator columns based on parameters
        ema_key = f'ema_{params["ema_period"]}'
        sma_short_key = f'sma_{params["sma_short"]}'
        sma_long_key = f'sma_{params["sma_long"]}'
        rsi_key = f'rsi_{params["rsi_period"]}'
        bb_num_std_str = f"{params['bb_num_std']:.1f}"
        bb_period = params['bb_period']
        upper_bb_key = f'upper_bb_{bb_period}_{bb_num_std_str}'
        lower_bb_key = f'lower_bb_{bb_period}_{bb_num_std_str}'
        macd_key = f'macd_line_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'
        signal_key = f'signal_line_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'
        atr_key = f'atr_{params["atr_period"]}'
        stoch_k_key = f'stoch_k_{params["stoch_k_period"]}_{params["stoch_d_period"]}'
        stoch_d_key = f'stoch_d_{params["stoch_k_period"]}_{params["stoch_d_period"]}'
        
        required_columns = [
            ema_key, sma_short_key, sma_long_key, rsi_key,
            upper_bb_key, lower_bb_key, macd_key, signal_key,
            atr_key, stoch_k_key, stoch_d_key
        ]
        
        for col in required_columns:
            if col not in test_res.columns:
                raise KeyError(col)
        
        sma_short = test_res[sma_short_key].values
        sma_long = test_res[sma_long_key].values
        rsi = test_res[rsi_key].values
        upper_bb = test_res[upper_bb_key].values
        lower_bb = test_res[lower_bb_key].values
        macd_line = test_res[macd_key].values
        signal_line = test_res[signal_key].values
        ema = test_res[ema_key].values
        atr = test_res[atr_key].values
        stoch_k = test_res[stoch_k_key].values
        stoch_d = test_res[stoch_d_key].values

        slices = get_random_slices(test_res, num_slices=params.get('num_slices',5), slice_length=params.get('slice_length',3000))

        numba_params = create_numba_params(params)
        total_roi = 0
        total_trades = 0

        for data_slice in slices:
            data_slice = data_slice.dropna()

            # Ensure the slice indices align with the pre-extracted arrays
            # (Assuming slices are contiguous and aligned)

            final_balance, trades = trading_strategy_enhanced(
                close=data_slice['close'].values,
                high=data_slice['high'].values,
                low=data_slice['low'].values,
                sma_short=sma_short,
                sma_long=sma_long,
                ema=ema,
                rsi=rsi,
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                upper_bb=upper_bb,
                lower_bb=lower_bb,
                macd_line=macd_line,
                signal_line=signal_line,
                atr=atr,
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
                initial_condition_flags=(0.0, 0.0)
            )

            initial_balance = params.get('starting_balance', 250)
            roi = (final_balance - initial_balance) / initial_balance * 100.0
            total_roi += roi
            total_trades += trades

        avg_roi = total_roi / len(slices) if len(slices) > 0 else 0
        avg_trades = total_trades / len(slices) if len(slices) > 0 else 0

        fold_results.append({
            'fold': i, 
            'roi': avg_roi, 
            'trades': avg_trades
        })
    
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
    # Treat each fold's ROI as the return of a period. For simplicity, assume a risk-free rate of zero.
    if len(rois) > 1:
        mean_roi = np.mean(rois)
        std_roi = np.std(rois, ddof=1)
        sharpe = mean_roi / (std_roi + 1e-9)
    else:
        # With only one fold, the Sharpe ratio is not meaningful
        sharpe = np.nan
    
    # Max Drawdown:
    # Convert fold ROIs into a sequence of cumulative growth factors (1 + roi/100)
    # Then calculate the drawdown.
    growth_factors = 1 + (rois / 100.0)
    cumulative = np.cumprod(growth_factors)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min() * 100.0  # in percentage
    
    # Profit Factor:
    # Approximate gains and losses from each fold's ROI
    gains = rois[rois > 0].sum()
    losses = -rois[rois < 0].sum()
    if losses > 0:
        profit_factor = gains / losses
    else:
        profit_factor = np.inf if gains > 0 else 1.0
    
    # Add trade metrics:
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
        'support_resistance_window': support_resistance_window,
        'num_slices': 5,
        'slice_length': 3000
    }

    try:
        fold_results = run_walk_forward_backtest(data, params, precomputed_data, n_splits=4)
    except KeyError as e:
        print(f"KeyError encountered: {e}")
        raise optuna.exceptions.TrialPruned()

    # Calculate metrics
    metrics = calculate_metrics(fold_results)

    # For optimization, we'll use the Sharpe ratio as the main metric
    # The optimizer tries to maximize the Sharpe ratio, so return it directly
    if np.isnan(metrics['sharpe_ratio']):
        return float('-inf')  # Treat as worst if Sharpe ratio is not available

    return metrics['sharpe_ratio']

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    n_jobs = multiprocessing.cpu_count() - 1

    # Collect all indicator parameters based on Optuna's search space
    indicator_params = collect_all_indicator_params()

    # Precompute indicators for all walk-forward splits and all possible parameter values
    n_splits = 4
    folds = walk_forward_splits(data, n_splits=n_splits, train_size=0.7)
    precomputed_data = []
    for train_data, test_data in folds:
        data_dict = precompute_data(
            test_data,
            sma_periods_short=indicator_params['sma_periods_short'],
            sma_periods_long=indicator_params['sma_periods_long'],
            ema_periods=indicator_params['ema_periods'],
            rsi_periods=indicator_params['rsi_periods'],
            bb_periods=indicator_params['bb_periods'],
            bb_num_std_values=indicator_params['bb_num_std_values'],
            macd_fast_periods=indicator_params['macd_fast_periods'],
            macd_slow_periods=indicator_params['macd_slow_periods'],
            macd_signal_periods=indicator_params['macd_signal_periods'],
            stoch_k_periods=indicator_params['stoch_k_periods'],
            stoch_d_periods=indicator_params['stoch_d_periods'],
            atr_periods=indicator_params['atr_periods'],
            support_resistance_timeframes=indicator_params['support_resistance_timeframes'],
            support_resistance_windows=indicator_params['support_resistance_windows']
        )
        precomputed_data.append(data_dict)

    try:
        study.optimize(lambda trial: objective(trial, precomputed_data), n_trials=5000, n_jobs=n_jobs)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected! Saving the best parameters so far...")
    finally:
        if study.best_trial is not None:
            print("Best parameters so far:")
            print(study.best_params)
            
            best_params = study.best_params
            try:
                fold_results = run_walk_forward_backtest(data, best_params, precomputed_data, n_splits=n_splits)
                final_metrics = calculate_metrics(fold_results)
                print("Final Metrics:")
                print(final_metrics)
            except KeyError as e:
                print(f"KeyError encountered during final backtest: {e}")
                final_metrics = {}
            
            # Get the top 100 trials
            sorted_trials = sorted(
                study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True
            )
            top_100 = sorted_trials[:100]

            # Save the top 100 trials to a JSON file
            now = datetime.now()
            filename = f"src/results/backtest_{now.strftime('%Y%m%d_%H%M')}.json"
            top_100_results = [
                {'rank': i + 1, 'value': trial.value, 'params': trial.params} 
                for i, trial in enumerate(top_100)
            ]

            with open(filename, 'w') as f:
                json.dump(top_100_results, f, indent=4)

            print(f"\nTop 100 Trials saved to {filename}")
        else:
            print("No successful trials were completed.")
