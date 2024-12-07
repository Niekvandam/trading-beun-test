import pandas as pd
from tqdm import tqdm
from strategy import trading_strategy
from utils import (
    generate_param_combinations,
    get_random_slices,
    create_numba_params
)

def backtest_strategy(data_dict, param_grid, timeframes, num_slices=5, slice_length=3000):
    results = []
    # Generate param combinations once
    if isinstance(param_grid, dict):
        param_combinations = list(generate_param_combinations(param_grid))
    elif isinstance(param_grid, list):
        param_combinations = param_grid
    else:
        raise ValueError("Invalid param_list format. Expected dict or list.")

    total_iterations = len(timeframes) * len(param_combinations)

    with tqdm(total=total_iterations, desc="Backtesting Progress") as pbar:
        for timeframe in timeframes:
            resampled_data = data_dict[timeframe]
            slices = get_random_slices(resampled_data, num_slices=num_slices, slice_length=slice_length)
            
            for params in param_combinations:  # use the precomputed list directly
                numba_params = create_numba_params(params)
                total_roi = 0
                total_trades = 0

                for data_slice in slices:
                    data_slice = data_slice.dropna()

                    # Extract necessary columns
                    close = data_slice['close'].values
                    high = data_slice['high'].values
                    low = data_slice['low'].values
                    sma_short = data_slice[f'sma_{params["sma_periods"][0]}'].values
                    sma_long = data_slice[f'sma_{params["sma_periods"][1]}'].values
                    rsi = data_slice[f'rsi_{params["rsi_period"]}'].values
                    num_std_str = f"{params['bb_num_std']:.1f}"
                    upper_bb = data_slice[f'upper_bb_{params["bb_period"]}_{num_std_str}'].values
                    lower_bb = data_slice[f'lower_bb_{params["bb_period"]}_{num_std_str}'].values
                    macd_line = data_slice[f'macd_line_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'].values
                    signal_line = data_slice[f'signal_line_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'].values
                    ema = data_slice[f'ema_{params["ema_period"]}'].values
                    atr = data_slice[f'atr_{params["atr_period"]}'].values
                    stoch_k = data_slice[f'stoch_k_{params["stoch_k_period"]}_{params["stoch_d_period"]}'].values
                    stoch_d = data_slice[f'stoch_d_{params["stoch_k_period"]}_{params["stoch_d_period"]}'].values

                    # Ensure correct argument order as per the trading_strategy signature
                    final_balance, trades = trading_strategy(
                        close,
                        high,
                        low,
                        sma_short,
                        sma_long,
                        ema,
                        rsi,
                        stoch_k,
                        stoch_d,
                        upper_bb,
                        lower_bb,
                        macd_line,
                        signal_line,
                        atr,
                        numba_params
                    )

                    initial_balance = params.get('starting_balance', 250)
                    roi = (final_balance - initial_balance) / initial_balance * 100
                    total_roi += roi
                    total_trades += trades

                avg_roi = total_roi / num_slices if num_slices > 0 else 0
                avg_trades = total_trades / num_slices if num_slices > 0 else 0

                results.append({
                    'timeframe': timeframe,
                    'params': params,
                    'avg_roi': avg_roi,
                    'avg_trades': avg_trades
                })
                pbar.update(1)

    return pd.DataFrame(results)


# backtest.py

from tqdm import tqdm

def backtest_strategy_hp(data_dict, param_list, timeframes, num_slices=5, slice_length=3000):
    results = []
    for timeframe in timeframes:
        resampled_data = data_dict[timeframe]
        slices = get_random_slices(resampled_data, num_slices=num_slices, slice_length=slice_length)
        for params in param_list:
            numba_params = create_numba_params(params)
            total_roi = 0
            total_trades = 0
            for data_slice in slices:
                data_slice = data_slice.dropna()
                
                # Extract necessary data
                close = data_slice['close'].values
                high = data_slice['high'].values  # Include high prices
                low = data_slice['low'].values    # Include low prices
                sma_short = data_slice[f'sma_{params["sma_short"]}'].values
                sma_long = data_slice[f'sma_{params["sma_long"]}'].values
                rsi = data_slice[f'rsi_{params["rsi_period"]}'].values
                upper_bb = data_slice[f'upper_bb_{params["bb_period"]}_{params["bb_num_std"]:.1f}'].values
                lower_bb = data_slice[f'lower_bb_{params["bb_period"]}_{params["bb_num_std"]:.1f}'].values
                macd_line = data_slice[f'macd_line_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'].values
                signal_line = data_slice[f'signal_line_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'].values
                ema = data_slice[f'ema_{params["ema_period"]}'].values
                atr = data_slice[f'atr_{14}'].values  # ATR period is fixed at 14
                stoch_k = data_slice[f'stoch_k_{params["stoch_k_period"]}_{params["stoch_d_period"]}'].values
                stoch_d = data_slice[f'stoch_d_{params["stoch_k_period"]}_{params["stoch_d_period"]}'].values

                # Pass all required parameters to trading_strategy
                final_balance, trades = trading_strategy(
                    close, high, low, sma_short, sma_long, ema, rsi, stoch_k, stoch_d,
                    upper_bb, lower_bb, macd_line, signal_line, atr, numba_params
                )

                initial_balance = params.get('starting_balance', 250)
                roi = (final_balance - initial_balance) / initial_balance * 100
                total_roi += roi
                total_trades += trades

            avg_roi = total_roi / num_slices
            avg_trades = total_trades / num_slices
            result = {
                'timeframe': timeframe,
                'params': params,
                'avg_roi': avg_roi,
                'avg_trades': avg_trades
            }
            results.append(result)
    return pd.DataFrame(results)
