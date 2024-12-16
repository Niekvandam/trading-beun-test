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
            
            # Define 'close', 'high', 'low' from resampled_data
            close = resampled_data['close'].values
            high = resampled_data['high'].values
            low = resampled_data['low'].values

            # Pre-extract necessary columns to avoid repeated access
            close_dict = {}
            high_dict = {}
            low_dict = {}
            sma_short_dict = {}
            sma_long_dict = {}
            rsi_dict = {}
            upper_bb_dict = {}
            lower_bb_dict = {}
            macd_line_dict = {}
            signal_line_dict = {}
            ema_dict = {}
            atr_dict = {}
            stoch_k_dict = {}
            stoch_d_dict = {}
            support_levels = data_dict['support_levels']
            resistance_levels = data_dict['resistance_levels']

            # Pre-extract columns for all parameter combinations
            for params in param_combinations:
                sma_short_key = f'sma_{params["sma_periods"][0]}'
                sma_long_key = f'sma_{params["sma_periods"][1]}'
                rsi_key = f'rsi_{params["rsi_period"]}'
                num_std_str = f"{params['bb_num_std']:.1f}"
                upper_bb_key = f'upper_bb_{params["bb_period"]}_{num_std_str}'
                lower_bb_key = f'lower_bb_{params["bb_period"]}_{num_std_str}'
                macd_key = f'macd_line_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'
                signal_key = f'signal_line_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'
                ema_key = f'ema_{params["ema_period"]}'
                atr_key = f'atr_{params["atr_period"]}'
                stoch_k_key = f'stoch_k_{params["stoch_k_period"]}_{params["stoch_d_period"]}'
                stoch_d_key = f'stoch_d_{params["stoch_k_period"]}_{params["stoch_d_period"]}'

                if sma_short_key not in sma_short_dict:
                    sma_short_dict[sma_short_key] = resampled_data[sma_short_key].values
                if sma_long_key not in sma_long_dict:
                    sma_long_dict[sma_long_key] = resampled_data[sma_long_key].values
                if rsi_key not in rsi_dict:
                    rsi_dict[rsi_key] = resampled_data[rsi_key].values
                if upper_bb_key not in upper_bb_dict:
                    upper_bb_dict[upper_bb_key] = resampled_data[upper_bb_key].values
                if lower_bb_key not in lower_bb_dict:
                    lower_bb_dict[lower_bb_key] = resampled_data[lower_bb_key].values
                if macd_key not in macd_line_dict:
                    macd_line_dict[macd_key] = resampled_data[macd_key].values
                if signal_key not in signal_line_dict:
                    signal_line_dict[signal_key] = resampled_data[signal_key].values
                if ema_key not in ema_dict:
                    ema_dict[ema_key] = resampled_data[ema_key].values
                if atr_key not in atr_dict:
                    atr_dict[atr_key] = resampled_data[atr_key].values
                if stoch_k_key not in stoch_k_dict:
                    stoch_k_dict[stoch_k_key] = resampled_data[stoch_k_key].values
                if stoch_d_key not in stoch_d_dict:
                    stoch_d_dict[stoch_d_key] = resampled_data[stoch_d_key].values

            for params in param_combinations:
                numba_params = create_numba_params(params)
                total_roi = 0
                total_trades = 0

                sma_short = sma_short_dict[f'sma_{params["sma_periods"][0]}']
                sma_long = sma_long_dict[f'sma_{params["sma_periods"][1]}']
                rsi = rsi_dict[f'rsi_{params["rsi_period"]}']
                upper_bb = upper_bb_dict[f'upper_bb_{params["bb_period"]}_{params["bb_num_std"]:.1f}']
                lower_bb = lower_bb_dict[f'lower_bb_{params["bb_period"]}_{params["bb_num_std"]:.1f}']
                macd_line = macd_line_dict[f'macd_line_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}']
                signal_line = signal_line_dict[f'signal_line_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}']
                ema = ema_dict[f'ema_{params["ema_period"]}']
                atr = atr_dict[f'atr_{params["atr_period"]}']
                stoch_k = stoch_k_dict[f'stoch_k_{params["stoch_k_period"]}_{params["stoch_d_period"]}']
                stoch_d = stoch_d_dict[f'stoch_d_{params["stoch_k_period"]}_{params["stoch_d_period"]}']

                for data_slice in slices:
                    data_slice = data_slice.dropna()

                    # Ensure the slice indices align with the pre-extracted arrays
                    # (Assuming slices are contiguous and aligned)

                    # Retrieve support and resistance levels
                    # Already extracted outside the loop

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
                        support_levels,
                        resistance_levels,
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
        # Pre-extract columns outside the param loop to avoid redundant access
        close_all = resampled_data['close'].values
        high_all = resampled_data['high'].values
        low_all = resampled_data['low'].values
        support_levels = data_dict['support_levels']
        resistance_levels = data_dict['resistance_levels']

        # Pre-extract indicators for all params
        indicator_data = {}
        for params in param_list:
            sma_short_key = f'sma_{params["sma_short"]}'
            sma_long_key = f'sma_{params["sma_long"]}'
            rsi_key = f'rsi_{params["rsi_period"]}'
            num_std_str = f"{params['bb_num_std']:.1f}"
            upper_bb_key = f'upper_bb_{params["bb_period"]}_{num_std_str}'
            lower_bb_key = f'lower_bb_{params["bb_period"]}_{num_std_str}'
            macd_key = f'macd_line_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'
            signal_key = f'signal_line_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'
            ema_key = f'ema_{params["ema_period"]}'
            atr_key = f'atr_{params["atr_period"]}'
            stoch_k_key = f'stoch_k_{params["stoch_k_period"]}_{params["stoch_d_period"]}'
            stoch_d_key = f'stoch_d_{params["stoch_k_period"]}_{params["stoch_d_period"]}'

            if sma_short_key not in indicator_data:
                indicator_data[sma_short_key] = resampled_data[sma_short_key].values
            if sma_long_key not in indicator_data:
                indicator_data[sma_long_key] = resampled_data[sma_long_key].values
            if rsi_key not in indicator_data:
                indicator_data[rsi_key] = resampled_data[rsi_key].values
            if upper_bb_key not in indicator_data:
                indicator_data[upper_bb_key] = resampled_data[upper_bb_key].values
            if lower_bb_key not in indicator_data:
                indicator_data[lower_bb_key] = resampled_data[lower_bb_key].values
            if macd_key not in indicator_data:
                indicator_data[macd_key] = resampled_data[macd_key].values
            if signal_key not in indicator_data:
                indicator_data[signal_key] = resampled_data[signal_key].values
            if ema_key not in indicator_data:
                indicator_data[ema_key] = resampled_data[ema_key].values
            if atr_key not in indicator_data:
                indicator_data[atr_key] = resampled_data[atr_key].values
            if stoch_k_key not in indicator_data:
                indicator_data[stoch_k_key] = resampled_data[stoch_k_key].values
            if stoch_d_key not in indicator_data:
                indicator_data[stoch_d_key] = resampled_data[stoch_d_key].values

    for params in param_list:
        numba_params = create_numba_params(params)
        total_roi = 0
        total_trades = 0

        sma_short = indicator_data[f'sma_{params["sma_short"]}']
        sma_long = indicator_data[f'sma_{params["sma_long"]}']
        rsi = indicator_data[f'rsi_{params["rsi_period"]}']
        upper_bb = indicator_data[f'upper_bb_{params["bb_period"]}_{params["bb_num_std"]:.1f}']
        lower_bb = indicator_data[f'lower_bb_{params["bb_period"]}_{params["bb_num_std"]:.1f}']
        macd_line = indicator_data[f'macd_line_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}']
        signal_line = indicator_data[f'signal_line_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}']
        ema = indicator_data[f'ema_{params["ema_period"]}']
        atr = indicator_data[f'atr_{params["atr_period"]}']
        stoch_k = indicator_data[f'stoch_k_{params["stoch_k_period"]}_{params["stoch_d_period"]}']
        stoch_d = indicator_data[f'stoch_d_{params["stoch_k_period"]}_{params["stoch_d_period"]}']

        for data_slice in slices:
            data_slice = data_slice.dropna()
            
            # Retrieve support and resistance levels

            # Ensure correct argument order as per the trading_strategy signature
            final_balance, trades = trading_strategy(
                close_all,
                high_all,
                low_all,
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
                support_levels,
                resistance_levels,
                numba_params
            )

            initial_balance = params.get('starting_balance', 250)
            roi = (final_balance - initial_balance) / initial_balance * 100.0
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
