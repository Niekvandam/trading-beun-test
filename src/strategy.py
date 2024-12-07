from numba import njit
import numpy as np

@njit
def trading_strategy(
    close, high, low, sma_short, sma_long, ema, rsi, stoch_k, stoch_d,
    upper_bb, lower_bb, macd_line, signal_line, atr,
    params
):
    # Extract parameters
    starting_balance = params['starting_balance']
    rsi_threshold_low = params['rsi_threshold_low']
    rsi_threshold_high = params['rsi_threshold_high']
    stoch_threshold_low = params['stoch_threshold_low']
    stoch_threshold_high = params['stoch_threshold_high']
    risk_per_trade = params['risk_per_trade']
    stop_loss_pct = params['stop_loss']
    take_profit_pct = params['take_profit']
    trailing_stop_loss_pct = params['trailing_stop_loss']
    trailing_take_profit_pct = params['trailing_take_profit']
    threshold = params['threshold']
    broker_fee = params['broker_fee']
    slippage = params['slippage']
    hold_time_limit = params['hold_time_limit']
    atr_period = params['atr_period']

    balance = starting_balance
    position = 0  # 0: no position, 1: long
    position_size = 0.0
    entry_price = 0.0
    stop_loss_price = 0.0
    take_profit_price = 0.0
    entry_index = 0
    trades = 0

    n = len(close)
    for i in range(1, n):
        condition_score = 0.0

        # Weighted Conditions
        # Condition: SMA Crossover (Weight: 1.5)
        if sma_short[i - 1] < sma_long[i - 1] and sma_short[i] >= sma_long[i]:
            condition_score += 1.5

        # Condition: RSI Oversold (Weight: 1.0)
        if rsi[i] < rsi_threshold_low:
            condition_score += 1.0

        # Condition: Stochastic Oversold (Weight: 1.0)
        if stoch_k[i] < stoch_threshold_low and stoch_d[i] < stoch_threshold_low:
            condition_score += 1.0

        # Condition: Price Below Lower Bollinger Band (Weight: 0.5)
        if close[i] < lower_bb[i]:
            condition_score += 0.5

        # Condition: MACD Crossover (Weight: 1.0)
        if macd_line[i - 1] < signal_line[i - 1] and macd_line[i] >= signal_line[i]:
            condition_score += 1.0

        # Condition: Price Above EMA (Weight: 0.5)
        if close[i] > ema[i]:
            condition_score += 0.5

        # Entry Logic
        if position == 0 and condition_score >= threshold:
            entry_price = close[i] * (1 + slippage)

            # Calculate ATR-based stop loss and take profit
            atr_value = atr[i]
            if np.isnan(atr_value) or atr_value == 0:
                continue  # Skip if ATR is not available

            stop_loss_distance = atr_value * stop_loss_pct
            take_profit_distance = atr_value * take_profit_pct

            # Calculate position size
            position_size = (balance * risk_per_trade) / stop_loss_distance
            max_position_size = balance / entry_price
            position_size = min(position_size, max_position_size)

            # Update stop loss and take profit prices
            stop_loss_price = entry_price - stop_loss_distance
            take_profit_price = entry_price + take_profit_distance

            # Deduct cost and fees from balance
            total_cost = position_size * entry_price
            balance -= total_cost
            balance -= total_cost * broker_fee  # Broker fee

            position = 1
            entry_index = i
            trades += 1

        elif position == 1:
            # Update trailing stop-loss and take-profit
            current_price = close[i]
            # Trailing Stop Loss
            new_stop_loss_price = current_price - (atr[i] * trailing_stop_loss_pct)
            if new_stop_loss_price > stop_loss_price:
                stop_loss_price = new_stop_loss_price

            # Trailing Take Profit
            new_take_profit_price = current_price + (atr[i] * trailing_take_profit_pct)
            if new_take_profit_price > take_profit_price:
                take_profit_price = new_take_profit_price

            # Exit Logic
            exit_reason = 0
            if hold_time_limit != -1 and (i - entry_index) >= hold_time_limit:
                exit_reason = 1  # Time limit reached
            elif low[i] <= stop_loss_price:
                exit_reason = 2  # Stop loss hit
                exit_price = stop_loss_price
            elif high[i] >= take_profit_price:
                exit_reason = 3  # Take profit hit
                exit_price = take_profit_price

            if exit_reason > 0:
                exit_price = exit_price * (1 - slippage)
                balance += position_size * exit_price
                balance -= position_size * exit_price * broker_fee  # Broker fee
                position = 0

    return balance, trades
