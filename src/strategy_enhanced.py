from numba import njit
import numpy as np

@njit
def trading_strategy_enhanced(
    close, high, low,
    sma_short, sma_long, ema, rsi, stoch_k, stoch_d,
    upper_bb, lower_bb, macd_line, signal_line, atr,
    starting_balance, rsi_threshold_low, rsi_threshold_high,
    stoch_threshold_low, stoch_threshold_high, risk_per_trade,
    stop_loss_pct, take_profit_pct, trailing_stop_loss_pct,
    trailing_take_profit_pct, threshold, broker_fee, slippage,
    hold_time_limit, atr_period, allow_short, trend_filter,
    dynamic_sizing_factor,
    higher_tf_ema,
    initial_condition_flags
):

    balance = starting_balance
    position = 0  # 0: no position, 1: long, -1: short
    position_size = 0.0
    entry_price = 0.0
    stop_loss_price = 0.0
    take_profit_price = 0.0
    entry_index = 0
    trades = 0
    
    # Track recent P/L for dynamic sizing
    recent_pnl = 0.0
    trade_count = 0
    
    n = len(close)
    for i in range(1, n):
        # Determine market trend from a higher timeframe EMA (e.g., bullish if close > higher_tf_ema)
        bullish_trend = close[i] > higher_tf_ema[i] if trend_filter else True
        bearish_trend = close[i] < higher_tf_ema[i] if trend_filter else True
        
        # Compute condition score for long
        condition_score_long = 0.0
        if sma_short[i - 1] < sma_long[i - 1] and sma_short[i] >= sma_long[i]:
            condition_score_long += 1.5
        if rsi[i] < rsi_threshold_low:
            condition_score_long += 1.0
        if stoch_k[i] < stoch_threshold_low and stoch_d[i] < stoch_threshold_low:
            condition_score_long += 1.0
        if close[i] < lower_bb[i]:
            condition_score_long += 0.5
        if macd_line[i - 1] < signal_line[i - 1] and macd_line[i] >= signal_line[i]:
            condition_score_long += 1.0
        if close[i] > ema[i]:
            condition_score_long += 0.5
        
        # Compute condition score for short
        condition_score_short = 0.0
        if sma_short[i - 1] > sma_long[i - 1] and sma_short[i] <= sma_long[i]:
            condition_score_short += 1.5
        if rsi[i] > rsi_threshold_high:
            condition_score_short += 1.0
        if stoch_k[i] > stoch_threshold_high and stoch_d[i] > stoch_threshold_high:
            condition_score_short += 1.0
        if close[i] > upper_bb[i]:
            condition_score_short += 0.5
        if macd_line[i - 1] > signal_line[i - 1] and macd_line[i] <= signal_line[i]:
            condition_score_short += 1.0
        if close[i] < ema[i]:
            condition_score_short += 0.5
        
        # Dynamic position sizing adjustment based on recent performance
        # If recent pnl > 0, slightly increase position size risk_per_trade, else decrease
        adjusted_risk_per_trade = risk_per_trade * (1 + dynamic_sizing_factor * (recent_pnl / (trade_count+1e-9)))
        adjusted_risk_per_trade = max(min(adjusted_risk_per_trade, risk_per_trade * 2), risk_per_trade * 0.5)
        
        if position == 0:
            # Entry Logic
            # Go long if conditions met and trend is bullish
            if condition_score_long >= threshold and bullish_trend:
                atr_value = atr[i]
                if np.isnan(atr_value) or atr_value == 0:
                    continue
                entry_price = close[i] * (1 + slippage)
                stop_loss_distance = atr_value * stop_loss_pct
                take_profit_distance = atr_value * take_profit_pct
                position_size = (balance * adjusted_risk_per_trade) / stop_loss_distance
                max_position_size = balance / entry_price
                position_size = min(position_size, max_position_size)
                stop_loss_price = entry_price - stop_loss_distance
                take_profit_price = entry_price + take_profit_distance
                total_cost = position_size * entry_price
                balance -= total_cost
                balance -= total_cost * broker_fee
                position = 1
                entry_index = i
                trades += 1
                
                # Store initial conditions that caused entry
                initial_condition_flags = (condition_score_long, condition_score_short)
            
            # Go short if conditions met and allowed and trend is bearish
            elif allow_short and condition_score_short >= threshold and bearish_trend:
                atr_value = atr[i]
                if np.isnan(atr_value) or atr_value == 0:
                    continue
                entry_price = close[i] * (1 - slippage)
                stop_loss_distance = atr_value * stop_loss_pct
                take_profit_distance = atr_value * take_profit_pct
                position_size = (balance * adjusted_risk_per_trade) / stop_loss_distance
                max_position_size = balance / entry_price
                position_size = min(position_size, max_position_size)
                stop_loss_price = entry_price + stop_loss_distance
                take_profit_price = entry_price - take_profit_distance
                total_cost = position_size * entry_price
                balance += total_cost # Because entering short effectively gives cash
                balance -= total_cost * broker_fee
                position = -1
                entry_index = i
                trades += 1
                
                # Store initial conditions that caused entry
                initial_condition_flags = (condition_score_long, condition_score_short)
        
        else:
            # Manage open position
            current_price = close[i]
            atr_value = atr[i]
            if np.isnan(atr_value) or atr_value == 0:
                atr_value = (stop_loss_price - take_profit_price) / 2  # fallback
            
            # Update trailing stops
            if position == 1:
                new_stop_loss_price = current_price - (atr_value * trailing_stop_loss_pct)
                if new_stop_loss_price > stop_loss_price:
                    stop_loss_price = new_stop_loss_price
                new_take_profit_price = current_price + (atr_value * trailing_take_profit_pct)
                if new_take_profit_price > take_profit_price:
                    take_profit_price = new_take_profit_price
            else:  # position == -1
                new_stop_loss_price = current_price + (atr_value * trailing_stop_loss_pct)
                if new_stop_loss_price < stop_loss_price:
                    stop_loss_price = new_stop_loss_price
                new_take_profit_price = current_price - (atr_value * trailing_take_profit_pct)
                if new_take_profit_price < take_profit_price:
                    take_profit_price = new_take_profit_price
            
            # Early exit if initial conditions are no longer valid
            # Example: if we went long but now condition_score_long < threshold
            if position == 1 and condition_score_long < threshold:
                exit_price = current_price * (1 - slippage)
                pnl = (exit_price - entry_price) * position_size
                balance += position_size * exit_price
                balance -= position_size * exit_price * broker_fee
                position = 0
                recent_pnl += pnl
                trade_count += 1
            
            elif position == -1 and condition_score_short < threshold:
                exit_price = current_price * (1 + slippage)
                pnl = (entry_price - exit_price) * position_size
                balance -= position_size * exit_price
                balance -= position_size * exit_price * broker_fee
                position = 0
                recent_pnl += pnl
                trade_count += 1
            
            # Exit logic: Time limit, stop loss, take profit
            exit_reason = 0
            if hold_time_limit != -1 and (i - entry_index) >= hold_time_limit:
                exit_reason = 1
            elif position == 1:
                if low[i] <= stop_loss_price:
                    exit_reason = 2
                    exit_price = stop_loss_price
                elif high[i] >= take_profit_price:
                    exit_reason = 3
                    exit_price = take_profit_price
            else: # position == -1
                if high[i] >= stop_loss_price:
                    exit_reason = 2
                    exit_price = stop_loss_price
                elif low[i] <= take_profit_price:
                    exit_reason = 3
                    exit_price = take_profit_price
            
            if exit_reason > 0:
                if position == 1:
                    exit_price *= (1 - slippage)
                    pnl = (exit_price - entry_price) * position_size
                    balance += position_size * exit_price
                    balance -= position_size * exit_price * broker_fee
                else:
                    exit_price *= (1 + slippage)
                    pnl = (entry_price - exit_price) * position_size
                    balance -= position_size * exit_price
                    balance -= position_size * exit_price * broker_fee
                position = 0
                recent_pnl += pnl
                trade_count += 1
    
    return balance, trades
