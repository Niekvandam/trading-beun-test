import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from binance.client import Client
from indicators import (
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_atr,
    calculate_stochastic_oscillator,
    calculate_ema,
    calculate_sma
)
from strategy import trading_strategy  # Assume this is the strategy provided earlier
import os

# Set up logging
logging.basicConfig(
    filename='results/paper_trade.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info("Starting paper trading bot...")

# Ensure the results directory exists
if not os.path.exists('results'):
    os.makedirs('results')

# Initialize Binance Client (no API key needed for public data)
client = Client()

# Parameters
params = {
    'timeframe': '2h',
    'starting_balance': 250,
    'rsi_threshold_low': 30,
    'rsi_threshold_high': 70,
    'stoch_threshold_low': 20,
    'stoch_threshold_high': 80,
    'risk_per_trade': 0.02,
    'stop_loss': 0.01,
    'take_profit': 0.05,
    'trailing_stop_loss': 0.005,
    'trailing_take_profit': 0.01,
    'threshold': 1.0,
    'broker_fee': 0.0002,
    'slippage': 0.00005,
    'hold_time_limit': 48,  # 48 intervals for 2-hour timeframe = 4 days
}

# Map timeframe to Binance interval
timeframe_mapping = {
    '1min': Client.KLINE_INTERVAL_1MINUTE,
    '5min': Client.KLINE_INTERVAL_5MINUTE,
    '15min': Client.KLINE_INTERVAL_15MINUTE,
    '30min': Client.KLINE_INTERVAL_30MINUTE,
    '1h': Client.KLINE_INTERVAL_1HOUR,
    '2h': Client.KLINE_INTERVAL_2HOUR,
    '4h': Client.KLINE_INTERVAL_4HOUR,
    '1d': Client.KLINE_INTERVAL_1DAY,
}

# Get Binance interval
binance_interval = timeframe_mapping.get(params['timeframe'])
if not binance_interval:
    logging.error(f"Timeframe {params['timeframe']} is not supported.")
    raise ValueError(f"Timeframe {params['timeframe']} is not supported.")

# Prepopulate historical data with the last two weeks
start_time = (datetime.utcnow() - timedelta(weeks=2)).strftime('%Y-%m-%d %H:%M:%S')
logging.info(f"Fetching OHLCV data for the last two weeks starting from {start_time}...")

def fetch_historical_data(symbol, interval, start_time):
    klines = client.get_historical_klines(symbol, interval, start_time)
    data = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    data = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return data

symbol = 'BTCUSDT'
historical_data = fetch_historical_data(symbol, binance_interval, start_time)
if historical_data.empty:
    logging.error("Failed to fetch historical data.")
    raise ValueError("No historical data fetched.")
logging.info(f"Preloaded {len(historical_data)} rows of historical data.")

# Indicators and trading variables
historical_data['sma_short'] = calculate_sma(historical_data['close'], 75)
historical_data['sma_long'] = calculate_sma(historical_data['close'], 140)
historical_data['ema'] = calculate_ema(historical_data['close'], 75)
historical_data['rsi'] = calculate_rsi(historical_data['close'], 14)
historical_data['upper_bb'], historical_data['lower_bb'] = calculate_bollinger_bands(historical_data['close'], 20, 2.5)
historical_data['macd_line'], historical_data['signal_line'] = calculate_macd(historical_data['close'], 12, 26, 9)
historical_data['atr'] = calculate_atr(historical_data['high'], historical_data['low'], historical_data['close'], 14)
historical_data['stoch_k'], historical_data['stoch_d'] = calculate_stochastic_oscillator(
    historical_data['close'], historical_data['high'], historical_data['low'], 14, 3
)

balance, trades = trading_strategy(
    historical_data['close'].values,
    historical_data['high'].values,
    historical_data['low'].values,
    historical_data['sma_short'].values,
    historical_data['sma_long'].values,
    historical_data['ema'].values,
    historical_data['rsi'].values,
    historical_data['stoch_k'].values,
    historical_data['stoch_d'].values,
    historical_data['upper_bb'].values,
    historical_data['lower_bb'].values,
    historical_data['macd_line'].values,
    historical_data['signal_line'].values,
    historical_data['atr'].values,
    params
)

# End-of-script metrics
net_profit = balance - params['starting_balance']
roi = (net_profit / params['starting_balance']) * 100

logging.info("=== End of Paper Trading Metrics ===")
logging.info(f"Final Balance: ${balance:.2f}")
logging.info(f"Net Profit: ${net_profit:.2f}")
logging.info(f"Return on Investment (ROI): {roi:.2f}%")
logging.info(f"Total Trades: {trades}")
