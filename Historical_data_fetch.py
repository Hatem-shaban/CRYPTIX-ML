import ccxt
import pandas as pd
import talib
from datetime import datetime
import time

# Symbols you requested
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT",
           "SOL/USDT", "MATIC/USDT", "DOT/USDT", "ADA/USDT"]

exchange = ccxt.binance()
timeframe = '1h'

def fetch_full_ohlcv(symbol, timeframe):
    """Fetch full available OHLCV data for a symbol from Binance"""
    since = None  # Fetch from earliest possible
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        since = ohlcv[-1][0] + 1  # Continue from last timestamp
        time.sleep(exchange.rateLimit / 1000)  # Respect rate limit
        # Stop if last fetched chunk < 1000 (no more data)
        if len(ohlcv) < 1000:
            break
    return all_ohlcv

def calculate_indicators(df):
    df['price'] = df['close']
    df['rsi'] = talib.RSI(df['price'], timeperiod=14)
    macd, macd_signal, _ = talib.MACD(df['price'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_trend'] = ['BULLISH' if m > s else 'BEARISH' if m < s else 'NEUTRAL'
                        for m, s in zip(macd, macd_signal)]
    df['sma5'] = df['price'].rolling(5).mean()
    df['sma20'] = df['price'].rolling(20).mean()
    return df

# Collect and combine data
combined_df = pd.DataFrame()
for sym in symbols:
    print(f"Fetching {sym}...")
    ohlcv = fetch_full_ohlcv(sym, timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = calculate_indicators(df)
    df['symbol'] = sym.split('/')[0]
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Save to CSV
combined_df.dropna().to_csv("trade_history_combined.csv", index=False)
print("✅ trade_history_combined.csv saved with all symbols and indicators.")
