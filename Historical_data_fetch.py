import ccxt
import pandas as pd
import numpy as np
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

def calculate_comprehensive_indicators(df):
    """
    Calculate all technical indicators used in the ML model training
    This matches the indicators in train_ml_model.py
    """
    # Basic price column
    df['price'] = df['close']
    
    # RSI calculation
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['rsi'] = df['rsi'].fillna(50)
    
    # MACD calculation
    macd, macd_signal, macd_histogram = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_histogram'] = macd_histogram
    
    # MACD trend (numeric for ML)
    df['macd_trend'] = np.where(df['macd'] > df['macd_signal'], 1,
                       np.where(df['macd'] < df['macd_signal'], -1, 0))
    
    # SMA calculations
    df['sma5'] = talib.SMA(df['close'], timeperiod=5)
    df['sma20'] = talib.SMA(df['close'], timeperiod=20)
    df['sma50'] = talib.SMA(df['close'], timeperiod=50)
    df['sma100'] = talib.SMA(df['close'], timeperiod=100)
    
    # EMA calculations
    df['ema12'] = talib.EMA(df['close'], timeperiod=12)
    df['ema26'] = talib.EMA(df['close'], timeperiod=26)
    df['ema50'] = talib.EMA(df['close'], timeperiod=50)
    df['ema200'] = talib.EMA(df['close'], timeperiod=200)
    
    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Average True Range (ATR)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['atr'] = df['atr'].fillna(df['atr'].mean() if not df['atr'].isna().all() else 1.0)
    
    # Stochastic Oscillator
    df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'], 
                                              fastk_period=14, slowk_period=3, slowk_matype=0,
                                              slowd_period=3, slowd_matype=0)
    df['stoch_k'] = df['stoch_k'].fillna(50)
    df['stoch_d'] = df['stoch_d'].fillna(50)
    
    # VWAP (Volume Weighted Average Price)
    try:
        if 'volume' in df.columns and df['volume'].sum() > 0:
            # Typical price (HLC/3)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['cumulative_pv'] = (typical_price * df['volume']).cumsum()
            df['cumulative_volume'] = df['volume'].cumsum()
            # VWAP = Sum(Typical Price * Volume) / Sum(Volume)
            df['vwap'] = df['cumulative_pv'] / df['cumulative_volume']
            
            # Rolling window VWAP for better signals
            window = min(50, len(df))
            df['vwap_rolling'] = (typical_price * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
            df['vwap'] = df['vwap'].fillna(df['close'])
            df['vwap_rolling'] = df['vwap_rolling'].fillna(df['close'])
        else:
            df['vwap'] = df['close']
            df['vwap_rolling'] = df['close']
    except Exception:
        df['vwap'] = df['close']
        df['vwap_rolling'] = df['close']
    
    # ADX (Average Directional Index) and Directional Indicators
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Fill NaN values with neutral values
    df['adx'] = df['adx'].fillna(25)
    df['plus_di'] = df['plus_di'].fillna(25)
    df['minus_di'] = df['minus_di'].fillna(25)
    
    # Volatility (rolling standard deviation)
    df['volatility'] = df['close'].rolling(window=20).std()
    df['volatility'] = df['volatility'].fillna(df['volatility'].mean() if not df['volatility'].isna().all() else 0.01)
    
    # Fill any remaining NaN values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].ffill().bfill()
    
    return df

# Collect and combine data
combined_df = pd.DataFrame()
total_symbols = len(symbols)

for i, sym in enumerate(symbols):
    print(f"Fetching {sym}... ({i+1}/{total_symbols})")
    try:
        ohlcv = fetch_full_ohlcv(sym, timeframe)
        if not ohlcv:
            print(f"  ⚠️ No data found for {sym}")
            continue
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print(f"  📊 Calculating {sym} indicators...")
        df = calculate_comprehensive_indicators(df)
        df['symbol'] = sym.split('/')[0]
        
        # Add to combined dataframe
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        print(f"  ✅ {sym} completed - {len(df)} records added")
        
    except Exception as e:
        print(f"  ❌ Error fetching {sym}: {e}")
        continue

# Save to CSV with comprehensive data
print(f"\n📈 Processing complete! Total records: {len(combined_df)}")
print("🧹 Cleaning data and removing NaN values...")

# Remove rows with too many NaN values
cleaned_df = combined_df.dropna(subset=['close', 'volume', 'rsi', 'macd'])
print(f"📊 After cleaning: {len(cleaned_df)} records")

# Save to CSV in logs directory
import os
os.makedirs('logs', exist_ok=True)
output_file = "logs/trade_history_combined.csv"
cleaned_df.to_csv(output_file, index=False)

print(f"✅ {output_file} saved with comprehensive technical indicators!")
print(f"📋 Indicators included: RSI, MACD, SMA/EMA, Bollinger Bands, Stochastic, VWAP, ADX, ATR, and more")
print(f"🎯 Data ready for ML model training with {len(cleaned_df)} samples")
