import pandas as pd
import numpy as np
from ml_predictor import PriceTrendPredictor

# Path to your historical data CSV (adjust as needed)
data_path = 'logs/trade_history_combined.csv'  # Or your OHLCV data file

def load_data(path):
    df = pd.read_csv(path)
    # Example: Use last N rows, drop NaNs
    df = df.dropna()
    return df

def calculate_comprehensive_indicators(df):
    """
    Calculate all technical indicators used in the trading bot
    """
    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        # If standard OHLCV columns are missing, try to use price column
        if 'price' in df.columns:
            df['close'] = df['price']
            df['open'] = df['price']
            df['high'] = df['price'] * 1.002  # Small variation for high
            df['low'] = df['price'] * 0.998   # Small variation for low
            if 'volume' not in df.columns:
                df['volume'] = 1000  # Default volume
        else:
            print(f"Warning: Missing required columns {missing_cols}")
            return df
    
    # RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)
    
    # MACD calculation
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # MACD trend
    df['macd_trend'] = np.where(df['macd'] > df['macd_signal'], 1,
                       np.where(df['macd'] < df['macd_signal'], -1, 0))
    
    # SMA calculations
    df['sma5'] = df['close'].rolling(window=5).mean()
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()
    df['sma100'] = df['close'].rolling(window=100).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Average True Range (ATR)
    try:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(14).mean()
        df['atr'] = df['atr'].fillna(df['atr'].mean() if not df['atr'].isna().all() else 1.0)
    except Exception:
        df['atr'] = 1.0
    
    # EMA calculations (12, 26, 50, 200)
    df['ema12'] = df['close'].ewm(span=12).mean()
    df['ema26'] = df['close'].ewm(span=26).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    
    # Stochastic Oscillator
    try:
        lowest_low = df['low'].rolling(window=14).min()
        highest_high = df['high'].rolling(window=14).max()
        if not lowest_low.isna().all() and not highest_high.isna().all():
            df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
            # %D is a 3-period moving average of %K
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            # Fill NaN values
            df['stoch_k'] = df['stoch_k'].fillna(50)
            df['stoch_d'] = df['stoch_d'].fillna(50)
        else:
            df['stoch_k'] = 50
            df['stoch_d'] = 50
    except Exception:
        df['stoch_k'] = 50
        df['stoch_d'] = 50
    
    # VWAP (Volume Weighted Average Price)
    try:
        if 'volume' in df.columns and df['volume'].sum() > 0:
            # Typical price (HLC/3)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['cumulative_pv'] = (typical_price * df['volume']).cumsum()
            df['cumulative_volume'] = df['volume'].cumsum()
            # VWAP = Sum(Typical Price * Volume) / Sum(Volume)
            df['vwap'] = df['cumulative_pv'] / df['cumulative_volume']
            
            # For intraday VWAP, reset at start of each day
            # Using rolling window VWAP for better signals
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
    try:
        # True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
                          np.maximum(df['high'] - df['high'].shift(), 0), 0)
        minus_dm = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
                           np.maximum(df['low'].shift() - df['low'], 0), 0)
        
        # Smooth the values
        atr_period = 14
        tr_smooth = pd.Series(tr).rolling(atr_period).mean()
        plus_dm_smooth = pd.Series(plus_dm).rolling(atr_period).mean()
        minus_dm_smooth = pd.Series(minus_dm).rolling(atr_period).mean()
        
        # Directional Indicators
        df['plus_di'] = 100 * (plus_dm_smooth / tr_smooth)
        df['minus_di'] = 100 * (minus_dm_smooth / tr_smooth)
        
        # ADX calculation
        dx = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = dx.rolling(atr_period).mean()
        
        # Fill NaN values with neutral values
        df['plus_di'] = df['plus_di'].fillna(25)
        df['minus_di'] = df['minus_di'].fillna(25)
        df['adx'] = df['adx'].fillna(25)
    except Exception:
        df['plus_di'] = 25
        df['minus_di'] = 25
        df['adx'] = 25
    
    # Volatility (rolling standard deviation)
    df['volatility'] = df['close'].rolling(window=20).std()
    df['volatility'] = df['volatility'].fillna(df['volatility'].mean() if not df['volatility'].isna().all() else 0.01)
    
    return df

def add_trend_label(df, price_col='close', window=3):
    """
    Add trend labels based on future price movement
    1: upward trend, -1: downward trend, 0: sideways
    """
    df = df.copy()
    
    # Use 'close' if available, fallback to 'price'
    if price_col not in df.columns:
        if 'close' in df.columns:
            price_col = 'close'
        elif 'price' in df.columns:
            price_col = 'price'
        else:
            raise ValueError("No price column found. Expected 'close' or 'price'")
    
    df['future_price'] = df[price_col].shift(-window)
    
    # Calculate percentage change for more robust trend detection
    pct_change = (df['future_price'] - df[price_col]) / df[price_col] * 100
    
    # Define trend thresholds (adjust as needed)
    up_threshold = 0.5    # 0.5% increase
    down_threshold = -0.5  # 0.5% decrease
    
    df['trend'] = np.where(pct_change > up_threshold, 1,
                  np.where(pct_change < down_threshold, -1, 0))
    
    # Remove rows where we can't calculate future price
    df = df.dropna(subset=['future_price', 'trend'])
    
    return df

def main():
    print("Loading historical data...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} rows of data")
    
    print("Calculating comprehensive technical indicators...")
    df = calculate_comprehensive_indicators(df)
    
    print("Adding trend labels...")
    df = add_trend_label(df, price_col='close', window=3)
    
    # Define comprehensive feature columns including all new indicators
    feature_cols = [
        # Price and basic indicators
        'close', 'volume', 'volatility',
        # RSI and MACD
        'rsi', 'macd', 'macd_trend', 'macd_histogram',
        # Moving Averages
        'sma5', 'sma20', 'sma50', 'sma100',
        'ema12', 'ema26', 'ema50', 'ema200',
        # Bollinger Bands
        'bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position',
        # Stochastic Oscillator
        'stoch_k', 'stoch_d',
        # VWAP
        'vwap', 'vwap_rolling',
        # ADX and Directional Indicators
        'adx', 'plus_di', 'minus_di',
        # ATR
        'atr'
    ]
    
    # Filter to only include columns that exist in the dataframe
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
    
    print(f"Using {len(available_features)} features for training:")
    for feature in available_features:
        print(f"  - {feature}")
    
    # Convert categorical macd_trend to numeric if present
    if 'macd_trend' in available_features and df['macd_trend'].dtype == object:
        df['macd_trend'] = df['macd_trend'].map({'BULLISH': 1, 'BEARISH': -1, 'NEUTRAL': 0}).fillna(0)
    
    target_col = 'trend'
    
    # Remove rows with NaN values in features or target
    df_clean = df[available_features + [target_col]].dropna()
    print(f"After cleaning: {len(df_clean)} rows available for training")
    
    if len(df_clean) < 100:
        print("Warning: Very few samples available for training. Consider using more historical data.")
    
    print("Training ML model...")
    predictor = PriceTrendPredictor()
    score = predictor.train(df_clean, available_features, target_col)
    print(f"Model trained successfully!")
    print(f"Test accuracy: {score:.3f}")
    print(f"Model and scaler saved for use in trading bot.")
    
    # Print feature importance if available
    try:
        if hasattr(predictor.model, 'feature_importances_'):
            feature_importance = list(zip(available_features, predictor.model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            print("\nTop 10 Most Important Features:")
            for feature, importance in feature_importance[:10]:
                print(f"  {feature}: {importance:.4f}")
    except Exception as e:
        print(f"Could not display feature importance: {e}")
    
    # Print class distribution
    try:
        print(f"\nTarget variable distribution:")
        print(df_clean[target_col].value_counts().sort_index())
    except Exception:
        pass

if __name__ == '__main__':
    main()
