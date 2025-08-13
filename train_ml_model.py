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

def add_trend_label(df, price_col='price', window=3):
    # Simple trend: 1 if price increases after window, -1 if decreases, 0 otherwise
    df = df.copy()
    df['future_price'] = df[price_col].shift(-window)
    df['trend'] = np.where(df['future_price'] > df[price_col], 1,
                   np.where(df['future_price'] < df[price_col], -1, 0))
    df = df.dropna()
    return df

def main():
    df = load_data(data_path)
    df = add_trend_label(df, price_col='price', window=3)
    feature_cols = [col for col in ['price', 'rsi', 'macd', 'macd_trend', 'sma5', 'sma20', 'volume'] if col in df.columns]
    # Convert categorical macd_trend to numeric if present
    if 'macd_trend' in feature_cols and df['macd_trend'].dtype == object:
        df['macd_trend'] = df['macd_trend'].map({'BULLISH': 1, 'BEARISH': -1, 'NEUTRAL': 0}).fillna(0)
    target_col = 'trend'
    predictor = PriceTrendPredictor()
    score = predictor.train(df, feature_cols, target_col)
    print(f"Model trained. Test accuracy: {score:.3f}")
    print(f"Model and scaler saved for use in trading bot.")

if __name__ == '__main__':
    main()
