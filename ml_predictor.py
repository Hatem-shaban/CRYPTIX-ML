import numpy as np
import pandas as pd

# Minimal import guard to avoid crashing if sklearn is missing
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None
    StandardScaler = None
    joblib = None

class PriceTrendPredictor:
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.model_path = model_path or 'rf_price_trend_model.pkl'
        self.scaler_path = 'rf_scaler.pkl'
        self._load_model()

    def _load_model(self):
        if not SKLEARN_AVAILABLE:
            self.model = None
            self.scaler = None
            return
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        except Exception:
            self.model = None
            self.scaler = None

    def train(self, df, feature_cols, target_col):
        if not SKLEARN_AVAILABLE:
            return 0.0
        X = df[feature_cols].values
        y = df[target_col].values
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        return self.model.score(X_test, y_test)

    def predict(self, df, feature_cols):
        if not SKLEARN_AVAILABLE or self.model is None or self.scaler is None:
            return None
        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, df, feature_cols):
        if not SKLEARN_AVAILABLE or self.model is None or self.scaler is None:
            return None
        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
