import ta
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ForecastingFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X['ret_lag1'] = X['SP500_ret'].shift(1)
        X['vol_lag1'] = X['SP500_vol_21d'].shift(1)
        X['vix_lag1'] = X['^VIX'].shift(1)

        X['ret_lag2'] = X['SP500_ret'].shift(2)
        X['ret_5d'] = X['SP500_ret'].rolling(5).sum()
        X['vol_5d'] = X['SP500_ret'].rolling(5).std()

        X['zscore_ret'] = (X['SP500_ret'] - X['SP500_ret'].rolling(21).mean()) / X['SP500_ret'].rolling(21).std()

        X['rsi'] = ta.momentum.RSIIndicator(X['^GSPC'], window=14).rsi()

        macd = ta.trend.MACD(X['^GSPC'])
        X['macd_diff'] = macd.macd_diff()

        bb = ta.volatility.BollingerBands(X['^GSPC'], window=20, window_dev=2)
        X['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()

        return X.dropna()
