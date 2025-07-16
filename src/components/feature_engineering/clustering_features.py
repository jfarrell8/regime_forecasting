import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ClusteringFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X['SP500_vol_21d'] = X['SP500_ret'].rolling(21).std()
        X['VIX_level'] = X['^VIX']

        return X.dropna()