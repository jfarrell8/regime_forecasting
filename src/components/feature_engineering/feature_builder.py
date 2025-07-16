from src.components.feature_engineering.clustering_features import ClusteringFeatureTransformer
from src.components.feature_engineering.forecasting_features import ForecastingFeatureTransformer
import pandas as pd

class FeatureBuilder:
    def __init__(self, mode="forecasting"):
        if mode == "clustering":
            self.transformer = ClusteringFeatureTransformer()
        elif mode == "forecasting":
            self.transformer = ForecastingFeatureTransformer()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.transformer.transform(data)
