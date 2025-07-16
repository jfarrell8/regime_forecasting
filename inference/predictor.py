# inference/predictor.py

import os
import pandas as pd
import numpy as np
import traceback
from typing import Union, List
from sklearn.base import BaseEstimator

from src.utils.utils import load_object, get_classification_score
from src.exception.exception import RegimeForecastingException
from src.components.regime_model_forecasting.regime_model_forecasting import RegimeModelForecasting
from src.components.market_data_loader.market_data_loader import MarketDataLoader
from src.components.feature_engineering.feature_builder import FeatureBuilder

MODEL_PATH = os.getenv("MODEL_PATH", "final_model/model.pkl")


class RegimePredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model: BaseEstimator = load_object(model_path)
        self.clustering_builder = FeatureBuilder(mode='clustering')      # for shared inputs
        self.forecasting_builder = FeatureBuilder(mode='forecasting')    # for downstream features

    def load_market_data(self) -> pd.DataFrame:
        loader = MarketDataLoader(
            tickers=["^GSPC", "^VIX"],
            start="2015-01-01"
        )
        return loader.load()


    def preprocess(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        # 1. Generate intermediate features from "clustering" step
        base_features = self.clustering_builder.transform(raw_data)

        # 2. Build forecasting features on top
        full_features = self.forecasting_builder.transform(base_features)

        if full_features.empty:
            raise ValueError("Insufficient data to generate forecast features")
        
        return full_features

    def predict(self) -> int:
        try:
            raw_data = self.load_market_data()
            X = self.preprocess(raw_data)

            latest_row = X.iloc[[-1]]  # must be 2D for scikit-learn
            prediction = self.model.predict(latest_row)[0]
            
            return prediction

        except Exception as e:
            print("Error during prediction:", e)
            traceback.print_exc()
            return -1
