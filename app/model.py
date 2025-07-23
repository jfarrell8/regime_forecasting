import joblib
import os

MODEL_PATH = "final_model/model.pkl"

class ModelService:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
    
    def predict(self, features: list[list[float]]) -> list[float]:
        return self.model.predict(features).tolist()