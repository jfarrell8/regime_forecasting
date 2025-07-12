import sys
import pandas as pd
from src.entity.config_entity import RegimeModelForecastingConfig
from src.entity.artifact_entity import ClusteringArtifact, RegimeModelForecastingArtifact
from src.logger.logger import logging
from src.exception.exception import RegimeForecastingException


class RegimeModelForecasting:
    def __init__(self, regime_model_forecasting_config: RegimeModelForecastingConfig, 
                 clustering_artifact: ClusteringArtifact):
        self.regime_model_forecasting_config = regime_model_forecasting_config
        self.clustering_artifact = clustering_artifact

    