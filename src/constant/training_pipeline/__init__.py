import os

"""
defining common constant variable for training pipeline
"""
TARGET_COLUMN: str = "Result"
PIPELINE_NAME: str = "RegimeForecasting"
ARTIFACT_DIR: str = "artifacts"
DATA_FILE_NAME: str = "close_prices.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR = os.path.join("saved_models")
MODEL_FILE_NAME: str = "model.pkl"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR: str = "raw"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


"""
Feature Engineering related constants
"""

FEATURE_ENG_DIR_NAME: str = "feature_engineering"
FEATURE_ENG_DATA_DIR: str = "processed"


"""
Clustering related constants
"""

CLUSTERING_DIR_NAME: str = "clustering"
REGIMES_FILE_NAME: str = "regimes.csv"
REGIME_STATS_FILE_NAME: str = "regime_stats.csv"


"""
Regime model forecasting constants
"""

REGIME_MODEL_FORECASTING_DIR_NAME: str = "regime_model_forecasting"
REGIME_MODEL_FORECASTING_FILE_NAME: str = "regime_forecasting.csv"
NAIVE_BASELINE_FILE_NAME: str = "naive_forecasting.csv"
NAIVE_BASELINE_YTEST_FILE_NAME: str = "naive_ytest.csv"