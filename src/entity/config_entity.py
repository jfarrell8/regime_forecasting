from datetime import datetime
import os
from src.constant import training_pipeline

class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name=training_pipeline.PIPELINE_NAME
        self.artifact_dir=training_pipeline.ARTIFACT_DIR
        # self.artifact_name=training_pipeline.ARTIFACT_DIR
        # self.artifact_dir=os.path.join(self.artifact_name, timestamp)
        self.model_dir=os.path.join("models")
        self.timestamp: str=timestamp

class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.training_pipeline_config = training_pipeline_config

        # ./artifacts/{timestamp}/data_ingestion/
        self.data_ingestion_dir:str = os.path.join(
            training_pipeline_config.artifact_dir,training_pipeline.DATA_INGESTION_DIR_NAME
        )
        os.makedirs(self.data_ingestion_dir, exist_ok=True)

        # ./artifacts/{timestamp}/data_ingestion/raw
        self.raw_ingestion_dir: str = os.path.join(self.data_ingestion_dir, training_pipeline.DATA_INGESTION_RAW_DATA_DIR)
        os.makedirs(self.raw_ingestion_dir, exist_ok=True)

        # ./artifacts/{timestamp}/data_ingestion/raw/close_prices.csv
        self.raw_file_path: str = os.path.join(self.raw_ingestion_dir, training_pipeline.DATA_FILE_NAME)


class FeatureEngineeringConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        
        self.training_pipeline_config = training_pipeline_config
        
        # ./artifacts/{timestamp}/feature_engineering/
        self.feature_engineering_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.FEATURE_ENG_DIR_NAME
        )
        os.makedirs(self.feature_engineering_dir, exist_ok=True)

        # ./artifacts/{timestamp}/feature_engineering/processed
        self.processed_feature_eng_dir: str = os.path.join(self.feature_engineering_dir, training_pipeline.FEATURE_ENG_DATA_DIR)
        os.makedirs(self.processed_feature_eng_dir, exist_ok=True)

        # ./artifacts/{timestamp}/feature_engineering/processed/close_prices.csv
        self.feature_eng_data_path: str = os.path.join(self.processed_feature_eng_dir, training_pipeline.DATA_FILE_NAME)


class ClusteringConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        
        self.training_pipeline_config = training_pipeline_config
        
        # ./artifacts/{timestamp}/clustering/
        self.clustering_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.CLUSTERING_DIR_NAME
        )
        os.makedirs(self.clustering_dir, exist_ok=True)

        # ./artifacts/{timestamp}/clustering/regimes.csv
        self.regimes_data_path: str = os.path.join(self.clustering_dir, training_pipeline.REGIMES_FILE_NAME)

        # ./artifacts/clustering/regime_stats.csv
        self.regimes_stats_data_path: str = os.path.join(self.clustering_dir, training_pipeline.REGIME_STATS_FILE_NAME)


class RegimeModelForecastingConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        
        self.training_pipeline_config = training_pipeline_config
        
        # ./artifacts/{timestamp}/regime_model_forecasting/
        self.regime_model_forecasting_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.REGIME_MODEL_FORECASTING_DIR_NAME
        )
        os.makedirs(self.regime_model_forecasting_dir, exist_ok=True)

        # ./artifacts/{timestamp}/regime_model_forecasting/regime_forecasting.csv
        self.regime_forecasting_data_path = os.path.join(self.regime_model_forecasting_dir, training_pipeline.REGIME_MODEL_FORECASTING_FILE_NAME)

        # ./artifacts/regime_model_forecasting/naive_baseline.csv
        self.naive_baseline_data_path = os.path.join(self.regime_model_forecasting_dir, training_pipeline.NAIVE_BASELINE_FILE_NAME)
        self.naive_baseline_ytest_data_path = os.path.join(self.regime_model_forecasting_dir, training_pipeline.NAIVE_BASELINE_YTEST_FILE_NAME)