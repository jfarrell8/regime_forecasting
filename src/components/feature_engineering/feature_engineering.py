import sys
import pandas as pd
from src.entity.config_entity import FeatureEngineeringConfig
from src.entity.artifact_entity import FeatureEngineeringArtifact, DataIngestionArtifact
from src.logger.logger import logging
from src.exception.exception import RegimeForecastingException
from src.constant.training_pipeline import FEATURE_ENG_DATA_DIR



class FeatureEngineering:
    def __init__(self, feature_engineering_config: FeatureEngineeringConfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.feature_engineering_config = feature_engineering_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            raise RegimeForecastingException(e, sys)
        
    def add_features(self, data):
        data['SP500_vol_21d'] = data['SP500_ret'].rolling(21).std()
        data['VIX_level'] = data['^VIX']
        data = data.dropna()

        return data
    
    def save_data_to_local(self, data):
        try:
            logging.info('Saving processed data locally...')

            # current_file = Path(__file__).resolve()
            # project_root = current_file.parents[2] # up two levels to regimeforecasting/

            # # make path to regimeforecasting/data/raw
            # raw_data_dir = project_root / 'data' / FEATURE_ENG_DATA_DIR
            # raw_data_dir.mkdir(parents=True, exist_ok=True)

            # output_path = raw_data_dir / 'close_prices.csv'
            data.to_csv(self.feature_engineering_config.feature_eng_data_path)

            logging.info('Processed data saved successfully!')
        
        except Exception as e:
            raise RegimeForecastingException(e, sys)
        
    def initiate_feature_engineering(self):
        try:
            data = pd.read_csv(self.data_ingestion_artifact.rawdata_file_path)
            data = self.add_features(data)
            self.save_data_to_local(data)
            feature_engineering_artifact = FeatureEngineeringArtifact(feature_engineering_file_path = self.feature_engineering_config.feature_eng_data_path)

            return feature_engineering_artifact

        except Exception as e:
            raise RegimeForecastingException(e, sys)