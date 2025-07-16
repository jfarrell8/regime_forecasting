from src.entity.config_entity import (TrainingPipelineConfig, 
                                      DataIngestionConfig, 
                                      FeatureEngineeringConfig, 
                                      ClusteringConfig, 
                                      RegimeModelForecastingConfig)
from src.components.data_ingestion.data_ingestion import DataIngestion
from src.components.feature_engineering.feature_engineering import FeatureEngineering
from src.components.clustering.clustering import Clustering
from src.components.regime_model_forecasting.regime_model_forecasting import RegimeModelForecasting
from src.logger.logger import logging
from src.exception.exception import RegimeForecastingException
import sys


if __name__=='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()

        ## DATA INGESTION
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)

        logging.info("Initiating the data ingestion...")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed!")

        ## FEATURE ENGINEERING

        featureengineeringconfig = FeatureEngineeringConfig(trainingpipelineconfig)
        feature_engineering = FeatureEngineering(feature_engineering_config=featureengineeringconfig,
                                                 data_ingestion_artifact=dataingestionartifact)

        logging.info("Initiating feature engineering...")
        featureengineering_artifact = feature_engineering.initiate_feature_engineering()
        logging.info("Feature engineering completed!")


        ## CLUSTERING TO ASSIGN REGIMES

        clusteringconfig = ClusteringConfig(trainingpipelineconfig)
        clustering = Clustering(clustering_config=clusteringconfig, 
                                feature_engineering_artifact=featureengineering_artifact)

        logging.info("Initiating clustering...")
        clustering_artifact = clustering.initiate_clustering()
        logging.info("Clustering regimes completed!")


        ## REGIME FORECASTING/MODEL TRAINING
        logging.info("Model Training started...")
        regime_model_forecasting_config = RegimeModelForecastingConfig(trainingpipelineconfig)
        regime_model_forecasting=RegimeModelForecasting(regime_model_forecasting_config=regime_model_forecasting_config,
                                             clustering_artifact=clustering_artifact)
        regime_model_forecasting_artifact = regime_model_forecasting.initiate_model_trainer()

        logging.info("Model training artifact created!")

    except Exception as e:
        raise RegimeForecastingException(e, sys)