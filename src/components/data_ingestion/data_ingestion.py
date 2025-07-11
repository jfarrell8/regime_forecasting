import yfinance as yf
import sys
from pathlib import Path
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.logger.logger import logging
from src.exception.exception import RegimeForecastingException
from src.constant.training_pipeline import DATA_INGESTION_RAW_DATA_DIR

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise RegimeForecastingException(e, sys)
        
    def download_price_data(self):
        try:
            logging.info('Downloading price data from Yahoo...')

            # Download daily S&P 500 and VIX data
            symbols = ['^GSPC', '^VIX']
            data = yf.download(symbols, start="2015-01-01")['Close']
            data = data.dropna()

            # Calculate daily returns
            data['SP500_ret'] = data['^GSPC'].pct_change()
            data['VIX_change'] = data['^VIX'].pct_change()
            data = data.dropna()

            logging.info('Price data succesfully downloaded.')

            return data
        
        except Exception as e:
            raise RegimeForecastingException(e, sys)
    
    def save_data_to_local(self, data):
        try:
            logging.info('Saving raw data locally...')

            # current_file = Path(__file__).resolve()
            # project_root = current_file.parents[2] # up two levels to regimeforecasting/

            # # make path to regime_forecasting/data/raw
            # raw_data_dir = project_root / 'data' / DATA_INGESTION_RAW_DATA_DIR
            # raw_data_dir.mkdir(parents=True, exist_ok=True)

            # output_path = raw_data_dir / 'close_prices.csv'
            data.to_csv(self.data_ingestion_config.raw_file_path)

            logging.info('Raw data saved successfully!')
        
        except Exception as e:
            raise RegimeForecastingException(e, sys)

        
    def initiate_data_ingestion(self):
        try:
            data = self.download_price_data()
            self.save_data_to_local(data)
            dataingestionartifact = DataIngestionArtifact(rawdata_file_path=self.data_ingestion_config.raw_file_path)

            return dataingestionartifact

        except Exception as e:
            raise RegimeForecastingException(e, sys)