import yfinance as yf
import sys
from pathlib import Path
from regimeforecasting.entity.config_entity import DataIngestionConfig
from regimeforecasting.entity.artifact_entity import DataIngestionArtifact

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise RegimeForecastingException(e,sys)
        
    def initiate_data_ingestion(self):
        try:
            dataframe=self.export_collection_as_dataframe()
            dataframe=self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                        test_file_path=self.data_ingestion_config.testing_file_path)
            return dataingestionartifact

        except Exception as e:
            raise RegimeForecastingException(e, sys)


def main():
    # Download daily S&P 500 and VIX data
    symbols = ['^GSPC', '^VIX']
    data = yf.download(symbols, start="2015-01-01")['Close']
    data = data.dropna()

    # Calculate daily returns
    data['SP500_ret'] = data['^GSPC'].pct_change()
    data['VIX_change'] = data['^VIX'].pct_change()
    data = data.dropna()

    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2] # up two levels to regime_forecasting/

    # make path to regime_forecasting/data/raw
    raw_data_dir = project_root / 'data' / 'raw'
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    output_path = raw_data_dir / 'close_prices.csv'
    data.to_csv(output_path)

if __name__ == "__main__":
    sys.exit(main())