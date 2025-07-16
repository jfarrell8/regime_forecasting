import sys
import pandas as pd
import numpy as np
from src.entity.config_entity import RegimeModelForecastingConfig
from src.entity.artifact_entity import ClusteringArtifact, RegimeModelForecastingArtifact
from src.logger.logger import logging
from src.exception.exception import RegimeForecastingException
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.utils.utils import get_classification_score, load_object, save_object
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import mlflow
from urllib.parse import urlparse
import dagshub
import ta
import os
from dotenv import load_dotenv


dagshub.init(repo_owner='jfarrell8',
             repo_name='regime_forecasting',
             mlflow=True)

class RegimeModelForecasting:
    def __init__(self, regime_model_forecasting_config: RegimeModelForecastingConfig, 
                 clustering_artifact: ClusteringArtifact):
        try:
            self.regime_model_forecasting_config = regime_model_forecasting_config
            self.clustering_artifact = clustering_artifact
        except Exception as e:
            raise RegimeForecastingException(e, sys)

    def track_mlflow(self, model_name: str, best_model, classificationmetric):
        """
        Logs metrics and model to MLflow using DagsHub integration.
        Assumes MLFLOW_TRACKING_URI, DAGSHUB_USERNAME, and DAGSHUB_TOKEN are defined in a `.env` file.
        """
        try:
            # Load environment variables from .env
            load_dotenv()

            # Set MLflow tracking URI and credentials for DagsHub
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

            os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
            os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run(run_name=model_name):
                # Log classification metrics
                mlflow.log_metric("f1_score", classificationmetric.f1_score)
                mlflow.log_metric("precision", classificationmetric.precision_score)
                mlflow.log_metric("recall", classificationmetric.recall_score)
                mlflow.log_metric("accuracy", classificationmetric.accuracy_score)

                # Log model parameters if available
                if hasattr(best_model, "get_params"):
                    mlflow.log_params(best_model.get_params())

                # Log the model and (if possible) register it
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        sk_model=best_model,
                        artifact_path="model",
                        registered_model_name=model_name
                    )
                else:
                    mlflow.sklearn.log_model(best_model, "model")

        except Exception as e:
            raise RegimeForecastingException(e, sys)

    def add_features(self, data):
        # 1-day lag
        data['ret_lag1'] = data['SP500_ret'].shift(1)
        data['vol_lag1'] = data['SP500_vol_21d'].shift(1)
        data['vix_lag1'] = data['^VIX'].shift(1)

        # 2-day lag
        data['ret_lag2'] = data['SP500_ret'].shift(2)

        # 5-day momentum
        data['ret_5d'] = data['SP500_ret'].rolling(5).sum()

        # 5-day rolling std
        data['vol_5d'] = data['SP500_ret'].rolling(5).std()

        # z-score (rolling normalization)
        data['zscore_ret'] = (data['SP500_ret'] - data['SP500_ret'].rolling(21).mean()) / data['SP500_ret'].rolling(21).std()
    
        # RSI (momentum)
        data['rsi'] = ta.momentum.RSIIndicator(data['^GSPC'], window=14).rsi()

        # MACD
        macd = ta.trend.MACD(data['^GSPC'])
        data['macd_diff'] = macd.macd_diff()

        # Bollinger bands
        bb = ta.volatility.BollingerBands(data['^GSPC'], window=20, window_dev=2)
        data['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()

        # current  regime
        data['regime_t'] = data['regime'].shift(1)

        data = data.dropna()

        return data
    
    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params):
        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                param_grid = params[list(models.keys())[i]]

                tscv = TimeSeriesSplit(n_splits=5)

                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=50,
                    scoring='f1_macro', # since we have class imbalance
                    cv=tscv,
                    verbose=1,
                    random_state=42,
                    n_jobs=-1
                )

                search.fit(X_train, y_train)

                model.set_params(**search.best_params_)
                
                y_train_pred = model.predict(X_train)

                y_test_pred = model.predict(X_test)

                train_model_score = f1_score(y_train, y_train_pred)

                test_model_score = f1_score(y_test, y_test_pred)

                report[list(models.keys())[i]] = test_model_score

            return report

        except Exception as e:
            raise RegimeForecastingException(e, sys)            

    def train_model(self, X_train, y_train, X_test, y_test):

        models = {
                "XGBoost": XGBClassifier(),
                "Random Forest": RandomForestClassifier(verbose=1),
                "Logistic Regression": Pipeline([
                    ('scaler', StandardScaler()),
                    ('logreg', LogisticRegression(verbose=1))
                ]),
            }
        
        params={
            "XGBoost": {
                "n_estimators": [100, 200, 300],
                "max_depth": [2, 3, 4, 5],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "gamma": [0, 0.1, 1],
                "reg_alpha": [0, 0.01, 0.1],
                "reg_lambda": [0.5, 1, 2],
            },
            "Random Forest":{
                'criterion':['gini', 'entropy', 'log_loss'],
                'max_features':['sqrt','log2', None],
                'n_estimators': [8, 16, 32, 128, 256]
            },
            "Logistic Regression": {
                'logreg__multi_class': ['multinomial'],
                'logreg__solver': ['lbfgs'],
                'logreg__max_iter': [1000]
            }
        }


        model_report:dict = self.evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, 
                                                 y_test=y_test, models=models, params=params)
        
        ## To get best model score from dict
        best_model_score = max(sorted(model_report.values()))

        ## To get best model name from dict
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        y_train_pred = best_model.predict(X_train)

        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        
        ## Track the experiements with mlflow
        self.track_mlflow(best_model, classification_train_metric, best_model_name)


        y_test_pred = best_model.predict(X_test)
        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)

        self.track_mlflow(best_model, classification_test_metric, best_model_name)

        # Save training artifact (for traceability)
        save_object(self.regime_model_forecasting_config.regime_forecasting_data_path, best_model)

        # Save production-ready model for inference
        deployment_model_path = os.path.join("final_model", "model.pkl")
        os.makedirs(os.path.dirname(deployment_model_path), exist_ok=True)
        save_object(deployment_model_path, best_model)


        ## Model Trainer Artifact
        model_trainer_artifact = RegimeModelForecastingArtifact(
                            trained_model_file_path=self.regime_model_forecasting_config.regime_forecasting_data_path,
                            train_metric_artifact=classification_train_metric,
                            test_metric_artifact=classification_test_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")

        return model_trainer_artifact

    def initiate_model_trainer(self)->RegimeModelForecastingArtifact:
        try:
            data = pd.read_csv(self.clustering_artifact.clustering_file_path, index_col='Date')

            data['next_regime'] = data['regime'].shift(-1)
            data = data.dropna(subset=['next_regime'])  # Drop last row

            data = self.add_features(data)
            X = data.drop(['regime', 'next_regime'], axis=1)
            y = data['next_regime']

            split_date = '2020-01-01'
            X_train, X_test = X[X.index < split_date], X[X.index >= split_date]
            y_train, y_test = y[y.index < split_date], y[y.index >= split_date]

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)

            return model_trainer_artifact
            
        except Exception as e:
            raise RegimeForecastingException(e, sys)