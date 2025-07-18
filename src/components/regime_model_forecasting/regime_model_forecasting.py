import sys
import pandas as pd
import numpy as np
from src.entity.config_entity import RegimeModelForecastingConfig
from src.entity.artifact_entity import ClusteringArtifact, RegimeModelForecastingArtifact
from src.logger.logger import logging
from src.exception.exception import RegimeForecastingException
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.utils.utils import get_classification_score, load_object, save_object
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import mlflow
from urllib.parse import urlparse
import ta
import os
from dotenv import load_dotenv
import time
import subprocess
import atexit
import socket
from sklearn.model_selection import ParameterSampler
from copy import deepcopy



class RegimeModelForecasting:
    def __init__(self, regime_model_forecasting_config: RegimeModelForecastingConfig, 
                 clustering_artifact: ClusteringArtifact):
        self.regime_model_forecasting_config = regime_model_forecasting_config
        self.clustering_artifact = clustering_artifact
        self.mlflow_proc = None

    @staticmethod
    def is_port_open(host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0

    def start_mlflow_server(self, port=5000):
        try:
            if not self.is_port_open("localhost", 5000):
                cmd = f"start cmd /k mlflow ui --backend-store-uri ./mlruns --port {port}"
                subprocess.Popen(cmd, shell=True)

            time.sleep(3)  # Give server time to start

        except Exception as e:
            raise RegimeForecastingException(e, sys)

    def shutdown_mlflow_server(self):
        if self.mlflow_proc:
            self.mlflow_proc.terminate()
            print("MLflow tracking server stopped.")

    def track_mlflow(self, model, metrics: dict, params: dict, model_name: str, register: bool = False):
        try:
            self.start_mlflow_server()
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("regimeforecasting")

            with mlflow.start_run(run_name=model_name):
                mlflow.log_params(params)
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)

                if register:
                    mlflow.sklearn.log_model(model, "model", registered_model_name="RegimeBestModel")
                else:
                    mlflow.sklearn.log_model(model, "model")

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

        data = data.dropna()

        return data    


    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params):
        try:
            report = {}
            best_f1 = -1
            best_model = None
            best_model_name = None
            best_params = None

            for model_name, model in models.items():
                param_grid = params[model_name]
                sampler = list(ParameterSampler(param_grid, n_iter=50, random_state=42))

                for i, sampled_params in enumerate(sampler):
                    model.set_params(**sampled_params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    f1 = f1_score(y_test, y_pred, average="macro")
                    precision = precision_score(y_test, y_pred, average="macro")
                    recall = recall_score(y_test, y_pred)
                    accuracy = accuracy_score(y_test, y_pred)

                    metrics = {
                        "f1_score": f1,
                        "precision": precision,
                        "recall": recall,
                        "accuracy": accuracy,
                    }

                    self.track_mlflow(model, metrics, sampled_params, f"{model_name}_trial_{i}", register=False)

                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = deepcopy(model)
                        best_model_name = model_name
                        best_params = sampled_params

                report[model_name] = best_f1

            return report, best_model_name, best_model, best_params

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

        params = {
            "XGBoost": {
                "n_estimators": [100, 200, 300],
                # "max_depth": [2, 3, 4, 5],
                # "learning_rate": [0.01, 0.05, 0.1],
                # "subsample": [0.6, 0.8, 1.0],
                # "colsample_bytree": [0.6, 0.8, 1.0],
                # "gamma": [0, 0.1, 1],
                # "reg_alpha": [0, 0.01, 0.1],
                # "reg_lambda": [0.5, 1, 2],
            },
            "Random Forest": {
                # 'criterion': ['gini', 'entropy', 'log_loss'],
                'max_features': ['sqrt', 'log2', None],
                # 'n_estimators': [8, 16, 32, 128, 256]
            },
            "Logistic Regression": {
                'logreg__multi_class': ['multinomial'],
                'logreg__solver': ['lbfgs'],
                'logreg__max_iter': [1000]
            }
        }

        model_report, best_model_name, best_model, best_params = self.evaluate_models(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            models=models, params=params
        )

        # Final training
        best_model.fit(X_train, y_train)
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        train_metric = get_classification_score(y_train, y_train_pred)
        test_metric = get_classification_score(y_test, y_test_pred)

        final_metrics = {
            "train_f1_score": train_metric.f1_score,
            "test_f1_score": test_metric.f1_score,
            "test_accuracy": test_metric.accuracy_score
        }

        # Track and register best model
        self.track_mlflow(best_model, final_metrics, best_params, f"{best_model_name}_final", register=True)

        # Save locally
        save_object(self.regime_model_forecasting_config.regime_forecasting_data_path, best_model)
        deployment_model_path = os.path.join("final_model", "model.pkl")
        os.makedirs(os.path.dirname(deployment_model_path), exist_ok=True)
        save_object(deployment_model_path, best_model)

        return RegimeModelForecastingArtifact(
            regime_forecasting_file_path=self.regime_model_forecasting_config.regime_forecasting_data_path,
            train_metric_artifact=train_metric,
            test_metric_artifact=test_metric
        )

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

            # get naive benchmark data
            y_naive = data['regime'].loc[y_test.index]  # today’s regime = tomorrow's prediction
            y_naive.to_csv(self.regime_model_forecasting_config.naive_baseline_data_path, index=False)

            y_test.to_csv(self.regime_model_forecasting_config.naive_baseline_ytest_data_path, index=False)

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)

            self.shutdown_mlflow_server()

            return model_trainer_artifact
            
        except Exception as e:
            raise RegimeForecastingException(e, sys)