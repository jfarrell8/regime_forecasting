from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from src.exception.exception import RegimeForecastingException
from src.entity.artifact_entity import ClassificationMetricArtifact
from src.logger.logger import logging
import sys
import os
import pickle


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise RegimeForecastingException(e, sys)
    
def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise RegimeForecastingException(e, sys)


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
            
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)
        model_accuracy_score = accuracy_score(y_true, y_pred)

        classification_metric = ClassificationMetricArtifact(
                    f1_score=model_f1_score,
                    precision_score=model_precision_score, 
                    recall_score=model_recall_score,
                    accuracy_score=model_accuracy_score)
        return classification_metric
    
    except Exception as e:
        raise RegimeForecastingException(e,sys)