from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    rawdata_file_path: str

@dataclass
class FeatureEngineeringArtifact:
    feature_engineering_file_path: str

@dataclass
class ClusteringArtifact:
    clustering_file_path: str

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float
    accuracy_score: float
    
@dataclass
class RegimeModelForecastingArtifact:
    regime_forecasting_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact