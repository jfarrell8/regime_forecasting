from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    rawdata_file_path: str

@dataclass
class FeatureEngineeringArtifact:
    feature_engineering_file_path: str

