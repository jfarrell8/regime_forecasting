import sys
import pandas as pd
from src.entity.config_entity import ClusteringConfig
from src.entity.artifact_entity import FeatureEngineeringArtifact, ClusteringArtifact
from src.logger.logger import logging
from src.exception.exception import RegimeForecastingException
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class Clustering:
    def __init__(self, clustering_config: ClusteringConfig, 
                 feature_engineering_artifact: FeatureEngineeringArtifact):
        try:
            self.clustering_config = clustering_config
            self.feature_engineering_artifact = feature_engineering_artifact
            self.data = pd.read_csv(self.feature_engineering_artifact.feature_engineering_file_path)

        except Exception as e:
            raise RegimeForecastingException(e, sys)
    
    def select_best_k(self, X, k_range=range(2, 6)):
        scores = []
        for k in k_range:
            model = KMeans(n_clusters=k, random_state=42).fit(X)
            labels = model.labels_
            score = silhouette_score(X, labels)
            scores.append((k, score))
        
        best_k, best_score = max(scores, key=lambda x: x[1])
        
        return best_k, scores

    def identify_regimes(self):
        try:
            features = self.data[['SP500_ret', 'SP500_vol_21d', 'VIX_level']].dropna()

            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features)

            # cluster into n regimes
            n_clusters, _ = self.select_best_k(X=X_scaled)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            features['regime'] = kmeans.fit_predict(X_scaled)

            # combine with original data
            self.data = self.data.merge(features['regime'], left_index=True, right_index=True)

            self.data.to_csv(self.clustering_config.regimes_data_path)
        
        except Exception as e:
            raise RegimeForecastingException(e, sys)
        
    def initiate_clustering(self):
        try:
            self.identify_regimes()
            clustering_artifact = ClusteringArtifact(clustering_file_path=self.clustering_config.regimes_data_path)

            return clustering_artifact
        
        except Exception as e:
            raise RegimeForecastingException(e, sys)