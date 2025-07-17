import sys
import pandas as pd
from src.entity.config_entity import ClusteringConfig
from src.entity.artifact_entity import FeatureEngineeringArtifact, ClusteringArtifact
from src.logger.logger import logging
from src.exception.exception import RegimeForecastingException
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os


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

            # save best num of clusters
            with open(os.path.join(self.clustering_config.clustering_dir, 'num_clusters.txt'), "w") as f:
                f.write(str(n_clusters))

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            features['regime'] = kmeans.fit_predict(X_scaled)

            # combine with original data
            self.data = self.data.merge(features['regime'], left_index=True, right_index=True)

            # send clustering data to artifacts/
            self.data.to_csv(self.clustering_config.regimes_data_path, index=False)

            # compileand send regime stats to artifacts/
            regime_stats = self.data.groupby('regime').agg({
                'SP500_ret': ['mean', 'std'],
                'SP500_vol_21d': 'mean',
                '^VIX': 'mean'
            }).rename(columns={
                'SP500_ret': 'Return',
                'SP500_vol_21d': 'Volatility (21d)',
                '^VIX': 'VIX Level'
            })

            # Flatten multi-index columns
            regime_stats.columns = ['_'.join(col).strip() for col in regime_stats.columns.values]
            regime_stats = regime_stats.reset_index()

            # Rename columns for clarity
            regime_stats.columns = ['Regime', 'Mean Return', 'Return Std Dev', 'Mean Volatility (21d)', 'Mean VIX Level']

            regime_stats.to_csv(self.clustering_config.regimes_stats_data_path, index=False)
        
        except Exception as e:
            raise RegimeForecastingException(e, sys)
        
    def initiate_clustering(self):
        try:
            self.identify_regimes()
            clustering_artifact = ClusteringArtifact(clustering_file_path=self.clustering_config.regimes_data_path)

            return clustering_artifact
        
        except Exception as e:
            raise RegimeForecastingException(e, sys)